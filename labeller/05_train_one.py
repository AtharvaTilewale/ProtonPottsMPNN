"""EV5 stage 53 — train ONE model (residue x feature-set) and persist everything. Built for SLURM.

    python 53_train_one.py <HIS|ACID> <features|encoder> [budget_seconds]

Same design for all four models, so they are directly comparable:

    NESTED out-of-fold — for each of 5 folds (grouped by PDB, stratified on the label):
        FLAML tunes AND fits on the other 4 folds  <- the held-out PDBs are invisible even to the
                                                      hyper-parameter search
        predict this fold
    then ONE model fitted on ALL the data — that is the deployable one.

Every acid/His therefore carries a prediction from a model that never saw its protein, and the AP we
report is honest. The 5 fold models ARE what produced that number, so they are kept too.

Saved into  sandbox/ev5/automl_{features|encoder}_{HIS|acid}/ :
    fold1..5.pkl   the models behind the out-of-fold predictions
    full.pkl       the deployable model + feature schema + metadata + the prevalence WARNING
    oof.npy        the out-of-fold probability for every residue
    metrics.json   AP / AUC / MCC / best threshold, all measured OUT-OF-FOLD

Base rates differ enormously — HIS 34%, ACID 1.6% — so AP must always be read against the base rate.
"""
import json
import pickle
import sys
import time
import warnings
import os
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent           # this stage's own folder — figures/CSVs land here
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)            # the shared parquet inputs, built once by stages 00/12/13/35/38/43/44/47/49
MODELS = HERE / "models"; MODELS.mkdir(exist_ok=True)        # the trained AutoML artefacts
KEY = ["pdb", "chain", "res_id"]
SEED = 42

RES = sys.argv[1].upper() if len(sys.argv) > 1 else "HIS"
KIND = sys.argv[2].lower() if len(sys.argv) > 2 else "features"
BUDGET = int(sys.argv[3]) if len(sys.argv) > 3 else 1200
assert RES in ("HIS", "ACID") and KIND in ("features", "encoder")

from flaml import AutoML
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (average_precision_score, roc_auc_score, matthews_corrcoef,
                             precision_recall_curve)

SUFFIX = "HIS" if RES == "HIS" else "acid"
OUT = MODELS / f"automl_{'feature' if KIND == 'features' else 'encoder'}_{SUFFIX}"
OUT.mkdir(exist_ok=True)

# ── data ────────────────────────────────────────────────────────────────────────────────
_rich = "his_rich.parquet" if RES == "HIS" else "acid_rich.parquet"
R = pd.read_parquet(DATA / _rich)
drop = KEY + ["truth", "y", "d_metal"] + ([] if RES == "HIS" else ["res_name"])
# the encoder-embedding feature set is the analysis-only branch and is NOT shipped; only load it for KIND=encoder
if KIND == "encoder":
    E = pd.read_parquet(DATA / ("his_encoder.parquet" if RES == "HIS" else "acid_encoder.parquet"))
    M = R.merge(E.drop(columns=["truth", "y"]), on=KEY, how="inner")
else:
    M = R
y = M.y.values
groups = M.pdb.values
COLS = ([c for c in R.columns if c not in drop] if KIND == "features"
        else [c for c in E.columns if c not in KEY + ["truth", "y"]])
CATS = [c for c in COLS if M[c].dtype == object]

X = M[COLS].copy()
for c in CATS:
    X[c] = X[c].astype("category")

base = y.mean()
npos_pdb = int((M.groupby("pdb").y.max() > 0).sum())
print(f"{RES} · {KIND}", flush=True)
print(f"  {len(M)} residues · {M.pdb.nunique()} PDBs · {int(y.sum())} positive "
      f"({100*base:.2f}%) · {len(COLS)} features ({len(CATS)} categorical)", flush=True)
print(f"  EFFECTIVE SAMPLE SIZE: {int(y.sum())} positives in {npos_pdb} PDBs", flush=True)
print(f"  FLAML budget: {BUDGET}s per fit × 6 fits (5 folds + 1 full)  ≈ {6*BUDGET/3600:.1f} h\n",
      flush=True)

FIT = dict(task="classification", metric="ap", eval_method="cv", n_splits=5, split_type="group",
           time_budget=BUDGET, early_stop=True,
           estimator_list=["lgbm", "xgboost", "xgb_limitdepth", "rf", "extra_tree"],
           seed=SEED, verbose=0)

# ── nested out-of-fold ──────────────────────────────────────────────────────────────────
t0 = time.time()
oof = np.zeros(len(M))
picks = []
for k, (a, b) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
                           .split(X, y, groups)):
    am = AutoML()
    am.fit(X_train=X.iloc[a], y_train=y[a], groups=groups[a],
           log_file_name=str(OUT / f"flaml_fold{k+1}.log"), **FIT)
    oof[b] = am.predict_proba(X.iloc[b])[:, 1]
    picks.append(am.best_estimator)
    with open(OUT / f"fold{k+1}.pkl", "wb") as fh:
        pickle.dump(dict(model=am, estimator=am.model.estimator, fold=k + 1,
                         train_idx=a, test_idx=b, best_estimator=am.best_estimator,
                         best_config=am.best_config, feature_columns=list(COLS),
                         categorical_columns=CATS), fh)
    print(f"  fold {k+1}/5  {am.best_estimator:<14} ({int(y[b].sum())} positives held out)  "
          f"[{(time.time()-t0)/60:.0f} min]", flush=True)

np.save(OUT / "oof.npy", oof)          # cheap, and it IS the result — write it before the pickles
ap = float(average_precision_score(y, oof))
auc = float(roc_auc_score(y, oof))
cuts = np.unique(np.quantile(oof, np.linspace(0.5 if RES == "HIS" else 0.90, 0.9995, 90)))
mcc, thr = max((matthews_corrcoef(y, oof >= t), float(t)) for t in cuts)
pm = oof >= thr
n_lab, n_right = int(pm.sum()), int(y[pm].sum())
print(f"\n  OUT-OF-FOLD:  AP {ap:.3f} ({ap/base:.0f}× base) · AUC {auc:.3f} · "
      f"MCC {mcc:.3f} @ P≥{thr:.2f}", flush=True)
print(f"                {n_lab} labels, {n_right} right → {100*n_right/max(n_lab,1):.0f}% precision, "
      f"{100*n_right/y.sum():.0f}% recall", flush=True)

# ── the deployable model ────────────────────────────────────────────────────────────────
full = AutoML()
full.fit(X_train=X, y_train=y, groups=groups, log_file_name=str(OUT / "flaml_full.log"), **FIT)
print(f"\n  full-data model: {full.best_estimator}  [{(time.time()-t0)/60:.0f} min total]", flush=True)

WARN = ("Thresholds were chosen at the NEUTRON base rate. The neutron set is ENRICHED — for HIS the "
        "real PDB prevalence is ~6% vs 34% here (~5x). This model WILL over-call if the threshold is "
        "ported unchanged. Recalibrate the prior/threshold to the target prevalence first.")

with open(OUT / "full.pkl", "wb") as fh:
    pickle.dump(dict(
        model=full, estimator=full.model.estimator,
        residue=RES, kind=KIND,
        feature_columns=list(COLS),          # EXACT order — reindex any new data to this
        categorical_columns=CATS,
        best_estimator=full.best_estimator, best_config=full.best_config,
        oof_ap=ap, oof_ap_lift=ap / base, oof_auc=auc, oof_mcc=mcc, oof_threshold=thr,
        trained_base_rate=float(base),
        real_world_prevalence=(0.06 if RES == "HIS" else None),
        n_train=len(M), n_positive=int(y.sum()), n_pdbs=int(M.pdb.nunique()),
        n_positive_pdbs=npos_pdb, flaml_budget_s=BUDGET, seed=SEED,
        WARNING=WARN,
    ), fh)
np.save(OUT / "oof.npy", oof)
json.dump(dict(residue=RES, kind=KIND, n=len(M), n_positive=int(y.sum()),
               base_rate=float(base), n_features=len(COLS),
               oof_ap=ap, oof_ap_lift=ap / base, oof_auc=auc, oof_mcc=mcc, oof_threshold=thr,
               oof_labels=n_lab, oof_right=n_right,
               oof_precision=n_right / max(n_lab, 1), oof_recall=n_right / float(y.sum()),
               fold_estimators=picks, full_estimator=full.best_estimator,
               flaml_budget_s=BUDGET, minutes=round((time.time() - t0) / 60, 1)),
          open(OUT / "metrics.json", "w"), indent=2)
print(f"\nsaved → {OUT}/  (fold1-5.pkl, full.pkl, oof.npy, metrics.json)", flush=True)
