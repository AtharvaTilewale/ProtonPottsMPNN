# %%
# ─────────────────────────────────────────────────────────────────────────────
# G_bind ON-TARGET binder/non-binder BENCHMARK — PER-TARGET performance vs the other predicted scores.
#
# Reads data/crossbinding/mb_gbind_benchmark.parquet (sandbox/mb_gbind_score.py --aggregate): per
# intended-target complex, v6 Potts −G_bind + the experimental binder/non-binder label (y) + a panel of
# other predicted scores (AF3/Boltz1/Colab ipTM, pLDDT, ipSAE, LIS, dockQ, and the Rosetta interface ΔG).
# Every score is oriented so HIGHER = more likely a binder.
#
# EVALUATION IS PER TARGET, NOT GLOBAL, and the headline metric is AVERAGE PRECISION (AUPR). G_bind
# carries a large per-target offset (it scales with the interface/target size), so a pooled AP/ROC over
# heterogeneous targets is dominated by between-target variation and is NOT a fair comparison. Every
# number here is computed WITHIN a single target and reported per target (no macro / global rollup).
# Two model scores are benchmarked: −G_bind (Potts binding energy) and the MPNN decoder log-likelihood.
# Run cell-by-cell in the dpo_potts venv.
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score

pd.set_option("display.max_columns", None); pd.set_option("display.width", 250)
PROJECT_ROOT = Path("/novo/users/cpjb/PHD/diffusion_DPO_gefion")
GBIND_PARQUET = PROJECT_ROOT / "data/crossbinding/mb_gbind_benchmark.parquet"
PLOT_DIR = PROJECT_ROOT / "sandbox/gbind/figures"; PLOT_DIR.mkdir(parents=True, exist_ok=True)

MIN_N, MIN_POS = 15, 3          # a target is evaluable only with enough designs and enough binders

CANDIDATE_SCORES = {
    "neg_g_bind": "−G_bind", "neg_g_bind_per_binder_res": "−G_bind/res",
    "mpnn_ll_mean": "MPNN logP", "mpnn_ll_sum": "MPNN logP(sum)",
    "iptm": "ipTM(fold)", "af3_iptm": "AF3 ipTM", "boltz1_iptm": "Boltz1 ipTM", "colab_iptm": "Colab ipTM",
    "colab_actifptm": "Colab actifpTM", "af3_dockQ": "AF3 dockQ",
    "af2_plddt_binder": "AF2 pLDDT", "boltz1_plddt": "Boltz1 pLDDT",
    "af3_ipSAE_max": "AF3 ipSAE", "boltz1_ipSAE_max": "Boltz1 ipSAE", "af3_LIS": "AF3 LIS",
    "neg_af3_rosetta_dG": "−AF3 Rosetta ΔG", "neg_boltz1_rosetta_dG": "−Boltz1 Rosetta ΔG",
    "neg_af2_rosetta_dG": "−AF2 Rosetta ΔG", "neg_input_rosetta_dG": "−input Rosetta ΔG",
}


def _save(fig, name):
    fig.savefig(PLOT_DIR / name, dpi=140, bbox_inches="tight"); plt.show(); print("saved ->", PLOT_DIR / name)


def _metrics(d, score, y="y"):
    d = d.dropna(subset=[score, y])
    n_pos = int(d[y].sum())
    if d[y].nunique() < 2 or len(d) < MIN_N or n_pos < MIN_POS or (len(d) - n_pos) < MIN_POS:
        return dict(AP=np.nan, ROC=np.nan, n=len(d), n_pos=n_pos)
    return dict(AP=average_precision_score(d[y], d[score]), ROC=roc_auc_score(d[y], d[score]),
                n=len(d), n_pos=n_pos)


# %% load
df = pd.read_parquet(GBIND_PARQUET)
df = df.dropna(subset=["y"]).copy(); df["y"] = df["y"].astype(int)
SCORES = {k: v for k, v in CANDIDATE_SCORES.items() if k in df.columns}
print(f"{len(df)} labelled complexes | {df.target_id.nunique()} targets | scores: {list(SCORES.values())}")


# %% PER-TARGET ROC/AP for every score  (this is the evaluation — no global pooling)
rows = []
for (src, tgt), g in df.groupby(["source", "target_id"]):
    prev = g["y"].mean()
    for sc, lbl in SCORES.items():
        m = _metrics(g, sc)
        rows.append(dict(source=src, target=tgt, score=lbl, ROC=m["ROC"], AP=m["AP"],
                         prevalence=round(prev, 3), AP_lift=(m["AP"] / prev if prev and m["AP"] == m["AP"] else np.nan),
                         n=m["n"], n_pos=m["n_pos"]))
M = pd.DataFrame(rows)
M.to_csv(PLOT_DIR / "mb_gbind_per_target_metrics.csv", index=False)
ev = M.dropna(subset=["ROC"])
print(f"evaluable targets (n≥{MIN_N}, ≥{MIN_POS} per class): "
      f"{ev.groupby('source')['target'].nunique().to_dict()}")


# %% PER-TARGET table: targets × scores (AVERAGE PRECISION), per source.
# AP is the headline metric (the classes are imbalanced: prevalence 0.10–0.24). NOTE AP's random
# baseline IS the target's prevalence, so `prev` is printed alongside; AP_lift = AP/prevalence makes
# values comparable ACROSS targets (1.0 = no better than random for that target).
PREV = ev.drop_duplicates(["source", "target"]).set_index(["source", "target"])["prevalence"]
for src in sorted(ev.source.unique()):
    piv = ev[ev.source == src].pivot_table(index="target", columns="score", values="AP")
    piv = piv[piv.mean().sort_values(ascending=False).index]
    piv.insert(0, "prev", PREV.loc[src].reindex(piv.index))
    print(f"\n=== {src}: AVERAGE PRECISION PER TARGET (rows=target, cols=score; prev = random baseline) ===")
    print(piv.round(3).to_string())


# %% PER-TARGET verdict: for EACH target — its prevalence, the best score, −G_bind's AUPR, its lift
# over the target's random baseline, and its RANK among the scores FOR THAT TARGET. No macro/global
# aggregation: every row is a self-contained per-target result.
GB = "−G_bind"
for src in sorted(ev.source.unique()):
    ap = ev[ev.source == src].pivot_table(index="target", columns="score", values="AP")
    rows = []
    for t, r in ap.iterrows():
        r = r.dropna()
        rank = int(r.rank(ascending=False)[GB]) if GB in r else np.nan
        rows.append(dict(target=t, n=int(ev[(ev.source == src) & (ev.target == t)]["n"].iloc[0]),
                         prev=PREV.loc[src, t], best_score=r.idxmax(), best_AP=r.max(),
                         gbind_AP=r.get(GB, np.nan),
                         gbind_lift=(r.get(GB, np.nan) / PREV.loc[src, t]),
                         gbind_rank=f"{rank}/{len(r)}"))
    print(f"\n=== {src}: PER-TARGET AUPR verdict (each row independent; lift = AP / prevalence) ===")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


# %% plot — −G_bind vs the best baselines, target by target (AP, with the prevalence floor).
# This is the ONLY figure the benchmark emits (per source); the full per-target numbers for every
# score live in mb_gbind_per_target_metrics.csv.
FOCUS = ["−G_bind", "MPNN logP", "ipTM(fold)", "−AF3 Rosetta ΔG"]
SPLIT_BY_TYPE = {"boltzgen"}          # boltzgen mixes nano/prot scaffolds → split (see below)
for src in sorted(s for s in ev.source.unique() if s not in SPLIT_BY_TYPE):
    e = ev[(ev.source == src) & (ev.score.isin(FOCUS))]
    piv = e.pivot_table(index="target", columns="score", values="AP")
    cols = [c for c in FOCUS if c in piv.columns]
    if not cols or piv.empty:
        continue
    piv = piv[cols].sort_values(cols[0], ascending=False)
    prev = PREV.loc[src].reindex(piv.index)
    x = np.arange(len(piv)); w = 0.8 / len(cols)
    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(piv) + 3), 5))
    for i, c in enumerate(cols):
        ax.bar(x + (i - (len(cols) - 1) / 2) * w, piv[c].values, w, label=c)
    ax.plot(x, prev.values, "k--", lw=1.2, marker="_", label="prevalence (random AP)")
    ax.set_xticks(x); ax.set_xticklabels(piv.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Average Precision (within target)")
    ax.set_title(f"{src}: per-target AP — G_bind vs baselines")
    ax.legend(fontsize=8); sns.despine(fig); fig.tight_layout()
    _save(fig, f"per_target_gbind_vs_baselines_AP_{src}.png")


# %% boltzgen SPLIT BY BINDER TYPE (nano vs prot).
# boltzgen mixes two scaffold classes and MPNN logP carries a large scaffold offset (nanobody
# frameworks score ~1.5–2.5 log-units above de-novo mini-proteins), so a target that mixes types
# confounds it: logP then partly ranks TYPE rather than binding. Splitting each target by type removes
# that. Cells are small, so each tick shows n and the number of binders (+n_pos) — read thin cells with care.
MIN_N_T, MIN_POS_T = 12, 2
for src in sorted(SPLIT_BY_TYPE & set(df.source.unique())):
    b = df[(df.source == src) & df.get("binder_type").notna()].copy()
    rows = []
    for (t, bt), g in b.groupby(["target_id", "binder_type"]):
        if len(g) < MIN_N_T or g.y.nunique() < 2 or g.y.sum() < MIN_POS_T:
            continue
        for sc, lbl in SCORES.items():
            d2 = g.dropna(subset=[sc])
            if d2.y.nunique() < 2:
                continue
            rows.append(dict(target=t, type=bt, score=lbl, n=len(d2), n_pos=int(d2.y.sum()),
                             prev=d2.y.mean(), AP=average_precision_score(d2.y, d2[sc])))
    B = pd.DataFrame(rows)
    if B.empty:
        continue
    B.to_csv(PLOT_DIR / f"per_target_by_type_metrics_{src}.csv", index=False)
    types = sorted(B.type.unique())
    fig, axes = plt.subplots(1, len(types), figsize=(7.5 * len(types), 5.2), squeeze=False)
    for ax, bt in zip(axes[0], types):
        sub = B[(B.type == bt) & (B.score.isin(FOCUS))]
        piv = sub.pivot_table(index="target", columns="score", values="AP")
        cols = [c for c in FOCUS if c in piv.columns]
        if not cols or piv.empty:
            continue
        piv = piv[cols].sort_values(cols[0], ascending=False)
        info = sub.drop_duplicates("target").set_index("target")[["prev", "n", "n_pos"]].reindex(piv.index)
        x = np.arange(len(piv)); w = 0.8 / len(cols)
        for i, c in enumerate(cols):
            ax.bar(x + (i - (len(cols) - 1) / 2) * w, piv[c].values, w, label=c)
        ax.plot(x, info["prev"].values, "k--", lw=1.2, marker="_", label="prevalence (random AP)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t}\nn={int(info.loc[t,'n'])}, +{int(info.loc[t,'n_pos'])}" for t in piv.index],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1.02); ax.set_ylabel("Average Precision (within target × type)")
        ax.set_title(f"{src} — {bt}"); ax.legend(fontsize=8)
    fig.suptitle(f"{src}: per-target AP split by binder type", y=1.01)
    sns.despine(fig); fig.tight_layout()
    _save(fig, f"per_target_gbind_vs_baselines_AP_{src}.png")


# %% SUMMARY — predictive performance across the 3 design classes (Boltzgen VHH / MB / metaanalysis).
# Collapses the per-target AP into a mean-per-target AP per predictor: −G_bind (the binding score) vs the two
# benchmarks common to all classes — the fold model's ipTM and the MPNN decoder logP. Boltzgen is split into
# VHH (nano) and MB (prot), removing the scaffold confound (see the by-type cell). Error bars = SEM across
# targets; the dashed line per class = mean prevalence (the random-AP floor). Reuses the per-target tables
# computed above: `ev` (pooled per-source) for metaanalysis, `B` (boltzgen split by binder type) for VHH/MB.
FOCUS3 = ["−G_bind", "ipTM(fold)", "MPNN logP"]
SUM_COLORS = {"−G_bind": "#c44e52", "ipTM(fold)": "#4c72b0", "MPNN logP": "#8172b3"}


def _class_summary(d, prevcol):
    d = d[d.score.isin(FOCUS3)].dropna(subset=["AP"])
    g = d.groupby("score").agg(AP=("AP", "mean"), SEM=("AP", "sem"))
    return g, d.drop_duplicates("target")[prevcol].mean(), d.target.nunique()


CLASSES = [
    ("Boltzgen VHH", *_class_summary(B[B.type == "nano"], "prev")),
    ("Boltzgen MB",  *_class_summary(B[B.type == "prot"], "prev")),
    ("metaanalysis", *_class_summary(ev[ev.source == "metaanalysis"], "prevalence")),
]
x = np.arange(len(CLASSES)); w = 0.8 / len(FOCUS3)
fig, ax = plt.subplots(figsize=(4.6, 4.4))
for i, sc in enumerate(FOCUS3):
    ys = [g["AP"].get(sc, np.nan) for _, g, _, _ in CLASSES]
    es = [g["SEM"].get(sc, np.nan) for _, g, _, _ in CLASSES]

    bars = ax.bar(x + (i - (len(FOCUS3) - 1) / 2) * w, ys, w, yerr=es, capsize=3,
                  color=SUM_COLORS[sc], edgecolor="k", label=sc)
    ax.bar_label(bars, fmt="%.2f", fontsize=6, padding=2)
for j, (lbl, g, prev, n) in enumerate(CLASSES):
    ax.plot([x[j] - 0.42, x[j] + 0.42], [prev] * 2, "k--", lw=1.0)   # per-class prevalence floor
ax.plot([], [], "k--", lw=1.0, label="prevalence")
ax.set_xticks(x); ax.set_xticklabels([f"{lbl}\n(n={n})" for lbl, _, _, n in CLASSES], fontsize=8)
ax.set_ylabel("mean per-target Average Precision"); ax.set_ylim(0, None)
ax.set_title("Binder / non-binder AP —\nG_bind vs benchmarks", fontsize=10)
ax.legend(fontsize=7); sns.despine(fig); fig.tight_layout()
_save(fig, "summary_binding_predictive_performance.png")
# plt.show()
# %%