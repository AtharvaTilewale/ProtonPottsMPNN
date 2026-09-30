# %% [markdown]
# Within-backbone predictive comparison for INVERSE-FOLDED-ONLY sequences (strategy='mpnn'):
#   Potts total-complex energy   vs   ProteinMPNN sequence score (sequence_decoded_prob_score)
# as predictors of RF3 fold outcomes (ipTM, binder pLDDT, scRMSD). Correlations are computed STRICTLY per
# backbone (origin_pdb_path) then aggregated. Both predictors are scored on the SAME folds (those carrying a
# ProteinMPNN score) for a fair head-to-head.
#
# Signs are RAW (no metric/predictor negation — reader knows the directions):
#   Potts energy LOWER=better ; ProteinMPNN prob HIGHER=better ; ipTM/pLDDT HIGHER=better ; scRMSD LOWER=better.
#   => GOOD Potts:       ρ<0 vs ipTM/pLDDT,  ρ>0 vs scRMSD
#      GOOD ProteinMPNN: ρ>0 vs ipTM/pLDDT,  ρ<0 vs scRMSD   (opposite to Potts by construction)

# %% config + load
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LIB = Path(__file__).resolve().parent / "data"
K = ["origin_pdb_path", "sequence"]
STRATEGY = "mpnn"          # inverse-folded only (non-optimized)
MIN_PER_BB = 5
MPNN_COL = "sequence_decoded_prob_score"
OUT_PNG = Path(__file__).resolve().parent / "results" / "within_backbone_inverse_outcomes.png"

fold = pd.read_parquet(LIB / "egfr_pdl1_folded.parquet")[K + ["rf3_iptm", "rf3_binder_plddt", "strategy"]]
resc = pd.read_parquet(LIB / "egfr_pdl1_rescored.parquet")[K + ["potts_energy"]]
scr = pd.read_parquet(LIB / "egfr_pdl1_scrmsd.parquet")[K + ["rf3_scrmsd"]]
df = fold.merge(resc, on=K, how="inner").merge(scr, on=K, how="inner")
df = df[df["strategy"] == STRATEGY].copy()

# ProteinMPNN score — already saved; pool the sweep-scores + curated tables and dedup on (backbone, sequence)
mpnn = pd.concat([pd.read_parquet(LIB / f)[K + [MPNN_COL]]
                  for f in ["optimized_sweep_scores.parquet", "curated_designs.parquet"]], ignore_index=True)
mpnn[MPNN_COL] = pd.to_numeric(mpnn[MPNN_COL], errors="coerce")
mpnn = mpnn.dropna(subset=[MPNN_COL]).drop_duplicates(K)
df = df.merge(mpnn, on=K, how="left")

for c in ["rf3_iptm", "rf3_binder_plddt", "rf3_scrmsd", "potts_energy", MPNN_COL]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
n_all = len(df)
df = df.dropna(subset=["rf3_iptm", "rf3_binder_plddt", "rf3_scrmsd", "potts_energy", MPNN_COL])   # fair shared subset
print(f"inverse-folded folds: {n_all}   with ProteinMPNN score (shared subset): {len(df)}   "
      f"backbones: {df['origin_pdb_path'].nunique()}")

PREDICTORS = [("potts_energy", "Potts total energy", "#0072B2"),
              (MPNN_COL,       "ProteinMPNN score",  "#D55E00")]
OUTCOMES = [("rf3_iptm",         "ipTM"),
            ("rf3_binder_plddt", "binder pLDDT"),
            ("rf3_scrmsd",       "scRMSD")]

# %% per-backbone WITHIN correlations (both predictors, same folds)
rows = []
for bb, g in df.groupby("origin_pdb_path"):
    if len(g) < MIN_PER_BB:
        continue
    rec = {"backbone": bb, "n": len(g)}
    for pc, _, _ in PREDICTORS:
        for oc, _ in OUTCOMES:
            rec[f"{pc}__{oc}"] = spearmanr(g[pc], g[oc]).correlation
    rows.append(rec)
res = pd.DataFrame(rows)
print(f"backbones with ≥{MIN_PER_BB} folds in the shared subset: {len(res)}\n")
hdr = f"{'predictor vs outcome':<40}{'median ρ':>10}"
print(hdr); print("-" * len(hdr))
for pc, plab, _ in PREDICTORS:
    for oc, olab in OUTCOMES:
        r = res[f"{pc}__{oc}"].dropna()
        print(f"{plab + ' vs ' + olab.replace(chr(10), ' '):<40}{r.median():>+10.3f}")

# %% plot — 3 outcome panels, 2 predictor BOXES each (no violin)
# FIGSIZE in one place so the figure can be resized without hunting through the plotting code. No
# suptitle: the panel titles + y-axis label already say what this is, and dropping it hands the freed
# strip back to the axes instead of leaving a blank band at the top.
FIGSIZE = (9, 12)          # canvas only — the panel shape is set by BOX_ASPECT below
# Per-panel HEIGHT:width. This, not FIGSIZE, decides how narrow the figure looks: three panels
# sit side by side, so the WHOLE image is ~3x wider than one panel. 4/3 gives 4:3 panels but a
# ~2.1:1 landscape figure; ~4.0 would be needed for the FIGURE itself to read 4:3 portrait.
BOX_ASPECT = 3.2
# type scale — kept together so the figure can be re-sized/re-scaled in one place
FS_TICK, FS_LABEL, FS_TITLE = 18, 22, 22
fig, axes = plt.subplots(1, len(OUTCOMES), figsize=FIGSIZE, sharey=True)
pos = np.arange(len(PREDICTORS))
cols = [c for _, _, c in PREDICTORS]
for ax, (oc, olab) in zip(axes, OUTCOMES):
    data = [res[f"{pc}__{oc}"].dropna().to_numpy() for pc, _, _ in PREDICTORS]
    ax.axhline(0, color="#888", lw=1, zorder=1)
    bp = ax.boxplot(data, positions=pos, widths=0.55, showfliers=False, patch_artist=True,
                    medianprops=dict(color="k", lw=2.2), whiskerprops=dict(color="#444"),
                    capprops=dict(color="#444"))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.85); patch.set_edgecolor("k")
    for i, (arr, c) in enumerate(zip(data, cols)):
        x = np.random.default_rng(0).normal(i, 0.07, len(arr))
        ax.scatter(x, arr, s=7, color=c, alpha=0.28, zorder=3, edgecolor="none")
    ax.set_xticks(pos)
    ax.set_xticklabels(["Potts\nenergy", "Protein\nMPNN"], fontsize=FS_TICK)
    ax.set_title(olab, fontsize=FS_TITLE, fontweight="semibold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_box_aspect(BOX_ASPECT)    # per-panel HEIGHT:width. 4/3 = 4:3 portrait panels;
                                     # THIS is what sets the shape, not FIGSIZE: savefig uses
                                     # bbox_inches="tight", which crops the canvas to the axes.
axes[0].set_ylabel("within-backbone Spearman ρ", fontsize=FS_LABEL)
axes[0].tick_params(axis="y", labelsize=FS_TICK)
fig.tight_layout()
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
print("\nsaved ->", OUT_PNG)
