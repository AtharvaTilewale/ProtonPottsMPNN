# %%
# GLOBAL binder/non-binder AP per design class, using SIZE-NORMALIZED scores (so they work as
# target-agnostic classifiers — no per-target size threshold, all labeled designs pooled within a class).
# Classes: Boltzgen VHH (nano) / Boltzgen MB (prot) / metaanalysis / bindcraft. Same oracle everywhere:
# ipTM from the (templated) Boltz-2 folds. Reads data/crossbinding/mb_gbind_benchmark.parquet
# (mb_gbind_score.py --aggregate). Manuscript figure -> figures/summary_global_ap.png
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

PR = Path(__file__).resolve().parent
PARQUET = PR / "data/mb_gbind_benchmark.parquet"
OUT = PR / "results/summary_global_ap.png"
RNG = np.random.default_rng(0)
N_BOOT = 2000

# score column -> (label, colour). All oriented so HIGHER = more likely a binder, all size-normalized/invariant.
SCORES = {                                                        # manuscript palette
    "neg_E_complex_per_res":     ("Potts-H / length", "#189486"),   # teal-green (Potts = highlight)
    "iptm":                      ("ipTM (Boltz-2)",   "#4153A4"),   # blue
    "mpnn_ll_mean":              ("MPNN logP",        "#828080"),   # grey
}


def class_frame(df):
    """Return [(label, subframe)] for the 4 design classes."""
    bg = df[df.source == "boltzgen"]
    return [
        ("Boltzgen VHH", bg[bg.get("binder_type") == "nano"]),
        ("Boltzgen MB",  bg[bg.get("binder_type") == "prot"]),
        ("metaanalysis", df[df.source == "metaanalysis"]),
        ("bindcraft",    df[df.source == "bindcraft"]),
    ]


def ap_ci(y, s):
    """Global AP + bootstrap 95% CI over pooled designs."""
    y = np.asarray(y, int); s = np.asarray(s, float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if len(y) < 4 or y.min() == y.max():
        return np.nan, np.nan, np.nan, len(y)
    ap = average_precision_score(y, s)
    boot = []
    for _ in range(N_BOOT):
        i = RNG.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            boot.append(average_precision_score(y[i], s[i]))
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
    return ap, lo, hi, len(y)


# %% load + compute
df = pd.read_parquet(PARQUET)
df = df.dropna(subset=["y"]).copy(); df["y"] = df["y"].astype(int)
classes = class_frame(df)

records = {}
print(f"{'class':<14}{'prev':>7}{'n':>6}   " + "  ".join(f"{v[0]:>16}" for v in SCORES.values()))
for lbl, sub in classes:
    prev = sub["y"].mean(); rec = {"prev": prev, "n": len(sub)}
    cells = []
    for col, (name, _) in SCORES.items():
        ap, lo, hi, n = ap_ci(sub["y"], sub[col]) if col in sub else (np.nan, np.nan, np.nan, 0)
        rec[col] = (ap, lo, hi)
        cells.append(f"{ap:>16.2f}" if ap == ap else f"{'—':>16}")
    records[lbl] = rec
    print(f"{lbl:<14}{prev:>7.2f}{len(sub):>6}   " + "  ".join(cells))

# %% grouped bar plot
labels = [l for l, _ in classes]
x = np.arange(len(labels)); w = 0.8 / len(SCORES)
fig, ax = plt.subplots(figsize=(7.8, 6.2))
for i, (col, (name, color)) in enumerate(SCORES.items()):
    ys = [records[l][col][0] for l in labels]
    los = [records[l][col][0] - records[l][col][1] for l in labels]
    his = [records[l][col][2] - records[l][col][0] for l in labels]
    xpos = x + (i - (len(SCORES) - 1) / 2) * w
    bars = ax.bar(xpos, ys, w, yerr=[los, his], capsize=2.5, color=color, edgecolor="k",
                  linewidth=0.6, label=name, error_kw=dict(lw=1))
    ax.bar_label(bars, fmt="%.2f", fontsize=13, padding=3, rotation=90)   # vertical: never clash side-by-side
for j, l in enumerate(labels):                                   # per-class prevalence (random-AP floor)
    ax.plot([x[j] - 0.42, x[j] + 0.42], [records[l]["prev"]] * 2, "k--", lw=1.1)
ax.plot([], [], "k--", lw=1.1, label="prevalence (random AP)")
ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n(n={records[l]['n']})" for l in labels], fontsize=16,
                   rotation=90, ha="center", va="top")
ax.tick_params(axis="y", labelsize=15)
ax.set_ylabel("Average Precision (macro)", fontsize=17)
ax.set_ylim(0, 1.0)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=15, ncol=1, loc="upper left", frameon=False, handlelength=1.8, handletextpad=0.6)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("\nsaved ->", OUT)
