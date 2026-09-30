"""Placement propensity by structural class — where the pipeline places each protonation state.

For every backbone in the library, the top-1 placement of each state's SELECTIVITY scan
(argmin of score_<state>, lower = preferred) and of the state-agnostic ΔE / IMPORTANCE scan
(argmax of imp_de) is assigned a structural class (interface / core / surface). We plot the
FRACTION OF BACKBONES whose top-1 lands in each class — the manuscript placement figure.

Reads the precomputed per-position scan `data/placement_scan.parquet` (823 backbones); **no model
needed**. The scan itself (engine over each backbone) is in `placement_reference/` — see README.
The *per-design* placement scan (the engine's own scan during a design run) is plotted by the design
notebook instead — see `inference/design_placement_scan.py`.

Run:  python benchmarks/placement_by_class.py     ->  benchmarks/placement_by_class.png
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CLASSES = ["interface", "core", "surface"]
OBS_STATES = ["HIS-P", "ASP-P", "GLU-P"]                       # the library only pins PROTONATED centres
STATE_COL = {"HIS-P": "#4153A4", "ASP-P": "#a3002e", "GLU-P": "#189486"}   # manuscript palette


def placement_fractions(df):
    """{series -> Series[class] = fraction of backbones}: the 3 states' selectivity scan + the ΔE scan."""
    frac = {st: df.loc[df.groupby("origin_pdb_path")[f"score_{st}"].idxmin(), "cls"]
                  .value_counts(normalize=True).reindex(CLASSES).fillna(0) for st in OBS_STATES}
    frac["importance"] = (df.loc[df.groupby("origin_pdb_path").imp_de.idxmax(), "cls"]
                          .value_counts(normalize=True).reindex(CLASSES).fillna(0))
    return frac


def plot_placement_by_class(df, ax=None):
    """Grouped bar: 3 selectivity states (coloured) + ΔE scan (grey), x = structural class."""
    frac = placement_fractions(df)
    series = [(st, frac[st], STATE_COL[st]) for st in OBS_STATES] + [("importance", frac["importance"], "#828080")]
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CLASSES)); bw = 0.20
    for i, (lab, f, c) in enumerate(series):
        off = (i - (len(series) - 1) / 2) * bw
        ax.bar(x + off, f.reindex(CLASSES).values, bw, color=c, edgecolor="white", linewidth=0.8,
               label=(r"$\Delta E$ scan" if lab == "importance" else f"{lab} (selectivity)"))
    ax.set_xticks(x); ax.set_xticklabels([c.capitalize() for c in CLASSES])
    ax.set_ylabel("Fraction of backbones (top-1 placement)"); ax.set_ylim(0, 1)
    ax.grid(axis="y", color="0.92", lw=0.8); ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=4, frameon=False, columnspacing=1.3)
    return ax


if __name__ == "__main__":
    df = pd.read_parquet(HERE / "data" / "placement_scan.parquet")
    ax = plot_placement_by_class(df)
    out = HERE / "placement_by_class.png"
    ax.figure.tight_layout(); ax.figure.savefig(out, dpi=300, bbox_inches="tight")
    print(f"{df.origin_pdb_path.nunique()} backbones -> {out}")
