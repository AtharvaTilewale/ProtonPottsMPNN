"""Labeller AUPR — out-of-fold precision–recall of the FLAML protonation classifiers.

Two figures (one per residue class), each recreating the manuscript PR plot: FLAML on the named GEOMETRIC
features vs on the ENCODER embedding (two curves, best-MCC dot on each), the base-rate floor, and the
nested geometric-RULE points (HBOND only → + salt bridge / buried → + charge network). Nothing is fitted —
everything comes from the shipped per-residue bundle `models/pr_bundle_<res>.npz`
(out-of-fold probs + truth + the few rule columns).

Run:  python labeller/pr_curve.py     ->  labeller/his_pr.png, labeller/acid_pr.png
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, matthews_corrcoef

HERE = Path(__file__).resolve().parent
C_FEAT, C_ENC = "#4153A4", "#189486"                      # geometric (blue) / encoder (teal)
C_V3, C_V4, C_EV5 = "#828080", "#FF477B", "#a3002e"       # HBOND only (grey) / salt-bridge|buried / charge network
MUTED, EDGE = "#555555", "white"


def rules(res, b):
    """The nested geometric rules per residue, as boolean masks over the bundle arrays (41/56)."""
    if res == "HIS":
        core = (b["n_donor_atoms"] >= 2) & (b["n_acceptor_atoms"] == 0)   # His donates from both ring N
        ion = b["d_carbox"] < 2.8                                          # carboxylate salt bridge / ion pair
        return {"HBOND only": core,
                "HBOND + salt bridge": core | ion,
                "HBOND + charge network": (core | ion) & (b["phi_min_6"] < 0)}
    hbond = (b["n_donor_atoms"] >= 1) & (b["best_donor_dha"] >= 140)       # carboxyl OH donates a good H-bond
    buried = hbond & (b["dens12"] > 250)
    return {"HBOND only": hbond,
            "HBOND + buried": buried,
            "HBOND + charge network": buried & (b["d_dyad"] < 2.8) & (b["net_q8"] < 0)}


def one_plot(res, title, out):
    b = dict(np.load(HERE / "models" / f"pr_bundle_{res}.npz"))
    y = b["y"]; base = float(y.mean())
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    curves, points = [], []
    for name, p, col in [("FLAML: geometric", b["feature_oof"], C_FEAT),
                         ("FLAML: encoder", b["encoder_oof"], C_ENC)]:
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ln, = ax.plot(rec, prec, lw=2.2, color=col, zorder=5, label=f"{name}  (AP {ap:.2f})")
        # best-MCC operating point (swept over the upper quantiles of p)
        cuts = np.unique(np.quantile(p, np.linspace(0.5 if res == "HIS" else 0.90, 0.9995, 80)))
        _, bt = max((matthews_corrcoef(y, p >= t), t) for t in cuts)
        pm = p >= bt
        ax.plot(pm[y == 1].sum() / max(y.sum(), 1), y[pm].sum() / max(pm.sum(), 1), "o",
                ms=10, color=col, mec=EDGE, mew=1.8, zorder=7)
        curves.append(ln)
    for nm, col in zip(rules(res, b), (C_V3, C_V4, C_EV5)):
        mask = rules(res, b)[nm]
        n, tp = int(mask.sum()), int(y[mask].sum())
        if n == 0:
            continue
        st, = ax.plot(tp / y.sum(), tp / n, "*", ms=20, color=col, mec=EDGE, mew=1.4,
                      zorder=6, label=nm)
        points.append(st)
    bl = ax.axhline(base, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2, label=f"Base rate  {100*base:.1f}%")
    ax.set_xlabel("Recall", fontsize=13); ax.set_ylabel("Precision", fontsize=13)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.25)); ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_xticklabels([f"{int(100*t)}%" for t in np.arange(0, 1.01, 0.25)])
    ax.set_yticklabels([f"{int(100*t)}%" for t in np.arange(0, 1.01, 0.25)])
    ax.set_title(title, fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=curves + [bl] + points, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=2, frameon=False, fontsize=12, handletextpad=0.6, columnspacing=1.4,
              labelspacing=0.6, borderaxespad=0)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    one_plot("HIS", "HIS — protonation labeller", HERE / "his_pr.png")
    one_plot("acid", "ASP / GLU — protonation labeller", HERE / "acid_pr.png")
