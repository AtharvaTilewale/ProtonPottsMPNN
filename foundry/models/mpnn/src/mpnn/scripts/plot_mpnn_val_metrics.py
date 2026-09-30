"""Plot validation metrics from a training run's val_metrics CSV.

Usage
-----
    python -m mpnn.scripts.plot_mpnn_val_metrics --model_dir MODEL_DIR

Outputs are written to MODEL_DIR/plot_mpnn_val_metrics/ by default,
or to a custom location with --output_dir.
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

WINDOW = 10  # rolling-average window (epochs)

sns.set_theme(style="ticks", context="paper", font_scale=1.15)
PALETTE = sns.color_palette("colorblind")


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_tensor_scalar(s):
    """Extract float from strings like \"tensor(1.23, device='cuda:0')\"."""
    if pd.isna(s):
        return float("nan")
    m = re.search(r"tensor\(([\d.eE+\-]+)", str(s))
    return float(m.group(1)) if m else float("nan")


def plot_with_rolling(ax, x, y, color, window=WINDOW):
    y = pd.Series(y.values if hasattr(y, "values") else y)
    ax.plot(x, y, color=color, alpha=0.7, linewidth=0.8, zorder=1)
    roll = y.rolling(window, center=True, min_periods=1)
    rm, rs = roll.mean(), roll.std()
    ax.plot(x, rm, color=color, linewidth=2.0, zorder=3)
    ax.fill_between(x, rm - rs, rm + rs, color=color, alpha=0.18, zorder=2)


def style_ax(ax, ylabel=None, xlabel="Epoch"):
    sns.despine(ax=ax)
    ax.set_xlabel(xlabel, labelpad=4)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=4)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))
    ax.grid(axis="y", linewidth=0.5, alpha=0.4, linestyle="--")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot validation metrics from a PottsMPNN training run"
    )
    parser.add_argument("--model_dir", type=Path, required=True,
                        help="Training output directory containing val_metrics/")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--window", type=int, default=WINDOW,
                        help=f"Rolling-average window in epochs (default: {WINDOW})")
    args = parser.parse_args()

    model_dir = args.model_dir
    if not model_dir.is_dir():
        sys.exit(f"model_dir does not exist: {model_dir}")

    output_dir = args.output_dir or (model_dir / "plot_mpnn_val_metrics")

    csv_path = model_dir / "val_metrics" / "validation_output_all_epochs.csv"
    if not csv_path.is_file():
        sys.exit(f"Validation CSV not found: {csv_path}")

    print(f"Model directory:  {model_dir}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and clean data ────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    for col in ["total_loss", "potts_nlcpl_agg", "label_smoothed_nll_loss_agg",
                "potts_log_edge_probs", "node_loss_weight", "potts_loss_weight"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_tensor_scalar)

    agg = df.groupby("epoch").agg(
        nll=("nll.mean_nll", "mean"),
        perplexity=("nll.mean_perplexity", "mean"),
        seq_rec_sampled=("sequence_recovery.mean_sequence_recovery_sampled", "mean"),
        seq_rec_argmax=("sequence_recovery.mean_sequence_recovery_argmax", "mean"),
        total_loss=("total_loss", "mean"),
        potts_nlcpl=("potts_nlcpl_agg", "mean"),
    ).reset_index()

    # ── MegaScale (optional) ───────────────────────────────────────────────────
    megascale_path = csv_path.parent / "megascale_metrics.csv"
    df_mega = None
    if megascale_path.is_file():
        df_mega = pd.read_csv(megascale_path)
        mega_agg = df_mega.groupby("epoch")["pearson_r"].mean().reset_index()

    # ── Main grid ──────────────────────────────────────────────────────────────
    panels = [
        ("total_loss",      "Total loss",              PALETTE[0]),
        ("nll",             "Mean NLL",                PALETTE[1]),
        ("perplexity",      "Perplexity",              PALETTE[2]),
        ("seq_rec_sampled", "Seq. recovery (sampled)", PALETTE[3]),
        ("seq_rec_argmax",  "Seq. recovery (argmax)",  PALETTE[4]),
        ("potts_nlcpl",     "Potts NLCPL",             PALETTE[5]),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 9), constrained_layout=True)

    for ax, (col, title, color) in zip(axes.flat, panels):
        if col in agg.columns and agg[col].notna().any():
            plot_with_rolling(ax, agg["epoch"], agg[col], color=color,
                              window=args.window)
        ax.set_title(title, fontweight="semibold", pad=6)
        style_ax(ax)

    if df_mega is not None:
        ax_mega = axes[2, 1]
        plot_with_rolling(ax_mega, mega_agg["epoch"], mega_agg["pearson_r"],
                          color=PALETTE[6], window=args.window)
        ax_mega.set_title("MegaScale Pearson r", fontweight="semibold", pad=6)
        style_ax(ax_mega)
        axes[2, 0].set_visible(False)
        axes[2, 2].set_visible(False)
    else:
        for ax in axes[2]:
            ax.set_visible(False)

    out = output_dir / "val_metrics_plot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}", flush=True)


if __name__ == "__main__":
    main()
