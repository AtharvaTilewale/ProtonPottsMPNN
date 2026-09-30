#!/usr/bin/env python
"""
PCA of PottsMPNN learned token embedding table (model.W_s.weight).

Extracts the [V=32, H=128] sequence embedding matrix from model parameters —
no structure data needed. Fits a joint PCA across all selected checkpoints so
axes are comparable, then plots the 32 token positions per epoch coloured by
biochemical grouping.
Outputs are saved in --model_dir by default.

Key question: do protonation variants (HID/HIE/HIS-P) diverge over training,
indicating the model learns protonation-state representations?

Usage:
    python -m mpnn.scripts.analyze_embeddings_pca \
        --model_dir mpnn_output_potts_mpnn_ev2 \
        [--n_ckpts 10] [--n_components 2]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import torch
from tqdm import tqdm

from mpnn.model.layers.graph_embeddings import PottsProteinFeatures
from mpnn.model.pottsmpnn import PottsMPNN

# Global aesthetics
sns.set_theme(style="white", context="paper", font_scale=1.05)
plt.rcParams.update({
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#e8e8e8",
    "grid.linewidth":     0.6,
    "grid.linestyle":     ":",
    "legend.frameon":     False,
    "legend.borderpad":   0.4,
    "figure.dpi":         150,
})

# ── Biochemical grouping and colours ─────────────────────────────────────────

TOKEN_GROUP: dict[str, str] = {
    "ALA": "hydrophobic", "VAL": "hydrophobic", "ILE": "hydrophobic",
    "LEU": "hydrophobic", "MET": "hydrophobic", "PHE": "hydrophobic",
    "TRP": "hydrophobic", "TYR": "hydrophobic", "PRO": "hydrophobic",
    "SER": "polar", "THR": "polar", "CYS": "polar",
    "ASN": "polar", "GLN": "polar",
    "LYS": "positive", "ARG": "positive",
    "ASP": "negative", "GLU": "negative",
    "HIS":   "HIS_family",
    "HID":   "HIS_family", "HIE":   "HIS_family",
    "HIS-P": "HIS_family", "HIS-D": "HIS_family", "HIS-A": "HIS_family",
    "ASP-P": "ASP_family", "ASP-D": "ASP_family", "ASP-A": "ASP_family",
    "GLU-P": "GLU_family", "GLU-D": "GLU_family", "GLU-A": "GLU_family",
    "GLY": "special",
    "UNK": "UNK",
}

GROUP_COLOR: dict[str, str] = {
    "hydrophobic": "#E07B39",
    "polar":       "#3A9F6E",
    "positive":    "#3B6FD4",
    "negative":    "#C0392B",
    "HIS_family":  "#7BAFD4",
    "ASP_family":  "#E8836A",
    "GLU_family":  "#B03030",
    "special":     "#888888",
    "UNK":         "#444444",
}

GROUP_MARKER: dict[str, str] = {
    "hydrophobic": "o", "polar": "s", "positive": "^", "negative": "v",
    "HIS_family": "D", "ASP_family": "P", "GLU_family": "X",
    "special": "h", "UNK": "x",
}


def load_embedding(ckpt_path: Path) -> tuple[np.ndarray, int, dict[str, int]]:
    """Load W_s.weight from checkpoint → (embedding_matrix [V,H], epoch, token_to_idx)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = PottsMPNN(
        graph_featurization_module=PottsProteinFeatures(),
        etab_source=PottsMPNN.infer_etab_source(ckpt["model"]),
    )
    model.apply(model.init_weights)
    model.load_state_dict(ckpt["model"])
    epoch = ckpt["current_epoch"]
    W = model.W_s.weight.detach().cpu().numpy()  # [V, H]
    token_to_idx = model.token_to_idx
    del model
    return W, epoch, token_to_idx


def resolve_ckpt_dir(model_dir: Path) -> Path:
    """Resolve a training output directory to its checkpoint directory."""
    candidates = [
        model_dir / "ckpt",
        model_dir / "checkpoints",
        model_dir,
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("epoch-*.ckpt")):
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"No epoch-*.ckpt files found. Looked in: {searched}"
    )


def _make_legend_handles() -> list:
    return [
        mlines.Line2D(
            [], [],
            color=GROUP_COLOR[g], marker=GROUP_MARKER[g],
            linestyle="None", markersize=6,
            label=g.replace("_", " "),
        )
        for g in GROUP_COLOR
    ]


def _scatter_epoch(
    ax: "plt.Axes",
    coords: np.ndarray,       # [V, 2]
    token_names: list[str],
    pca: PCA,
    epoch: int,
) -> None:
    for i, name in enumerate(token_names):
        group = TOKEN_GROUP.get(name, "UNK")
        color = GROUP_COLOR[group]
        marker = GROUP_MARKER[group]
        ax.scatter(
            coords[i, 0], coords[i, 1],
            c=color, marker=marker, s=72, zorder=3,
            linewidths=0.5, edgecolors="white", alpha=0.92,
        )
        txt = ax.text(
            coords[i, 0], coords[i, 1], f" {name}",
            fontsize=5.5, va="center", zorder=4, color="#1a1a1a",
        )
        txt.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])

    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1  ({ev[0]:.1%} var)", labelpad=4)
    ax.set_ylabel(f"PC2  ({ev[1]:.1%} var)", labelpad=4)
    ax.set_title(f"Epoch {epoch}", pad=6)
    ax.tick_params(labelsize=7, length=3)
    sns.despine(ax=ax)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PCA of PottsMPNN token embedding table across checkpoints"
    )
    parser.add_argument("--model_dir", type=Path, required=False,
                        help="Training output directory; checkpoints are found under model_dir/ckpt")
    parser.add_argument("--ckpt_dir", type=Path, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--output_dir", type=Path, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--n_ckpts", type=int, default=None,
                        help="Evenly-spaced checkpoints to include (default: all)")
    parser.add_argument("--n_components", type=int, default=2,
                        help="PCA dimensionality (default: 2)")
    args = parser.parse_args()

    if args.model_dir is None and args.ckpt_dir is None:
        parser.error("--model_dir is required")

    model_dir = args.model_dir
    if model_dir is None:
        model_dir = args.ckpt_dir.parent if args.ckpt_dir.name == "ckpt" else args.ckpt_dir

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else resolve_ckpt_dir(model_dir)
    output_dir = args.output_dir or (model_dir / "analyze_embeddings_pca")

    all_ckpts = sorted(ckpt_dir.glob("epoch-*.ckpt"))
    if not all_ckpts:
        sys.exit(f"No checkpoints found in {ckpt_dir}")

    if args.n_ckpts is None or args.n_ckpts >= len(all_ckpts):
        selected = all_ckpts
    else:
        indices = np.linspace(0, len(all_ckpts) - 1, args.n_ckpts, dtype=int)
        selected = [all_ckpts[i] for i in indices]

    print(f"Model directory:      {model_dir}", flush=True)
    print(f"Checkpoint directory: {ckpt_dir}", flush=True)
    print(f"Output directory:     {output_dir}", flush=True)
    print(f"Loading embeddings from {len(selected)} checkpoint(s)…", flush=True)

    # epoch 0 = random-init baseline; checkpoints are shifted to epoch N+1
    random_model = PottsMPNN(graph_featurization_module=PottsProteinFeatures())
    random_model.apply(random_model.init_weights)
    W_rand = random_model.W_s.weight.detach().cpu().numpy()
    t2i_rand = random_model.token_to_idx
    del random_model

    token_names: list[str] = []
    idx_to_token_rand = {v: k for k, v in t2i_rand.items()}
    token_names = [idx_to_token_rand[i] for i in range(W_rand.shape[0])]

    embeddings_per_ckpt: list[tuple[np.ndarray, int]] = [(W_rand, 0)]
    print("Random-init baseline collected (epoch = 0).", flush=True)

    for ckpt_path in tqdm(selected, desc="Loading", file=sys.stderr):
        W, ckpt_epoch, t2i = load_embedding(ckpt_path)
        embeddings_per_ckpt.append((W, ckpt_epoch + 1))

    V, H = embeddings_per_ckpt[0][0].shape
    n_ckpts = len(embeddings_per_ckpt)
    print(f"Embedding: {V} tokens × {H} dims, {n_ckpts} checkpoints", flush=True)

    # ── Joint PCA across all epochs (shared coordinate frame) ────────────────
    all_W = np.vstack([W for W, _ in embeddings_per_ckpt])  # [n_ckpts*V, H]
    pca = PCA(n_components=args.n_components)
    all_coords = pca.fit_transform(all_W)                   # [n_ckpts*V, n_components]
    coords_per_ckpt = [
        all_coords[i * V : (i + 1) * V] for i in range(n_ckpts)
    ]

    print(
        "Joint PCA explained variance: "
        + ", ".join(
            f"PC{j+1}={ev:.1%}"
            for j, ev in enumerate(pca.explained_variance_ratio_[:args.n_components])
        ),
        flush=True,
    )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for (W, epoch), coords in zip(embeddings_per_ckpt, coords_per_ckpt):
        for i, name in enumerate(token_names):
            row = {"epoch": epoch, "token": name, "group": TOKEN_GROUP.get(name, "UNK")}
            for j in range(args.n_components):
                row[f"pc{j+1}"] = coords[i, j]
            rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = output_dir / "token_embeddings_pca.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved PCA coordinates → {csv_path}", flush=True)

    legend_handles = _make_legend_handles()

    # ── Per-epoch scatter plots ───────────────────────────────────────────────
    for (W, epoch), coords in zip(embeddings_per_ckpt, coords_per_ckpt):
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        _scatter_epoch(ax, coords[:, :2], token_names, pca, epoch)
        ax.legend(
            handles=legend_handles, fontsize=6.5,
            loc="upper right", title="Group", title_fontsize=7,
            handletextpad=0.3, labelspacing=0.3,
        )
        fig.tight_layout()
        fig.savefig(
            output_dir / f"pca_scatter_{epoch:04d}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

    # ── Grid of all epochs ────────────────────────────────────────────────────
    n_cols = min(4, n_ckpts)
    n_rows = (n_ckpts + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 3.8 * n_rows),
        sharex=True, sharey=True,
    )
    axes_flat = np.array(axes).flatten()
    for ax, (W, epoch), coords in zip(axes_flat, embeddings_per_ckpt, coords_per_ckpt):
        _scatter_epoch(ax, coords[:, :2], token_names, pca, epoch)
    for ax in axes_flat[n_ckpts:]:
        ax.set_visible(False)
    fig.legend(
        handles=legend_handles, fontsize=6.5,
        loc="lower right", bbox_to_anchor=(1.0, 0.0),
        title="Group", title_fontsize=7,
        handletextpad=0.3, labelspacing=0.3,
    )
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    fig.savefig(output_dir / "pca_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid → {output_dir / 'pca_grid.png'}", flush=True)

    # ── Explained-variance bar chart ──────────────────────────────────────────
    n_bar = min(20, H)
    pca_full = PCA(n_components=n_bar).fit(all_W)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_) * 100
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(
        range(1, n_bar + 1),
        pca_full.explained_variance_ratio_ * 100,
        color=GROUP_COLOR["positive"], alpha=0.75,
        edgecolor="white", linewidth=0.5,
    )
    ax2 = ax.twinx()
    ax2.plot(range(1, n_bar + 1), cumulative, color="#333333",
             linewidth=1.2, marker=".", markersize=4)
    ax2.set_ylabel("Cumulative variance (%)", labelpad=4)
    ax2.set_ylim(0, 105)
    sns.despine(ax=ax)
    sns.despine(ax=ax2, right=False)
    ax.set_xlabel("Principal component", labelpad=4)
    ax.set_ylabel("Explained variance (%)", labelpad=4)
    ax.set_title("Token embedding — explained variance (joint PCA, all epochs)")
    ax.set_xticks(range(1, n_bar + 1))
    ax.tick_params(length=3)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_variance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Trajectory: how each token drifts in PC space over training ───────────
    if n_ckpts >= 2:
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        for i, name in enumerate(token_names):
            group = TOKEN_GROUP.get(name, "UNK")
            color = GROUP_COLOR[group]
            marker = GROUP_MARKER[group]
            xs = [coords_per_ckpt[j][i, 0] for j in range(n_ckpts)]
            ys = [coords_per_ckpt[j][i, 1] for j in range(n_ckpts)]
            ax.plot(xs, ys, color=color, alpha=0.35, linewidth=0.9, zorder=2)
            # open circle = start epoch
            ax.scatter(xs[0], ys[0], c="white", edgecolors=color,
                       marker=marker, s=55, zorder=3, linewidths=1.3)
            # filled = final epoch
            ax.scatter(xs[-1], ys[-1], c=color, marker=marker,
                       s=75, zorder=4, linewidths=0.5, edgecolors="white", alpha=0.92)
            txt = ax.text(xs[-1], ys[-1], f" {name}",
                          fontsize=5.5, va="center", zorder=5, color="#1a1a1a")
            txt.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])

        ax.set_xlabel(f"PC1  ({pca.explained_variance_ratio_[0]:.1%} var)", labelpad=4)
        ax.set_ylabel(f"PC2  ({pca.explained_variance_ratio_[1]:.1%} var)", labelpad=4)
        ax.set_title("Token embedding trajectory  (open = epoch 0, filled = last epoch)")
        ax.legend(
            handles=legend_handles, fontsize=6.5,
            loc="upper right", title="Group", title_fontsize=7,
            handletextpad=0.3, labelspacing=0.3,
        )
        ax.tick_params(labelsize=7, length=3)
        sns.despine(ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / "pca_trajectory.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"All outputs saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
