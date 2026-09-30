#!/usr/bin/env python
"""
Per-amino-acid log-likelihood analysis across PottsMPNN checkpoints.

Mirrors train.py's val setup exactly — same dataset, same fallback wrapper,
same Fabric/trainer infrastructure — then adds per-residue LL collection.

Usage:
    python -m mpnn.scripts.analyze_ll_per_aa \
        --model_dir mpnn_output_potts_mpnn_ev2 \
        [--n_ckpts 6] [--n_examples 100] [--max_batches 50]
"""

import argparse
import contextlib
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.utils.checkpoint  # force submodule load; model code accesses it lazily
from omegaconf import DictConfig
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from atomworks.io.parser import STANDARD_PARSER_ARGS
from atomworks.ml.datasets.pandas_dataset import PandasDataset, StructuralDatasetWrapper
from atomworks.ml.datasets.parsers.default_metadata_row_parsers import GenericDFParser
from atomworks.ml.samplers import (
    DistributedMixedSampler,
    calculate_weights_for_pdb_dataset_df,
)
from foundry.utils.datasets import wrap_dataset_and_sampler_with_fallbacks
from mpnn.collate.feature_collator import TokenBudgetAwareFeatureCollator
from mpnn.pipelines.potts_mpnn import build_mpnn_transform_pipeline as build_potts_pipeline
from mpnn.trainers.pottsmpnn import PottsMPNNTrainer

# ── Hardcoded paths (same as train.py) ───────────────────────────────────────
_VAL_PARQUET = Path(
    "/novo/users/cpjb/PHD/conditional_binding/ph/data/mpnn_split/val_df_filtered.parquet"
)

# Same filters as train.py
MPNN_FILTERS = [
    "resolution < 3.5 and ~method.str.contains('NMR')",
    "n_non_atomized_tokens >= 30",
    "cluster.notnull() and cluster != 'nan'",
    "method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY']",
    "n_non_atomized_tokens <= 10000",
    "n_prot == 1",
    "assembly_id == '1'",
]

# Protonation variants → canonical amino acid name
_PROTONATION_TO_CANONICAL = {
    "HID": "HIS", "HIE": "HIS", "HIS-P": "HIS", "HIS-D": "HIS", "HIS-A": "HIS",
    "ASP-P": "ASP", "ASP-D": "ASP", "ASP-A": "ASP",
    "GLU-P": "GLU", "GLU-D": "GLU", "GLU-A": "GLU",
}


def _canonical(token_name: str) -> str:
    return _PROTONATION_TO_CANONICAL.get(token_name, token_name)


# ── Silent loader — copied verbatim from train.py ────────────────────────────
def _silent(loader_fn):
    def wrapper(row):
        with open(os.devnull, "w") as _null, \
             contextlib.redirect_stdout(_null), \
             contextlib.redirect_stderr(_null):
            return loader_fn(row)
    return wrapper


def _move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_device(v, device) for v in obj]
    return obj


# ── Trainer + Fabric setup — mirrors train.py ────────────────────────────────
def build_trainer(output_dir: Path) -> PottsMPNNTrainer:
    """Instantiate PottsMPNNTrainer with Fabric, mirroring train.py (eval-only)."""
    n_gpus = torch.cuda.device_count()
    trainer = PottsMPNNTrainer(
        model_type="potts_mpnn",
        extended_vocab=True,
        accelerator="gpu" if n_gpus > 0 else "cpu",
        devices_per_node=1,  # single device for analysis — avoid DDP launch
        max_epochs=1,
        output_dir=output_dir,
        callbacks=[],
        precision="bf16-mixed",
        clip_grad_max_norm=None,
        verbose=False,
    )
    # Minimal train_cfg — only needed for initialize_or_update_trainer_state
    train_cfg = DictConfig({
        "model": {
            "optimizer": {
                "_target_": "torch.optim.Adam",
                "lr": 1.0,
                "betas": [0.9, 0.98],
                "eps": 1e-9,
                "weight_decay": 0.0,
            },
            "lr_scheduler": {
                "_target_": "torch.optim.lr_scheduler.LambdaLR",
                "lr_lambda": "lambda s: 1.0",
            },
        }
    })
    trainer.initialize_or_update_trainer_state({"train_cfg": train_cfg})
    trainer.fabric.launch()
    trainer.construct_model()
    return trainer


def load_checkpoint(trainer: PottsMPNNTrainer, ckpt_path: Path) -> int:
    """Load a checkpoint's weights into the Fabric-managed model. Returns epoch."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = trainer.state["model"]
    model.load_state_dict(ckpt["model"])
    model.eval()
    return ckpt["current_epoch"]


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


# ── Val dataloader — mirrors train.py lines 254-356 exactly ──────────────────
def build_val_dataloader(n_examples_per_epoch: int = 100) -> DataLoader:
    """Build val dataloader with the same fallback wrapper as train.py."""
    inference_pipeline = build_potts_pipeline(
        model_type="potts_mpnn",
        is_inference=True,
        minimal_return=True,
    )
    val_df = pd.read_parquet(_VAL_PARQUET)
    _cif_args = {
        **STANDARD_PARSER_ARGS,
        "add_bond_types_from_struct_conn": (),
        "load_from_cache": False,
        "save_to_cache": False,
        "cache_dir": None,
    }

    def _make_ds():
        ds = StructuralDatasetWrapper(
            dataset=PandasDataset(
                data=val_df,
                id_column="example_id",
                name="pn_units_df_val",
                filters=MPNN_FILTERS,
            ),
            dataset_parser=GenericDFParser(
                example_id_colname="example_id",
                path_colname="path",
                assembly_id_colname="assembly_id",
            ),
            transform=inference_pipeline,
            cif_parser_args=_cif_args,
        )
        ds.loader = _silent(ds.loader)
        return ds

    val_ds = _make_ds()
    val_fallback_ds = _make_ds()

    val_weights = calculate_weights_for_pdb_dataset_df(
        dataset_df=val_ds.data,
        beta=1.0,
        alphas={"a_prot": 1.0, "a_nuc": 0, "a_ligand": 0, "a_loi": 0},
    )
    val_sampler = DistributedMixedSampler(
        datasets_info=[{
            "sampler": WeightedRandomSampler(val_weights, len(val_ds)),
            "dataset": val_ds,
            "probability": 1.0,
        }],
        num_replicas=1,
        rank=0,
        n_examples_per_epoch=n_examples_per_epoch,
    )
    val_fallback_sampler = WeightedRandomSampler(val_weights, len(val_ds))

    val_ds_fb, val_sampler_fb = wrap_dataset_and_sampler_with_fallbacks(
        dataset_to_be_wrapped=val_ds,
        sampler_to_be_wrapped=val_sampler,
        dataset_to_fallback_to=val_fallback_ds,
        sampler_to_fallback_to=val_fallback_sampler,
        n_fallback_retries=10,
    )
    collator = TokenBudgetAwareFeatureCollator(max_tokens_with_padding=10000)
    return DataLoader(
        val_ds_fb,
        sampler=val_sampler_fb,
        num_workers=4,
        prefetch_factor=1,
        collate_fn=collator,
        persistent_workers=True,
    )


# ── Per-residue LL collection ─────────────────────────────────────────────────
def collect_ll_records(
    trainer: PottsMPNNTrainer,
    epoch: int,
    loader: DataLoader,
    max_batches: int | None,
    idx_to_token: dict[int, str],
) -> list[dict]:
    model = trainer.state["model"]
    device = next(model.parameters()).device
    records = []

    for batch_idx, batch in enumerate(
        tqdm(loader, desc=f"epoch {epoch}", leave=False, file=sys.stderr)
    ):
        if max_batches is not None and batch_idx >= max_batches:
            break

        # Save true tokens BEFORE moving to device / forward pass
        S = batch["input_features"]["S"].clone()  # [B, L]

        # Conditional marginal LL: P(s_i | s_{j≠i}, X) in one forward pass
        batch["input_features"]["decode_type"] = "teacher_forcing"
        batch["input_features"]["causality_pattern"] = "conditional_minus_self"

        batch = _move_to_device(batch, device)
        S = S.to(device)

        try:
            with contextlib.redirect_stdout(open(os.devnull, "w")), \
                 contextlib.redirect_stderr(open(os.devnull, "w")):
                with torch.no_grad():
                    out = model(batch)
        except Exception as e:
            tqdm.write(f"[warning] batch {batch_idx} forward failed: {e}", file=sys.stderr)
            continue

        log_probs = out["decoder_features"]["log_probs"]   # [B, L, V]
        mask = out["input_features"]["mask_for_loss"]      # [B, L]
        ll = log_probs.gather(-1, S.unsqueeze(-1)).squeeze(-1)  # [B, L]

        mask_bool = mask.bool()
        mask_f = mask_bool.float()
        valid_tokens = int(mask_bool.sum().item())
        if valid_tokens == 0:
            continue

        masked_ll = ll * mask_f
        valid_per_example = mask_f.sum(dim=-1)
        valid_examples = valid_per_example > 0
        ll_per_example = masked_ll.sum(dim=-1) / valid_per_example.clamp_min(1.0)

        batch_token_mean_ll = (masked_ll.sum() / mask_f.sum()).item()
        batch_mean_ll = ll_per_example[valid_examples].mean().item()

        B, L = S.shape
        for b in range(B):
            for l in range(L):
                if not mask_bool[b, l].item():
                    continue
                token_idx = S[b, l].item()
                token_name = idx_to_token.get(token_idx, f"IDX_{token_idx}")
                if token_name in ("UNK", "UNKNOWN_AA"):
                    continue
                ll_value = ll[b, l].item()
                records.append({
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "example_idx": b,
                    "position": l,
                    "token": token_name,
                    "canonical_aa": _canonical(token_name),
                    "log_likelihood": ll_value,
                    "nll": -ll_value,
                    "prob_true": float(np.exp(ll_value)),
                    "batch_mean_log_likelihood": batch_mean_ll,
                    "batch_mean_nll": -batch_mean_ll,
                    "batch_token_mean_log_likelihood": batch_token_mean_ll,
                    "batch_token_mean_nll": -batch_token_mean_ll,
                    "batch_valid_tokens": valid_tokens,
                })
    return records


# ── Plots ─────────────────────────────────────────────────────────────────────

def _boxplot_grid(df: pd.DataFrame, epochs: list[int], x_col: str,
                  x_order: list[str], palette: dict, path: Path) -> None:
    n_cols = 2
    n_rows = (len(epochs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4 * n_rows), sharey=True)
    axes = np.array(axes).flatten()
    for ax, ep in zip(axes, epochs):
        sns.boxplot(
            data=df[df["epoch"] == ep], x=x_col, y="log_likelihood",
            order=x_order, palette=palette, ax=ax,
            showfliers=False, linewidth=0.8,
        )
        ax.set_title(f"Epoch {ep}", fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("Log-likelihood" if ax == axes[0] else "")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        sns.despine(ax=ax)
    for ax in axes[len(epochs):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _heatmap(df: pd.DataFrame, epochs: list[int], x_col: str,
             x_order: list[str], path: Path) -> None:
    pivot = (
        df.groupby(["epoch", x_col])["log_likelihood"]
        .median()
        .unstack(x_col)
        .reindex(columns=x_order)
    )
    fig, ax = plt.subplots(
        figsize=(max(10, len(x_order) * 0.55), max(4, len(epochs) * 0.45))
    )
    sns.heatmap(
        pivot, ax=ax, cmap="RdYlGn", center=0, annot=False,
        linewidths=0.3, cbar_kws={"label": "Median log-likelihood"},
    )
    ax.set_title("Median log-likelihood over training")
    ax.set_xlabel("")
    ax.set_ylabel("Epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _violin(df: pd.DataFrame, epochs: list[int], x_col: str,
            x_order: list[str], path: Path) -> None:
    epoch_palette = dict(zip(epochs, sns.color_palette("viridis", len(epochs))))
    fig, ax = plt.subplots(figsize=(max(12, len(x_order) * 0.8), 5))
    sns.violinplot(
        data=df, x=x_col, y="log_likelihood", hue="epoch",
        order=x_order, palette=epoch_palette, ax=ax,
        inner=None, linewidth=0.5, density_norm="width", dodge=True,
    )
    ax.set_title("Log-likelihood distribution (all checkpoints)")
    ax.set_xlabel("")
    ax.set_ylabel("Log-likelihood")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(title="Epoch", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _nll_over_checkpoints(df: pd.DataFrame, path: Path) -> None:
    summary = (
        df.groupby("epoch")
        .agg(
            mean_nll=("nll", "mean"),
            median_nll=("nll", "median"),
            n_residues=("nll", "size"),
        )
        .reset_index()
        .sort_values("epoch")
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(
        data=summary,
        x="epoch",
        y="mean_nll",
        marker="o",
        linewidth=2,
        label="Mean NLL",
        ax=ax,
    )
    sns.lineplot(
        data=summary,
        x="epoch",
        y="median_nll",
        marker="s",
        linewidth=1.5,
        label="Median NLL",
        ax=ax,
    )
    ax.set_title("NLL over checkpoints")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL (-log likelihood)")
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)
    ax.legend(title="")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_plots(df: pd.DataFrame, epochs: list[int], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white", context="paper", font_scale=1.05)

    _nll_over_checkpoints(df, output_dir / "nll_over_checkpoints.png")

    latest = df[df["epoch"] == max(epochs)]

    # ── Canonical AA level (HID/HIE/HIS-P → HIS, etc.) ───────────────────────
    aa_order = (
        latest.groupby("canonical_aa")["log_likelihood"]
        .median()
        .sort_values()
        .index.tolist()
    )
    aa_palette = dict(zip(aa_order, sns.color_palette("tab20", len(aa_order))))

    _boxplot_grid(df, epochs, "canonical_aa", aa_order, aa_palette,
                  output_dir / "boxplot_grid.png")
    _heatmap(df, epochs, "canonical_aa", aa_order,
             output_dir / "ll_heatmap.png")
    _violin(df, epochs, "canonical_aa", aa_order,
            output_dir / "ll_violin.png")

    # ── Token level (all 32 tokens — protonation variants kept separate) ──────
    # Order: group by canonical family (sorted by canonical median LL),
    # then sort by token median LL within each family.
    canonical_median = latest.groupby("canonical_aa")["log_likelihood"].median()
    token_canonical = df[["token", "canonical_aa"]].drop_duplicates().set_index("token")["canonical_aa"]
    token_order = (
        latest.groupby("token")["log_likelihood"]
        .median()
        .reset_index()
        .assign(
            canonical_aa=lambda d: d["token"].map(token_canonical),
            canonical_rank=lambda d: d["canonical_aa"].map(canonical_median),
        )
        .sort_values(["canonical_rank", "log_likelihood"])["token"]
        .tolist()
    )
    # Colour each token by its canonical family so variants cluster visually
    token_palette = {tok: aa_palette.get(_canonical(tok), "#888888") for tok in token_order}

    _boxplot_grid(df, epochs, "token", token_order, token_palette,
                  output_dir / "boxplot_grid_tokens.png")
    _heatmap(df, epochs, "token", token_order,
             output_dir / "ll_heatmap_tokens.png")
    _violin(df, epochs, "token", token_order,
            output_dir / "ll_violin_tokens.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-AA log-likelihood analysis across PottsMPNN checkpoints"
    )
    parser.add_argument("--model_dir", type=Path, required=False,
                        help="Training output directory; checkpoints are found under model_dir/ckpt")
    parser.add_argument("--ckpt_dir", type=Path, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--output_dir", type=Path, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--n_ckpts", type=int, default=6,
                        help="Number of evenly-spaced checkpoints to evaluate (default: 6)")
    parser.add_argument("--n_examples", type=int, default=100,
                        help="Val examples per epoch drawn by sampler (default: 100)")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Cap on batches per checkpoint (default: all n_examples)")
    args = parser.parse_args()

    if args.model_dir is None and args.ckpt_dir is None:
        parser.error("--model_dir is required")

    model_dir = args.model_dir
    if model_dir is None:
        model_dir = args.ckpt_dir.parent if args.ckpt_dir.name == "ckpt" else args.ckpt_dir

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else resolve_ckpt_dir(model_dir)
    output_dir = args.output_dir or (model_dir / "analyze_ll_per_aa")

    all_ckpts = sorted(ckpt_dir.glob("epoch-*.ckpt"))
    if not all_ckpts:
        sys.exit(f"No checkpoints found in {ckpt_dir}")

    print(f"Model directory:      {model_dir}", flush=True)
    print(f"Checkpoint directory: {ckpt_dir}", flush=True)
    print(f"Output directory:     {output_dir}", flush=True)

    if args.n_ckpts >= len(all_ckpts):
        selected = all_ckpts
    else:
        indices = np.linspace(0, len(all_ckpts) - 1, args.n_ckpts, dtype=int)
        selected = [all_ckpts[i] for i in indices]

    print(f"Selected {len(selected)} checkpoint(s): {[p.name for p in selected]}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Setting up PottsMPNNTrainer + Fabric...", flush=True)
    trainer = build_trainer(output_dir)
    model = trainer.state["model"]
    idx_to_token = {v: k for k, v in model.token_to_idx.items()}

    print(f"Building val dataloader ({_VAL_PARQUET})...", flush=True)
    loader = build_val_dataloader(n_examples_per_epoch=args.n_examples)

    all_records: list[dict] = []
    epochs_seen: list[int] = []

    # epoch 0 = random-init baseline (build_trainer already calls init_weights,
    # so the model is freshly random before any checkpoint is loaded)
    print("Collecting random-init baseline (epoch = 0)…", flush=True)
    baseline_records = collect_ll_records(trainer, 0, loader, args.max_batches, idx_to_token)
    all_records.extend(baseline_records)
    epochs_seen.append(0)

    for ckpt_path in tqdm(selected, desc="Checkpoints", file=sys.stderr):
        ckpt_epoch = load_checkpoint(trainer, ckpt_path)
        epoch = ckpt_epoch + 1  # shift so epoch 0 is reserved for the random baseline
        records = collect_ll_records(trainer, epoch, loader, args.max_batches, idx_to_token)
        all_records.extend(records)
        epochs_seen.append(epoch)

    if not all_records:
        sys.exit("No records collected — check pipeline / data paths.")

    df = pd.DataFrame(all_records)
    csv_path = output_dir / "ll_per_aa_records.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df):,} residue records → {csv_path}", flush=True)

    summary = (
        df.groupby("epoch")
        .agg(
            mean_log_likelihood=("log_likelihood", "mean"),
            median_log_likelihood=("log_likelihood", "median"),
            mean_nll=("nll", "mean"),
            median_nll=("nll", "median"),
            mean_prob_true=("prob_true", "mean"),
            n_residues=("log_likelihood", "size"),
        )
        .reset_index()
    )
    summary_path = output_dir / "ll_epoch_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved epoch LL/NLL summary → {summary_path}", flush=True)

    print("Generating plots...", flush=True)
    make_plots(df, sorted(epochs_seen), output_dir)
    print(f"All outputs saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
