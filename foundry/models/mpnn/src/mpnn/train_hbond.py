#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../../../.ipd/shebang/mpnn_exec.sh" "$0" "$@"'

"""Train the linear H-bond / salt-bridge head on a FROZEN PottsMPNN (multi-GPU).

A trimmed copy of ``train.py`` with ONLY what the head needs: the extended-vocab Potts
data path with ``build_bond_labels=True`` (so every example carries its HBPLUS / salt
ground-truth as token-pair partner lists), the ``HBondModel`` (frozen PottsMPNN + linear
head — everything but the head is frozen), ``HBondHeadTrainer`` (Fabric/DDP, bf16), and
``HBondValidationCallback`` (per-epoch PR / ROC / per-type metrics).

Config is the block below (no argparse). Point ENCODER_CKPT/EXTENDED_VOCAB/DATASET there; the cache and
output dir come from the environment. Launch via scripts/slurm_train_hbond_v6.sh, or directly:
    MPNN_PRECOMPUTED_DIR=/novo/users/cpjb/rdd/cpjb/ev6_snapshots MPNN_OUTPUT_DIR=trained_models \
        EPOCHS=100 python -m mpnn.train_hbond
"""

import contextlib
import os
from pathlib import Path

import pandas as pd
import torch
from atomworks.io.parser import STANDARD_PARSER_ARGS
from atomworks.ml.datasets.pandas_dataset import PandasDataset, StructuralDatasetWrapper
from atomworks.ml.datasets.parsers.default_metadata_row_parsers import GenericDFParser
from atomworks.ml.samplers import (
    DistributedMixedSampler,
    calculate_weights_for_pdb_dataset_df,
)
from omegaconf import DictConfig
from torch.utils.data import DataLoader, WeightedRandomSampler

from foundry.utils.datasets import wrap_dataset_and_sampler_with_fallbacks
from mpnn.collate.feature_collator import TokenBudgetAwareFeatureCollator
from mpnn.pipelines.potts_mpnn import build_mpnn_transform_pipeline as build_potts_pipeline
from mpnn.samplers.samplers import PaddedTokenBudgetBatchSampler
from mpnn.callbacks.hbond_validation import HBondValidationCallback
from mpnn.trainers.hbond_head import HBondHeadTrainer
from mpnn.transforms.precomputed import make_snapshot_loader
from mpnn.transforms.extended_vocab import get_vocab

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit here. (Everything tunable lives in this one block.)
# ══════════════════════════════════════════════════════════════════════════════
_DATA = "/novo/users/cpjb/PHD/conditional_binding/ph/data/mpnn_split"

# SUBSET = True -> quick end-to-end check on the small *_subset.parquet (few structures,
# few examples/epoch). Flip to False for the real run. SUBSET is always PDB-only.
SUBSET = False

# Encoder vocabulary -- MUST match ENCODER_CKPT. "v6" -> 30-token, "v4"/"v3" -> 32-token.
EXTENDED_VOCAB = "v6"
# Training data for the (non-SUBSET) run: "afdb_pdb" (PDB + AFDB, matches the v6 encoder) or "pdb".
DATASET = "afdb_pdb"

# _node5 = 5-part head input concat(h_V[i], W_s(S_i), h_E, W_s(S_j), h_V[j]).
RUN_NAME = "hbond_head_v6_subset" if SUBSET else "hbond_head_v6_afdb_edge"
# Pretrained PottsMPNN checkpoint to FREEZE (the v6 edge-coupling encoder).
ENCODER_CKPT = ("/novo/users/cpjb/PHD/conditional_binding/ph/trained_models/"
                "mpnn_output_potts_v6_afdb_edge/ckpt/epoch-0085.ckpt")

# Data split.
if SUBSET:
    train_path = f"{_DATA}/train_subset.parquet"
    val_path = f"{_DATA}/val_subset.parquet"
    TRAIN_EXAMPLES_PER_EPOCH = 64       # structures sampled per epoch (across all GPUs)
    VAL_EXAMPLES_PER_EPOCH = 48
    EPOCHS = 3
else:
    train_path = f"{_DATA}/train_df_filtered.parquet"
    val_path = f"{_DATA}/val_df_filtered.parquet"
    TRAIN_EXAMPLES_PER_EPOCH = 4000
    VAL_EXAMPLES_PER_EPOCH = 300
    EPOCHS = int(os.environ.get("EPOCHS", 100))   # env-overridable; auto-resume runs to this target

# Head / optimisation.
N_HIDDEN = 256         # hidden width; HBondHead is a 2-hidden-layer MLP (n_layers=2):
                       #   concat(5H) -> Linear -> ReLU -> Linear -> ReLU -> Linear(->4). 0 => pure linear.
NEG_PER_POS = 10       # negative:positive subsample for the BCE terms (training)
LR = 1e-3
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 4))   # under precomputed the loader is light (snapshot
                       # load + encode, no HBPLUS); keep <= --cpus-per-task in the submit script.

# Featurization (kept in sync with the potts data path).
batch_size = 10000                 # token budget per batch (same as potts)
train_structure_noise_default = 0.2
complex_max_atoms = 6000
complex_pair_probability = 0.9
train_date_cutoff = "2022-12-16"
# ══════════════════════════════════════════════════════════════════════════════

run_name = RUN_NAME
encoder_ckpt = ENCODER_CKPT

# Precomputed annotation cache (built by scripts/build_snapshots.py). Set MPNN_PRECOMPUTED_DIR so the
# loader returns the cleaned/cropped/annotated snapshot -- which already carries hbond_pairs/salt_pairs, so
# BuildBondEdgeLabels regenerates the H-bond/salt labels at train time with NO live HBPLUS in the workers.
# A miss raises and the fallback wrapper draws another cached example. Off (None) => today's live pipeline.
precomputed_dir = os.environ.get("MPNN_PRECOMPUTED_DIR") or None
precomputed = bool(precomputed_dir)


def create_noam_scheduler(optimizer, d_model, warmup_steps=4000, factor=2):
    def noam_lambda(step):
        base_lr = factor * (d_model ** (-0.5))
        if step == 0:
            return 0.0
        return base_lr * min(step ** (-0.5), step * warmup_steps ** (-1.5))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)


def get_num_tokens(df, idx):
    if isinstance(idx, (list, tuple)):
        idx = idx[0]
    return df.iloc[idx]["n_non_atomized_tokens"]


# Filters identical to train.py. COMMON_FILTERS gate structure regardless of source.
COMMON_FILTERS = [
    "n_non_atomized_tokens >= 30",
    "cluster.notnull() and cluster != 'nan'",
    f"n_non_atomized_tokens < {batch_size}",
    "n_prot >= 1",
    "assembly_id == '1'",
]
if DATASET == "afdb_pdb" and not SUBSET:
    # PDB + AFDB in one table: gate the experimental rows ONLY (an AFDB row has resolution NaN and
    # deposition_date NaT, so a bare PDB expression would silently drop every prediction). Mirrors train.py.
    MPNN_FILTERS = [
        "method == 'ALPHAFOLD' or "
        "(resolution < 3.5 and method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY'])"
    ] + COMMON_FILTERS
    MPNN_TRAIN_FILTERS = [
        f"method == 'ALPHAFOLD' or deposition_date < '{train_date_cutoff}'"
    ] + MPNN_FILTERS
else:
    MPNN_FILTERS = [
        "resolution < 3.5 and ~method.str.contains('NMR')",
        "method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY']",
    ] + COMMON_FILTERS
    MPNN_TRAIN_FILTERS = [f"deposition_date < '{train_date_cutoff}'"] + MPNN_FILTERS

n_gpus = torch.cuda.device_count()
# Under `srun --ntasks-per-node=N`, SLURM (not torchrun) launches the ranks, so the
# DistributedMixedSampler must read SLURM_PROCID / SLURM_NTASKS — RANK/LOCAL_RANK are unset
# by srun, which would collapse every task to rank 0 and mis-shard the data. (Falls back to
# torchrun env, then single-process, when not under srun.)
world_size = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", max(n_gpus, 1))))
global_rank = int(os.environ.get("SLURM_PROCID",
                                 os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))))
print(f"Using {n_gpus} GPU(s); world_size={world_size} rank={global_rank}; "
      f"EPOCHS={EPOCHS}; encoder_ckpt={encoder_ckpt}")

if DATASET == "afdb_pdb" and not SUBSET:
    # PDB experimental + AFDB predictions in one table (leak-free AFDB split from scripts/split_afdb.py).
    # `subset` tags let the sampler weight per source. Mirrors train.py's afdb_pdb branch.
    afdb_dir = Path("/novo/users/cpjb/rdd/cpjb/afdb_complexes")
    pdb_train = pd.read_parquet(train_path); pdb_train["subset"] = "pdb"
    afdb_train = pd.read_parquet(afdb_dir / "afdb_train.parquet")
    train_df = pd.concat([pdb_train, afdb_train[pdb_train.columns]], ignore_index=True)
    pdb_val = pd.read_parquet(val_path); pdb_val["subset"] = "pdb"
    afdb_val = pd.read_parquet(afdb_dir / "afdb_val.parquet")
    val_df = pd.concat([pdb_val, afdb_val[pdb_val.columns]], ignore_index=True)
    print(f"AFDB+PDB: train {len(train_df):,} ({dict(train_df['subset'].value_counts())}) | "
          f"val {len(val_df):,} rows")
else:
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

# H-bond label scope: "sc_any" keeps side-chain<->side-chain AND side-chain<->backbone bonds
# (still drops the identity-agnostic backbone<->backbone secondary-structure bonds). Adds positives
# vs "sc_sc", so raise the partner cap to avoid truncation (watch log's bond_label_truncated_partners).
# Train and val MUST use the same scope so metrics compare. Change here + re-train (labels are built
# on-the-fly per example, load_from_cache=False, so no dataset regeneration is needed).
HBOND_LABEL_SCOPE = "sc_any"
MAX_HBOND_PARTNERS = 24

# Potts pipeline WITH bond labels. Train: complex pairing + noise augmentation.
# Val: full structure (complex_pair_probability=1.0), no noise. Both rate=1.0 so S carries
# the true protonation microstate (the head's inference setting); *-A edges are masked.
train_pipeline = build_potts_pipeline(
    model_type="potts_mpnn", is_inference=False, minimal_return=True,
    train_structure_noise_default=train_structure_noise_default,
    protonation_label_rate=1.0,
    build_bond_labels=True, hbond_scope=HBOND_LABEL_SCOPE, max_hbond_partners=MAX_HBOND_PARTNERS,
    extended_vocab=EXTENDED_VOCAB,
    complex_pair_probability=complex_pair_probability,   # inert under precomputed (p=1 crop is baked in)
    complex_max_atoms=complex_max_atoms, complex_min_atom_contacts=10, complex_max_chains=None,
    precomputed=precomputed, precomputed_dir=precomputed_dir,
)
inference_pipeline = build_potts_pipeline(
    model_type="potts_mpnn", is_inference=False, minimal_return=True,
    train_structure_noise_default=0.0,
    protonation_label_rate=1.0,
    build_bond_labels=True, hbond_scope=HBOND_LABEL_SCOPE, max_hbond_partners=MAX_HBOND_PARTNERS,
    extended_vocab=EXTENDED_VOCAB,
    complex_pair_probability=1.0,
    complex_max_atoms=complex_max_atoms, complex_min_atom_contacts=10,
    precomputed=precomputed, precomputed_dir=precomputed_dir,
)


def _silent(loader_fn):
    def wrapper(row):
        with open(os.devnull, "w") as _null, \
             contextlib.redirect_stdout(_null), contextlib.redirect_stderr(_null):
            return loader_fn(row)
    return wrapper


def make_structural_dataset(df, name, filters, transform):
    ds = StructuralDatasetWrapper(
        dataset=PandasDataset(data=df, id_column="example_id", name=name, filters=filters),
        dataset_parser=GenericDFParser(example_id_colname="example_id",
                                       path_colname="path", assembly_id_colname="assembly_id"),
        transform=transform,
        cif_parser_args={**STANDARD_PARSER_ARGS, "add_bond_types_from_struct_conn": (),
                         "load_from_cache": False, "save_to_cache": False, "cache_dir": None},
    )
    # Precomputed: return the annotated snapshot for a row (skip the CIF parse + live HBPLUS); a miss
    # raises and the fallback wrapper draws another cached example. Wrap BEFORE _silent.
    if precomputed_dir:
        ds.loader = make_snapshot_loader(precomputed_dir)
    ds.loader = _silent(ds.loader)
    return ds


# ── Train dataset / sampler / loader (with fallbacks) ──────────────────────────
train_structural_dataset = make_structural_dataset(train_df, "pn_units_df_train", MPNN_TRAIN_FILTERS, train_pipeline)
train_fallback_dataset = make_structural_dataset(train_df, "pn_units_df_train", MPNN_TRAIN_FILTERS, train_pipeline)

train_weights = calculate_weights_for_pdb_dataset_df(
    dataset_df=train_structural_dataset.data, beta=1.0,
    alphas={"a_prot": 1.0, "a_nuc": 0, "a_ligand": 0, "a_loi": 0},
)
train_sampler = DistributedMixedSampler(
    datasets_info=[{
        "sampler": WeightedRandomSampler(train_weights, len(train_structural_dataset)),
        "dataset": train_structural_dataset, "probability": 1.0,
    }],
    num_replicas=world_size, rank=global_rank, n_examples_per_epoch=TRAIN_EXAMPLES_PER_EPOCH,
)
train_fallback_sampler = WeightedRandomSampler(train_weights, len(train_structural_dataset))
train_dataset_with_fallback, train_sampler_with_fallback = wrap_dataset_and_sampler_with_fallbacks(
    dataset_to_be_wrapped=train_structural_dataset, sampler_to_be_wrapped=train_sampler,
    dataset_to_fallback_to=train_fallback_dataset, sampler_to_fallback_to=train_fallback_sampler,
    n_fallback_retries=10,
)
# Complex crop -> parquet token count is query-chain only; bound by the atom budget.
_MAX_TOKENS_PER_SAMPLE = complex_max_atoms // 7
batched_train_sampler = PaddedTokenBudgetBatchSampler(
    sampler=train_sampler_with_fallback,
    get_num_tokens=lambda idx: _MAX_TOKENS_PER_SAMPLE,
    max_tokens_with_padding=batch_size, shuffle_batches=True,
)

# ── Val dataset / sampler / loader (with fallbacks) ────────────────────────────
val_structural_dataset = make_structural_dataset(val_df, "pn_units_df_val", MPNN_FILTERS, inference_pipeline)
val_fallback_dataset = make_structural_dataset(val_df, "pn_units_df_val", MPNN_FILTERS, inference_pipeline)
val_weights = calculate_weights_for_pdb_dataset_df(
    dataset_df=val_structural_dataset.data, beta=1.0,
    alphas={"a_prot": 1.0, "a_nuc": 0, "a_ligand": 0, "a_loi": 0},
)
val_sampler = DistributedMixedSampler(
    datasets_info=[{
        "sampler": WeightedRandomSampler(val_weights, len(val_structural_dataset)),
        "dataset": val_structural_dataset, "probability": 1.0,
    }],
    num_replicas=world_size, rank=global_rank, n_examples_per_epoch=VAL_EXAMPLES_PER_EPOCH,
)
val_fallback_sampler = WeightedRandomSampler(val_weights, len(val_structural_dataset))
val_dataset_with_fallback, val_sampler_with_fallback = wrap_dataset_and_sampler_with_fallbacks(
    dataset_to_be_wrapped=val_structural_dataset, sampler_to_be_wrapped=val_sampler,
    dataset_to_fallback_to=val_fallback_dataset, sampler_to_fallback_to=val_fallback_sampler,
    n_fallback_retries=10,
)

collator = TokenBudgetAwareFeatureCollator(max_tokens_with_padding=batch_size)
train_loader = DataLoader(train_dataset_with_fallback, batch_sampler=batched_train_sampler,
                          num_workers=NUM_WORKERS, prefetch_factor=2, collate_fn=collator, persistent_workers=True)
val_loaders = {"test_val": DataLoader(val_dataset_with_fallback, sampler=val_sampler_with_fallback,
                                      num_workers=NUM_WORKERS, prefetch_factor=2, collate_fn=collator,
                                      persistent_workers=True)}

# ── Trainer ────────────────────────────────────────────────────────────────────
output_dir = Path(os.environ.get("MPNN_OUTPUT_DIR", ".")) / f"mpnn_output_{run_name}"
output_dir.mkdir(parents=True, exist_ok=True)
# HBondValidationCallback handles all logging (per-epoch PR/ROC/per-type CSV + plots).
# The generic StoreValidationMetricsInDFCallback is NOT used: it expects an example_id from
# the MetricManager path, which the head trainer doesn't use.
callbacks = [HBondValidationCallback(save_dir=output_dir / "val_metrics",
                                     encoding=get_vocab(EXTENDED_VOCAB)["token_encoding"])]
trainer = HBondHeadTrainer(
    extended_vocab=EXTENDED_VOCAB, n_hidden=N_HIDDEN, neg_per_pos=NEG_PER_POS,
    encoder_checkpoint=encoder_ckpt,
    accelerator="gpu", devices_per_node=n_gpus, max_epochs=EPOCHS,
    n_examples_per_epoch=TRAIN_EXAMPLES_PER_EPOCH,
    output_dir=output_dir, callbacks=callbacks, precision="bf16-mixed",
    clip_grad_max_norm=None, find_unused_parameters=True, verbose=True,
)

# Simple constant-LR Adam for the tiny head (no Noam warmup needed); no scheduler.
train_cfg = DictConfig({
    "model": {
        "optimizer": {"_target_": "torch.optim.Adam", "lr": LR,
                      "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
        "lr_scheduler": None,
    }
})

trainer.initialize_or_update_trainer_state({"train_cfg": train_cfg})
trainer.fabric.launch()
trainer.construct_model()
trainer.construct_optimizer()
trainer.construct_scheduler()


class CkptConfig:
    def __init__(self, path, weight_loading_config=None, reset_optimizer=False):
        self.path = path
        self.weight_loading_config = weight_loading_config
        self.reset_optimizer = reset_optimizer


# Resume from the latest checkpoint if one exists (train from where the last run left off).
ckpt_dir = output_dir / "ckpt"
ckpt_config = CkptConfig(path=ckpt_dir) if ckpt_dir.exists() else None

print("Starting H-bond head training...")
trainer.fit(train_loader=train_loader, val_loaders=val_loaders, ckpt_config=ckpt_config)
print("Training completed!")
