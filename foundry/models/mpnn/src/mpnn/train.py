#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../../../.ipd/shebang/mpnn_exec.sh" "$0" "$@"'

import contextlib
import os
import sys
from pathlib import Path
import numpy as np
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
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from foundry.callbacks.metrics_logging import StoreValidationMetricsInDFCallback
from foundry.utils.datasets import wrap_dataset_and_sampler_with_fallbacks
from mpnn.collate.feature_collator import TokenBudgetAwareFeatureCollator
from mpnn.pipelines.mpnn import build_mpnn_transform_pipeline
from mpnn.pipelines.potts_mpnn import build_mpnn_transform_pipeline as build_potts_pipeline
from mpnn.transforms.precomputed import make_snapshot_loader
from mpnn.samplers.samplers import PaddedTokenBudgetBatchSampler
from mpnn.callbacks.megascale_benchmark import MegaScaleEnergyCallback
from mpnn.callbacks.pkad_benchmark import PKADBenchmarkCallback
from mpnn.callbacks.neutron_benchmark import NeutronRecoveryCallback
from mpnn.trainers.mpnn import MPNNTrainer
from mpnn.trainers.pottsmpnn import PottsMPNNTrainer
from mpnn.transforms.extended_vocab import get_vocab

args = sys.argv[1:]
model_type = args[0]
run_name = next((a for a in args[1:] if not a.startswith("--")), model_type)
# Source of the Potts single-body field h_i(a): the diagonal of the self-edge's
# pairwise table (default), or a dedicated linear head on the node embedding.
field_source = "node" if "--node-field" in args else "self_edge"

# What the Potts pairwise coupling head sees:
#   "edge"            (default) the edge embedding h_E[i,k] alone            -> head input H
#   "node_edge_node"  concat(h_V[i], h_E[i,k], h_V[j])                       -> head input 3H
# Like field_source, this changes etab_out's weight SHAPE, so it must match the checkpoint on resume --
# keep it in the SLURM launch line, not inferred per-job.
etab_source = args[args.index("--etab-source") + 1] if "--etab-source" in args else "edge"

# Depth of the two Potts heads, set per-job from the SLURM environment, e.g.
#   export POTTS_ETAB_HIDDEN="256 256"   # edge embedding -> 256 -> 256 -> V*V
#   export POTTS_FIELD_HIDDEN="128"      # node embedding -> 128 -> V  (field_source="node" only)
# UNSET (the default) -> None -> a single Linear each, i.e. the original architecture. It is read from
# the environment rather than hard-coded here so that a job RESUMING a 1-layer checkpoint always
# rebuilds 1-layer heads: an MLP head has different state_dict keys (etab_out.0.weight vs
# etab_out.weight) and would fail the strict=True load.
_etab_env = os.environ.get("POTTS_ETAB_HIDDEN", "")
_field_env = os.environ.get("POTTS_FIELD_HIDDEN", "")
etab_hidden = [int(x) for x in _etab_env.split()] if _etab_env else None
field_hidden = [int(x) for x in _field_env.split()] if _field_env else None

# WHICH protonation vocabulary to label with -- a NAME, not a flag. See transforms/extended_vocab.py:
#   "v3"  ev3's original salt-bridge labeller (cutoff_HA 3.0, no capability filter, deterministic)
#   "v4"  the rewritten atom-level labeller + salt-bridge fallback (what potts_sb_* trains on)
# The vocabulary carries its own H-bond cutoff, capability filter and train-time determinism, because
# those decide which bonds the classifier ever sees -- they are part of the vocabulary, not free knobs.
#
# IT MUST MATCH THE CHECKPOINT: the encoder sees these tokens in S, and unlike field_source there is no
# state_dict key to catch a mismatch. `--ev` (the legacy flag, still used by the queued jobs) selects
# the default vocabulary; `--vocab v3` overrides it.
_vocab_arg = args[args.index("--vocab") + 1] if "--vocab" in args else None
extended_vocab = _vocab_arg or ("v6" if "--ev" in args else None)   # None -> standard 21-token vocab

# Protonation OPERATING POINT (v6 only). EV6_HIS_PROB_THR / EV6_ACID_PROB_THR override the P-vs-D
# probability cut baked in thresholds.json, so the labels the model trains on can be swept per run WITHOUT
# rebuilding the snapshot cache: ApplyProtonationThreshold re-derives protonation_label from each snapshot's
# persisted flaml_p just before S is built (requires scripts/augment_snapshots_flaml.py to have run). Unset
# -> None -> the baked default label is used unchanged. Only prob_thr is swept; sd_cut stays fixed. Read
# from the env (not a flag) so the SAME value reaches the featurization tail AND the v6 live predictor in the
# callbacks (extended_vocab_v6._env_prob_thr), keeping every label path at one operating point.
_his_thr_env = os.environ.get("EV6_HIS_PROB_THR", "").strip()
_acid_thr_env = os.environ.get("EV6_ACID_PROB_THR", "").strip()
his_prob_thr = float(_his_thr_env) if _his_thr_env else None
acid_prob_thr = float(_acid_thr_env) if _acid_thr_env else None

# PROTONATION-STATE auxiliary loss (L_state), v6 / extended-vocab only. The full-vocabulary NLL at a
# titratable position factorises as -log p(state | parent) + -log p(parent), and the second term carries
# ~80% of it, so the protonation call -- the only thing the model is used for at design time -- is a
# low-single-digit fraction of the objective. MPNN_STATE_LOSS_WEIGHT scales the group-renormalised state
# term (0 = off, and then the loss is bit-for-bit the pre-existing one); MPNN_STATE_LOSS_HEADS picks which
# heads it supervises ("pot,dec" by default -- the Potts head is what the design engine reads, the decoder
# head what the autoregressive arm reads). Env-driven for the same reason as the thresholds above: one
# knob reaches every path, and it is recorded in train_cfg so the checkpoint self-documents its objective.
_state_w_env = os.environ.get("MPNN_STATE_LOSS_WEIGHT", "").strip()
_state_heads_env = os.environ.get("MPNN_STATE_LOSS_HEADS", "").strip()
state_loss_weight = float(_state_w_env) if _state_w_env else 0.0
state_loss_heads = tuple(h.strip() for h in _state_heads_env.split(",") if h.strip()) or ("pot", "dec")

if model_type == "protein_mpnn":
    batch_size = 10000
    train_date_cutoff = "2021-08-02"
    clip_grad_max_norm = None
    train_structure_noise_default = 0.2
elif model_type == "ligand_mpnn":
    batch_size = 6000
    train_date_cutoff = "2022-12-16"
    clip_grad_max_norm = 1.0
    train_structure_noise_default = 0.1
elif model_type == "potts_mpnn":
    batch_size = 10000
    train_date_cutoff = "2022-12-16"
    clip_grad_max_norm = None
    train_structure_noise_default = 0.2
    # Complex crop knobs (extended-vocab only). complex_max_atoms must stay in
    # sync with the batch sampler's per-sample token estimate below.
    complex_max_atoms = 6000
    complex_pair_probability = 0.9
else:
    raise ValueError(f"Unknown model_type: {model_type}")


def create_noam_scheduler(optimizer, d_model, warmup_steps=4000, factor=2):
    """
    Create a NoamOpt-style scheduler using standard PyTorch components.

    Args:
        optimizer: PyTorch optimizer
        d_model: Model dimension (for scaling)
        warmup_steps: Number of warmup steps
        factor: Scaling factor

    Returns:
        LambdaLR scheduler that implements NoamOpt schedule
    """

    def noam_lambda(step):
        # NoamOpt formula: factor * (d_model ** (-0.5)) * min(step ** (-0.5), step * warmup ** (-1.5))
        base_lr = factor * (d_model ** (-0.5))
        if step == 0:
            return 0.0  # Start with zero learning rate

        # Calculate the schedule component
        schedule = min(step ** (-0.5), step * warmup_steps ** (-1.5))
        return base_lr * schedule

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)


def get_num_tokens(df, idx):
    """
    Extract the number of non-atomized tokens for a given index.

    Args:
        df: DataFrame containing the dataset
        idx: Index to extract token count from

    Returns:
        Number of non-atomized tokens for the sample at idx
    """
    if isinstance(idx, (list, tuple)):
        # If idx is a list/tuple, return the first element's token count
        idx = idx[0]
    return df.iloc[idx]["n_non_atomized_tokens"]


# Which dataset to train on. "pdb" (default) keeps the original experimental-structure
# behavior untouched; "afdb" trains on the AlphaFold-DB dimer complexes.
DATASET = os.environ.get("MPNN_DATASET", "pdb")

# Filters that apply to any dataset.
COMMON_FILTERS = [
    "n_non_atomized_tokens >= 30",
    f"n_non_atomized_tokens < {batch_size}",
    "cluster.notnull() and cluster != 'nan'",
    "n_prot >= 1",
    "assembly_id == '1'",
]
# Experimental-quality gates. AFDB entries are *predictions*: they have no resolution, no
# deposition date and no experimental method, so these gates only make sense for the PDB.
PDB_QUALITY_FILTERS = [
    "resolution < 3.5 and ~method.str.contains('NMR')",
    "method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY']",
]

# ── Batch composition (afdb_pdb only) ──────────────────────────────────────────────────
# These become the `probability` of each child sampler in the DistributedMixedSampler, i.e.
# the expected share of each batch drawn from that subset — independent of how many rows the
# subset actually has. Heterodimers are only ~11% of the AFDB structures, so the default
# below deliberately over-samples them relative to their natural abundance.
#   AFDB_FRACTION=0.5        -> half of every batch is AFDB, half PDB
#   HETERODIMER_FRACTION=0.5 -> heterodimers are half of the AFDB half
AFDB_FRACTION = float(os.environ.get("AFDB_FRACTION", "0.5"))
HETERODIMER_FRACTION = float(os.environ.get("HETERODIMER_FRACTION", "0.5"))

n_gpus = torch.cuda.device_count()

# Distributed metadata. READ SLURM FIRST — this is load-bearing:
#   * `sbatch` sets SLURM_NTASKS, so Lightning's SLURMEnvironment.detect() fires even when the job is
#     launched with a plain `python` (no srun). It then reports world_size=SLURM_NTASKS and, because
#     creates_processes_externally=True, NEVER spawns the per-GPU workers. With --ntasks=1 that means
#     one process on one GPU while the rest idle — silently, since Lightning only warns when
#     SLURM_NTASKS is absent. The job MUST therefore be launched as `srun` with one task per GPU.
#   * Under srun, SLURM sets SLURM_PROCID/SLURM_LOCALID but NOT RANK/LOCAL_RANK, so falling back to
#     those would collapse every task to rank 0 and hand every GPU the same DistributedSampler shard.
# Falls back to torchrun's env, then single-process, when not under SLURM.
world_size = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", max(n_gpus, 1))))
global_rank = int(os.environ.get("SLURM_PROCID",
                                os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))))
num_nodes = int(os.environ.get("SLURM_NNODES", 1))
# Lightning requires devices * num_nodes == world_size.
devices_per_node = max(1, world_size // num_nodes)
print(f"Using {n_gpus} visible GPU(s); world_size={world_size} rank={global_rank} "
      f"num_nodes={num_nodes} devices_per_node={devices_per_node}")

if DATASET == "pdb":
    # Pre-split, pre-filtered PDB parquets (run filter_data.py once to generate these).
    train_path = "/novo/users/cpjb/PHD/conditional_binding/ph/data/mpnn_split/train_df_filtered.parquet"
    val_path = "/novo/users/cpjb/PHD/conditional_binding/ph/data/mpnn_split/val_df_filtered.parquet"
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    MPNN_FILTERS = PDB_QUALITY_FILTERS + COMMON_FILTERS
    MPNN_TRAIN_FILTERS = [f"deposition_date < '{train_date_cutoff}'"] + MPNN_FILTERS

elif DATASET == "afdb_pdb":
    # PDB experimental + AFDB predictions, MIXED per batch by train_mixture: AFDB_FRACTION sets AFDB's share
    # of each batch, HETERODIMER_FRACTION the homo/hetero split within AFDB. Built in memory from the two
    # already-split source tables + `subset` tags -- no pre-clustered joint parquet needed. The AFDB half is
    # the LEAK-FREE split (scripts/split_afdb.py): every AFDB cluster that touches PDB val is held out of
    # afdb_train (shared PDB+AFDB namespace), so mixing it with PDB train never leaks the PDB val set. PDB
    # keeps its own cluster namespace -- fine, since per-subset weighting never compares across subsets.
    afdb_dir = Path("/novo/users/cpjb/rdd/cpjb/afdb_complexes")
    pdb_train = pd.read_parquet("data/mpnn_split/train_df_filtered.parquet")
    pdb_train["subset"] = "pdb"
    afdb_train = pd.read_parquet(afdb_dir / "afdb_train.parquet")   # subset (homo/hetero) already tagged
    train_df = pd.concat([pdb_train, afdb_train[pdb_train.columns]], ignore_index=True)

    pdb_val = pd.read_parquet("data/mpnn_split/val_df_filtered.parquet")
    pdb_val["subset"] = "pdb"
    afdb_val = pd.read_parquet(afdb_dir / "afdb_val.parquet")
    val_df = pd.concat([pdb_val, afdb_val[pdb_val.columns]], ignore_index=True)
    print(f"AFDB+PDB: train {len(train_df):,} ({dict(train_df['subset'].value_counts())}) | "
          f"val {len(val_df):,} rows")

    # Both kinds of row live in one table, so the experimental gates must apply to PDB rows
    # only: an AFDB row has resolution NaN (and NaN < 3.5 is False) and deposition_date NaT,
    # so the PDB expressions on their own would silently drop every predicted model.
    MPNN_FILTERS = [
        "method == 'ALPHAFOLD' or "
        "(resolution < 3.5 and method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY'])"
    ] + COMMON_FILTERS
    MPNN_TRAIN_FILTERS = [
        f"method == 'ALPHAFOLD' or deposition_date < '{train_date_cutoff}'"
    ] + MPNN_FILTERS

elif DATASET == "afdb":
    # AlphaFold-DB structures ONLY, split leak-free vs the PDB val set: scripts/split_afdb.py assigns every
    # AFDB row its SHARED PDB+AFDB sequence cluster and keeps any cluster that touches PDB val out of
    # afdb_train (so AFDB training can't leak the PDB benchmark). `subset` (afdb_homodimer/afdb_heterodimer)
    # is preserved, so train_mixture's non-pdb branch mixes homo/hetero by HETERODIMER_FRACTION (the empty
    # `pdb` subset drops out and the rest renormalise).
    afdb_dir = Path("/novo/users/cpjb/rdd/cpjb/afdb_complexes")
    train_df = pd.read_parquet(afdb_dir / "afdb_train.parquet")
    val_df = pd.read_parquet(afdb_dir / "afdb_val.parquet")
    print(f"AFDB: train {len(train_df):,} ({dict(train_df['subset'].value_counts())}) | val {len(val_df):,} rows")

    # AFDB rows are all predictions (no resolution / deposition_date gates); just the common structural gates.
    MPNN_FILTERS = ["method == 'ALPHAFOLD'"] + COMMON_FILTERS
    MPNN_TRAIN_FILTERS = MPNN_FILTERS

else:
    raise ValueError(f"Unknown MPNN_DATASET: {DATASET!r} (expected 'pdb', 'afdb' or 'afdb_pdb')")


# Precomputed annotations. Set MPNN_PRECOMPUTED_DIR to a snapshot cache (built by
# scripts/build_snapshots.py) so the loader returns the cleaned, cropped, annotated structure directly.
# Training/validation are PRECOMPUTED-ONLY: an uncached example is SKIPPED (the dataset fallback draws
# another), never annotated live in a DataLoader worker -- EV6's OpenMP tree models deadlock after fork.
# So EVERYTHING is precomputed offline; EV6 runs live only in the PKAD/MegaScale callbacks (main process).
# Off (None) => today's live pipeline, unchanged.
precomputed_dir = os.environ.get("MPNN_PRECOMPUTED_DIR") or None
precomputed = bool(precomputed_dir)


if extended_vocab and model_type == "potts_mpnn":
    train_pipeline = build_potts_pipeline(
        model_type="potts_mpnn",
        is_inference=False,
        minimal_return=True,
        train_structure_noise_default=train_structure_noise_default,
        protonation_label_rate=0.5,
        extended_vocab=extended_vocab,
        # AUGMENTATION: when the vocabulary allows it, sample the genuinely-ambiguous protonation calls
        # each epoch (ambiguous His/acid H-bond roles, and symmetric shared-proton dyads -> one -P /
        # one -D); seed=None re-samples per call. This is PART OF THE VOCABULARY: v3 trained fully
        # deterministically, v4 samples. Validation/inference is always deterministic (stable S).
        deterministic=get_vocab(extended_vocab)["train_deterministic"],
        # Sweepable protonation operating point (v6 + augmented snapshots); None/None -> baked label.
        his_prob_thr=his_prob_thr,
        acid_prob_thr=acid_prob_thr,
        complex_pair_probability=complex_pair_probability,
        complex_max_atoms=complex_max_atoms,
        complex_min_atom_contacts=10,
        complex_max_chains=None,
        # When precomputed, the crop + annotation are baked into the snapshot at a FIXED p=1 multimer
        # composition, so complex_pair_probability above is unused (no crop runs) -- training is effectively
        # p=1. Only noise + label-masking augment per epoch.
        precomputed=precomputed,
        precomputed_dir=precomputed_dir,
    )
    inference_pipeline = build_potts_pipeline(
        model_type="potts_mpnn",
        is_inference=False,
        minimal_return=True,
        train_structure_noise_default=0.0,
        protonation_label_rate=1.0,
        extended_vocab=extended_vocab,
        deterministic=True,          # reproducible validation labels
        # Validation labels must sit at the SAME operating point the model trains on.
        his_prob_thr=his_prob_thr,
        acid_prob_thr=acid_prob_thr,
        complex_pair_probability=1.0,
        complex_max_atoms=complex_max_atoms,
        complex_min_atom_contacts=10,
        precomputed=precomputed,
        precomputed_dir=precomputed_dir,
    )
else:
    pipeline_model = "protein_mpnn" if model_type != "ligand_mpnn" else model_type
    train_pipeline = build_mpnn_transform_pipeline(
        model_type=pipeline_model,
        is_inference=False,
        minimal_return=True,
        train_structure_noise_default=train_structure_noise_default,
    )
    inference_pipeline = build_mpnn_transform_pipeline(
        model_type=pipeline_model, is_inference=True, minimal_return=True
    )

def _silent(loader_fn):
    def wrapper(row):
        with open(os.devnull, "w") as _null, \
             contextlib.redirect_stdout(_null), \
             contextlib.redirect_stderr(_null):
            return loader_fn(row)
    return wrapper


def build_structural_dataset(data, name, filters, transform):
    ds = StructuralDatasetWrapper(
        dataset=PandasDataset(
            data=data,
            id_column="example_id",
            name=name,
            filters=filters,
        ),
        dataset_parser=GenericDFParser(
            example_id_colname="example_id",
            path_colname="path",
            assembly_id_colname="assembly_id",
        ),
        transform=transform,
        cif_parser_args={
            **STANDARD_PARSER_ARGS,
            "add_bond_types_from_struct_conn": (),
            "load_from_cache": False,
            "save_to_cache": False,
            "cache_dir": None,
        },
    )
    # When precomputed, return the snapshot for a row (skipping the CIF parse) with a live fall-back that
    # self-populates on a miss. Wrap BEFORE _silent so the miss-path annotation stays quiet too.
    if precomputed_dir:
        ds.loader = make_snapshot_loader(precomputed_dir)   # hit -> snapshot; miss -> raise (skipped)
    ds.loader = _silent(ds.loader)
    return ds


def train_mixture(df):
    """(name, rows, probability) per subset the train sampler mixes over.

    `probability` is the share of each batch drawn from that subset, so it is set by the
    AFDB/heterodimer knobs rather than by how many rows the subset happens to contain.
    """
    if DATASET == "pdb":
        return [("pn_units_df_train", df, 1.0)]

    probs = {
        "pdb": 1.0 - AFDB_FRACTION,
        "afdb_homodimer": AFDB_FRACTION * (1.0 - HETERODIMER_FRACTION),
        "afdb_heterodimer": AFDB_FRACTION * HETERODIMER_FRACTION,
    }
    mixture = [(name, df[df["subset"] == name], p) for name, p in probs.items()]
    # A subset that is empty or switched off would make the probabilities stop summing to 1.
    mixture = [(n, rows, p) for n, rows, p in mixture if p > 0 and len(rows) > 0]
    total = sum(p for _, _, p in mixture)
    return [(n, rows, p / total) for n, rows, p in mixture]


# One dataset per subset. DistributedMixedSampler maps each child sampler's indices into the
# ConcatDataset by cumulative length, so datasets_info and the ConcatDataset MUST be built in
# the same order — hence both are driven off this one list.
train_subsets = train_mixture(train_df)

train_datasets, train_datasets_info, train_weights_per_subset = [], [], []
for name, rows, probability in train_subsets:
    ds = build_structural_dataset(rows, name, MPNN_TRAIN_FILTERS, train_pipeline)
    weights = calculate_weights_for_pdb_dataset_df(
        dataset_df=ds.data,
        beta=1.0,  # For chains
        alphas={"a_prot": 1.0, "a_nuc": 0, "a_ligand": 0, "a_loi": 0},
    )
    train_datasets.append(ds)
    train_weights_per_subset.append(np.asarray(weights, dtype=float))
    train_datasets_info.append(
        {
            "sampler": WeightedRandomSampler(weights, len(ds)),
            "dataset": ds,
            "probability": probability,
        }
    )
    print(f"  train subset [{name}]: {len(ds):,} examples, {100 * probability:.1f}% of each batch")

# Fallback mirrors the mixture exactly, so a retry redraws from the same distribution.
train_fallback_datasets = [
    build_structural_dataset(rows, f"{name}_fallback", MPNN_TRAIN_FILTERS, train_pipeline)
    for name, rows, _ in train_subsets
]

train_structural_dataset = ConcatDataset(train_datasets)
train_fallback_dataset = ConcatDataset(train_fallback_datasets)
# Indices coming out of the sampler are ConcatDataset indices, so the token lookup has to see
# the subset frames concatenated in the same order.
train_tokens_df = pd.concat([ds.data for ds in train_datasets], ignore_index=True)
train_weights = np.concatenate(train_weights_per_subset)

train_sampler = DistributedMixedSampler(
    datasets_info=train_datasets_info,
    num_replicas=world_size,
    rank=global_rank,
    n_examples_per_epoch=20000, # 20000
)

train_fallback_sampler = WeightedRandomSampler(
    train_weights, len(train_structural_dataset)
)

train_dataset_with_fallback, train_sampler_with_fallback = (
    wrap_dataset_and_sampler_with_fallbacks(
        dataset_to_be_wrapped=train_structural_dataset,
        sampler_to_be_wrapped=train_sampler,
        dataset_to_fallback_to=train_fallback_dataset,
        sampler_to_fallback_to=train_fallback_sampler,
        n_fallback_retries=10,
    )
)

# For complex training the parquet token count is query-chain only.
# Use the atom budget as the upper-bound estimate: max_atoms / 7 heavy atoms per residue.
# This prevents the batch sampler from over-packing and silently dropping samples.
_MAX_TOKENS_PER_SAMPLE = (complex_max_atoms // 7) if (extended_vocab and model_type == "potts_mpnn") else None
batched_train_sampler = PaddedTokenBudgetBatchSampler(
    sampler=train_sampler_with_fallback,
    get_num_tokens=lambda idx: (
        _MAX_TOKENS_PER_SAMPLE
        if _MAX_TOKENS_PER_SAMPLE is not None
        else get_num_tokens(train_tokens_df, idx)
    ),
    max_tokens_with_padding=batch_size,
    shuffle_batches=True,
)

# Create val dataset with fallback
val_structural_dataset = StructuralDatasetWrapper(
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
    cif_parser_args={
        **STANDARD_PARSER_ARGS,
        "add_bond_types_from_struct_conn": (),
        "load_from_cache": False,
        "save_to_cache": False,
        "cache_dir": None,
    },
)

if precomputed_dir:
    val_structural_dataset.loader = make_snapshot_loader(precomputed_dir)   # miss -> raise (skipped)
val_structural_dataset.loader = _silent(val_structural_dataset.loader)

# Create val sampler with fallback
val_weights = calculate_weights_for_pdb_dataset_df(
    dataset_df=val_structural_dataset.data,
    beta=1.0,  # For chains
    alphas={"a_prot": 1.0, "a_nuc": 0, "a_ligand": 0, "a_loi": 0},
)

val_sampler = DistributedMixedSampler(
    datasets_info=[
        {
            "sampler": WeightedRandomSampler(val_weights, len(val_structural_dataset)),
            "dataset": val_structural_dataset,
            "probability": 1.0,
        }
    ],
    num_replicas=world_size,
    rank=global_rank,
    n_examples_per_epoch=100,
)   

val_fallback_dataset = StructuralDatasetWrapper(
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
    cif_parser_args={
        **STANDARD_PARSER_ARGS,
        "add_bond_types_from_struct_conn": (),
        "load_from_cache": False,
        "save_to_cache": False,
        "cache_dir": None,
    },
)
val_fallback_dataset.loader = _silent(val_fallback_dataset.loader)

val_fallback_sampler = WeightedRandomSampler(val_weights, len(val_structural_dataset))

val_dataset_with_fallback, val_sampler_with_fallback = (
    wrap_dataset_and_sampler_with_fallbacks(
        dataset_to_be_wrapped=val_structural_dataset,
        sampler_to_be_wrapped=val_sampler,
        dataset_to_fallback_to=val_fallback_dataset,
        sampler_to_fallback_to=val_fallback_sampler,
        n_fallback_retries=10,
    )
)

# Create collator
collator = TokenBudgetAwareFeatureCollator(max_tokens_with_padding=batch_size)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset_with_fallback,
    batch_sampler=batched_train_sampler,
    num_workers=4,
    prefetch_factor=1,
    collate_fn=collator,
    persistent_workers=True,
)

val_loaders = {
    "test_val": DataLoader(
        val_dataset_with_fallback,
        sampler=val_sampler_with_fallback,
        num_workers=4,
        prefetch_factor=1,
        collate_fn=collator,
        persistent_workers=True,
    )
}

# Create output directory for logs and checkpoints
# MPNN_OUTPUT_DIR relocates runs off the project root (default ".") -- e.g. to the `trained_models`
# symlink on the big filesystem. The ckpt/val_metrics dirs hang off this, so RESUMING a run must use the
# same MPNN_OUTPUT_DIR (the checkpoint is looked up at output_dir/ckpt).
output_dir = Path(os.environ.get("MPNN_OUTPUT_DIR", ".")) / f"mpnn_output_{run_name}"
output_dir.mkdir(parents=True, exist_ok=True)   # parents=True: MPNN_OUTPUT_DIR base may not exist yet

# Create CSV logging callback
csv_callback = StoreValidationMetricsInDFCallback(
    save_dir=output_dir / "val_metrics", metrics_to_save="all"
)

# Create trainer with minimal configuration for testing
if model_type == "potts_mpnn":
    megascale_callback = MegaScaleEnergyCallback(
        csv_path=Path("external/PottsMPNN/energy_benchmark_datasets/megascale_test_subset.csv"),
        pdb_dir=Path("data/energy_benchmark_datasets/megascale_pdbs"),
        model_type="potts_mpnn",
        save_dir=output_dir / "val_metrics",
        extended_vocab=extended_vocab,
    )
    potts_callbacks = [csv_callback, megascale_callback]
    if extended_vocab:
        pkad_callback = PKADBenchmarkCallback(
            csv_path=Path("data/PKAD/PKAD-R-v1.0_2026-04-22T16_0755.441Z.csv"),
            pdb_dir=Path("data/PKAD/pdb_cache"),
            save_dir=output_dir / "val_metrics",
            # score in the same vocabulary the model trains on
            extended_vocab=extended_vocab,
        )
        potts_callbacks.append(pkad_callback)
        # Held-out protonation-state RECOVERY vs neutron truth (companion to PKAD's pKa correlation).
        neutron_callback = NeutronRecoveryCallback(
            save_dir=output_dir / "val_metrics",
            extended_vocab=extended_vocab,
        )
        potts_callbacks.append(neutron_callback)
    trainer = PottsMPNNTrainer(
        model_type=model_type,
        extended_vocab=extended_vocab,
        field_source=field_source,
        etab_source=etab_source,
        etab_hidden=etab_hidden,
        field_hidden=field_hidden,
        # Weight 0 (the default) does not even construct the state term, so the objective is unchanged.
        loss={
            "state_loss_weight": state_loss_weight,
            "state_loss_heads": state_loss_heads,
        },
        accelerator="gpu",
        devices_per_node=devices_per_node,
        num_nodes=num_nodes,
        max_epochs=500,
        output_dir=output_dir,
        callbacks=potts_callbacks,
        precision="bf16-mixed",
        clip_grad_max_norm=clip_grad_max_norm,
        verbose=True,
    )

else:
    trainer = MPNNTrainer(
        model_type=model_type,  
        accelerator="gpu",
        devices_per_node=1,
        max_epochs=500,
        output_dir=output_dir,
        callbacks=[csv_callback],
        precision="bf16-mixed",
        clip_grad_max_norm=clip_grad_max_norm,
    )

# Create minimal train_cfg for optimizer and scheduler construction
train_cfg = DictConfig(
    {
        # The vocabulary this run trains on, recorded so the CHECKPOINT SELF-IDENTIFIES. Everything else
        # follows from this name: the token set (and hence vocab_size -- v6 is 30, v3/v4 are 32) and the
        # aa_protonated / aa_deprotonated maps. Without it a checkpoint's vocabulary is unrecoverable
        # (v3 and v4 are both 32-token, so the weights cannot tell them apart) and scoring it later --
        # PKAD, sequence recovery -- could silently use the wrong tokens.
        "extended_vocab": extended_vocab,
        # The protonation operating point this run trained on, recorded so the CHECKPOINT SELF-DOCUMENTS it
        # (None => thresholds.json default). Without it a swept-threshold model is indistinguishable from a
        # default one, and later scoring could silently compare it at the wrong prob_thr.
        "his_prob_thr": his_prob_thr,
        "acid_prob_thr": acid_prob_thr,
        # The OBJECTIVE this run trained on, recorded for the same reason as the operating point above:
        # without it a state-loss model is indistinguishable from a baseline one at scoring time.
        "state_loss_weight": state_loss_weight,
        "state_loss_heads": list(state_loss_heads),
        "model": {
            "optimizer": {
                "_target_": "torch.optim.Adam",
                "lr": 1.0,  # This will be overridden by the NoamOpt scheduler
                "betas": [0.9, 0.98],  # NoamOpt uses (0.9, 0.98)
                "eps": 1e-9,  # NoamOpt uses 1e-9
                "weight_decay": 0.0,
            },
            "lr_scheduler": {
                "_target_": "__main__.create_noam_scheduler",
                "d_model": 128,  # Adjust based on your model's hidden dimension
                "warmup_steps": 4000,
                "factor": 2,
            },
        }
    }
)

# Initialize trainer state with train_cfg
trainer.initialize_or_update_trainer_state({"train_cfg": train_cfg})

# Launch Fabric (this sets up the distributed environment)
trainer.fabric.launch()

# Construct model
trainer.construct_model()

# Construct optimizer and scheduler
trainer.construct_optimizer()
trainer.construct_scheduler()


class CkptConfig:
    def __init__(self, path, weight_loading_config=None, reset_optimizer=False):
        self.path = path
        self.weight_loading_config = weight_loading_config
        self.reset_optimizer = reset_optimizer


ckpt_dir = output_dir / "ckpt"
if ckpt_dir.exists():
    ckpt_config = CkptConfig(
        path=ckpt_dir, weight_loading_config=None, reset_optimizer=False
    )
else:
    ckpt_config = None

# Run the full training using fit method
print("Starting training...")
trainer.fit(train_loader=train_loader, val_loaders=val_loaders, ckpt_config=ckpt_config)
print("Training completed!")
