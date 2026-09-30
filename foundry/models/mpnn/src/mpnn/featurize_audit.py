#!/usr/bin/env python3
"""Standalone featurization audit: run the full transform pipeline on each example
without starting model training, logging which structures pass/fail and why.

Usage:
    python foundry/models/mpnn/src/mpnn/featurize_audit.py [--split train|val] \
        [--max-examples N] [--workers N]

Outputs:
    data/mpnn_split/featurization_audit_{split}.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from atomworks.io.parser import STANDARD_PARSER_ARGS
from atomworks.ml.datasets.pandas_dataset import PandasDataset, StructuralDatasetWrapper
from atomworks.ml.datasets.parsers.default_metadata_row_parsers import GenericDFParser

from mpnn.pipelines.mpnn import build_mpnn_transform_pipeline

PROJECT_DIR = Path("/novo/users/cpjb/PHD/conditional_binding/ph")
DATA_DIR = PROJECT_DIR / "data" / "mpnn_split"

BATCH_SIZE = 10000
TRAIN_DATE_CUTOFF = "2021-08-02"

MPNN_FILTERS = [
    "resolution < 3.5 and ~method.str.contains('NMR')",
    "n_non_atomized_tokens >= 30",
    "cluster.notnull() and cluster != 'nan'",
    "method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY']",
    f"n_non_atomized_tokens <= {BATCH_SIZE}",
    "n_prot == 1",
    "assembly_id == '1'",
]

MPNN_TRAIN_FILTERS = [f"deposition_date < '{TRAIN_DATE_CUTOFF}'"] + MPNN_FILTERS


def build_dataset(df: pd.DataFrame, filters: list[str], is_inference: bool) -> StructuralDatasetWrapper:
    pipeline = build_mpnn_transform_pipeline(
        model_type="protein_mpnn",
        is_inference=is_inference,
        minimal_return=True,
        train_structure_noise_default=0.2,
    )
    return StructuralDatasetWrapper(
        dataset=PandasDataset(
            data=df,
            id_column="example_id",
            name="featurize_audit",
            filters=filters,
        ),
        dataset_parser=GenericDFParser(
            example_id_colname="example_id",
            path_colname="path",
            assembly_id_colname="assembly_id",
        ),
        transform=pipeline,
        cif_parser_args={
            **STANDARD_PARSER_ARGS,
            "add_bond_types_from_struct_conn": (),
            "load_from_cache": False,
            "save_to_cache": False,
            "cache_dir": None,
        },
    )


def process_chunk(
    indices: list[int],
    filtered_df: pd.DataFrame,
    is_inference: bool,
    worker_id: int,
) -> list[dict]:
    """Featurize a subset of indices. Each worker builds its own pipeline and
    dataset so StructuralDatasetWrapper never needs to be pickled."""
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    dataset = build_dataset(filtered_df, filters=[], is_inference=is_inference)
    meta_df = filtered_df.reset_index(drop=True)

    records = []
    for i in tqdm(indices, desc=f"worker-{worker_id}", position=worker_id, leave=False):
        row = meta_df.iloc[i]
        record = {
            "example_id": row.get("example_id", str(i)),
            "path": row.get("path", ""),
            "n_tokens_expected": row.get("n_non_atomized_tokens", None),
            "n_tokens_actual": None,
            "status": None,
            "error_type": None,
            "error_msg": None,
        }
        try:
            sample = dataset[i]
            record["n_tokens_actual"] = int(sample["input_features"]["S"].shape[0])
            record["status"] = "ok"
        except Exception as e:
            record["status"] = "error"
            record["error_type"] = type(e).__name__
            record["error_msg"] = str(e).split("\n")[0][:200]
        records.append(record)

    return records


def run_audit(
    filtered_df: pd.DataFrame,
    is_inference: bool,
    max_examples: int | None,
    n_workers: int,
    out_path: Path,
) -> pd.DataFrame:
    """Run featurization audit, writing results incrementally as chunks complete."""
    n = min(len(filtered_df), max_examples) if max_examples else len(filtered_df)
    print(f"Auditing {n} / {len(filtered_df)} examples with {n_workers} worker(s)...")

    all_indices = list(range(n))
    chunk_size = (n + n_workers - 1) // n_workers
    chunks = [all_indices[i : i + chunk_size] for i in range(0, n, chunk_size)]

    def _iter_chunks():
        if n_workers == 1:
            yield process_chunk(all_indices, filtered_df, is_inference, worker_id=0)
        else:
            # return_as="generator_unordered" yields each chunk result as it finishes.
            yield from Parallel(
                n_jobs=n_workers, prefer="processes", return_as="generator_unordered"
            )(
                delayed(process_chunk)(chunk, filtered_df, is_inference, wid)
                for wid, chunk in enumerate(chunks)
            )

    all_records = []
    nonlocal_header = [True]  # mutable flag for write_header inside loop

    for chunk_records in tqdm(_iter_chunks(), total=len(chunks), desc="chunks done", position=n_workers):
        chunk_df = pd.DataFrame(chunk_records)
        chunk_df.to_csv(out_path, mode="a", header=nonlocal_header[0], index=False)
        nonlocal_header[0] = False
        all_records.extend(chunk_records)

    return pd.DataFrame(all_records)


def print_summary(results: pd.DataFrame) -> None:
    total = len(results)
    ok = (results["status"] == "ok").sum()
    errors = results[results["status"] == "error"]

    print(f"\n{'='*60}")
    print(f"FEATURIZATION AUDIT SUMMARY")
    print(f"{'='*60}")
    print(f"Total examples audited : {total}")
    print(f"Successful             : {ok} ({100*ok/total:.1f}%)")
    print(f"Failed                 : {len(errors)} ({100*len(errors)/total:.1f}%)")

    if len(errors):
        print(f"\nError breakdown:")
        for error_type, count in errors["error_type"].value_counts().items():
            print(f"  {error_type:<40} {count:>5}")

        print(f"\nTop error messages:")
        for msg, count in errors["error_msg"].value_counts().head(10).items():
            print(f"  [{count:>4}x] {msg}")

    ok_results = results[results["status"] == "ok"]["n_tokens_actual"].dropna()
    if len(ok_results):
        print(f"\nToken length distribution (successful):")
        print(f"  min    : {ok_results.min():.0f}")
        print(f"  median : {ok_results.median():.0f}")
        print(f"  p95    : {ok_results.quantile(0.95):.0f}")
        print(f"  max    : {ok_results.max():.0f}")
        over_budget = (ok_results > BATCH_SIZE).sum()
        print(f"  > {BATCH_SIZE} tokens: {over_budget} ({100*over_budget/len(ok_results):.1f}%)")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit featurization pipeline per example")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of examples (for quick checks)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes")
    args = parser.parse_args()

    parquet_name = "train_df_filtered.parquet" if args.split == "train" else "val_df_filtered.parquet"
    parquet_path = DATA_DIR / parquet_name
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"{parquet_path} not found — run filter_data.py first:\n"
            "  python foundry/models/mpnn/src/mpnn/filter_data.py"
        )

    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows from parquet")

    # Apply filters once in the main process; workers receive the pre-filtered df.
    filters = MPNN_TRAIN_FILTERS if args.split == "train" else MPNN_FILTERS
    is_inference = args.split == "val"
    tmp_dataset = build_dataset(df, filters, is_inference)
    filtered_df = tmp_dataset.data.reset_index(drop=True)
    print(f"Examples after filters: {len(filtered_df)}")

    out_path = DATA_DIR / f"featurization_audit_{args.split}.csv"
    out_path.unlink(missing_ok=True)  # start fresh so append mode doesn't stack old runs

    results = run_audit(filtered_df, is_inference, args.max_examples, args.workers, out_path)
    print_summary(results)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
