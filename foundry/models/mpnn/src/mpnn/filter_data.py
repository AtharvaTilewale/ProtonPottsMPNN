#!/usr/bin/env python3
"""Pre-filter train/val parquet files by checking structure files exist on disk.

Run once before training:
    python foundry/models/mpnn/src/mpnn/filter_data.py
"""

import os
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd

TRAIN_PATH = "/novo/users/cpjb/PHD/conditional_binding/ph/data/mpnn_split/train.parquet"
VAL_PATH = "/novo/users/cpjb/PHD/conditional_binding/ph/data/mpnn_split/val.parquet"
TRAIN_OUT = "/novo/users/cpjb/PHD/conditional_binding/ph/data/mpnn_split/train_df_filtered.parquet"
VAL_OUT = "/novo/users/cpjb/PHD/conditional_binding/ph/data/mpnn_split/val_df_filtered.parquet"


def filter_missing_pdb_files(df: pd.DataFrame, split: str) -> pd.DataFrame:
    exists_mask = pd.Series(
        Parallel(n_jobs=-1, prefer="threads")(
            delayed(os.path.exists)(p)
            for p in tqdm(df["path"], total=len(df), desc=f"## Checking file paths [{split}]")
        ),
        index=df.index,
    )
    n_missing = (~exists_mask).sum()
    if n_missing:
        print(f"[{split}] WARNING: dropping {n_missing}/{len(df)} rows with missing PDB/CIF files.")
    return df[exists_mask].reset_index(drop=True)


if __name__ == "__main__":
    print("Loading train...")
    train_df = pd.read_parquet(TRAIN_PATH)
    train_df = filter_missing_pdb_files(train_df, "train")
    train_df.to_parquet(TRAIN_OUT, index=False)
    print(f"Saved {len(train_df)} rows -> {TRAIN_OUT}")

    print("Loading val...")
    val_df = pd.read_parquet(VAL_PATH)
    val_df = filter_missing_pdb_files(val_df, "val")
    val_df.to_parquet(VAL_OUT, index=False)
    print(f"Saved {len(val_df)} rows -> {VAL_OUT}")
