import contextlib
import os
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from atomworks.constants import DICT_THREE_TO_ONE
from atomworks.io.parser import STANDARD_PARSER_ARGS
from atomworks.ml.datasets.pandas_dataset import PandasDataset, StructuralDatasetWrapper
from atomworks.ml.datasets.parsers.default_metadata_row_parsers import GenericDFParser
from foundry.callbacks.callback import BaseCallback
from foundry.utils.ddp import RankedLogger
from mpnn.pipelines.mpnn import build_mpnn_transform_pipeline as _build_standard_pipeline
from mpnn.pipelines.potts_mpnn import build_mpnn_transform_pipeline as _build_potts_pipeline

ranked_logger = RankedLogger(__name__, rank_zero_only=True)

# Derived from atomworks.constants.DICT_THREE_TO_ONE at import time — stays in sync with vocabulary changes.
_ONE_TO_THREE: dict[str, str] = {v: k for k, v in DICT_THREE_TO_ONE.items()}


class MegaScaleEnergy:
    """Utilities for MegaScale mutation parsing using foundry vocabulary."""

    @staticmethod
    def parse_mut_type(mut_type_str: str) -> tuple[int, str, str]:
        """Parse 'E0Q' → (pos=0, wt_aa='E', mut_aa='Q'). Position is 0-indexed."""
        m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mut_type_str)
        if m is None:
            raise ValueError(f"Cannot parse mut_type: {mut_type_str!r}")
        return int(m.group(2)), m.group(1), m.group(3)

    @staticmethod
    def build_one_to_foundry_int(token_to_idx: dict[str, int]) -> dict[str, int]:
        """Map one-letter AA codes to foundry integer indices via token_to_idx (3-letter → int)."""
        return {one: token_to_idx[three] for one, three in _ONE_TO_THREE.items() if three in token_to_idx}

    @staticmethod
    def build_mutant_seq(
        wt_seq_int: torch.Tensor,
        pos: int,
        mut_aa: str,
        one_to_foundry_int: dict[str, int],
    ) -> torch.Tensor:
        """Return a new [L] tensor with residue at pos mutated to mut_aa (foundry vocab)."""
        mut_seq = wt_seq_int.clone()
        mut_seq[pos] = one_to_foundry_int[mut_aa]
        return mut_seq

    @staticmethod
    def unsqueeze_input_features(input_features: dict) -> dict:
        """Add batch dim to all tensor values: [L, ...] → [1, L, ...]."""
        return {
            k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in input_features.items()
        }


class MegaScaleEnergyCallback(BaseCallback):
    """After each validation epoch, compute Pearson r between predicted and experimental ΔΔG.

    Uses the MegaScale single-point mutation dataset (202,804 mutations, 232 PDBs).
    Featurizes each PDB through the standard foundry pipeline, runs the PottsMPNN encoder
    once per PDB, then scores all mutations using the direct Potts Hamiltonian.
    Results are written to output_dir/energy_benchmark/megascale_epoch_{epoch}.csv.
    """

    def __init__(
        self,
        csv_path: Path,
        pdb_dir: Path,
        model_type: str = "protein_mpnn",
        save_dir: Path | None = None,
        extended_vocab: str | bool = False,
    ):
        pdb_dir = Path(pdb_dir)
        self.save_dir = Path(save_dir) if save_dir is not None else None

        mut_df = pd.read_csv(csv_path)
        self.pdb_groups = {pdb_id: grp for pdb_id, grp in mut_df.groupby("pdb")}

        # Match the featurization pipeline to the model's actual vocab size, BY NAME. extended_vocab is a
        # vocab name ("v4"/"v6") or None/False. The pipeline defaults to v4 (32 tokens), so a v6 model (30
        # tokens) MUST get extended_vocab passed through -- otherwise S is encoded with v4 tokens at indices
        # 30-31 that overflow the v6 W_s embedding, triggering the CUDA index-out-of-bounds assert in the
        # encoder. (Same reason a non-extended 21-token model uses the standard pipeline below.)
        if extended_vocab:
            pipeline = _build_potts_pipeline(
                model_type=model_type,
                is_inference=True,
                minimal_return=True,
                train_structure_noise_default=0.0,
                extended_vocab=extended_vocab,
            )
        else:
            pipeline = _build_standard_pipeline(
                model_type="protein_mpnn",
                is_inference=True,
                minimal_return=True,
                train_structure_noise_default=0.0,
            )

        # Build a small DataFrame of the 232 unique PDB files — mirrors the val_df setup in train.py.
        pdb_ids = list(self.pdb_groups.keys())
        benchmark_df = pd.DataFrame({
            "example_id": pdb_ids,
            "path": [str(pdb_dir / f"{pdb_id}.pdb") for pdb_id in pdb_ids],
            "assembly_id": "1",
        })

        self._benchmark_dataset = StructuralDatasetWrapper(
            dataset=PandasDataset(
                data=benchmark_df,
                id_column="example_id",
                name="megascale_benchmark",
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
        self._pdb_id_to_idx: dict[str, int] = {pdb_id: i for i, pdb_id in enumerate(pdb_ids)}
        self._one_to_foundry_int: dict[str, int] | None = None
        self._raw_features: dict[str, dict] = {}  # pdb_id → parsed input features (lazy, cached)

    def _featurize_pdb(self, pdb_id: str) -> dict | None:
        """Return cached raw input features for pdb_id, loading lazily on first access."""
        if pdb_id in self._raw_features:
            return self._raw_features[pdb_id]
        idx = self._pdb_id_to_idx[pdb_id]
        try:
            with open(os.devnull, "w") as _null, \
                 contextlib.redirect_stdout(_null), \
                 contextlib.redirect_stderr(_null):
                out = self._benchmark_dataset[idx]
            features = MegaScaleEnergy.unsqueeze_input_features(out["input_features"])
            self._raw_features[pdb_id] = features
            return features
        except Exception as e:
            ranked_logger.warning(f"[MegaScale] Failed to featurize {pdb_id}: {e}")
            return None

    @staticmethod
    def _unwrap_model(model):
        """Peel Fabric and DDP wrappers to reach the raw PottsMPNN module.

        Fabric wraps the model in _FabricModule (which holds _forward_module).
        DDP wraps it in DistributedDataParallel (which holds .module).
        Calling encoder forward on either wrapper can fire NCCL collectives;
        the raw module has no such side effects.
        """
        while hasattr(model, "_forward_module"):
            model = model._forward_module
        while hasattr(model, "module"):
            model = model.module
        return model

    def on_validation_epoch_end(self, trainer) -> None:
        rank = trainer.fabric.global_rank
        print(f"[MegaScale] rank={rank} ENTER on_validation_epoch_end", flush=True)

        # Non-zero ranks skip computation but must hit the barrier so they don't race
        # ahead to the next training step while rank 0 is still inside the MegaScale loop.
        if not trainer.fabric.is_global_zero:
            print(f"[MegaScale] rank={rank} PRE-BARRIER (early exit)", flush=True)
            trainer.fabric.barrier()
            print(f"[MegaScale] rank={rank} POST-BARRIER, returning", flush=True)
            return

        print(f"[MegaScale] rank=0 importing PottsMPNN and unwrapping model", flush=True)
        from mpnn.model.pottsmpnn import PottsMPNN

        # Unwrap Fabric (_FabricModule) and DDP (DistributedDataParallel) wrappers so that
        # run_potts_encoder() fires no NCCL collectives — rank 0 must not trigger any
        # collective while rank 1 is sitting at the barrier above.
        raw_model = self._unwrap_model(trainer.state["model"])
        epoch = trainer.state["current_epoch"]

        print(
            f"[MegaScale] rank=0 model unwrapped: type={type(raw_model).__name__}"
            f"  epoch={epoch}  cached_features={len(self._raw_features)}",
            flush=True,
        )

        raw_model.eval()
        print(f"[MegaScale] rank=0 model set to eval mode, entering no_grad block", flush=True)
        with torch.no_grad():
            if self._one_to_foundry_int is None:
                self._one_to_foundry_int = MegaScaleEnergy.build_one_to_foundry_int(raw_model.token_to_idx)
            print(f"[MegaScale] rank=0 vocab built ({len(self._one_to_foundry_int)} tokens),"
                  f" starting tqdm over {len(self.pdb_groups)} PDBs", flush=True)

            device = next(raw_model.parameters()).device
            all_pred: list[torch.Tensor] = []
            all_expt: list[float] = []
            n_skipped = 0
            n_feat_miss = 0
            n_encoder_fail = 0

            n_pdbs = len(self.pdb_groups)
            for pdb_id, df in tqdm(self.pdb_groups.items(), total=n_pdbs, desc="[MegaScale] scoring PDBs", file=sys.stderr):
                raw_features = self._featurize_pdb(pdb_id)
                if raw_features is None:
                    n_feat_miss += 1
                    continue

                # Clone tensors before moving to device: run_potts_encoder calls
                # sample_and_construct_masks which mutates input_features in-place.
                # The cached raw_features must remain pristine for future epochs.
                input_features = {
                    k: v.clone().to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in raw_features.items()
                }

                try:
                    etab_out, E_idx = raw_model.run_potts_encoder(input_features)
                except Exception as e:
                    n_encoder_fail += 1
                    print(f"[MegaScale] rank=0 encoder failed for {pdb_id}: {e}", flush=True)
                    continue

                wt_seq_int = input_features["S"].squeeze(0)  # [L]
                L = wt_seq_int.shape[0]

                # --- Build mutant sequences ---
                mutant_seqs: list[torch.Tensor] = []
                ddg_expt: list[float] = []

                for _, row in df.iterrows():
                    try:
                        pos, wt_aa, mut_aa = MegaScaleEnergy.parse_mut_type(row["mut_type"])
                    except ValueError:
                        n_skipped += 1
                        continue

                    if pos >= L:
                        n_skipped += 1
                        continue

                    wt_idx = self._one_to_foundry_int.get(wt_aa)
                    mut_idx = self._one_to_foundry_int.get(mut_aa)
                    if wt_idx is None or mut_idx is None:
                        n_skipped += 1
                        continue

                    # Sanity check: wt_aa in CSV must match featurized sequence at pos
                    if int(wt_seq_int[pos].item()) != wt_idx:
                        n_skipped += 1
                        continue

                    mutant_seqs.append(
                        MegaScaleEnergy.build_mutant_seq(wt_seq_int, pos, mut_aa, self._one_to_foundry_int)
                    )
                    ddg_expt.append(float(row["ddG_expt"]))

                if not mutant_seqs:
                    continue

                # --- Score sequences ---
                wt_batch = wt_seq_int.unsqueeze(0)       # [1, L]
                mut_batch = torch.stack(mutant_seqs)      # [N, L]

                e_wt = PottsMPNN.calc_potts_eners(etab_out, E_idx, wt_batch)   # [1]
                e_mut = PottsMPNN.calc_potts_eners(etab_out, E_idx, mut_batch)  # [N]

                all_pred.append((e_mut - e_wt).cpu().float())  # [N]
                all_expt.extend(ddg_expt)

        print(
            f"[MegaScale] rank=0 tqdm loop done: n_pred={len(all_pred)} n_expt={len(all_expt)}"
            f"  feat_miss={n_feat_miss} enc_fail={n_encoder_fail} skipped={n_skipped}",
            flush=True,
        )

        if not all_pred:
            ranked_logger.warning("[MegaScale] No predictions computed.")
        else:
            print(f"[MegaScale] rank=0 computing Pearson r over {len(all_pred)} batches", flush=True)
            ddg_pred = torch.cat(all_pred)                                    # [N_total]
            ddg_expt = torch.tensor(all_expt, dtype=torch.float32)            # [N_total]
            pearson_r = float(torch.corrcoef(torch.stack([ddg_pred, ddg_expt]))[0, 1])
            print(f"[MegaScale] rank=0 pearson_r={pearson_r:.4f}  n={len(ddg_pred)}", flush=True)

            ranked_logger.info(
                f"[MegaScale] epoch={epoch}  pearson_r={pearson_r:.4f}  n={len(ddg_pred)}  skipped={n_skipped}"
            )

            if self.save_dir is not None:
                print(f"[MegaScale] rank=0 saving metrics CSV to {self.save_dir}", flush=True)
                self.save_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = self.save_dir / "megascale_metrics.csv"
                row = pd.DataFrame([{"epoch": epoch, "pearson_r": pearson_r, "n": len(ddg_pred), "n_skipped": n_skipped}])
                row.to_csv(metrics_path, mode="a", header=not metrics_path.exists(), index=False)
                print(f"[MegaScale] rank=0 metrics CSV saved", flush=True)

        print(f"[MegaScale] rank=0 PRE-BARRIER (final)", flush=True)
        # Rank 1 is waiting at its barrier (early return above). Rank 0 must always reach
        # this point so both ranks can proceed to the next training step together.
        trainer.fabric.barrier()
        print(f"[MegaScale] rank=0 POST-BARRIER, EXIT", flush=True)
