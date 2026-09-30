"""Per-epoch NEUTRON protonation-recovery benchmark callback.

After each validation epoch, score the live model against neutron / joint X-ray–neutron structures, whose
OBSERVED H/D positions give the true protonation state of every HIS/ASP/GLU. These structures are excluded
from training (train.py's method filter), so this is genuine HELD-OUT generalization — the natural companion
to the PKAD pKa-correlation benchmark, but a hard argmax RECOVERY against the true state.

Design mirrors ``PKADBenchmarkCallback`` exactly (the near-identical template): rank-0-only + fabric.barrier
so the OpenMP/HBPLUS annotation never runs in a forked worker; ``_unwrap_model`` to reach the raw PottsMPNN
without firing NCCL collectives; ``get_vocab(extended_vocab)`` to resolve the protonated / deprotonated
tokens; featurize-once caching; teacher-forcing + conditional-minus-self forward for both the decoder head
and the Potts single-site+pairwise head; one appended row per epoch in ``save_dir/neutron_metrics.csv``.

Metric: hard ARGMAX recovery against the true token, scored per head (there is NO alpha blend -- the
``alpha`` column is vestigial, kept only for plot compatibility: 0.0 = Potts head, 1.0 = decoder head).
Two forms per residue: ``inter`` = the true token is the argmax over the WHOLE vocabulary (identity and
protonation state must both be right); ``intra`` = the argmax within the residue's own three variants
{X-P, X-S/D, X-A}, so the ambiguous token competes and an unresolved call counts as an error. Both are
reported separately for truly-protonated and truly-deprotonated residues, per residue type and pooled.
The Potts head's distribution is ``log_softmax(-potts_candidate_energies(...))``, i.e. field + outgoing
+ incoming couplings -- the same conditional energy the pKa derivation and the design engine use.
"""
from __future__ import annotations

import contextlib
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from atomworks.io.parser import STANDARD_PARSER_ARGS
from atomworks.ml.datasets.pandas_dataset import PandasDataset, StructuralDatasetWrapper
from atomworks.ml.datasets.parsers.default_metadata_row_parsers import GenericDFParser
from foundry.callbacks.callback import BaseCallback
from foundry.utils.ddp import RankedLogger
from mpnn.benchmarks.neutron import (
    build_position_map,
    load_neutron_benchmark,
    prepare_neutron_benchmark,
    prob_of_group,
)
from mpnn.model.pottsmpnn import PottsMPNN
from mpnn.pipelines.potts_mpnn import build_mpnn_transform_pipeline
from mpnn.transforms.extended_vocab import get_vocab

ranked_logger = RankedLogger(__name__, rank_zero_only=True)

_TITRATABLE = ("HIS", "ASP", "GLU")
_ALPHAS = (0.0, 0.5, 1.0)
_EPS = 1e-12
_DEFAULT_PARQUET = "data/mpnn_split/train_df_filtered.parquet"
_DEFAULT_HEAVY_DIR = "benchmarks/data/neutron_benchmark/heavy"
_DEFAULT_TRUTH_CSV = "benchmarks/data/neutron_benchmark/truth.csv"
_DEFAULT_LIST_CSV = "benchmarks/data/neutron_benchmark_list.csv"


def _make_token_idx_tensor(token_names, token_to_idx: dict[str, int]) -> torch.Tensor:
    return torch.tensor([token_to_idx[t] for t in token_names if t in token_to_idx], dtype=torch.long)


class NeutronRecoveryCallback(BaseCallback):
    """Score protonation-state RECOVERY vs neutron truth after each validation epoch."""

    def __init__(
        self,
        save_dir: Path | None = None,
        extended_vocab: str = "v6",
        parquet_path: str | Path = _DEFAULT_PARQUET,
        heavy_dir: str | Path = _DEFAULT_HEAVY_DIR,
        truth_csv: str | Path = _DEFAULT_TRUTH_CSV,
        list_csv: str | Path = _DEFAULT_LIST_CSV,
    ):
        self.save_dir = Path(save_dir) if save_dir is not None else None
        # MUST match the model being validated: it fixes both the labels the heavy structures are featurized
        # with (S is teacher-forced) and which tokens count as protonated / deprotonated below.
        self.extended_vocab = extended_vocab
        # Config only — NO file I/O or dataset construction here. The one-time freeze (reading 235 gzipped
        # PDBs) and the benchmark dataset are built LAZILY on rank 0 inside on_validation_epoch_end, so under
        # srun the non-zero ranks (which only barrier) never touch the files: no write race, no 8x redundant
        # freeze, no cross-rank ordering dependency. Mirrors the rank-0-only forward below.
        self.parquet_path = parquet_path
        self.heavy_dir = Path(heavy_dir)
        self.truth_csv = Path(truth_csv)
        self.list_csv = Path(list_csv)

        self._prepared = False
        self.truth_by_pdb: dict[str, dict[tuple, str]] = {}
        self.heavy_path: dict[str, Path] = {}
        self._benchmark_dataset = None
        self._pdb_id_to_idx: dict[str, int] = {}
        self._cache: dict[str, tuple[dict, dict[tuple, int]]] = {}
        # Built on first epoch, once the model's token_to_idx is known.
        self._prot_idx: dict[str, torch.Tensor] | None = None
        self._deprot_idx: dict[str, torch.Tensor] | None = None

    def _ensure_prepared(self) -> None:
        """Freeze (once) + build the benchmark dataset. RANK-0 ONLY — called after the is_global_zero gate."""
        if self._prepared:
            return
        if not self.truth_csv.exists():
            prepare_neutron_benchmark(self.parquet_path, self.heavy_dir, self.truth_csv, self.list_csv)
        self.truth_by_pdb, self.heavy_path = load_neutron_benchmark(self.heavy_dir, self.truth_csv)
        ranked_logger.info(
            f"[neutron] benchmark: {len(self.truth_by_pdb)} structures, "
            f"{sum(len(t) for t in self.truth_by_pdb.values())} truth residues"
        )
        # minimal_return=False preserves atom_array for the position map.
        pipeline = build_mpnn_transform_pipeline(
            model_type="potts_mpnn",
            is_inference=True,
            minimal_return=False,
            extended_vocab=self.extended_vocab,   # label + encode in the model's own vocabulary
        )
        pids = list(self.truth_by_pdb.keys())
        benchmark_df = pd.DataFrame({
            "example_id": pids,
            "path": [str(self.heavy_path[p]) for p in pids],
            "assembly_id": "1",
        })
        self._benchmark_dataset = StructuralDatasetWrapper(
            dataset=PandasDataset(data=benchmark_df, id_column="example_id", name="neutron_benchmark"),
            dataset_parser=GenericDFParser(
                example_id_colname="example_id", path_colname="path", assembly_id_colname="assembly_id",
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
        self._pdb_id_to_idx = {pid: i for i, pid in enumerate(pids)}
        self._prepared = True

    def _build_token_indices(self, token_to_idx: dict[str, int]) -> None:
        vocab = get_vocab(self.extended_vocab)
        prot, deprot, amb = vocab["aa_protonated"], vocab["aa_deprotonated"], vocab["aa_ambiguous"]
        # The single TRUE token index per state (v6 groups are single-token): a true X-P residue is
        # "recovered" iff this exact token is the argmax.
        self._true_prot_idx = {res: token_to_idx[prot[res][0]] for res in _TITRATABLE}
        self._true_dep_idx = {res: token_to_idx[deprot[res][0]] for res in _TITRATABLE}
        # The residue's FULL protonation-token set {X-P, X-S/D, X-A} — the candidate set for the INTRA-token
        # argmax ("given it's an Asp, which protonation state gets the top score", incl. the ambiguous X-A).
        self._intra_idx = {
            res: torch.tensor(
                sorted({token_to_idx[t] for grp in (prot[res], deprot[res], amb[res])
                        for t in grp if t in token_to_idx}),
                dtype=torch.long,
            )
            for res in _TITRATABLE
        }
        self._prot_idx = {res: t for res, t in self._true_prot_idx.items()}  # sentinel: "indices built"
        for res in _TITRATABLE:
            missing = [t for t in (*prot[res], *deprot[res]) if t not in token_to_idx]
            if missing:
                raise ValueError(
                    f"extended_vocab={self.extended_vocab!r} names {missing} for {res}, but the model's "
                    f"encoding has no such token(s). The benchmark's vocabulary does not match the model."
                )

    def _featurize_pdb(self, pdb_id: str):
        if pdb_id in self._cache:
            return self._cache[pdb_id]
        idx = self._pdb_id_to_idx.get(pdb_id)
        if idx is None:
            return None
        try:
            with open(os.devnull, "w") as _null, \
                 contextlib.redirect_stdout(_null), contextlib.redirect_stderr(_null):
                out = self._benchmark_dataset[idx]
            pos_map = build_position_map(out)
            input_features = {
                k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                for k, v in out["input_features"].items()
            }
            self._cache[pdb_id] = (input_features, pos_map)
            return input_features, pos_map
        except Exception as e:
            ranked_logger.warning(f"[neutron] Failed to featurize {pdb_id}: {e}")
            return None

    @staticmethod
    def _unwrap_model(model):
        while hasattr(model, "_forward_module"):
            model = model._forward_module
        while hasattr(model, "module"):
            model = model.module
        return model

    def on_validation_epoch_end(self, trainer) -> None:
        if not trainer.fabric.is_global_zero:
            trainer.fabric.barrier()
            return

        # Rank 0 only past this point: freeze + build the dataset lazily on the first epoch (no fork here).
        self._ensure_prepared()

        raw_model = self._unwrap_model(trainer.state["model"])
        epoch = trainer.state["current_epoch"]

        if self._prot_idx is None:
            self._build_token_indices(raw_model.token_to_idx)

        raw_model.eval()
        device = next(raw_model.parameters()).device
        intra_idx = {res: t.to(device) for res, t in self._intra_idx.items()}

        # records: per residue, whether the TRUE token is the argmax — over ALL tokens (inter) and over the
        # residue's own protonation variants (intra) — for each head.
        records: list[dict] = []
        n_miss = n_fail = n_skip = 0

        with torch.no_grad():
            for pdb_id, truth in tqdm(
                self.truth_by_pdb.items(), total=len(self.truth_by_pdb),
                desc="[neutron] scoring", file=__import__("sys").stderr,
            ):
                cached = self._featurize_pdb(pdb_id)
                if cached is None:
                    n_miss += 1
                    continue
                raw_features, pos_map = cached

                hits = [(pos_map[k], k[2], lbl) for k, lbl in truth.items() if k in pos_map]
                n_skip += len(truth) - len(hits)
                if not hits:
                    continue

                input_features = {
                    k: v.clone().to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in raw_features.items()
                }
                input_features["decode_type"] = "teacher_forcing"
                input_features["causality_pattern"] = "conditional_minus_self"

                try:
                    network_output = raw_model({"input_features": input_features})
                except Exception as e:
                    n_fail += 1
                    ranked_logger.warning(f"[neutron] forward failed for {pdb_id}: {e}")
                    continue

                log_probs_dec = network_output["decoder_features"]["log_probs"]  # [1, L, V]
                etab_out = network_output["potts_context"].etab_out               # [1, L, K, V, V]
                E_idx = network_output["potts_context"].E_idx                     # [1, L, K]
                S = input_features["S"]                                           # [1, L]

                # Per-position conditional energies: single-site field + OUTGOING pair couplings
                # i->j + INCOMING pair couplings m->i (the graph transpose). The kNN graph is
                # ~16% non-reciprocal, so an outgoing-only sum is not the residue's conditional
                # energy and does not match the Hamiltonian's finite difference. This helper is
                # the same one the pKa derivation and the design engine use, so all three read
                # the same quantity.
                E_cand = PottsMPNN.potts_candidate_energies(etab_out, E_idx, S)    # [1, L, V]
                log_probs_pot = F.log_softmax(-E_cand, dim=-1)                     # [1, L, V]

                for pos, res_name, label in hits:
                    is_prot = label.endswith("-P")
                    true_idx = (self._true_prot_idx[res_name] if is_prot
                                else self._true_dep_idx[res_name])
                    cand = intra_idx[res_name]
                    rec = {"res_name": res_name, "is_prot": is_prot}
                    p_i, d_i = self._true_prot_idx[res_name], self._true_dep_idx[res_name]
                    for head, lp in (("pot", log_probs_pot[0, pos, :]),
                                     ("dec", log_probs_dec[0, pos, :])):
                        # inter: is the true token the single top token over the WHOLE vocabulary?
                        rec[f"inter_{head}"] = int(lp.argmax().item() == true_idx)
                        # intra: within {X-P, X-S/D, X-A}, is the true state the top?
                        rec[f"intra_{head}"] = int(cand[lp[cand].argmax()].item() == true_idx)
                        # SOFT reads, so the epoch-wise signal matches how the model is actually scored
                        # and selected: the log-ratio that feeds auPR, and the two-state NLL of the true
                        # state (identity cost renormalised away -- the L_state quantity).
                        lp_p, lp_d = lp[p_i], lp[d_i]
                        rec[f"score_{head}"] = float(lp_p - lp_d)
                        rec[f"state_nll_{head}"] = float(
                            -((lp_p if is_prot else lp_d) - torch.logaddexp(lp_p, lp_d))
                        )
                    records.append(rec)

        ranked_logger.info(
            f"[neutron] epoch={epoch}  n_records={len(records)}  "
            f"n_miss={n_miss}  n_fail={n_fail}  n_skip={n_skip}"
        )
        if not records:
            trainer.fabric.barrier()
            return

        rdf = pd.DataFrame(records)
        res_arr = rdf["res_name"].to_numpy()
        is_prot = rdf["is_prot"].to_numpy(dtype=bool)
        csv_rows = []
        # One row per HEAD (not an alpha blend): argmax recovery is a per-head quantity. alpha column kept
        # for plot compatibility — Potts head = 0.0, decoder/MPNN head = 1.0.
        for head, alpha in (("pot", 0.0), ("dec", 1.0)):
            inter = rdf[f"inter_{head}"].to_numpy(dtype=float)
            intra = rdf[f"intra_{head}"].to_numpy(dtype=float)
            row: dict = {"epoch": epoch, "alpha": alpha, "n": len(rdf)}

            def _add_group(mask, tag: str) -> None:
                """Per-state sequence recovery on a residue subset, conditioned on the true label:
                inter = true token is the argmax over ALL vocab tokens; intra = argmax within the residue's
                own protonation variants {X-P, X-S/D, X-A}."""
                for state, sel in (("prot", is_prot), ("dep", ~is_prot)):
                    m = mask & sel
                    n = int(m.sum())
                    row[f"n_{state}_{tag}"] = n
                    row[f"inter_recov_{state}_{tag}"] = float(inter[m].mean()) if n else float("nan")
                    row[f"intra_recov_{state}_{tag}"] = float(intra[m].mean()) if n else float("nan")

            for res in _TITRATABLE:
                _add_group(res_arr == res, res)
            _add_group(np.ones(len(rdf), dtype=bool), "all")

            # auPR of the PROTONATED class from the soft log-ratio, plus the two-state NLL. auPR needs both
            # classes present, so it is computed per residue TYPE (not per true state like _add_group) --
            # this is the metric the threshold choice and the manuscript figures are read on, so it belongs
            # in the per-epoch trace rather than only in the offline re-scoring.
            score = rdf[f"score_{head}"].to_numpy(dtype=float)
            snll = rdf[f"state_nll_{head}"].to_numpy(dtype=float)
            for res in (*_TITRATABLE, "all"):
                m = np.ones(len(rdf), dtype=bool) if res == "all" else (res_arr == res)
                y = is_prot[m]
                row[f"aupr_{res}"] = (
                    float(average_precision_score(y, score[m]))
                    if 0 < int(y.sum()) < int(m.sum()) else float("nan")
                )
                row[f"state_nll_{res}"] = float(snll[m].mean()) if m.any() else float("nan")
                row[f"state_nll_prot_{res}"] = (
                    float(snll[m & is_prot].mean()) if (m & is_prot).any() else float("nan")
                )
            csv_rows.append(row)
            ranked_logger.info(
                f"[neutron] epoch={epoch}  head={head}  HIS "
                f"inter(P={row['inter_recov_prot_HIS']:.3f} S={row['inter_recov_dep_HIS']:.3f})  "
                f"intra(P={row['intra_recov_prot_HIS']:.3f} S={row['intra_recov_dep_HIS']:.3f})  "
                f"[n_P={row['n_prot_HIS']} n_S={row['n_dep_HIS']}]"
            )

        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = self.save_dir / "neutron_metrics.csv"
            new_df = pd.DataFrame(csv_rows)
            # Schema-safe append: if the metric columns changed since this CSV was started (e.g. the callback
            # was extended and a run RESUMED into the same file), rotate the old file aside instead of
            # appending misaligned rows. Keeps every resume corruption-free without a manual archive step.
            if metrics_path.exists():
                old_cols = pd.read_csv(metrics_path, nrows=0).columns.tolist()
                if old_cols != list(new_df.columns):
                    bak = metrics_path.with_name(f"neutron_metrics_prev_{int(time.time())}.csv")
                    metrics_path.rename(bak)
                    ranked_logger.warning(
                        f"[neutron] metric schema changed; rotated old CSV to {bak.name}"
                    )
            new_df.to_csv(
                metrics_path, mode="a", header=not metrics_path.exists(), index=False
            )

        trainer.fabric.barrier()
