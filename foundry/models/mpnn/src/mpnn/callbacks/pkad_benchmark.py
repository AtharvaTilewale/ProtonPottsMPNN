import contextlib
import math
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from atomworks.io.parser import STANDARD_PARSER_ARGS
from atomworks.ml.datasets.pandas_dataset import PandasDataset, StructuralDatasetWrapper
from atomworks.ml.datasets.parsers.default_metadata_row_parsers import GenericDFParser
from atomworks.ml.utils.token import token_iter
from foundry.callbacks.callback import BaseCallback
from foundry.utils.ddp import RankedLogger
from mpnn.pipelines.potts_mpnn import build_mpnn_transform_pipeline
from mpnn.transforms.extended_vocab import get_vocab

ranked_logger = RankedLogger(__name__, rank_zero_only=True)

_TITRATABLE = {"HIS", "ASP", "GLU"}

# The token groups come from THE VOCABULARY THE MODEL WAS TRAINED ON — `get_vocab(extended_vocab)` gives
# `aa_protonated` / `aa_deprotonated` / `aa_ambiguous`, and those dicts ARE what is scored here. They must
# not be hardcoded: v3/v4 call neutral His the HID/HIE tautomers, v6 calls it HIS-S, and naming either one
# here would silently drop the other's probability mass (`_make_token_idx_tensor` skips names the encoding
# lacks) — the His pKa signal would go quietly wrong. See transforms/extended_vocab.py.
#
# TWO signals are scored every epoch and both are written to the CSV.
#
# 1. STRICT (columns `pearson_*` / `spearman_*`) — ambiguous EXCLUDED:
#        signal = log10( P(deprot) / P(prot) )      -> correlates NEGATIVELY with pKa
#
# 2. PA (columns `pearson_*_pa` / `spearman_*_pa`) — protonated over everything-else, ambiguous
#    INCLUDED in the denominator:
#        signal = log10( P(prot) / (P(deprot) + P(amb)) )   -> correlates POSITIVELY with pKa
#    Rationale: *-A is the MODAL label for acids (measured on the PKAD structures: ASP-A 60%,
#    GLU-A 52%, HIS-A 40%), so the strict metric throws away most of the model's probability mass
#    and rests the signal on the ~5% of acids labelled *-P. Putting *-A in the denominator is also
#    the right chemical prior, and the one the labeller itself makes (charge_network.acid_amb_w
#    = -1.0: "an acid is deprotonated at almost any relevant pH, so 'we found nothing' is safest
#    read as 'still a carboxylate'").
#
# Both sides are exactly the pKa equilibrium the vocabulary defines: protonated <-> deprotonated. The
# imidazolate HIS-D is in NEITHER map (a separate, far higher-pKa deprotonation), so it no longer leaks
# into the His strict numerator the way the old hardcoded set did.

_ALPHAS = (0.0, 0.5, 1.0)


def _make_token_idx_tensor(token_names: frozenset[str], token_to_idx: dict[str, int]) -> torch.Tensor:
    return torch.tensor(
        [token_to_idx[t] for t in token_names if t in token_to_idx],
        dtype=torch.long,
    )

def _log10_ratio(log_probs_1d: torch.Tensor, dep_idx: torch.Tensor, prot_idx: torch.Tensor) -> float:
    """log10(P_deprot / P_prot) from a [V] log-probability vector."""
    log_p_dep = torch.logsumexp(log_probs_1d[dep_idx], dim=0)
    log_p_prot = torch.logsumexp(log_probs_1d[prot_idx], dim=0)
    return (log_p_dep - log_p_prot).item() / math.log(10)


def _total_titratable_prob(
    log_probs_1d: torch.Tensor,
    dep_idx: torch.Tensor,
    prot_idx: torch.Tensor,
) -> float:
    """Total softmax probability on all protonation-state tokens (deprot + prot).

    This is the confidence score: high means the model has committed mass to
    the titratable variants of this residue; low means it prefers other token
    types, making the pKa log-ratio unreliable.
    """
    probs = log_probs_1d.exp()
    return (probs[dep_idx].sum() + probs[prot_idx].sum()).item()


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    t = torch.tensor([x, y], dtype=torch.float32)
    return float(torch.corrcoef(t)[0, 1])


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")

    def _rank(v: list[float]) -> torch.Tensor:
        t = torch.tensor(v, dtype=torch.float32)
        return t.argsort().argsort().float()

    return float(torch.corrcoef(torch.stack([_rank(x), _rank(y)]))[0, 1])


def _r2(x: list[float], y: list[float]) -> float:
    """R² = Pearson r² (equivalent to OLS R² with free intercept)."""
    r = _pearson(x, y)
    return float(r ** 2) if r == r else float("nan")


class PKADBenchmarkCallback(BaseCallback):
    """After each validation epoch, evaluate pKa prediction signal vs PKAD-R experimental values.

    Computes log10(P_dep / P_prot) from two sources:
      - Decoder head: teacher-forced forward pass with conditional-minus-self causality,
        giving P(s_i | s_{j≠i}, X) for every position simultaneously.
      - Potts head:   single-site fields from the diagonal of etab_out[:,:,0,:,:].

    Combined: pKa_signal(α) = α · decoder_term + (1-α) · potts_term
    Evaluated at α ∈ {0.0, 0.5, 1.0} for HIS, ASP, GLU residues.
    Writes per-epoch metrics to save_dir/pkad_metrics.csv.
    """

    def __init__(
        self,
        csv_path: Path,
        pdb_dir: Path,
        save_dir: Path | None = None,
        extended_vocab: str = "v4",
    ):
        csv_path = Path(csv_path)
        pdb_dir = Path(pdb_dir)
        self.save_dir = Path(save_dir) if save_dir is not None else None
        # MUST match the model being validated. It drives BOTH halves of this benchmark: the labels the
        # structures are featurized with (S is teacher-forced into the model, so v4 tokens fed to a v6
        # model would be nonsense) and which tokens count as protonated / deprotonated below.
        self.extended_vocab = extended_vocab

        pka_df = pd.read_csv(csv_path)
        pka_df = pka_df[
            (pka_df["pKa Classification"] == "Main")
            & (pka_df["ResName"].isin(_TITRATABLE))
        ].copy()
        pka_df["Expt. pKa"] = pd.to_numeric(pka_df["Expt. pKa"], errors="coerce")
        pka_df = pka_df.dropna(subset=["Expt. pKa"])

        self.pdb_groups: dict[str, pd.DataFrame] = {
            pdb_id: grp for pdb_id, grp in pka_df.groupby("PDB")
        }

        # minimal_return=False preserves atom_array in pipeline output for position mapping
        pipeline = build_mpnn_transform_pipeline(
            model_type="potts_mpnn",
            is_inference=True,
            minimal_return=False,
            extended_vocab=extended_vocab,   # label + encode in the model's own vocabulary
        )

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
                name="pkad_benchmark",
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
        # Cache: pdb_id → (unsqueezed input_features, position_map)
        self._cache: dict[str, tuple[dict, dict[tuple, int]]] = {}
        # Token index tensors — built on first epoch once we have the model's token_to_idx
        self._prot_idx: dict[str, torch.Tensor] | None = None
        self._deprot_idx: dict[str, torch.Tensor] | None = None
        self._pa_denom_idx: dict[str, torch.Tensor] | None = None

    def _build_token_indices(self, token_to_idx: dict[str, int]) -> None:
        """Resolve the vocabulary's protonated / deprotonated / ambiguous tokens to model indices."""
        vocab = get_vocab(self.extended_vocab)
        prot, deprot, amb = (vocab["aa_protonated"], vocab["aa_deprotonated"],
                             vocab["aa_ambiguous"])
        self._prot_idx = {
            res: _make_token_idx_tensor(frozenset(prot[res]), token_to_idx)
            for res in _TITRATABLE
        }
        self._deprot_idx = {
            res: _make_token_idx_tensor(frozenset(deprot[res]), token_to_idx)
            for res in _TITRATABLE
        }
        # PA denominator: the deprotonated side plus the ambiguous token, straight from the vocabulary.
        self._pa_denom_idx = {
            res: _make_token_idx_tensor(frozenset(deprot[res]) | frozenset(amb[res]), token_to_idx)
            for res in _TITRATABLE
        }
        # Every token the vocabulary names must exist in the model's encoding, or the signal is silently
        # computed over a subset. This is exactly the v4-tokens-into-a-v6-model failure, made loud.
        for res in _TITRATABLE:
            missing = [t for t in (*prot[res], *deprot[res], *amb[res]) if t not in token_to_idx]
            if missing:
                raise ValueError(
                    f"extended_vocab={self.extended_vocab!r} names {missing} for {res}, but the model's "
                    f"encoding has no such token(s). The benchmark's vocabulary does not match the model."
                )

    def _featurize_pdb(self, pdb_id: str) -> tuple[dict, dict[tuple, int]] | None:
        """Lazily featurize a PDB and cache (unsqueezed_input_features, pos_map)."""
        if pdb_id in self._cache:
            return self._cache[pdb_id]
        idx = self._pdb_id_to_idx.get(pdb_id)
        if idx is None:
            return None
        try:
            with open(os.devnull, "w") as _null, \
                 contextlib.redirect_stdout(_null), \
                 contextlib.redirect_stderr(_null):
                out = self._benchmark_dataset[idx]

            pos_map = self._build_position_map(out)
            input_features = {
                k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                for k, v in out["input_features"].items()
            }
            self._cache[pdb_id] = (input_features, pos_map)
            return input_features, pos_map
        except Exception as e:
            ranked_logger.warning(f"[PKAD] Failed to featurize {pdb_id}: {e}")
            return None

    @staticmethod
    def _build_position_map(out: dict) -> dict[tuple, int]:
        """Map (chain_id, res_id, res_name) → token index from the processed atom_array."""
        atom_array = out["atom_array"]
        non_atomized = atom_array[~atom_array.atomize]
        pos_map: dict[tuple, int] = {}
        for i, token in enumerate(token_iter(non_atomized)):
            key = (token.chain_id[0], int(token.res_id[0]), token.res_name[0])
            pos_map[key] = i
        return pos_map

    @staticmethod
    def _unwrap_model(model):
        while hasattr(model, "_forward_module"):
            model = model._forward_module
        while hasattr(model, "module"):
            model = model.module
        return model

    def on_validation_epoch_end(self, trainer) -> None:
        rank = trainer.fabric.global_rank
        print(f"[PKAD] rank={rank} ENTER on_validation_epoch_end", flush=True)

        if not trainer.fabric.is_global_zero:
            print(f"[PKAD] rank={rank} PRE-BARRIER (early exit)", flush=True)
            trainer.fabric.barrier()
            print(f"[PKAD] rank={rank} POST-BARRIER, returning", flush=True)
            return

        raw_model = self._unwrap_model(trainer.state["model"])
        epoch = trainer.state["current_epoch"]
        print(
            f"[PKAD] rank=0 model unwrapped: type={type(raw_model).__name__}"
            f"  epoch={epoch}  cached={len(self._cache)}",
            flush=True,
        )

        if self._prot_idx is None:
            self._build_token_indices(raw_model.token_to_idx)
            print(
                f"[PKAD] rank=0 token indices built:"
                f"  prot={ {k: v.tolist() for k, v in self._prot_idx.items()} }",
                flush=True,
            )

        raw_model.eval()
        device = next(raw_model.parameters()).device

        prot_idx = {res: t.to(device) for res, t in self._prot_idx.items()}
        deprot_idx = {res: t.to(device) for res, t in self._deprot_idx.items()}
        pa_denom_idx = {res: t.to(device) for res, t in self._pa_denom_idx.items()}

        records: list[dict] = []
        n_miss = n_fail = n_skip = 0

        print(f"[PKAD] rank=0 starting loop over {len(self.pdb_groups)} PDBs", flush=True)
        with torch.no_grad():
            for pdb_id, df in tqdm(
                self.pdb_groups.items(), total=len(self.pdb_groups), desc="[PKAD] scoring PDBs",
                file=__import__("sys").stderr,
            ):
                cached = self._featurize_pdb(pdb_id)
                if cached is None:
                    n_miss += 1
                    continue
                raw_features, pos_map = cached

                rows: list[tuple[int, str, float]] = []
                for _, row in df.iterrows():
                    key = (str(row["Chain"]), int(row["ResID in PDB"]), str(row["ResName"]))
                    pos = pos_map.get(key)
                    if pos is None:
                        n_skip += 1
                        continue
                    rows.append((pos, str(row["ResName"]), float(row["Expt. pKa"])))

                if not rows:
                    continue

                # Clone + move to device — forward mutates input_features in-place
                input_features = {
                    k: v.clone().to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in raw_features.items()
                }

                # Teacher forcing with conditional-minus-self causality gives
                # P(s_i | s_{j≠i}, X) for all positions in one forward pass —
                # equivalent to single-residue masking without a per-residue loop.
                input_features["decode_type"] = "teacher_forcing"
                input_features["causality_pattern"] = "conditional_minus_self"

                try:
                    network_output = raw_model({"input_features": input_features})
                except Exception as e:
                    n_fail += 1
                    print(f"[PKAD] rank=0 forward failed for {pdb_id}: {e}", flush=True)
                    continue

                log_probs_dec = network_output["decoder_features"]["log_probs"]  # [1, L, V]
                etab_out = network_output["potts_context"].etab_out               # [1, L, K, V, V]
                E_idx    = network_output["potts_context"].E_idx                  # [1, L, K]
                S        = input_features["S"]                                    # [1, L]

                B, L_seq, K = E_idx.shape
                V = etab_out.shape[-1]
                h_i = torch.diagonal(
                    etab_out[:, :, 0:1, :, :], offset=0, dim1=-2, dim2=-1
                ).squeeze(2)                                        # [1, L, V]

                # Full Potts energy: h_i plus pairwise couplings at current sequence context.
                S_neigh = S[torch.arange(B, device=S.device)[:, None, None], E_idx]
                s_idx = (
                    S_neigh[:, :, 1:]          # [B, L, K-1]
                    .unsqueeze(-1)             # [B, L, K-1, 1]
                    .unsqueeze(-1)             # [B, L, K-1, 1, 1]
                    .expand(-1, -1, -1, V, 1)  # [B, L, K-1, V, 1]
                )
                J_context = etab_out[:, :, 1:].gather(-1, s_idx).squeeze(-1)  # [1, L, K-1, V]
                h_full = h_i + J_context.sum(dim=2)                # [1, L, V]
                log_probs_pot = F.log_softmax(-h_full, dim=-1)      # [1, L, V]

                for pos, res_name, expt in rows:
                    records.append({
                        "res_name": res_name,
                        "expt_pka": expt,
                        "log_ratio_dec": _log10_ratio(
                            log_probs_dec[0, pos, :], deprot_idx[res_name], prot_idx[res_name]
                        ),
                        "log_ratio_pot": _log10_ratio(
                            log_probs_pot[0, pos, :], deprot_idx[res_name], prot_idx[res_name]
                        ),
                        # PA: log10( P(prot) / (P(deprot) + P(amb)) ). Note the argument order is
                        # (numerator=prot, denominator=deprot+amb), i.e. INVERTED vs the strict
                        # ratio above -- so this correlates POSITIVELY with pKa.
                        "log_ratio_dec_pa": _log10_ratio(
                            log_probs_dec[0, pos, :], prot_idx[res_name], pa_denom_idx[res_name]
                        ),
                        "log_ratio_pot_pa": _log10_ratio(
                            log_probs_pot[0, pos, :], prot_idx[res_name], pa_denom_idx[res_name]
                        ),
                    })

        print(
            f"[PKAD] rank=0 loop done: n_records={len(records)}"
            f"  n_miss={n_miss}  n_fail={n_fail}  n_skip={n_skip}",
            flush=True,
        )
        ranked_logger.info(
            f"[PKAD] epoch={epoch}  n_records={len(records)}"
            f"  n_miss={n_miss}  n_fail={n_fail}  n_skip={n_skip}"
        )

        if not records:
            print(f"[PKAD] rank=0 no records — skipping CSV write, going to barrier", flush=True)
            trainer.fabric.barrier()
            return

        results_df = pd.DataFrame(records)
        csv_rows = []

        for alpha in _ALPHAS:
            row: dict = {"epoch": epoch, "alpha": alpha, "n": len(results_df)}

            # Score both signals. suffix "" = strict, log10(deprot/prot), NEGATIVE vs pKa (same
            # meaning as every CSV written before). "_pa" = log10(prot/(deprot+amb)), POSITIVE vs pKa.
            for suffix, dec_col, pot_col in (
                ("", "log_ratio_dec", "log_ratio_pot"),
                ("_pa", "log_ratio_dec_pa", "log_ratio_pot_pa"),
            ):
                results_df["signal"] = (
                    alpha * results_df[dec_col] + (1.0 - alpha) * results_df[pot_col]
                )
                for res in ("HIS", "ASP", "GLU"):
                    mask = results_df["res_name"] == res
                    row[f"pearson_{res}{suffix}"] = _pearson(
                        results_df.loc[mask, "signal"].tolist(),
                        results_df.loc[mask, "expt_pka"].tolist(),
                    )
                    row[f"spearman_{res}{suffix}"] = _spearman(
                        results_df.loc[mask, "signal"].tolist(),
                        results_df.loc[mask, "expt_pka"].tolist(),
                    )
                row[f"pearson_all{suffix}"] = _pearson(
                    results_df["signal"].tolist(), results_df["expt_pka"].tolist()
                )
                row[f"spearman_all{suffix}"] = _spearman(
                    results_df["signal"].tolist(), results_df["expt_pka"].tolist()
                )
            csv_rows.append(row)

            ranked_logger.info(
                f"[PKAD] epoch={epoch}  alpha={alpha:.1f}"
                f"  pearson_all={row['pearson_all']:.4f}"
                f"  spearman_all={row['spearman_all']:.4f}"
                f"  pearson_HIS={row['pearson_HIS']:.4f}"
            )

        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = self.save_dir / "pkad_metrics.csv"
            pd.DataFrame(csv_rows).to_csv(
                metrics_path, mode="a", header=not metrics_path.exists(), index=False
            )

        trainer.fabric.barrier()
