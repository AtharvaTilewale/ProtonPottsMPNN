import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from atomworks.constants import (
    DICT_THREE_TO_ONE,
    PROTEIN_BACKBONE_ATOM_NAMES,
    UNKNOWN_AA,
)
from atomworks.ml.utils.token import get_token_starts, spread_token_wise
from biotite.structure import AtomArray
from mpnn.collate.feature_collator import FeatureCollator
from mpnn.metrics.sequence_recovery import (
    InterfaceSequenceRecovery,
    SequenceRecovery,
)
from mpnn.model.mpnn import LigandMPNN, ProteinMPNN
from mpnn.model.pottsmpnn import PottsMPNN
from mpnn.model.layers.graph_embeddings import PottsProteinFeatures

from mpnn.pipelines.mpnn import build_mpnn_transform_pipeline
from mpnn.transforms.feature_aggregation.token_encodings import MPNN_TOKEN_ENCODING
from mpnn.utils.inference import (
    MPNN_GLOBAL_INFERENCE_DEFAULTS,
    MPNN_PER_INPUT_INFERENCE_DEFAULTS,
    MPNNInferenceInput,
    MPNNInferenceOutput,
    _absolute_path_or_none,
)

from foundry.inference_engines.checkpoint_registry import REGISTERED_CHECKPOINTS

from foundry.metrics.metric import MetricManager
from foundry.utils.ddp import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


# ---------------------------------------------------------------------------
# Standalone input-preparation helper
# ---------------------------------------------------------------------------

@dataclass
class PottsMPNNBatch:
    """Featurized structure ready to pass directly into a PottsMPNN model.

    Attributes
    ----------
    network_input:
        Collated dict accepted by ``model(network_input)`` or
        ``model.run_potts_encoder(network_input["input_features"])``.
    atom_array:
        Processed AtomArray after the full transform pipeline.  Keep this
        around if you want to decode designed sequences back onto a structure.
    input_dict:
        The fully-defaulted per-input settings dict (design scope, temperature,
        etc.) that was used to build this batch.
    """

    network_input: dict
    atom_array: AtomArray
    input_dict: dict


def prepare_potts_mpnn_input(
    structure: str | Path | AtomArray,
    *,
    fixed_residues: list[str] | None = None,
    designed_residues: list[str] | None = None,
    fixed_chains: list[str] | None = None,
    designed_chains: list[str] | None = None,
    temperature: float = 0.1,
    structure_noise: float = 0.0,
    decode_type: str = "auto_regressive",
    causality_pattern: str = "auto_regressive",
    extended_vocab: str | None = None,
    device: str | torch.device | None = None,
) -> PottsMPNNBatch:
    """Featurize a structure for PottsMPNN inference.

    Accepts either a file path (PDB/CIF) or a pre-loaded AtomArray and runs
    the full transform pipeline, returning a ``PottsMPNNBatch`` whose
    ``network_input`` can be passed directly to::

        model(batch.network_input)                          # full forward
        model.run_potts_encoder(batch.network_input["input_features"])  # encoder only

    Parameters
    ----------
    structure:
        Path to a PDB/CIF file *or* an already-loaded ``AtomArray``.
    fixed_residues:
        Residue IDs to keep fixed, e.g. ``["A5", "A12"]``.
    designed_residues:
        Residue IDs to design.  If both ``fixed_residues`` and
        ``designed_residues`` are None the whole chain is designed.
    fixed_chains:
        Chain IDs to keep fixed entirely, e.g. ``["B"]``.
    designed_chains:
        Chain IDs to design entirely.
    temperature:
        Sampling temperature written into the feature tensor.
    structure_noise:
        Standard deviation of coordinate noise added during featurization (Å).
    decode_type:
        ``"auto_regressive"`` or ``"teacher_forcing"``.
    causality_pattern:
        One of ``"auto_regressive"``, ``"unconditional"``, ``"conditional"``,
        ``"conditional_minus_self"``.
    extended_vocab:
        Whether to use the extended protonation-state vocabulary (32 tokens).
    device:
        Target device for output tensors.  Defaults to CUDA if available.

    Returns
    -------
    PottsMPNNBatch
    """
    # Resolve device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device("xpu")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    # Build a minimal per-input dict with only the keys we override
    user_overrides: dict[str, Any] = {
        "fixed_residues": fixed_residues,
        "designed_residues": designed_residues,
        "fixed_chains": fixed_chains,
        "designed_chains": designed_chains,
        "temperature": temperature,
        "structure_noise": structure_noise,
        "decode_type": decode_type,
        "causality_pattern": causality_pattern,
        # sensible single-design defaults
        "batch_size": 1,
        "number_of_batches": 1,
        "repeat_sample_num": None,
        "atomize_side_chains": False,
        "initialize_sequence_embedding_with_ground_truth": False,
        "features_to_return": None,
        "occupancy_threshold_sidechain": None,
        "occupancy_threshold_backbone": None,
        "undesired_res_names": None,
    }

    # Resolve atom_array and per-input dict through the standard helper so
    # fixed/designed residue annotations are correctly applied to the array.
    if isinstance(structure, (str, Path)):
        inference_input = MPNNInferenceInput.from_atom_array_and_dict(
            input_dict={"structure_path": str(structure), **user_overrides},
        )
    else:
        inference_input = MPNNInferenceInput.from_atom_array_and_dict(
            atom_array=structure,
            input_dict=user_overrides,
        )

    model_type = "potts_mpnn"

    # Build pipeline args — pass occupancy overrides only when explicitly set
    pipeline_args: dict[str, Any] = {}
    if inference_input.input_dict["occupancy_threshold_sidechain"] is not None:
        pipeline_args["occupancy_threshold_sidechain"] = (
            inference_input.input_dict["occupancy_threshold_sidechain"]
        )
    if inference_input.input_dict["occupancy_threshold_backbone"] is not None:
        pipeline_args["occupancy_threshold_backbone"] = (
            inference_input.input_dict["occupancy_threshold_backbone"]
        )
    if inference_input.input_dict["undesired_res_names"] is not None:
        pipeline_args["undesired_res_names"] = (
            inference_input.input_dict["undesired_res_names"]
        )

    pipeline = build_mpnn_transform_pipeline(
        model_type=model_type,
        is_inference=True,
        minimal_return=True,
        device=device,
        **pipeline_args,
    )
    collator = FeatureCollator()

    data: dict[str, Any] = {
        "atom_array": inference_input.atom_array.copy(),
        "structure_noise": inference_input.input_dict["structure_noise"],
        "decode_type": inference_input.input_dict["decode_type"],
        "causality_pattern": inference_input.input_dict["causality_pattern"],
        "initialize_sequence_embedding_with_ground_truth": (
            inference_input.input_dict["initialize_sequence_embedding_with_ground_truth"]
        ),
        "atomize_side_chains": inference_input.input_dict["atomize_side_chains"],
        "repeat_sample_num": inference_input.input_dict["repeat_sample_num"],
        "features_to_return": inference_input.input_dict["features_to_return"],
    }

    pipeline_output = pipeline(data)
    network_input = collator([pipeline_output])

    return PottsMPNNBatch(
        network_input=network_input,
        atom_array=pipeline_output["atom_array"],
        input_dict=inference_input.input_dict,
    )


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class MPNNInferenceEngine:
    """Inference engine for PottsMPNN — autoregressive sampling and Potts
    Gibbs-sweep sequence optimisation."""

    model_type = "potts_mpnn"
    is_legacy_weights = False

    def __init__(
        self,
        *,
        checkpoint_path: str = MPNN_GLOBAL_INFERENCE_DEFAULTS["checkpoint_path"],
        out_directory: str | None = MPNN_GLOBAL_INFERENCE_DEFAULTS["out_directory"],
        write_fasta: bool = MPNN_GLOBAL_INFERENCE_DEFAULTS["write_fasta"],
        write_structures: bool = MPNN_GLOBAL_INFERENCE_DEFAULTS["write_structures"],
        device: str | torch.device | None = None,
        extended_vocab: str | None = None,
        field_source: str = "self_edge",
        etab_source: str | None = None,
        etab_hidden: list[int] | None = None,
        field_hidden: list[int] | None = None,
    ):
        self.out_directory = out_directory
        self.write_fasta = write_fasta
        self.write_structures = write_structures
        self.extended_vocab = extended_vocab
        # Must match the checkpoint: a "node" model carries node_field.* weights
        # that a "self_edge" model has no slot for (loads are strict=True). The head depths
        # must match too -- an MLP head has `<name>.0.weight` keys, a single Linear `<name>.weight`.
        self.field_source = field_source
        self.etab_hidden = etab_hidden
        self.field_hidden = field_hidden
        # etab_source ("edge" vs "node_edge_node") also has to match, but unlike the above it is
        # READABLE off the checkpoint -- etab_out's in-features are H or 3H. None => auto-detect.
        self.etab_source = etab_source

        self.checkpoint_path = (
            str(
                REGISTERED_CHECKPOINTS[
                    self.model_type.replace("_", "")
                ].get_default_path()
            )
            if not checkpoint_path
            else checkpoint_path
        )

        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = torch.device("xpu")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self._validate_all()
        self._post_process_engine_config()
        self.model = self._build_and_load_model().to(self.device)
        self.metrics = self._build_metrics_manager()

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _validate_model_config(self) -> None:
        if not isinstance(self.checkpoint_path, str):
            raise TypeError("checkpoint_path must be a string path.")
        ckpt_path = Path(_absolute_path_or_none(self.checkpoint_path))
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"checkpoint_path does not exist: {self.checkpoint_path}"
            )
        if not isinstance(self.is_legacy_weights, bool):
            raise TypeError("is_legacy_weights must be a bool.")

    def _validate_output_config(self) -> None:
        if self.out_directory is not None and not isinstance(self.out_directory, str):
            raise TypeError("out_directory must be a string when provided.")
        for name in ("write_fasta", "write_structures"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool.")
            if value and self.out_directory is None:
                raise ValueError(f"{name} is True, but out_directory is not set.")

    def _validate_all(self) -> None:
        self._validate_model_config()
        self._validate_output_config()

    def _post_process_engine_config(self) -> None:
        self.checkpoint_path = _absolute_path_or_none(self.checkpoint_path)
        if self.out_directory is not None:
            self.out_directory = _absolute_path_or_none(self.out_directory)

    def _build_and_load_model(self) -> torch.nn.Module:
        # Load first: the checkpoint tells us the head's arity, so the caller doesn't have to.
        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise TypeError("Expected checkpoint to be a dict with a 'model' key.")

        etab_source = self.etab_source
        if etab_source is None:
            etab_source = PottsMPNN.infer_etab_source(checkpoint["model"])

        potts_kwargs = dict(
            field_source=self.field_source,
            etab_source=etab_source,
            etab_hidden=self.etab_hidden,
            field_hidden=self.field_hidden,
        )
        if self.extended_vocab:
            model = PottsMPNN(
                graph_featurization_module=PottsProteinFeatures(), **potts_kwargs
            )
        else:
            model = PottsMPNN(**potts_kwargs)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        return model

    def _build_metrics_manager(self) -> MetricManager:
        metrics: dict[str, Any] = {
            "sequence_recovery": SequenceRecovery(return_per_example_metrics=True),
        }
        return MetricManager.from_metrics(metrics, raise_errors=True)

    def _build_network_input(
        self,
        atom_array: AtomArray,
        input_dict: dict[str, Any],
    ) -> tuple[dict, Any]:
        """Run the transform pipeline and collator, returning
        ``(network_input, pipeline_output)``.  ``pipeline_output`` is kept
        so callers can decode sequences back onto the atom array."""

        pipeline_args: dict[str, Any] = {}
        if input_dict.get("occupancy_threshold_sidechain") is not None:
            pipeline_args["occupancy_threshold_sidechain"] = input_dict[
                "occupancy_threshold_sidechain"
            ]
        if input_dict.get("occupancy_threshold_backbone") is not None:
            pipeline_args["occupancy_threshold_backbone"] = input_dict[
                "occupancy_threshold_backbone"
            ]
        if input_dict.get("undesired_res_names") is not None:
            pipeline_args["undesired_res_names"] = input_dict["undesired_res_names"]

        pipeline = build_mpnn_transform_pipeline(
            model_type=self.model_type,
            is_inference=True,
            minimal_return=True,
            device=self.device,
            **pipeline_args,
        )
        collator = FeatureCollator()

        data: dict[str, Any] = {
            "atom_array": atom_array.copy(),
            "structure_noise": input_dict["structure_noise"],
            "decode_type": input_dict["decode_type"],
            "causality_pattern": input_dict["causality_pattern"],
            "initialize_sequence_embedding_with_ground_truth": input_dict[
                "initialize_sequence_embedding_with_ground_truth"
            ],
            "atomize_side_chains": input_dict["atomize_side_chains"],
            "repeat_sample_num": input_dict["repeat_sample_num"],
            "features_to_return": input_dict["features_to_return"],
        }

        pipeline_output = pipeline(data)
        network_input = collator([pipeline_output])
        return network_input, pipeline_output

    def _decode_sequences(
        self,
        seq_int: np.ndarray,
        pipeline_output: Any,
        input_dict: dict[str, Any],
        batch_idx: int | None,
        extra_output_fields: dict | None = None,
    ) -> list[MPNNInferenceOutput]:
        """Convert integer-encoded sequences [N, L] into MPNNInferenceOutput
        objects, applying the designed residues back onto the atom array.

        Parameters
        ----------
        seq_int:
            Integer-encoded sequences as a numpy array of shape [N, L].
        pipeline_output:
            The dict returned by the transform pipeline (contains atom_array).
        input_dict:
            Fully-defaulted per-input settings dict.
        batch_idx:
            Batch index written into each output dict.
        extra_output_fields:
            Optional additional fields (e.g. ``potts_energy``) written into
            each output dict. If a list is provided per key its i-th element
            is used for design i; scalars are broadcast to all designs.
        """
        idx_to_token = MPNN_TOKEN_ENCODING.idx_to_token
        N = seq_int.shape[0]
        outputs: list[MPNNInferenceOutput] = []

        for design_idx in range(N):
            design_atom_array = pipeline_output["atom_array"].copy()

            design_non_atomized_array = design_atom_array[~design_atom_array.atomize]
            design_non_atomized_token_starts = get_token_starts(
                design_non_atomized_array
            )
            design_non_atomized_token_level = design_non_atomized_array[
                design_non_atomized_token_starts
            ]

            designed_resnames = np.array(
                [idx_to_token[int(t)] for t in seq_int[design_idx]],
                dtype=design_atom_array.res_name.dtype,
            )

            if len(design_non_atomized_token_level) != len(designed_resnames):
                raise ValueError(
                    "Mismatch between number of non-atomized tokens and "
                    "decoded sequence length."
                )

            designed_resnames_atom = spread_token_wise(
                design_non_atomized_array, designed_resnames
            )
            full_resnames = design_atom_array.res_name.copy()
            full_resnames[~design_atom_array.atomize] = designed_resnames_atom
            design_atom_array.set_annotation("res_name", full_resnames)

            design_is_backbone_atom = np.isin(
                design_atom_array.atom_name, PROTEIN_BACKBONE_ATOM_NAMES
            )
            if (
                "mpnn_designed_residue_mask"
                in design_atom_array.get_annotation_categories()
            ):
                design_is_fixed_atom = ~design_atom_array.mpnn_designed_residue_mask
            else:
                design_is_fixed_atom = np.zeros(len(design_atom_array), dtype=bool)
            design_atom_array = design_atom_array[
                design_atom_array.atomize
                | design_is_backbone_atom
                | design_is_fixed_atom
            ]

            one_letter_seq = "".join(
                DICT_THREE_TO_ONE.get(r, DICT_THREE_TO_ONE[UNKNOWN_AA])
                for r in designed_resnames
            )

            output_dict: dict[str, Any] = {
                "batch_idx": batch_idx,
                "design_idx": design_idx,
                "designed_sequence": one_letter_seq,
                "model_type": self.model_type,
                "checkpoint_path": self.checkpoint_path,
                "is_legacy_weights": self.is_legacy_weights,
            }

            if extra_output_fields is not None:
                for k, v in extra_output_fields.items():
                    output_dict[k] = v[design_idx] if isinstance(v, list) else v

            outputs.append(
                MPNNInferenceOutput(
                    atom_array=design_atom_array,
                    output_dict=output_dict,
                    input_dict=copy.deepcopy(input_dict),
                )
            )

        return outputs

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run_autoregeressive(
        self,
        *,
        input_dicts: list[dict[str, Any]] | None = None,
        atom_arrays: list[AtomArray] | None = None,
    ) -> list[MPNNInferenceOutput]:
        """Sample sequences autoregressively (standard ProteinMPNN decoding).

        Returns a flat list of MPNNInferenceOutput objects, one per design.
        """
        num_inputs = len(input_dicts) if input_dicts is not None else len(atom_arrays)
        results: list[MPNNInferenceOutput] = []

        for input_idx in range(num_inputs):
            inference_input = MPNNInferenceInput.from_atom_array_and_dict(
                atom_array=atom_arrays[input_idx] if atom_arrays is not None else None,
                input_dict=input_dicts[input_idx] if input_dicts is not None else None,
            )

            seed = inference_input.input_dict["seed"]
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                elif torch.backends.mps.is_available():
                    torch.mps.manual_seed(seed)

            for batch_idx in range(inference_input.input_dict["number_of_batches"]):
                ranked_logger.info(
                    f"Running MPNN autoregressive inference for input {input_idx}, "
                    f"batch {batch_idx}..."
                )
                result = self._run_batch(
                    atom_array=inference_input.atom_array,
                    input_dict=inference_input.input_dict,
                    batch_idx=batch_idx,
                )
                results.extend(result)

        self._write_outputs(results)
        return results

    def run_potts_optimize(
        self,
        *,
        input_dicts: list[dict[str, Any]] | None = None,
        atom_arrays: list[AtomArray] | None = None,
        num_sequences: int = 10,
        temperature: float = 0.01,
        max_iters: int = 1000,
        convergence_mode: bool = True,
        init_mode: str = "sample",
    ) -> list[MPNNInferenceOutput]:
        """Optimise sequences using Potts Gibbs sweeps.

        For each input structure this method:
        1. Featurizes the structure through the transform pipeline.
        2. Runs the model encoder to obtain structure-conditioned Potts
           energy tables (etab_out, E_idx).
        3. Builds ``num_sequences`` initial sequences via autoregressive
           sampling (``init_mode="sample"``), argmax (``init_mode="argmax"``),
           or the native sequence (``init_mode="ground_truth"``).
        4. Runs Gibbs sweeps independently on each sequence until convergence
           or ``max_iters`` sweeps, updating one position at a time.
        5. Returns the optimised sequences sorted by Potts energy (lowest first).

        Fixed / designed residues are read from ``input_dict["fixed_residues"]``
        and related keys — the same as for autoregressive inference.

        Parameters
        ----------
        input_dicts:
            Per-input JSON-style settings dicts.
        atom_arrays:
            Pre-loaded AtomArray objects, aligned one-to-one with input_dicts.
        num_sequences:
            Number of independent Gibbs chains (N) to run per input.
        temperature:
            Boltzmann temperature for Gibbs sampling. As T→0 the sampler
            becomes a greedy coordinate-descent minimiser.
        max_iters:
            Maximum number of Gibbs sweeps per sequence.
        convergence_mode:
            Stop a chain early when a full sweep produces zero mutations.
        init_mode:
            How to initialise the N sequences before Gibbs:
            - ``"sample"``       — N autoregressive samples from the model.
            - ``"argmax"``       — argmax of the model logits, repeated N times.
            - ``"ground_truth"`` — native sequence from the structure, repeated N times.

        Returns
        -------
        list[MPNNInferenceOutput]
            One output per optimised sequence, sorted lowest Potts energy first
            within each input.
        """
        if init_mode not in ("sample", "argmax", "ground_truth"):
            raise ValueError(
                f"init_mode must be 'sample', 'argmax', or 'ground_truth', "
                f"got {init_mode!r}."
            )

        num_inputs = len(input_dicts) if input_dicts is not None else len(atom_arrays)
        results: list[MPNNInferenceOutput] = []

        for input_idx in range(num_inputs):
            inference_input = MPNNInferenceInput.from_atom_array_and_dict(
                atom_array=atom_arrays[input_idx] if atom_arrays is not None else None,
                input_dict=input_dicts[input_idx] if input_dicts is not None else None,
            )

            seed = inference_input.input_dict["seed"]
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                elif torch.backends.mps.is_available():
                    torch.mps.manual_seed(seed)

            ranked_logger.info(
                f"Running Potts optimisation for input {input_idx} "
                f"({num_sequences} sequences, init={init_mode})..."
            )

            # ── Step 1: featurize ──────────────────────────────────────────
            # For init_mode "sample"/"argmax" we do a full forward pass with
            # repeat_sample_num=num_sequences so the model returns N sampled
            # sequences AND the Potts context in one shot.
            # For "ground_truth" we only need the encoder.

            input_dict = inference_input.input_dict

            if init_mode in ("sample", "argmax"):
                # Temporarily set repeat_sample_num so the forward pass
                # returns N sequences without re-encoding.
                fd_input_dict = copy.deepcopy(input_dict)
                fd_input_dict["repeat_sample_num"] = num_sequences

                network_input, pipeline_output = self._build_network_input(
                    inference_input.atom_array, fd_input_dict
                )

                with torch.no_grad():
                    network_output = self.model(network_input)

                # Potts tables — computed before repeat_along_batch, so B=1.
                etab_out = network_output["potts_context"].etab_out  # [1, L, K, V, V]
                E_idx = network_output["potts_context"].E_idx         # [1, L, K]

                if init_mode == "sample":
                    seq_init = network_output["decoder_features"]["S_sampled"]  # [N, L]
                else:
                    seq_argmax = network_output["decoder_features"]["S_argmax"]  # [1, L]
                    seq_init = seq_argmax.expand(num_sequences, -1)

            else:  # ground_truth
                network_input, pipeline_output = self._build_network_input(
                    inference_input.atom_array, input_dict
                )

                # Encoder-only pass — no decoder needed.
                input_features_copy = copy.deepcopy(
                    network_input["input_features"]
                )
                with torch.no_grad():
                    etab_out, E_idx = self.model.run_potts_encoder(
                        input_features_copy
                    )

                # Use the native sequence as initialisation.
                S_native = network_input["input_features"]["S"]  # [1, L]
                seq_init = S_native.expand(num_sequences, -1)

            # ── Step 2: build masks ────────────────────────────────────────
            # designed_residue_mask [1, L]: True = free to design / mutate.
            free_mask = (
                network_input["input_features"]["designed_residue_mask"]
                .squeeze(0)
                .bool()
            )  # [L]

            # Exclude UNK and any other invalid tokens from Gibbs proposals.
            V = self.model.potts_vocab_size
            valid_aa_mask = torch.ones(V, dtype=torch.bool, device=self.device)
            for idx in self.model.unknown_token_indices:
                valid_aa_mask[idx] = False

            # ── Step 3: Gibbs optimisation ─────────────────────────────────
            ranked_logger.info("Starting Gibbs sweeps...")
            seq_opt, energies = PottsMPNN.potts_gibbs_optimize(
                etab_out=etab_out,
                E_idx=E_idx,
                seq_init=seq_init.clone(),
                free_mask=free_mask,
                temperature=temperature,
                max_iters=max_iters,
                convergence_mode=convergence_mode,
                valid_aa_mask=valid_aa_mask,
            )

            # ── Step 4: sort by energy (lowest = best) and decode ──────────
            order = energies.argsort()
            seq_opt = seq_opt[order].detach().cpu().numpy()      # [N, L]
            energies_sorted = energies[order].detach().cpu().tolist()

            batch_outputs = self._decode_sequences(
                seq_int=seq_opt,
                pipeline_output=pipeline_output,
                input_dict=input_dict,
                batch_idx=0,
                extra_output_fields={
                    "potts_energy": energies_sorted,
                },
            )
            results.extend(batch_outputs)

        self._write_outputs(results)
        return results

    # ------------------------------------------------------------------ #
    # Internal batch runner (autoregressive path)
    # ------------------------------------------------------------------ #

    def _run_batch(
        self,
        atom_array: AtomArray,
        input_dict: dict[str, Any],
        batch_idx: int | None = None,
    ) -> list[MPNNInferenceOutput]:
        """Featurize, run the full forward pass, decode, and return outputs."""

        network_input, pipeline_output = self._build_network_input(
            atom_array, input_dict
        )

        with torch.no_grad():
            network_output = self.model(network_input)

        metrics_output = self.metrics(
            network_input=network_input,
            network_output=network_output,
            extra_info={},
        )

        S_sampled = (
            network_output["decoder_features"]["S_sampled"].detach().cpu().numpy()
        )
        B = S_sampled.shape[0]
        if B != input_dict["batch_size"]:
            raise ValueError(
                "Mismatch between network output batch size and input_dict batch_size."
            )

        sequence_recovery_per_design = (
            metrics_output[
                "sequence_recovery.sequence_recovery_per_example_sampled"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        return self._decode_sequences(
            seq_int=S_sampled,
            pipeline_output=pipeline_output,
            input_dict=input_dict,
            batch_idx=batch_idx,
            extra_output_fields={
                "sequence_recovery": sequence_recovery_per_design.tolist(),
                "ligand_interface_sequence_recovery": None,
            },
        )

    # ------------------------------------------------------------------ #
    # Output writing
    # ------------------------------------------------------------------ #

    def _write_outputs(self, results: list[MPNNInferenceOutput]) -> None:
        out_directory = self.out_directory
        if not out_directory and (self.write_fasta or self.write_structures):
            raise ValueError(
                "Output directory is not set, but writing of outputs was requested."
            )
        if not out_directory:
            return

        out_dir_path = Path(out_directory)
        out_dir_path.mkdir(parents=True, exist_ok=True)

        if self.write_structures:
            for idx, result in enumerate(results):
                name = result.input_dict["name"]
                if name is None:
                    raise ValueError(
                        f"Cannot write structure for result {idx}: 'name' is "
                        "not set in input_dict."
                    )
                batch_idx = result.output_dict["batch_idx"]
                design_idx = result.output_dict["design_idx"]
                file_stem = f"{name}_b{batch_idx}_d{design_idx}"
                result.write_structure(base_path=out_dir_path / file_stem)

        if self.write_fasta:
            grouped: dict[str, list[MPNNInferenceOutput]] = {}
            for result in results:
                name = result.input_dict["name"]
                if name is None:
                    raise ValueError(
                        "Cannot write FASTA output: 'name' is not set in input_dict."
                    )
                grouped.setdefault(name, []).append(result)

            for name, group in grouped.items():
                fasta_path = out_dir_path / f"{name}.fa"
                with fasta_path.open("a") as handle:
                    for result in group:
                        result.write_fasta(handle=handle)
