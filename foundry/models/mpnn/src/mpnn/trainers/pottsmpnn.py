import torch
from beartype.typing import Any
from lightning_utilities import apply_to_collection
from mpnn.loss.nll_loss import LabelSmoothedNLLLoss
from mpnn.loss.potts_loss import PottsJointLoss
from mpnn.metrics.nll import NLL, InterfaceNLL
from mpnn.metrics.sequence_recovery import (
    InterfaceSequenceRecovery,
    PottsSequenceRecovery,
    SequenceRecovery,
)
from mpnn.model.layers.graph_embeddings import PottsProteinFeatures
from mpnn.model.mpnn import LigandMPNN, ProteinMPNN
from mpnn.model.pottsmpnn import PottsMPNN
from omegaconf import DictConfig

from foundry.metrics.metric import MetricManager
from foundry.trainers.fabric import FabricTrainer
from foundry.utils.ddp import RankedLogger
from foundry.utils.torch import assert_no_nans

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


class PottsMPNNTrainer(FabricTrainer):
    """Standard Trainer for MPNN-style models"""

    def __init__(
        self,
        *,
        model_type: str,
        extended_vocab: str | None = None,
        field_source: str = "self_edge",
        etab_source: str = "edge",
        etab_hidden: list[int] | None = None,
        field_hidden: list[int] | None = None,
        loss: DictConfig | dict | None = None,
        metrics: DictConfig | dict | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        See `FabricTrainer` for the additional initialization arguments.

        Args:
            model_type (str): Type of model to use ("protein_mpnn",
                "ligand_mpnn", or "potts_mpnn").
            extended_vocab (str | None): Protonation-vocabulary NAME ("v3"/"v4", see
                transforms/extended_vocab.py) or None for the standard 21-token vocab. When set,
                also adds an ``extended_recovery``
                metric using PottsSequenceRecovery (canonical-parent-aware
                correctness for the 32-token protonation vocabulary).
            field_source (str): How PottsMPNN produces the single-body field
                h_i(a). "self_edge" (default) takes it from the diagonal of the
                self-edge's pairwise table; "node" maps the node embedding
                through a dedicated linear head. Ignored for non-Potts models.
            etab_source (str): What the pairwise coupling head sees. "edge" (default) is the
                edge embedding h_E alone; "node_edge_node" is concat(h_V[i], h_E[i,k], h_V[j]),
                which widens the head from H to 3H. Ignored for non-Potts models.
            etab_hidden (list[int] | None): Hidden dims for the coupling head
                (head input -> hidden... -> V*V). None/[] = a single Linear (default).
            field_hidden (list[int] | None): Hidden dims for the node-field head
                (node embedding -> hidden... -> V). Only used when field_source="node".
            loss (DictConfig | dict | None): Configuration for the loss
                function. If None, default parameters will be used.
            metrics (DictConfig | dict | None): Configuration for the metrics.
                Ignored - metrics are hard-coded.
        """
        super().__init__(**kwargs)

        self.model_type = model_type
        self.extended_vocab = extended_vocab
        self.field_source = field_source
        self.etab_source = etab_source
        self.etab_hidden = etab_hidden
        self.field_hidden = field_hidden
        self.verbose = verbose

        # Metrics
        metrics = {
            "nll": NLL(),
            "sequence_recovery": SequenceRecovery(),
        }
        if extended_vocab:
            # the canonical map is indexed by token index, so it must be built from the SAME encoding the
            # model is (v6 = 30 tokens); a bare bool keeps the 32-token default.
            _enc = None
            if isinstance(extended_vocab, str):
                from mpnn.transforms.extended_vocab import get_vocab
                _enc = get_vocab(extended_vocab)["token_encoding"]
            metrics["extended_recovery"] = PottsSequenceRecovery(token_encoding=_enc)
        if self.model_type == "ligand_mpnn":
            metrics["interface_nll"] = InterfaceNLL()
            metrics["interface_sequence_recovery"] = InterfaceSequenceRecovery()
        self.metrics = MetricManager(metrics)

        # Loss
        loss_params = dict(loss) if loss else {}
        if self.model_type == "potts_mpnn":
            # The protonation-state term names its P/D token groups through the vocabulary, so the loss
            # has to know which one this run trains on. Passed from the trainer rather than the caller so
            # it can never disagree with the vocabulary the model is built with.
            loss_params.setdefault("extended_vocab", self.extended_vocab)
            self.loss = PottsJointLoss(**loss_params)
        else:
            self.loss = LabelSmoothedNLLLoss(**loss_params)

    def construct_model(self):
        """Construct the model with hard-coded parameters."""
        with self.fabric.init_module():
            ranked_logger.info(f"Instantiating {self.model_type} model...")

            # Hard-coded model selection
            if self.model_type == "potts_mpnn":
                ranked_logger.info(
                    f"Potts field_source: {self.field_source}  "
                    f"etab_source: {self.etab_source}  "
                    f"etab_hidden: {self.etab_hidden}  field_hidden: {self.field_hidden}"
                )
                potts_kwargs = dict(
                    field_source=self.field_source,
                    etab_source=self.etab_source,
                    etab_hidden=self.etab_hidden,
                    field_hidden=self.field_hidden,
                )
                if self.extended_vocab:
                    # extended_vocab is a NAME ("v3"/"v4"/"v6"); a bare bool (legacy) keeps the 32-token
                    # default. The vocab's token_encoding sizes the whole encoder (v6 -> 30 tokens).
                    vocab_name = self.extended_vocab if isinstance(self.extended_vocab, str) else None
                    if vocab_name:
                        from mpnn.transforms.extended_vocab import get_vocab
                        feats = PottsProteinFeatures(
                            token_encoding=get_vocab(vocab_name)["token_encoding"]
                        )
                    else:
                        feats = PottsProteinFeatures()
                    model = PottsMPNN(graph_featurization_module=feats, **potts_kwargs)
                    # the model self-identifies its vocabulary, so validation converts through the right maps
                    model.extended_vocab_name = vocab_name
                else:
                    model = PottsMPNN(**potts_kwargs)
                    model.extended_vocab_name = None
            elif self.model_type == "protein_mpnn":
                model = ProteinMPNN()
            elif self.model_type == "ligand_mpnn":
                model = LigandMPNN()
            else:
                raise ValueError(f"Invalid model type: {self.model_type}")

            # Initialize model weights
            model.apply(model.init_weights)

        self.initialize_or_update_trainer_state({"model": model})

    def _assert_forward_outputs_finite(
        self,
        network_output: dict,
        *,
        batch_idx: int,
    ) -> None:
        """Check decoder outputs and, when present, Potts outputs for NaNs."""
        assert_no_nans(
            network_output["decoder_features"],
            msg="network_output['decoder_features'] "
            + f"for batch_idx: {batch_idx}",
        )

        if self.model_type == "potts_mpnn":
            if "potts_context" not in network_output:
                raise ValueError(
                    "Potts model output must contain 'potts_context'."
                )
            assert_no_nans(
                network_output["potts_context"].etab_out,
                msg="network_output['potts_context'].etab_out "
                + f"for batch_idx: {batch_idx}",
            )

    def training_step(
        self,
        batch: Any,
        batch_idx: int,
        is_accumulating: bool,
    ) -> None:
        """
        Training step, running forward and backward passes.

        Args:
            batch (Any): The current batch; can be of any form.
            batch_idx (int): The index of the current batch.
            is_accumulating (bool): Whether we are accumulating gradients
                (i.e., not yet calling optimizer.step()). If this is the case,
                we should skip the synchronization during the backward pass.

        Returns:
            None; we call `loss.backward()` directly, and store the outputs in
                `self._current_train_return`.
        """
        model = self.state["model"]
        assert model.training, "Model must be training!"

        network_input = batch

        with self.fabric.no_backward_sync(model, enabled=is_accumulating):
            # Forward pass
            network_output = model.forward(network_input)
            self._assert_forward_outputs_finite(
                network_output,
                batch_idx=batch_idx,
            )

            total_loss, loss_dict = self.loss(
                network_input=batch,
                network_output=network_output,
                loss_input={},
            )

            if self.verbose:
                epoch = self.state.get("current_epoch", "?")
                loss_str = "  ".join(f"{k}={v.item():.4f}" for k, v in loss_dict.items() if hasattr(v, "item") and v.numel() == 1)
                print(f"[train] epoch={epoch}  batch={batch_idx}  total_loss={total_loss.item():.4f}  {loss_str}", flush=True)

            # Backward pass
            self.fabric.backward(total_loss)

            # Optionally compute training metrics
            train_return = {"total_loss": total_loss, "loss_dict": loss_dict}

            # Store the outputs without gradients for use in logging,
            # callbacks, learning rate schedulers, etc.
            self._current_train_return = apply_to_collection(
                train_return,
                dtype=torch.Tensor,
                function=lambda x: x.detach(),
            )

    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
        compute_metrics: bool = True,
    ) -> dict:
        """
        Validation step, running forward pass and computing validation
        metrics.

        Args:
            batch (Any): The current batch; can be of any form.
            batch_idx (int): The index of the current batch.
            compute_metrics (bool): Whether to compute metrics. If False, we
                will not compute metrics, and the output will be None. Set to
                False during the inference pipeline, where we need the network
                output but cannot compute metrics (since we do not have the
                ground truth).

        Returns:
            dict: Output dictionary containing the validation metrics and
                network output.
        """
        model = self.state["model"]
        assert not model.training, "Model must be in evaluation mode during validation!"

        network_input = batch

        # Forward pass
        network_output = model.forward(network_input)

        self._assert_forward_outputs_finite(
            network_output,
            batch_idx=batch_idx,
        )

        total_loss, loss_dict = self.loss(
            network_input=batch,
            network_output=network_output,
            loss_input={},
        )
        loss_dict = apply_to_collection(loss_dict, torch.Tensor, lambda x: x.detach())

        metrics_output = {}
        if compute_metrics:
            # Compute all metrics using MetricManager
            metrics_output = self.metrics(
                network_input=batch,
                network_output=network_output,
                extra_info={},
            )

            # Avoid gradients in stored values to prevent memory leaks
            if metrics_output:
                metrics_output = apply_to_collection(
                    metrics_output, torch.Tensor, lambda x: x.detach()
                )

        if self.verbose:
            epoch = self.state.get("current_epoch", "?")
            met_str = "  ".join(
                f"{k}={v.item():.4f}" for k, v in metrics_output.items() if hasattr(v, "item") and v.numel() == 1
            )
            print(f"[val]   epoch={epoch}  batch={batch_idx}  total_loss={total_loss.item():.4f}  {met_str}", flush=True)

        # Only merge scalar loss entries: dense per-residue/per-edge tensors
        # (e.g. potts_log_edge_probs [B,L,K-1], label_smoothed_nll_loss_per_residue [B,L])
        # must not reach the CSV callback — they would bloat or crash serialization.
        scalar_loss_metrics = {
            k: v
            for k, v in loss_dict.items()
            if not isinstance(v, torch.Tensor) or v.numel() == 1
        }
        metrics_output = {
            **metrics_output,
            **scalar_loss_metrics,
        }

        network_output = apply_to_collection(
            network_output, torch.Tensor, lambda x: x.detach()
        )

        validation_return = {
            "metrics_output": metrics_output,
            "loss_dict": loss_dict,
            "total_loss": total_loss.detach(),
            "network_output": network_output,
        }

        return validation_return
