"""Fabric/DDP trainer for the linear H-bond / salt-bridge head.

Analogous to ``PottsMPNNTrainer`` but trains ONLY the small ``HBondHead`` on top of a
FROZEN PottsMPNN (encoder + token embedding + decoder + heads all frozen — see
``HBondModel``). Multi-GPU via the same Fabric/DDP harness:

    trainer = HBondHeadTrainer(accelerator="gpu", devices_per_node=n_gpus,
                               precision="bf16-mixed", ...)

``find_unused_parameters=True`` is required: the head's donor/acceptor (and salt) output
rows receive no gradient on batches without the corresponding positive edges, so DDP must
tolerate unused parameters.
"""

import numpy as np
import torch
from beartype.typing import Any
from lightning_utilities import apply_to_collection

from mpnn.loss.hbond_loss import HBondHeadLoss
from mpnn.model.hbond_model import HBondModel
from mpnn.model.pottsmpnn import PottsMPNN
from mpnn.model.layers.graph_embeddings import PottsProteinFeatures
from mpnn.hbond_head import (
    bond_labels_from_partners, edge_masks, ambiguous_token_mask, reverse_edge_index,
)

from foundry.trainers.fabric import FabricTrainer
from foundry.utils.ddp import RankedLogger
from foundry.utils.torch import assert_no_nans

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


class HBondHeadTrainer(FabricTrainer):
    """Trainer that fits only the HBondHead over a frozen PottsMPNN."""

    def __init__(
        self,
        *,
        extended_vocab: str | bool = True,
        n_hidden: int = 0,
        neg_per_pos: int | None = 10,
        encoder_checkpoint: str | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Args:
            extended_vocab: the encoder's vocabulary; must match ``encoder_checkpoint``. A vocab
                NAME (``"v6"`` -> 30-token, ``"v3"``/``"v4"`` -> 32-token) sizes ``W_s`` and the
                ambiguous-token mask correctly; a bare ``True`` keeps the default 32-token vocab.
            n_hidden: HBondHead hidden width (0 => single linear layer).
            neg_per_pos: negative:positive subsample ratio for the BCE terms (training).
            encoder_checkpoint: path to the pretrained PottsMPNN ``.ckpt`` whose weights
                are loaded (frozen) into ``HBondModel.potts``. Required.
            verbose: print per-batch losses.
        """
        super().__init__(**kwargs)
        self.extended_vocab = extended_vocab
        self.n_hidden = n_hidden
        self.encoder_checkpoint = encoder_checkpoint
        self.verbose = verbose
        self.metrics = None  # PR/ROC/per-type handled by HBondValidationCallback
        # The ambiguous (*-A/UNK) mask in the loss must use the encoder's own vocab: a vocab NAME
        # ("v6") -> that vocab's token_encoding; a bare bool -> the default 32-token encoding (v3/v4).
        # Resolved ONCE and kept on self: the validation edge-collection below needs the same
        # encoding, and passing the 32-token default there under v6 masks ASP-D/GLU-D instead of
        # the *-A tokens -- silently corrupting the reported PR/ROC/per-type metrics.
        if isinstance(extended_vocab, str):
            from mpnn.transforms.extended_vocab import get_vocab
            self.token_encoding = get_vocab(extended_vocab)["token_encoding"]
            self.loss = HBondHeadLoss(neg_per_pos=neg_per_pos, encoding=self.token_encoding)
        else:
            self.token_encoding = None      # -> HBondHeadLoss / ambiguous_token_mask default (32-token)
            self.loss = HBondHeadLoss(neg_per_pos=neg_per_pos)

    def construct_model(self):
        """Build HBondModel and load the frozen pretrained PottsMPNN weights."""
        if not self.encoder_checkpoint:
            raise ValueError("HBondHeadTrainer requires encoder_checkpoint (pretrained PottsMPNN).")

        ckpt = torch.load(self.encoder_checkpoint, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        with self.fabric.init_module():
            ranked_logger.info("Instantiating HBondModel (frozen PottsMPNN + linear head)...")
            # The head only ever reads h_V/h_E, never etab_out -- but the frozen encoder is loaded
            # strict=True, so its coupling head AND W_s must match the checkpoint's shape. Build the
            # PottsMPNN here: arity (edge vs node_edge_node) read off the checkpoint, and the token
            # vocabulary sized by extended_vocab -- a NAME ("v6") -> 30 tokens, bool -> default 32.
            if self.extended_vocab:
                if isinstance(self.extended_vocab, str):
                    from mpnn.transforms.extended_vocab import get_vocab
                    feats = PottsProteinFeatures(
                        token_encoding=get_vocab(self.extended_vocab)["token_encoding"])
                else:
                    feats = PottsProteinFeatures()
                potts = PottsMPNN(graph_featurization_module=feats,
                                  etab_source=PottsMPNN.infer_etab_source(state))
            else:
                potts = PottsMPNN(etab_source=PottsMPNN.infer_etab_source(state))
            model = HBondModel(extended_vocab=self.extended_vocab, n_hidden=self.n_hidden,
                               potts=potts)

        model.potts.load_state_dict(state, strict=True)
        model.freeze_pretrained()  # re-assert: only the head trains
        ranked_logger.info(f"Loaded frozen PottsMPNN from {self.encoder_checkpoint}")

        self.initialize_or_update_trainer_state({"model": model})

    def construct_optimizer(self) -> None:
        """Optimize ONLY the trainable (head) parameters."""
        model = self.state["model"]
        params = [p for p in model.parameters() if p.requires_grad]
        n = sum(p.numel() for p in params)
        ranked_logger.info(f"Optimizing {n} trainable head params (everything else frozen).")
        import hydra
        optimizer = hydra.utils.instantiate(
            self.state["train_cfg"].model.optimizer, params=params
        )
        self.initialize_or_update_trainer_state({"optimizer": optimizer})

    # ------------------------------------------------------------------ #
    def training_step(self, batch: Any, batch_idx: int, is_accumulating: bool) -> None:
        model = self.state["model"]
        assert model.training, "Model must be training!"
        with self.fabric.no_backward_sync(model, enabled=is_accumulating):
            network_output = model(batch)
            assert_no_nans(network_output["logits"], msg=f"logits batch_idx={batch_idx}")
            total_loss, loss_dict = self.loss(
                network_input=batch, network_output=network_output, loss_input={}
            )
            if self.verbose:
                ep = self.state.get("current_epoch", "?")
                parts = "  ".join(f"{k}={v.item():.4f}" for k, v in loss_dict.items()
                                  if hasattr(v, "item") and v.numel() == 1)
                print(f"[train] epoch={ep} batch={batch_idx} {parts}", flush=True)
            self.fabric.backward(total_loss)
            self._current_train_return = apply_to_collection(
                {"total_loss": total_loss, "loss_dict": loss_dict},
                dtype=torch.Tensor, function=lambda x: x.detach(),
            )

    @torch.no_grad()
    def validation_step(self, batch: Any, batch_idx: int, compute_metrics: bool = True) -> dict:
        model = self.state["model"]
        assert not model.training, "Model must be in eval mode during validation!"
        network_output = model(batch)
        total_loss, loss_dict = self.loss(
            network_input=batch, network_output=network_output, loss_input={}
        )
        loss_dict = apply_to_collection(loss_dict, torch.Tensor, lambda x: x.detach())

        edge_eval = self._collect_edge_eval(batch, network_output)

        scalar_loss = {k: v for k, v in loss_dict.items()
                       if not isinstance(v, torch.Tensor) or v.numel() == 1}
        return {
            "metrics_output": scalar_loss,
            "loss_dict": loss_dict,
            "total_loss": total_loss.detach(),
            "edge_eval": edge_eval,
        }

    def _collect_edge_eval(self, batch, network_output) -> dict:
        """Per-edge predictions + labels for the validation callback (CPU numpy).

        Returns deduped undirected H-bond / salt arrays (PR), directed donor-vs-acceptor
        arrays on H-bond edges (ROC), and per directed H-bond-positive edge the endpoint
        token ids + predicted hbond prob (per-type accuracy / frequency / loss).
        """
        inp = batch["input_features"]
        logits = network_output["logits"]
        E_idx = network_output["E_idx"]
        S = inp["S"]

        # Probabilities directly from logits (avoid re-encoding).
        p_hbond = torch.sigmoid(logits[..., 0])
        p_dir = torch.softmax(logits[..., 1:3], dim=-1)
        p_acc = p_dir[..., 1]
        p_salt = torch.sigmoid(logits[..., 3])

        labels = bond_labels_from_partners(
            E_idx, inp["hbond_donates_to"], inp["hbond_accepts_from"], inp["salt_partners"]
        )
        amb = (ambiguous_token_mask(S, encoding=self.token_encoding)
               if self.token_encoding is not None else ambiguous_token_mask(S))
        masks = edge_masks(E_idx, inp["residue_mask"], amb, labels)

        B, L, K = E_idx.shape
        hb_p, hb_y, sb_p, sb_y, dir_p, dir_y = [], [], [], [], [], []
        ti, tj, tp = [], [], []   # per-type: donor tok, acceptor tok, p_hbond on +edges
        for b in range(B):
            rk, _ = reverse_edge_index(E_idx[b])
            rows = torch.arange(L, device=E_idx.device)[:, None].expand(L, K)
            j = E_idx[b]
            # undirected dedup (i<j), unambiguous endpoints
            keep = masks["hbond"][b] & (rows < j)
            comb = torch.maximum(p_hbond[b], p_hbond[b][j, rk])
            yb = labels["hbond"][b] | labels["hbond"][b][j, rk]
            hb_p.append(comb[keep]); hb_y.append(yb[keep].float())
            keeps = masks["salt"][b] & (rows < j)
            scomb = torch.maximum(p_salt[b], p_salt[b][j, rk])
            ys = labels["salt"][b] | labels["salt"][b][j, rk]
            sb_p.append(scomb[keeps]); sb_y.append(ys[keeps].float())
            # direction ROC (directed hbond edges, label 1 = acceptor)
            md = masks["dir"][b]
            dir_p.append(p_acc[b][md]); dir_y.append(labels["acceptor"][b][md].float())
            # per-type: donor-positive directed edges (i donates to j)
            dmask = masks["hbond"][b] & labels["donor"][b]
            di, dj = torch.where(dmask)
            ti.append(S[b][di]); tj.append(S[b][E_idx[b][di, dj]]); tp.append(p_hbond[b][di, dj])

        cat = lambda xs: torch.cat(xs).detach().cpu().numpy() if xs else np.zeros(0)
        return {
            "hbond_p": cat(hb_p), "hbond_y": cat(hb_y),
            "salt_p": cat(sb_p), "salt_y": cat(sb_y),
            "dir_p": cat(dir_p), "dir_y": cat(dir_y),
            "type_donor": cat(ti).astype(np.int64), "type_acceptor": cat(tj).astype(np.int64),
            "type_phbond": cat(tp),
        }
