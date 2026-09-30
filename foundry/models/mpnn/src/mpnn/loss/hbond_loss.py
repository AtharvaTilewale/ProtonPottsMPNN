"""Loss for the linear H-bond / salt-bridge head (HBondModel).

Follows the foundry trainer loss contract used by ``PottsJointLoss``::

    total_loss, loss_dict = loss(network_input=batch,
                                 network_output={"logits", "E_idx"},
                                 loss_input={})

The ground-truth bond graph is carried IN the batch as token-pair partner lists
(``hbond_donates_to`` / ``hbond_accepts_from`` / ``salt_partners``; produced by
``mpnn.transforms.bond_annotation.BuildBondEdgeLabels``). We expand them to per-edge
labels aligned to the model's own ``E_idx`` and combine:

    BCE(hbond presence) + BCE(salt presence) + CE(donor-vs-acceptor direction).

Edges touching an ambiguous protonation microstate (``*-A``) or ``UNK`` are masked out
of the H-bond and direction terms; salt-bridge presence is geometry-only and keeps them.
"""

import torch
import torch.nn as nn

from mpnn.hbond_head import (
    bond_labels_from_partners,
    edge_masks,
    ambiguous_token_mask,
    hbond_head_loss,
)
from mpnn.transforms.feature_aggregation.token_encodings import POTTS_MPNN_TOKEN_ENCODING


class HBondHeadLoss(nn.Module):
    """Joint per-edge BCE(hbond) + BCE(salt) + CE(direction) loss.

    Args:
        neg_per_pos: negative:positive subsample ratio for the two BCE terms during
            training (caps the heavy edge-class imbalance). ``None`` => use all edges.
        encoding: token encoding used to resolve the ambiguous (``*-A``/UNK) tokens that
            are masked out of the H-bond/direction terms. MUST match the encoder's vocab
            (v6 = 30-token; v3/v4 = the default 32-token). Wrong encoding => the wrong
            token indices get masked.
    """

    def __init__(self, neg_per_pos: int | None = 10, encoding=POTTS_MPNN_TOKEN_ENCODING):
        super().__init__()
        self.neg_per_pos = neg_per_pos
        self.encoding = encoding

    def forward(self, network_input, network_output, loss_input):
        inp = network_input["input_features"]
        for key in ("S", "residue_mask", "hbond_donates_to",
                    "hbond_accepts_from", "salt_partners"):
            if key not in inp:
                raise ValueError(f"Input features must contain '{key}' for HBondHeadLoss.")
        if "logits" not in network_output or "E_idx" not in network_output:
            raise ValueError("Network output must contain 'logits' and 'E_idx'.")

        logits = network_output["logits"]     # [B, L, K, 4]
        E_idx = network_output["E_idx"]        # [B, L, K]

        labels = bond_labels_from_partners(
            E_idx, inp["hbond_donates_to"], inp["hbond_accepts_from"], inp["salt_partners"]
        )
        amb = ambiguous_token_mask(inp["S"], self.encoding)
        masks = edge_masks(E_idx, inp["residue_mask"], amb, labels)

        total, parts = hbond_head_loss(
            logits, labels, masks,
            neg_per_pos=self.neg_per_pos if self.training else None,
        )

        loss_dict = {f"{k}_loss": v.detach() for k, v in parts.items()}
        loss_dict["total_loss"] = total.detach()
        # edge-count diagnostics (scalars; cheap and useful for monitoring imbalance)
        loss_dict["n_hbond_pos"] = (labels["hbond"] & masks["hbond"]).sum().detach()
        loss_dict["n_salt_pos"] = (labels["salt"] & masks["salt"]).sum().detach()
        loss_dict["n_dir_edges"] = masks["dir"].sum().detach()
        return total, loss_dict
