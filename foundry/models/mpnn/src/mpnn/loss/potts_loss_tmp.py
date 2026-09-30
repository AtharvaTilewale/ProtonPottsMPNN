import torch
import torch.nn as nn


class LabelSmoothedNLLLoss(nn.Module):
    def __init__(self, label_smoothing_eps=0.1, normalization_constant=6000.0):
        """
        Label smoothed negative log likelihood loss for Protein/Ligand MPNN.

        Args:
            label_smoothing_eps (float): The label smoothing factor. Default is
                0.1.
            normalization_constant (float): The normalization constant for the
                loss. As opposed to averaging per sample in the batch, or
                averaging across all tokens, this constant is used to normalize
                the loss. Default is 6000.0.
        """
        super(LabelSmoothedNLLLoss, self).__init__()

        self.label_smoothing_eps = label_smoothing_eps
        self.normalization_constant = normalization_constant

    def forward(self, network_input, network_output, loss_input):
        """
        Given the network_input (same as input_features to the model), network
        output, and loss input, compute the loss.
        """
        input_features = network_input["input_features"]

        if "S" not in input_features:
            raise ValueError("Input features must contain 'S' key.")
        if "input_features" not in network_output:
            raise ValueError("Network output must contain 'input_features' key.")
        if "mask_for_loss" not in network_output["input_features"]:
            raise ValueError(
                "Network output must contain'mask_for_loss' key in 'input_features'."
            )
        if "decoder_features" not in network_output:
            raise ValueError("Network output must contain 'decoder_features' key.")
        if "log_probs" not in network_output["decoder_features"]:
            raise ValueError(
                "Network output must contain'log_probs' key in 'decoder_features'."
            )

        _, _, vocab_size = network_output["decoder_features"]["log_probs"].shape

        S_onehot = torch.nn.functional.one_hot(
            input_features["S"], num_classes=vocab_size
        ).float()

        label_smoothed_S_onehot = (
            1 - self.label_smoothing_eps
        ) * S_onehot + self.label_smoothing_eps / vocab_size

        label_smoothed_nll_loss_per_residue = (
            -torch.sum(
                label_smoothed_S_onehot
                * network_output["decoder_features"]["log_probs"],
                dim=-1,
            )
            * network_output["input_features"]["mask_for_loss"]
        )

        label_smoothed_nll_loss_agg = (
            torch.sum(label_smoothed_nll_loss_per_residue)
            / self.normalization_constant
        )

        loss_dict = {
            "label_smoothed_nll_loss_per_residue": label_smoothed_nll_loss_per_residue.detach(),
            "label_smoothed_nll_loss_agg": label_smoothed_nll_loss_agg.detach(),
        }

        return label_smoothed_nll_loss_agg, loss_dict


class PottsNLCPLLoss(nn.Module):
    """
    Negative log composite pseudo-likelihood (NLCPL) for PottsMPNN.

    This follows the same computation pattern as
    `external/PottsMPNN/potts_mpnn_utils.py::nlcpl`, adapted to the local
    Potts context:

    - `ctx.etab_out` is already shaped `[B, L, K, V, V]`
    - `ctx.potts_loss_mask` is already an edge-level supervision mask
    - we therefore do not need the external repo's 20->22 padding step or
      extra unknown-residue mask
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, network_input, network_output, loss_input):
        input_features = network_input["input_features"]

        if "S" not in input_features:
            raise ValueError("Input features must contain 'S' key.")
        if "potts_context" not in network_output:
            raise ValueError("Network output must contain 'potts_context' key.")

        S = input_features["S"]
        ctx = network_output["potts_context"]

        etab_out = ctx.etab_out
        E_idx = ctx.E_idx # BxLxK
        potts_loss_mask = ctx.potts_loss_mask

        B, L, K, V, V2 = etab_out.shape
        if V != V2:
            raise ValueError("Expected square Potts tables in etab_out.")
        if K < 2:
            raise ValueError("PottsNLCPLLoss requires at least one non-self edge.")
        if int(S.max().item()) >= V:
            raise ValueError(
                f"Sequence index {int(S.max().item())} exceeds Potts vocab size {V}."
            )
        
        y_true = torch.zeros(B, L, K, V, V2)
        y_true[E_idx,:,:] = ...
