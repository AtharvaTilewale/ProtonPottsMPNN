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
        E_idx = ctx.E_idx
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

        # Match external nlcpl(): separate self and pair energies.
        self_etab = etab_out[:, :, 0:1, :, :]    # [B, L, 1, V, V]
        pair_etab = etab_out[:, :, 1:, :, :]     # [B, L, K-1, V, V]
        E_idx_jn = E_idx[:, :, 1:]               # [B, L, K-1]
        pair_mask = potts_loss_mask[:, :, 1:]    # [B, L, K-1]

        # Diagonal self-edge energies are the single-site field terms.
        self_nrgs_im = torch.diagonal(
            self_etab, offset=0, dim1=-2, dim2=-1
        )                                         # [B, L, 1, V]
        self_nrgs_im_expand = self_nrgs_im.expand(-1, -1, K - 1, -1)

        # Gather self-field of the neighbor residue j for each edge i->j.
        E_idx_jn_expand = E_idx_jn.unsqueeze(-1).expand(-1, -1, -1, V)
        self_nrgs_jn = torch.gather(
            self_nrgs_im_expand, 1, E_idx_jn_expand
        )                                         # [B, L, K-1, V]

        # Gather native amino-acid identities for each neighbor j.
        E_aa = torch.gather(
            S.unsqueeze(-1).expand(-1, -1, K - 1),
            1,
            E_idx_jn,
        )                                         # [B, L, K-1]

        # For each pair table J_ij(a, b), gather the column corresponding to
        # the native amino acid at neighbor j. This yields J_ij(a, s_j).
        E_aa_expand = (
            E_aa.unsqueeze(-1).unsqueeze(-1)
            .expand(-1, -1, -1, V, 1)
        )                                         # [B, L, K-1, V, 1]
        pair_nrgs_jn = torch.gather(
            pair_etab, 4, E_aa_expand
        ).squeeze(-1)                             # [B, L, K-1, V]

        # Sum pair terms across neighbors and subtract the current edge to get
        # the "all other neighbors of i" contribution.
        sum_pair_nrgs_jn = torch.sum(pair_nrgs_jn, dim=2)   # [B, L, V]
        pair_nrgs_im_u = (
            sum_pair_nrgs_jn.unsqueeze(2).expand(-1, -1, K - 1, -1)
            - pair_nrgs_jn
        )                                         # [B, L, K-1, V]

        # Map the analogous "all other neighbors of j" contribution back into
        # the current edge layout. This mirrors the external implementation.
        E_idx_imu_to_ujn = E_idx_jn.unsqueeze(-1).expand_as(pair_nrgs_im_u)
        pair_nrgs_u_jn = torch.gather(
            pair_nrgs_im_u, 1, E_idx_imu_to_ujn
        )                                         # [B, L, K-1, V]

        # Expand vector terms into full candidate pair tables over (a, b).
        self_nrgs_im_expand = self_nrgs_im_expand.unsqueeze(-1).expand(
            -1, -1, -1, -1, V
        )                                         # [B, L, K-1, V, V]
        self_nrgs_jn_expand = self_nrgs_jn.unsqueeze(-1).expand(
            -1, -1, -1, -1, V
        ).transpose(-2, -1)                       # [B, L, K-1, V, V]
        pair_nrgs_im_expand = pair_nrgs_im_u.unsqueeze(-1).expand(
            -1, -1, -1, -1, V
        )                                         # [B, L, K-1, V, V]
        pair_nrgs_jn_expand = pair_nrgs_u_jn.unsqueeze(-1).expand(
            -1, -1, -1, -1, V
        ).transpose(-2, -1)                       # [B, L, K-1, V, V]

        composite_nrgs = (
            self_nrgs_im_expand
            + self_nrgs_jn_expand
            + pair_etab
            + pair_nrgs_im_expand
            + pair_nrgs_jn_expand
        )                                         # [B, L, K-1, V, V]

        # Convert energies into log-probabilities over all V*V candidate pairs.
        composite_nrgs_reshape = composite_nrgs.view(B, L, K - 1, V * V, 1)
        log_composite_prob_dist = torch.log_softmax(
            -composite_nrgs_reshape,
            dim=-2,
        ).view(B, L, K - 1, V, V)

        # First gather the observed neighbor amino acid s_j for every candidate
        # center amino acid a, then gather the observed center amino acid s_i.
        im_probs = torch.gather(
            log_composite_prob_dist, 4, E_aa_expand
        ).squeeze(-1)                             # [B, L, K-1, V]
        ref_seqs_expand = S.view(B, L, 1, 1).expand(-1, -1, K - 1, 1)
        log_edge_probs = torch.gather(
            im_probs, 3, ref_seqs_expand
        ).squeeze(-1)                             # [B, L, K-1]

        pair_mask = pair_mask.to(log_edge_probs.dtype)
        masked_log_edge_probs = log_edge_probs * pair_mask

        n_edges = torch.sum(pair_mask)
        if n_edges <= 0:
            raise ValueError("PottsNLCPLLoss found zero valid supervised pair edges.")

        potts_nlcpl_agg = -torch.sum(masked_log_edge_probs) / n_edges.clamp_min(self.eps)

        loss_dict = {
            "potts_log_edge_probs": masked_log_edge_probs.detach(),
            "potts_nlcpl_agg": potts_nlcpl_agg.detach(),
        }

        return potts_nlcpl_agg, loss_dict


_TITRATABLE = ("HIS", "ASP", "GLU")


class ProtonationStateLoss(nn.Module):
    """Auxiliary loss on the PROTONATION STATE alone (``L_state``).

    The full-vocabulary NLL at a titratable position factorises exactly as

        -log p(true token) = -log p(state | parent) + -log p(parent)

    and on the selected v6 model the second term carries ~80% of the 4.01 nats at a HIS position. Since
    titratable residues are only ~1 position in 7, the state decision — the only thing the model is used
    for at design time — is a low-single-digit percentage of the training objective, and the Potts head
    receives no node-level state supervision at all (it is trained solely through NLCPL over edges).

    This module is the first term: the group-renormalised state cross-entropy, summed over titratable
    positions. For the two-state group ``G = {X-P, X-D}`` it is equivalent to a logistic loss on the gap
    ``g = l_P - l_D``, i.e. a contrastive term that rewards separating the two protonation states.

    Because it only reweights the split WITHIN a parent's token group, it leaves the amino-acid marginal
    untouched and cannot trade sequence recovery for protonation accuracy.

    Design decisions:

    - **Two-state.** ``G`` is ``{P, D}`` only. Ambiguous ``-A`` tokens and bare parent tokens are excluded
      automatically, because supervision requires ``S[i]`` to be one of the P/D token indices. This
      matches how the model is scored at inference (``logp_prot - logp_dep``).
    - **Token indices come from the vocabulary**, never hardcoded: ``aa_protonated`` / ``aa_deprotonated``
      are tuple-valued, so v3/v4's ``{HID, HIE}`` neutral-His group marginalises via ``logsumexp`` with no
      special case, while v6's single ``HIS-S`` is the one-member case of the same code path.
    - **Reduction** is the mean over supervised positions (not the node loss's ``/6000`` constant), so the
      weight has a stable meaning regardless of how many titratable residues a batch happens to contain.
      Heads are averaged, so the weight also does not change meaning when a head is added or removed.
    - **float32.** Training runs ``bf16-mixed``; the log-space reductions are done in float32.
    - No class weighting: acid-P is 1-2% of acids, so the acid part is dominated by the ``-D`` side. That
      is a known property of this (plain conditional-CE) form, not an oversight.
    """

    def __init__(
        self,
        extended_vocab: str,
        heads: tuple[str, ...] = ("pot", "dec"),
        eps: float = 1e-8,
    ):
        super().__init__()

        from mpnn.transforms.extended_vocab import get_vocab

        unknown = tuple(h for h in heads if h not in ("pot", "dec"))
        if unknown:
            raise ValueError(f"ProtonationStateLoss: unknown head(s) {unknown}; expected 'pot' / 'dec'.")
        if not heads:
            raise ValueError("ProtonationStateLoss: at least one head is required.")

        vocab = get_vocab(extended_vocab)
        t2i = vocab["token_encoding"].token_to_idx
        prot, deprot = vocab["aa_protonated"], vocab["aa_deprotonated"]

        # res -> (protonated token indices, deprotonated token indices). Tuple-valued so a multi-token
        # state group (v3/v4 neutral His = {HID, HIE}) marginalises rather than picking a member.
        self.groups: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
        for res in _TITRATABLE:
            p_idx = tuple(t2i[t] for t in prot[res] if t in t2i)
            d_idx = tuple(t2i[t] for t in deprot[res] if t in t2i)
            if not p_idx or not d_idx:
                raise ValueError(
                    f"extended_vocab={extended_vocab!r} gives {res} protonated={prot[res]} "
                    f"deprotonated={deprot[res]}, but the token encoding is missing some of them; "
                    f"the loss's vocabulary does not match the model's."
                )
            self.groups[res] = (p_idx, d_idx)

        self.extended_vocab = extended_vocab
        self.heads = tuple(heads)
        self.eps = eps

    @staticmethod
    def _head_log_probs(head: str, network_input, network_output) -> torch.Tensor:
        """Per-position log-probabilities over the vocabulary, for one head. [B, L, V], float32."""
        if head == "dec":
            return network_output["decoder_features"]["log_probs"].float()

        ctx = network_output["potts_context"]
        S = network_input["input_features"]["S"]
        # The Potts head's distribution is log_softmax(-(h_i + sum_j J_ij)) over the candidate axis:
        # field + outgoing + INCOMING couplings, the same conditional energy the neutron benchmark, the
        # pKa derivation and the design engine read.
        #
        # potts_candidate_energies handles ONE structure (it squeezes dim 0 and then unpacks 4 dims), so
        # it is applied per batch element rather than reimplemented for [B, ...]: reusing the audited
        # helper removes any chance of the training-time energy drifting from the deployed convention,
        # and B is a handful of structures under the token budget, so the loop is negligible. The helper
        # is differentiable (see tests/test_potts_energy.py), which is what makes this viable.
        from mpnn.model.pottsmpnn import PottsMPNN

        e_cand = torch.stack([
            PottsMPNN.potts_candidate_energies(
                ctx.etab_out[b : b + 1], ctx.E_idx[b : b + 1], S[b : b + 1]
            )[0]
            for b in range(S.shape[0])
        ])                                                          # [B, L, V]
        return torch.log_softmax(-e_cand.float(), dim=-1)

    def forward(self, network_input, network_output, loss_input):
        input_features = network_input["input_features"]
        if "S" not in input_features:
            raise ValueError("Input features must contain 'S' key.")
        if "mask_for_loss" not in network_output["input_features"]:
            raise ValueError("Network output must contain 'mask_for_loss' key in 'input_features'.")
        if "pot" in self.heads and "potts_context" not in network_output:
            raise ValueError("Network output must contain 'potts_context' for the 'pot' head.")

        S = input_features["S"]
        mask_for_loss = network_output["input_features"]["mask_for_loss"].bool()

        loss_dict: dict = {}
        head_terms = []

        # Deliberately BRANCH-FREE and free of host syncs: every head and every residue type contributes a
        # term on every rank, with an empty selection giving an exact 0 via a clamped denominator instead
        # of being skipped. Training runs 4-GPU DDP, so a data-dependent `if n == 0: continue` would let
        # ranks build different backward graphs whenever one of them happens to draw a batch with no HIS
        # (or no titratable residue at all) — a gradient-sync mismatch, not merely a slow path.
        for head in self.heads:
            log_probs = self._head_log_probs(head, network_input, network_output)
            head_sum = log_probs.new_zeros(())
            head_n = log_probs.new_zeros(())

            for res, (p_idx, d_idx) in self.groups.items():
                p_t = torch.as_tensor(p_idx, device=S.device)
                d_t = torch.as_tensor(d_idx, device=S.device)

                is_p = torch.isin(S, p_t)
                # Supervised iff the true token is one of this residue's P/D tokens: excludes the
                # ambiguous -A tokens, bare parent tokens and padding, with no extra bookkeeping.
                sel = (mask_for_loss & (is_p | torch.isin(S, d_t))).to(log_probs.dtype)

                lp_p = torch.logsumexp(log_probs[..., p_t], dim=-1)     # [B, L] log p(protonated group)
                lp_d = torch.logsumexp(log_probs[..., d_t], dim=-1)     # [B, L] log p(deprotonated group)
                denom = torch.logaddexp(lp_p, lp_d)                     # log p(P or D)
                per_pos = -(torch.where(is_p, lp_p, lp_d) - denom)      # [B, L], >= 0

                res_sum = (per_pos * sel).sum()
                res_n = sel.sum()
                head_sum = head_sum + res_sum
                head_n = head_n + res_n

                loss_dict[f"state_nll_{head}_{res}"] = (res_sum / res_n.clamp_min(1.0)).detach()
                loss_dict[f"state_n_supervised_{res}"] = res_n.detach()

            head_mean = head_sum / head_n.clamp_min(1.0)
            head_terms.append(head_mean)
            loss_dict[f"state_nll_{head}"] = head_mean.detach()
            loss_dict["state_n_supervised"] = head_n.detach()   # identical for every head

        # Mean over heads, so the weight keeps its meaning when a head is added or removed. With no
        # supervised position anywhere this is an exact 0.0 that still carries a graph.
        state_loss = torch.stack(head_terms).mean()
        loss_dict["state_loss_agg"] = state_loss.detach()
        return state_loss, loss_dict


class PottsJointLoss(nn.Module):
    """
    Joint ProteinMPNN + PottsMPNN loss.

    This is a thin wrapper that:
    - computes the standard label-smoothed node loss from decoder log-probs
    - computes the Potts edge NLCPL loss from `network_output["potts_context"]`
    - optionally computes the protonation-state auxiliary loss (``ProtonationStateLoss``)
    - returns the weighted sum of the terms

    ``state_loss_weight`` defaults to 0.0, in which case the state term is not constructed at all and the
    total is bit-for-bit what it was before the term existed.
    """

    def __init__(
        self,
        node_loss_weight: float = 1.0,
        potts_loss_weight: float = 1.0,
        label_smoothing_eps: float = 0.1,
        normalization_constant: float = 6000.0,
        eps: float = 1e-8,
        state_loss_weight: float = 0.0,
        state_loss_heads: tuple[str, ...] = ("pot", "dec"),
        extended_vocab: str | None = None,
    ):
        super().__init__()

        self.node_loss_weight = node_loss_weight
        self.potts_loss_weight = potts_loss_weight
        self.state_loss_weight = state_loss_weight

        self.node_loss = LabelSmoothedNLLLoss(
            label_smoothing_eps=label_smoothing_eps,
            normalization_constant=normalization_constant,
        )
        self.potts_loss = PottsNLCPLLoss(eps=eps)

        # Only built when actually requested: with weight 0 the total stays bit-for-bit identical to the
        # pre-existing loss, which is the regression guard for this change.
        self.state_loss = None
        if state_loss_weight:
            if not extended_vocab:
                raise ValueError(
                    "state_loss_weight > 0 requires extended_vocab (the protonation vocabulary naming "
                    "the P/D token groups); got None."
                )
            self.state_loss = ProtonationStateLoss(
                extended_vocab=extended_vocab,
                heads=tuple(state_loss_heads),
                eps=eps,
            )

    def forward(self, network_input, network_output, loss_input):
        node_loss_value, node_loss_dict = self.node_loss(
            network_input=network_input,
            network_output=network_output,
            loss_input=loss_input,
        )
        potts_loss_value, potts_loss_dict = self.potts_loss(
            network_input=network_input,
            network_output=network_output,
            loss_input=loss_input,
        )

        total_loss = (
            self.node_loss_weight * node_loss_value
            + self.potts_loss_weight * potts_loss_value
        )

        state_loss_dict: dict = {}
        if self.state_loss is not None:
            state_loss_value, state_loss_dict = self.state_loss(
                network_input=network_input,
                network_output=network_output,
                loss_input=loss_input,
            )
            total_loss = total_loss + self.state_loss_weight * state_loss_value

        loss_dict = {
            **node_loss_dict,
            **potts_loss_dict,
            **state_loss_dict,
            "node_loss_weight": torch.tensor(self.node_loss_weight),
            "potts_loss_weight": torch.tensor(self.potts_loss_weight),
            "state_loss_weight": torch.tensor(self.state_loss_weight),
            "total_loss": total_loss.detach(),
        }

        return total_loss, loss_dict
