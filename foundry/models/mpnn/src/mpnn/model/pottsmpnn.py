import torch
import torch.nn as nn
import torch.nn.functional as F
from atomworks.constants import UNKNOWN_AA
from mpnn.model.layers.graph_embeddings import (
    ProteinFeatures,
    ProteinFeaturesLigand,
    ProteinFeaturesMembrane,
    ProteinFeaturesPSSM,
)
from mpnn.model.layers.message_passing import (
    DecLayer,
    EncLayer,
    cat_neighbors_nodes,
    gather_nodes,
)
from mpnn.utils.probability import sample_bernoulli_rv
from mpnn.model.mpnn import ProteinMPNN
from dataclasses import dataclass

@dataclass
class PottsOutput:
    """Structured output from a PottsMPNN forward pass."""
    S_sampled: torch.Tensor   # [N, L]  autoregressive samples
    S_argmax:  torch.Tensor   # [1, L]  greedy argmax sequence
    etab_out:  torch.Tensor   # [1, L, K, V, V]  pair-energy tables
    E_idx:     torch.Tensor   # [1, L, K]  neighbour indices
    log_probs: torch.Tensor | None = None   # [*, L, V]  decoder log-probabilities (decode pass)


@dataclass
class PottsContext:
    etab_out: torch.Tensor      # [B, L, K, V, V]
    E_mask: torch.Tensor        # [B, L, K]
    E_idx: torch.Tensor         # [B, L, K]
    potts_loss_mask: torch.Tensor
    reverse_k: torch.Tensor | None = None
    has_reverse: torch.Tensor | None = None
    

FIELD_SOURCES = ("self_edge", "node")
ETAB_SOURCES = ("edge", "node_edge_node")


def get_pottshead_input(
    h_V: torch.Tensor,
    h_E: torch.Tensor,
    E_idx: torch.Tensor,
    etab_source: str = "edge",
) -> torch.Tensor:
    """Build the coupling head's input tensor for every directed edge i->k.

    - ``"edge"`` (default): ``h_E`` itself, ``[B, L, K, H]``. The coupling J_ij is a
      function of the edge embedding alone.
    - ``"node_edge_node"``: ``concat(h_V[i], h_E[i,k], h_V[j])``, ``[B, L, K, 3H]``, where
      ``j = E_idx[i, k]``. After the encoder's message passing, ``h_V`` carries each
      residue's aggregated structural neighbourhood; this hands that context to the head
      instead of making it reconstruct it from the 128-d edge vector.

    The ``[node_i, edge_ij, node_j]`` ordering matches the rest of the network: ``EncLayer``
    builds the same 3H tensor for its node and edge updates (message_passing.py), as does
    ``hbond_head.build_edge_features``.

    Keep the branch *inside* this function. Callers that re-implement the head (e.g.
    ``scripts/diag_potts_asymmetry.py``) must build the identical input, and duplicating the
    branch at each call site is how they silently drift out of sync.
    """
    if etab_source == "edge":
        return h_E
    if etab_source != "node_edge_node":
        raise ValueError(
            f"Invalid etab_source: {etab_source!r}. Expected one of {ETAB_SOURCES}."
        )
    B, L, K, H = h_E.shape
    return torch.cat(
        [
            h_V.unsqueeze(2).expand(B, L, K, H),  # node_i  (focal)
            h_E,                                  # edge_ij
            gather_nodes(h_V, E_idx),             # node_j  (neighbour)
        ],
        dim=-1,
    )  # [B, L, K, 3H]


def _mlp(in_dim: int, out_dim: int, hidden: list[int] | None) -> nn.Module:
    """``hidden=None``/``[]`` -> a bare ``nn.Linear``. Otherwise ``in -> hidden... -> out`` as an
    ``nn.Sequential``, e.g. ``_mlp(128, 1024, [256, 256])`` is
    ``Linear(128,256) GELU Linear(256,256) GELU Linear(256,1024)``. GELU matches EncLayer/DecLayer.

    The bare-Linear case is NOT cosmetic: wrapping a single Linear in a Sequential renames the
    state_dict keys (``etab_out.weight`` -> ``etab_out.0.weight``), which would stop every existing
    checkpoint from loading with ``strict=True``.
    """
    if not hidden:
        return nn.Linear(in_dim, out_dim)

    layers = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.GELU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class PottsMPNN(ProteinMPNN):
    """PottsMPNN.

    Hamiltonian: H(s) = sum_i h_i(s_i) + sum_{i,k>=1} J_ik(s_i, s_{E_idx[i,k]}).

    The single-body field h_i(a) always lives on the diagonal of the self-edge
    (neighbour slot k=0, where E_idx[i,0] == i), so every consumer reads it the
    same way: ``etab[:, :, 0].diagonal(...)``. ``field_source`` selects how that
    diagonal is produced:

    - "self_edge" (default): the self-edge embedding h_E[:, :, 0] is pushed
      through the pairwise ``etab_out`` head and the off-diagonal is zeroed, so
      the field is a by-product of the coupling head.
    - "node": a dedicated single-body head maps the node embedding h_V to a
      length-V field, which replaces the self-edge diagonal entirely. The field
      then has its own capacity, independent of the coupling head.

    ``etab_source`` selects what the pairwise coupling head sees (see
    ``get_pottshead_input``):

    - "edge" (default): the edge embedding h_E[i,k] alone.
    - "node_edge_node": concat(h_V[i], h_E[i,k], h_V[j]), matching the 3H convention
      EncLayer already uses internally.

    Note the two flags are not fully orthogonal. With field_source="self_edge" the field
    lives on the self-edge (k=0, where E_idx[i,0] == i), so under "node_edge_node" the
    concat there is [h_V_i, h_E_ii, h_V_i] and the field diagonal gains h_V as an input
    too. With field_source="node" the diagonal is overwritten by node_field regardless, so
    there is no interaction.
    """

    def __init__(self, vocab_size=None, field_source="self_edge", etab_source="edge",
                 etab_hidden=None, field_hidden=None, **kwargs):
        super().__init__(**kwargs)
        self.potts_vocab_size = self.vocab_size if vocab_size is None else vocab_size
        if field_source not in FIELD_SOURCES:
            raise ValueError(
                f"Invalid field_source: {field_source!r}. Expected one of {FIELD_SOURCES}."
            )
        if etab_source not in ETAB_SOURCES:
            raise ValueError(
                f"Invalid etab_source: {etab_source!r}. Expected one of {ETAB_SOURCES}."
            )
        self.field_source = field_source
        self.etab_source = etab_source
        self.etab_hidden = etab_hidden
        self.field_hidden = field_hidden

        # Pairwise couplings: head input -> [hidden...] -> V*V. The input is h_E ("edge", H)
        # or concat(h_V[i], h_E, h_V[j]) ("node_edge_node", 3H) -- see get_pottshead_input.
        etab_in = self.hidden_dim * (3 if self.etab_source == "node_edge_node" else 1)
        self.etab_out = _mlp(etab_in, self.potts_vocab_size**2, etab_hidden)
        if self.field_source == "node":
            # Dedicated single-body head: node embedding -> [hidden...] -> V.
            self.node_field = _mlp(self.hidden_dim, self.potts_vocab_size, field_hidden)

    @staticmethod
    def infer_etab_source(state_dict: dict, hidden_dim: int = 128) -> str:
        """Read ``etab_source`` off a checkpoint's ``etab_out`` weight shape.

        No checkpoint in this repo stores architecture hparams, and every loader does a
        ``strict=True`` ``load_state_dict``, so callers otherwise have to remember which flags a
        checkpoint was trained with. The head's first Linear gives it away: its in-features are
        ``H`` for "edge" and ``3H`` for "node_edge_node".

        Handles both key spellings -- a bare ``nn.Linear`` head is ``etab_out.weight``, an MLP head
        (``etab_hidden`` set) is ``etab_out.0.weight`` (see ``_mlp``).
        """
        for key in ("etab_out.weight", "etab_out.0.weight"):
            if key in state_dict:
                in_dim = state_dict[key].shape[1]
                break
        else:
            raise KeyError(
                "Checkpoint has no 'etab_out.weight' / 'etab_out.0.weight'; not a PottsMPNN "
                f"state_dict? Found keys: {sorted(state_dict)[:8]}..."
            )

        mult, remainder = divmod(in_dim, hidden_dim)
        if remainder or mult not in (1, 3):
            raise ValueError(
                f"etab_out in-features {in_dim} is not 1x or 3x hidden_dim={hidden_dim}; "
                "cannot infer etab_source."
            )
        return "node_edge_node" if mult == 3 else "edge"

    def compute_edge_mask(self, input_features, graph_features):
        residue_mask = input_features["residue_mask"]
        E_idx = graph_features["E_idx"]

        neighbor_mask = gather_nodes(residue_mask.unsqueeze(-1), E_idx).squeeze(-1)
        E_mask = residue_mask.unsqueeze(-1) & neighbor_mask
        return E_mask

    def compute_potts_loss_mask(self, input_features, graph_features):
        mask_for_loss = input_features["mask_for_loss"]
        E_idx = graph_features["E_idx"]

        neighbor_loss_mask = gather_nodes(
            mask_for_loss.unsqueeze(-1),
            E_idx,
        ).squeeze(-1)

        potts_loss_mask = mask_for_loss.unsqueeze(-1) & neighbor_loss_mask
        return potts_loss_mask

    def compute_potts_context(self, input_features, graph_features, encoder_features):
        h_E = encoder_features["h_E"]
        B, L, K, H = h_E.shape

        E_idx = graph_features["E_idx"]
        E_mask = self.compute_edge_mask(input_features,graph_features)

        head_in = get_pottshead_input(
            encoder_features["h_V"], h_E, E_idx, self.etab_source
        )                                            # [B, L, K, H] or [B, L, K, 3H]
        etab_out = self.etab_out(head_in)            # [B, L, K, Vp x Vp]
        etab_out = etab_out * E_mask.unsqueeze(-1).to(dtype=etab_out.dtype)
        etab_out = etab_out.view(B,L,K,self.potts_vocab_size, self.potts_vocab_size)  # [B, L, K, Vp, Vp]

        # The self-edge (k=0) carries the single-body field h_i(a) on its
        # diagonal; the off-diagonal is always zero. Only the source of that
        # diagonal differs between field_source modes. A diagonal slot-0 is
        # invariant under the reciprocal merge below (its reciprocal edge is
        # itself, and a diagonal matrix equals its own transpose).
        if self.field_source == "node":
            # h_i(a) from a dedicated linear map on the node embedding.
            field = self.node_field(encoder_features["h_V"])  # [B, L, Vp]
            field = field * E_mask[:, :, 0:1].to(dtype=field.dtype)
            etab_out[:, :, 0, :, :] = torch.diag_embed(field)
        else:
            # h_i(a) from the diagonal of the self-edge's pairwise table.
            eye = torch.eye(
                self.potts_vocab_size,
                device=etab_out.device,
                dtype=etab_out.dtype,
            )
            etab_out[:, :, 0, :, :] = etab_out[:, :, 0, :, :] * eye

        # Averaging coupled positiosn
        batch_idx = torch.arange(B, device=etab_out.device)[:, None, None].expand(B, L, K)
        src_idx = torch.arange(L, device=etab_out.device)[None, :, None].expand(B, L, K)

        # neighbor_lists[b, i, k, :] = E_idx[b, j, :] where j = E_idx[b, i, k]
        neighbor_lists = E_idx[batch_idx, E_idx, :]                     # [B, L, K, K]

        # reverse_matches[b, i, k, r] is True if edge slot r in residue j points back to i
        reverse_matches = neighbor_lists == src_idx.unsqueeze(-1)       # [B, L, K, K]
        has_reverse = reverse_matches.any(dim=-1)                       # [B, L, K]
        reverse_k = reverse_matches.long().argmax(dim=-1)               # [B, L, K]

        reverse_etab = etab_out[batch_idx, E_idx, reverse_k]            # [B, L, K, V, V]
        merged_etab = 0.5 * (etab_out + reverse_etab.transpose(-1, -2))

        if E_mask is not None:
            reverse_mask = E_mask[batch_idx, E_idx, reverse_k]
            valid_merge = has_reverse & E_mask.bool() & reverse_mask.bool()
        else:
            valid_merge = has_reverse

        etab_out = torch.where(
            valid_merge[..., None, None],
            merged_etab,
            etab_out,
        )
        potts_loss_mask = self.compute_potts_loss_mask(input_features, graph_features)
        context = PottsContext(
            etab_out=etab_out,
            E_idx=E_idx,
            E_mask=E_mask,
            reverse_k=reverse_k,
            has_reverse=has_reverse,
            potts_loss_mask=potts_loss_mask
        )

        return context

    @torch.no_grad()
    def run_potts_encoder(
        self, input_features: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encoder-only forward pass returning Potts energy tables. Skips decoder.

        Returns:
            etab_out: [B, L, K, V, V] structure-conditioned pair-interaction energy tables.
            E_idx:    [B, L, K] neighbor index tensor.
        """
        self.sample_and_construct_masks(input_features)
        graph_features = self.graph_featurization(input_features)
        encoder_features = self.encode(input_features, graph_features)
        ctx = self.compute_potts_context(input_features, graph_features, encoder_features)
        return ctx.etab_out, ctx.E_idx

    @staticmethod
    def calc_potts_eners(
        etab_out: torch.Tensor,  # [1, L, K, V, V]
        E_idx: torch.Tensor,     # [1, L, K]
        seqs_int: torch.Tensor,  # [N, L]
    ) -> torch.Tensor:
        """ Given Encoded graph and calcualted PottsHead return Potts Hamiltonian H(s) = Σ_i Σ_k etab[i,k,s_i,s_neighbor]. Returns [N]."""
        etab = etab_out.squeeze(0)   # [L, K, V, V]
        eidx = E_idx.squeeze(0)      # [L, K]
        N, L = seqs_int.shape
        K = eidx.shape[-1]
        E_aa_j = seqs_int[:, eidx]                           # [N, L, K] neighbor aa
        s_i = seqs_int.unsqueeze(-1).expand(N, L, K)         # [N, L, K] focal aa
        L_idx = torch.arange(L, device=etab.device)[None, :, None].expand(N, L, K)
        K_idx = torch.arange(K, device=etab.device)[None, None, :].expand(N, L, K)
        return etab[L_idx, K_idx, s_i, E_aa_j].sum(dim=(-1, -2))  # [N]

    @staticmethod
    def _directed_pair_edges(E_idx: torch.Tensor):
        """Flatten the non-self directed edges (slots k>=1).

        Returns ``(src, slot, tgt)`` 1-D long tensors enumerating every directed
        edge i->j with ``j = E_idx[i, k]`` for ``k >= 1`` and ``j != i``. These
        are the edges whose pair table depends jointly on ``s_i`` and ``s_j``;
        the self-edge (k=0) carries the single-site field and is handled
        separately. This is pure index bookkeeping on the model's own ``E_idx``
        — it does not rebuild the kNN graph.
        """
        eidx = E_idx.squeeze(0)                                   # [L, K]
        L, K = eidx.shape
        device = eidx.device
        src = torch.arange(L, device=device)[:, None].expand(L, K - 1).reshape(-1)
        slot = torch.arange(1, K, device=device)[None, :].expand(L, K - 1).reshape(-1)
        tgt = eidx[:, 1:].reshape(-1)
        keep = tgt != src
        return src[keep], slot[keep], tgt[keep]

    @staticmethod
    def _incoming_adjacency(E_idx: torch.Tensor):
        """Padded reverse adjacency for incoming non-self edges (graph transpose).

        For every position ``p`` this enumerates the ``(m, k)`` source/slot pairs
        with ``E_idx[m, k] == p`` (``k >= 1``) — the edges in which ``p`` appears
        as the *neighbour*. ``calc_potts_eners`` sums every directed edge, so
        ``p``'s identity also enters ``H`` through these incoming edges; the
        per-position conditional must add them (not just ``p``'s outgoing edges)
        to match the Hamiltonian's finite difference. The kNN graph is asymmetric
        (measured ~16% non-reciprocal edges on 9NNF), so ``reverse_k`` alone —
        which only covers reciprocal edges — is not sufficient; this inverts the
        full ``E_idx``.

        Returns ``(in_src, in_slot, in_mask)``, each ``[L, D]`` with
        ``D = max in-degree`` (zero-padded; ``in_mask`` flags valid entries).
        """
        src, slot, tgt = PottsMPNN._directed_pair_edges(E_idx)
        L = E_idx.shape[1]
        device = E_idx.device
        indeg = torch.zeros(L, dtype=torch.long, device=device)
        indeg.index_add_(0, tgt, torch.ones_like(tgt))
        D = int(indeg.max().item()) if indeg.numel() else 0
        in_src = torch.zeros(L, D, dtype=torch.long, device=device)
        in_slot = torch.zeros(L, D, dtype=torch.long, device=device)
        in_mask = torch.zeros(L, D, dtype=torch.bool, device=device)
        if D > 0 and tgt.numel():
            # Sort edges by target so each target's edges are contiguous, then
            # assign a within-group slot index 0..indeg[t]-1.
            order = torch.argsort(tgt, stable=True)
            tgt_s, src_s, slot_s = tgt[order], src[order], slot[order]
            group_start = torch.zeros(L, dtype=torch.long, device=device)
            group_start[1:] = torch.cumsum(indeg, 0)[:-1]
            within = torch.arange(tgt_s.numel(), device=device) - group_start[tgt_s]
            in_src[tgt_s, within] = src_s
            in_slot[tgt_s, within] = slot_s
            in_mask[tgt_s, within] = True
        return in_src, in_slot, in_mask

    @staticmethod
    def potts_candidate_energies(
        etab_out: torch.Tensor,   # [1, L, K, V, V]
        E_idx: torch.Tensor,      # [1, L, K]
        seqs_int: torch.Tensor,   # [N, L]
    ) -> torch.Tensor:
        """Per-position conditional energies consistent with ``calc_potts_eners``.

        Returns ``E`` with shape ``[N, L, V]`` such that for every sequence ``n``,
        position ``p`` and candidate amino acid ``a``::

            calc_potts_eners(s with s_p := a) - calc_potts_eners(s)
                == E[n, p, a] - E[n, p, s_p]

        ``E[n, p, :]`` gathers every Hamiltonian term that depends on ``s_p``:
        the single-site field, the outgoing pair couplings ``p->j``, and the
        incoming pair couplings ``m->p`` (the graph transpose; required because
        the kNN graph is asymmetric). Pure inference helper — it does not change
        the (double-counted) Hamiltonian convention.
        """
        etab = etab_out.squeeze(0)                               # [L, K, V, V]
        eidx = E_idx.squeeze(0)                                  # [L, K]
        L, K, V, _ = etab.shape
        N = seqs_int.shape[0]
        device = etab.device

        self_term = etab[:, 0].diagonal(dim1=-2, dim2=-1)        # [L, V]
        nbr_idx = eidx[:, 1:]                                    # [L, K-1]
        pe = etab[:, 1:]                                         # [L, K-1, V, V]
        src, slot, tgt = PottsMPNN._directed_pair_edges(E_idx)   # [E] each

        out = torch.empty(N, L, V, device=device, dtype=etab.dtype)
        for n in range(N):
            s = seqs_int[n].to(device)
            # Outgoing: out[p, a] = sum_k pe[p, k, a, s_{nbr(p,k)}].
            s_nbr = s[nbr_idx]                                   # [L, K-1]
            gi = s_nbr.unsqueeze(-1).unsqueeze(-1).expand(L, K - 1, V, 1)
            outgoing = torch.gather(pe, -1, gi).squeeze(-1).sum(1)   # [L, V]
            # Incoming (transpose): scatter etab[m, k, s_m, :] into target p.
            s_m = s[src]                                         # [E]
            edge_vals = etab[src, slot, s_m, :]                  # [E, V]
            incoming = torch.zeros(L, V, device=device, dtype=etab.dtype)
            incoming.index_add_(0, tgt, edge_vals)
            out[n] = self_term + outgoing + incoming
        return out

    @staticmethod
    def potts_mutation_delta(
        etab_out: torch.Tensor,
        E_idx: torch.Tensor,
        seqs_int: torch.Tensor,
    ) -> torch.Tensor:
        """Exact ΔH for substituting each position with each candidate AA.

        ``delta[n, p, a] = calc_potts_eners(s with s_p := a) - calc_potts_eners(s)``,
        shape ``[N, L, V]``. Derived from :meth:`potts_candidate_energies`, so it
        is guaranteed consistent with the reported Hamiltonian.
        """
        cand = PottsMPNN.potts_candidate_energies(etab_out, E_idx, seqs_int)  # [N,L,V]
        cur = cand.gather(-1, seqs_int.to(cand.device).unsqueeze(-1))         # [N,L,1]
        return cand - cur

    @staticmethod
    @torch.no_grad()
    def potts_gibbs_optimize(
        etab_out: torch.Tensor,
        E_idx: torch.Tensor,
        seq_init: torch.Tensor,
        free_mask: torch.Tensor,
        temperature: float = 0.01,
        max_iters: int = 1000,
        convergence_mode: bool = True,
        valid_aa_mask: torch.Tensor | None = None,
        track_trajectory: bool = False,
        track_per_mutation: bool = False,
    ) -> tuple:
        """Gibbs sweep over free positions to minimize the Potts Hamiltonian.

        Each sweep visits every free position exactly once in a random order.
        At each position we compute the conditional energy E_i(a) for all V
        amino acids given the current sequence at every other position, then
        immediately sample a new amino acid and update the sequence in-place
        before moving to the next position. The update from position i is
        therefore visible when position j is visited later in the same sweep.

        Args:
            etab_out:         [1, L, K, V, V] structure-conditioned energy tables.
            E_idx:            [1, L, K] neighbour index tensor.
            seq_init:         [N, L] integer-encoded initial sequences.
            free_mask:        [L] bool — True means the position may be mutated.
            temperature:      Boltzmann temperature. As T->0 becomes greedy argmin.
            max_iters:        Maximum number of sweeps per sequence.
            convergence_mode: Stop early when a full sweep produces zero mutations.
            valid_aa_mask:    [V] bool — True marks amino acids that may be proposed.

        Returns:
            seq_opt:  [N, L] optimised sequences.
            energies: [N] final Potts Hamiltonian H(s) for each sequence.
        """
        N, L = seq_init.shape
        V = etab_out.shape[-1]
        device = etab_out.device

        seqs = seq_init.clone().to(device)

        # Remove batch dim once — cheaper indexing on [L, K, V, V]
        etab = etab_out.squeeze(0)   # [L, K, V, V]
        eidx = E_idx.squeeze(0)      # [L, K]

        # Incoming (transpose) adjacency, built once: for each position the
        # edges m->pos that list pos as a neighbour. Needed so the conditional
        # matches calc_potts_eners (the kNN graph is asymmetric — see
        # _incoming_adjacency).
        in_src, in_slot, in_mask = PottsMPNN._incoming_adjacency(E_idx)

        free_positions = free_mask.nonzero(as_tuple=False).squeeze(1)  # [F]

        trajectory = [] if track_trajectory else None
        mut_log    = [] if track_per_mutation else None

        for n in range(N):
            for _iter in range(max_iters):
                mutations = 0

                # New random visitation order each sweep
                perm = torch.randperm(len(free_positions), device=device)
                order = free_positions[perm]

                for pos_t in order:
                    pos = pos_t.item()

                    # Self-field h_i(a): diagonal of self-edge (k=0).
                    # Off-diagonals are zeroed in compute_potts_context so
                    # diagonal() cleanly extracts h_i(a) for all a.
                    h_i = etab[pos, 0].diagonal()           # [V]

                    # Outgoing pair couplings i->j for neighbours k=1..K-1.
                    pair_etab = etab[pos, 1:]               # [K-1, V, V]
                    neighbor_pos = eidx[pos, 1:]            # [K-1]

                    # Current AA at each neighbour — reflects mutations already
                    # made earlier in this sweep (immediate update convention).
                    s_j = seqs[n, neighbor_pos]             # [K-1]

                    # For each candidate a: sum_k J_{i,j(k)}(a, s_{j(k)})
                    # Gather the column s_j[k] from pair_etab[k], sum over k.
                    s_j_idx = s_j.view(-1, 1, 1).expand(-1, V, 1)  # [K-1, V, 1]
                    out_nrg = pair_etab.gather(2, s_j_idx).squeeze(2).sum(0)  # [V]

                    # Incoming pair couplings m->i: positions m that list i as a
                    # neighbour. calc_potts_eners() sums every directed edge, so
                    # i's identity also enters through these reverse edges; the
                    # conditional must include them to match the Hamiltonian.
                    m = in_src[pos]                         # [D]
                    ksl = in_slot[pos]                      # [D]
                    msk = in_mask[pos]                      # [D]
                    s_m = seqs[n, m]                        # [D]
                    inc_nrg = (etab[m, ksl, s_m, :] * msk.unsqueeze(-1)).sum(0)  # [V]

                    # Conditional energy for every candidate AA at position i,
                    # consistent with H(s) = calc_potts_eners(s).
                    E_i = h_i + out_nrg + inc_nrg           # [V]

                    # Mask invalid tokens (e.g. UNK) out of proposals
                    if valid_aa_mask is not None:
                        E_i = E_i.masked_fill(~valid_aa_mask, float("inf"))

                    # Boltzmann sample: P(a) ∝ exp(-E_i(a) / T)
                    probs = F.softmax(-E_i / temperature, dim=-1)
                    new_aa = torch.multinomial(probs, 1).item()

                    old_aa = seqs[n, pos].item()
                    if new_aa != old_aa:
                        mutations += 1
                        if track_per_mutation:
                            mut_log.append({
                                "n":       n,
                                "sweep":   _iter,
                                "pos":     pos,
                                "old":     old_aa,
                                "new":     new_aa,
                                "delta_E": (E_i[new_aa] - E_i[old_aa]).item(),
                            })
                    seqs[n, pos] = new_aa

                if track_trajectory:
                    sweep_e = PottsMPNN.calc_potts_eners(etab_out, E_idx, seqs)
                    trajectory.append((seqs.clone(), sweep_e.clone()))

                if convergence_mode and mutations == 0:
                    break

        energies = PottsMPNN.calc_potts_eners(etab_out, E_idx, seqs)
        if track_trajectory and track_per_mutation:
            return seqs, energies, trajectory, mut_log
        if track_trajectory:
            return seqs, energies, trajectory
        if track_per_mutation:
            return seqs, energies, mut_log
        return seqs, energies

    def forward(self, network_input):
        """
        Forward pass of the ProteinMPNN model.

        A NOTE on shapes:
            - B = batch dimension size
            - L = sequence length (number of residues)
            - K = number of neighbors per residue
            - H = hidden dimension size
            - vocab_size = self.vocab_size
            - num_atoms =
                self.graph_featurization_module.TOKEN_ENCODING.n_atoms_per_token
            - num_backbone_atoms = len(
                self.graph_featurization_module.BACKBONE_ATOM_NAMES
            )
            - num_virtual_atoms = len(
                self.graph_featurization_module.DATA_TO_CALCULATE_VIRTUAL_ATOMS
            )
            - num_rep_atoms = len(
                self.graph_featurization_module.REPRESENTATIVE_ATOM_NAMES
            )
            - num_edge_output_features =
                self.graph_featurization_module.num_edge_output_features
            - num_node_output_features =
                self.graph_featurization_module.num_node_output_features

        Args:
            network_input (dict): Dictionary containing the input to the
                network.
                - input_features (dict): dictionary containing input features
                    and all necessary information for the model to run.
                    - X (torch.Tensor): [B, L, num_atoms, 3] - 3D coordinates of
                        polymer atoms.
                    - X_m (torch.Tensor): [B, L, num_atoms] - Mask indicating
                        which polymer atoms are valid.
                    - S (torch.Tensor): [B, L] - Sequence of the polymer
                        residues.
                    - R_idx (torch.Tensor): [B, L] - indices of the residues.
                    - chain_labels (torch.Tensor): [B, L] - chain labels for
                        each residue.
                    - residue_mask (torch.Tensor): [B, L] - Mask indicating
                        which residues are valid.
                    - designed_residue_mask (torch.Tensor): [B, L] - mask for
                        the designed residues.
                    - symmetry_equivalence_group (torch.Tensor, optional):
                        [B, L] - an integer for every residue, indicating the
                        symmetry group that it belongs to. If None, the
                        residues are not grouped by symmetry. For example, if
                        residue i and j should be decoded symmetrically, then
                        symmetry_equivalence_group[i] ==
                        symmetry_equivalence_group[j]. Must be torch.int64 to
                        allow for use as an index. These values should range
                        from 0 to the maximum number of symmetry groups - 1 for
                        each example. NOTE: bias, pair_bias, and temperature
                        should be the same for all residues in the symmetry
                        equivalence group; otherwise, the intended behavior may
                        not be achieved. The residues within a symmetry group
                        should all have the same validity and design/fixed
                        status.
                    - symmetry_weight (torch.Tensor, optional): [B, L] - the
                        weights for each residue, to be used when aggregating
                        across its respective symmetry group. If None, the
                        weights are assumed to be 1.0 for all residues.
                    - bias (torch.Tensor, optional): [B, L, 21] - the
                        per-residue bias to use for sampling. If None, the code
                        will implicitly use a bias of 0.0 for all residues.
                    - pair_bias (torch.Tensor, optional): [B, L, 21, L, 21] -
                        the per-residue pair bias to use for sampling. If None,
                        the code will implicitly use a pair bias of 0.0 for all
                        residue pairs.
                    - temperature (torch.Tensor, optional): [B, L] - the
                        per-residue temperature to use for sampling. If None,
                        the code will implicitly use a temperature of 1.0.
                    - structure_noise (float): Standard deviation of the
                        Gaussian noise to add to the input coordinates, in
                        Angstroms.
                    - decode_type (str): the type of decoding to use.
                        - "teacher_forcing": Use teacher forcing for the
                            decoder, where the decoder attends to the ground
                            truth sequence S for all previously decoded
                            residues.
                        - "auto_regressive": Use auto-regressive decoding,
                            where the decoder attends to the sequence and
                            decoder representation of residues that have
                            already been decoded (using the predicted sequence).
                    - causality_pattern (str): The pattern of causality to use
                        for the decoder. For all causality patterns, the
                        decoding order is randomized.
                        - "auto_regressive": Use an auto-regressive causality
                            pattern, where residues can attend to the sequence
                            and decoder representation of residues that have
                            already been decoded (NOTE: as mentioned above,
                            this will be randomized).
                        - "unconditional": Residues cannot attend to the
                            sequence or decoder representation of any other
                            residues.
                        - "conditional": Residues can attend to the sequence
                            and decoder representation of all other residues.
                        - "conditional_minus_self": Residues can attend to the
                            sequence and decoder representation of all other
                            residues, except for themselves (as destination
                            nodes).
                    - initialize_sequence_embedding_with_ground_truth (bool):
                        - True: Initialize the sequence embedding with the
                            ground truth sequence S.
                            - If doing auto-regressive decoding, also
                                initialize S_sampled with the ground truth
                                sequence S, which should only affect the
                                application of pair bias.
                        - False: Initialize the sequence embedding with zeros.
                            - If doing auto-regressive decoding, initialize
                                S_sampled with unknown residues.
                    - features_to_return (dict, optional): dictionary
                        determining which features to return from the model. If
                        None, return all features (including modified input
                        features, graph features, encoder features, and decoder
                        features). Otherwise, expects a dictionary with the
                        following key, value pairs:
                        - "input_features": list - the input features to return.
                        - "graph_features": list - the graph features to return.
                        - "encoder_features": list - the encoder features to
                            return.
                        - "decoder_features": list - the decoder features to
                            return.
                    - repeat_sample_num (int, optional): Number of times to
                        repeat the samples along the batch dimension. If None,
                        no repetition is performed. If greater than 1, the
                        samples are repeated along the batch dimension. If
                        greater than 1, B must be 1, since repeating samples
                        along the batch dimension is not supported when more
                        than one sample is provided in the batch.
        Side Effects:
            Any changes denoted below to input_features are also mutated on the
            original input features.
        Returns:
            network_output (dict): Output dictionary containing the requested
                features based on the input features' "features_to_return" key.
                - input_features (dict): The input features from above, with
                    the following keys added or modified:
                    - mask_for_loss (torch.Tensor): [B, L] - mask for loss,
                        where True is a residue that is included in the loss
                        calculation, and False is a residue that is not
                        included in the loss calculation.
                    - X (torch.Tensor): [B, L, num_atoms, 3] - 3D coordinates of
                        polymer atoms with added Gaussian noise.
                    - X_pre_noise (torch.Tensor): [B, L, num_atoms, 3] -
                        3D coordinates of polymer atoms before adding Gaussian
                        noise ('X' before noise).
                    - X_backbone (torch.Tensor): [B, L, num_backbone_atoms, 3] -
                        3D coordinates of the backbone atoms for each residue,
                        built from the noisy 'X' coordinates.
                    - X_m_backbone (torch.Tensor): [B, L, num_backbone_atoms] -
                        mask indicating which backbone atoms are valid.
                    - X_virtual_atoms (torch.Tensor):
                        [B, L, num_virtual_atoms, 3] - 3D coordinates of the
                        virtual atoms for each residue, built from the noisy
                        'X' coordinates.
                    - X_m_virtual_atoms (torch.Tensor):
                        [B, L, num_virtual_atoms] - mask indicating which
                        virtual atoms are valid.
                    - X_rep_atoms (torch.Tensor): [B, L, num_rep_atoms, 3] - 3D
                        coordinates of the representative atoms for each
                        residue, built from the noisy 'X' coordinates.
                    - X_m_rep_atoms (torch.Tensor): [B, L, num_rep_atoms] -
                        mask indicating which representative atoms are valid.
                - graph_features (dict): The graph features.
                    - E_idx (torch.Tensor): [B, L, K] - indices of the top K
                        nearest neighbors for each residue.
                    - E (torch.Tensor): [B, L, K, num_edge_output_features] -
                        Edge features for each pair of neighbors.
                - encoder_features (dict): The encoder features.
                    - h_V (torch.Tensor): [B, L, H] - the protein node features
                        after encoding message passing.
                    - h_E (torch.Tensor): [B, L, K, H] - the protein edge
                        features after encoding message passing.
                - decoder_features (dict): The decoder features.
                    - causal_mask (torch.Tensor): [B, L, K, 1] - the causal
                        mask for the decoder.
                    - anti_causal_mask (torch.Tensor): [B, L, K, 1] - the
                        anti-causal mask for the decoder.
                    - decoding_order (torch.Tensor): [B, L] - the order in
                        which the residues should be decoded.
                    - decode_last_mask (torch.Tensor): [B, L] - mask for
                        residues that should be decoded last, where False is a
                        residue that should be decoded first (invalid or
                        fixed), and True is a residue that should not be
                        decoded first (designed residues).
                    - h_V (torch.Tensor): [B, L, H] - the updated node features
                        for the decoder.
                    - logits (torch.Tensor): [B, L, vocab_size] - the logits
                        for the sequence.
                    - log_probs (torch.Tensor): [B, L, vocab_size] - the log
                        probabilities for the sequence.
                    - probs (torch.Tensor): [B, L, vocab_size] - the
                        probabilities for the sequence.
                    - probs_sample (torch.Tensor): [B, L, vocab_size] -
                        the probabilities for the sequence, with the unknown
                        residues zeroed out and the other residues normalized.
                    - S_sampled (torch.Tensor): [B, L] - the predicted
                        sequence, sampled from the probabilities (unknown
                        residues are not sampled).
                    - S_argmax (torch.Tensor): [B, L] - the predicted sequence,
                        obtained by taking the argmax of the probabilities
                        (unknown residues are not selected).
        """
        input_features = network_input["input_features"]

        # Check that the input features contains the necessary keys.
        if "decode_type" not in input_features:
            raise ValueError("Input features must contain 'decode_type' key.")

        # Setup masks (added to the input features).
        self.sample_and_construct_masks(input_features)

        # Graph featurization (also modifies/adds to input_features).
        graph_features = self.graph_featurization(input_features)

        # Run the encoder.
        encoder_features = self.encode(input_features, graph_features)

        # Calcualte Potts Head J
        potts_context = self.compute_potts_context(input_features=input_features,
                                                graph_features=graph_features,
                                                encoder_features=encoder_features)

        
        # Setup for decoder (repeat features along the batch dimension, modifies
        # input_features, graph_features, and encoder_features).
        self.repeat_along_batch(
            input_features,
            graph_features,
            encoder_features,
        )

        # Set up the causality masks.
        decoder_features = self.setup_causality_masks(input_features, graph_features)

        # Decoder, either teacher forcing or auto-regressive.
        if input_features["decode_type"] == "teacher_forcing":
            self.decode_teacher_forcing(
                input_features, graph_features, encoder_features, decoder_features
            )
        elif input_features["decode_type"] == "auto_regressive":
            self.decode_auto_regressive(
                input_features, graph_features, encoder_features, decoder_features
            )
        else:
            raise ValueError(f"Unknown decode_type: {input_features['decode_type']}.")

        # Create the output dictionary based on the requested features.
        network_output = self.construct_output_dictionary(
            input_features, graph_features, encoder_features, decoder_features
        )
        network_output["potts_context"] = potts_context
        return network_output
