"""Unit tests for the PottsMPNN energy convention.

These validate that the per-position conditional energy and the exact mutation
delta are consistent with ``calc_potts_eners`` (the reported Hamiltonian), which
sums every directed edge so a reciprocal pair is counted from both endpoints.
The kNN graph is asymmetric, so the conditional must include the full incoming
(transpose) adjacency, not just reverse_k reciprocal edges.

Pure unit tests on tiny synthetic tables — no checkpoint, no model forward.
"""

import torch

from mpnn.model.pottsmpnn import PottsMPNN


def _make_graph(L: int, K: int, V: int, seed: int):
    """Synthetic ``(etab_out [1,L,K,V,V], E_idx [1,L,K])``.

    Slot 0 is the self-edge (``E_idx[i,0] == i``, diagonalised to mimic the
    single-site field convention); slots 1..K-1 are distinct other residues
    chosen independently per residue, so the graph is generally asymmetric.
    """
    g = torch.Generator().manual_seed(seed)
    E_idx = torch.zeros(L, K, dtype=torch.long)
    for i in range(L):
        others = torch.tensor([j for j in range(L) if j != i])
        perm = torch.randperm(others.numel(), generator=g)[: K - 1]
        E_idx[i, 0] = i
        E_idx[i, 1:] = others[perm]
    etab = torch.randn(1, L, K, V, V, generator=g)
    etab[0, :, 0] = etab[0, :, 0] * torch.eye(V)  # diagonalise self-edge
    return etab, E_idx.unsqueeze(0)


def test_delta_matches_finite_difference():
    """delta[n,p,a] == calc_potts_eners(s with s_p:=a) - calc_potts_eners(s)."""
    L, K, V = 12, 5, 4
    etab, E_idx = _make_graph(L, K, V, seed=0)
    seqs = torch.randint(0, V, (3, L), generator=torch.Generator().manual_seed(1))

    delta = PottsMPNN.potts_mutation_delta(etab, E_idx, seqs)        # [N,L,V]
    base = PottsMPNN.calc_potts_eners(etab, E_idx, seqs)            # [N]

    for n in range(seqs.shape[0]):
        for p in range(L):
            for a in range(V):
                m = seqs[n].clone()
                m[p] = a
                true = PottsMPNN.calc_potts_eners(etab, E_idx, m[None])[0] - base[n]
                assert torch.allclose(delta[n, p, a], true, atol=1e-4), (
                    n, p, a, delta[n, p, a].item(), true.item()
                )


def test_candidate_energy_self_outgoing_incoming_decomposition():
    """Candidate energies equal calc_potts_eners up to a per-position constant."""
    L, K, V = 10, 4, 5
    etab, E_idx = _make_graph(L, K, V, seed=7)
    seqs = torch.randint(0, V, (2, L), generator=torch.Generator().manual_seed(8))
    cand = PottsMPNN.potts_candidate_energies(etab, E_idx, seqs)     # [N,L,V]
    H = PottsMPNN.calc_potts_eners(etab, E_idx, seqs)               # [N]
    # Energy of the actual identity at each position differs from H only by the
    # terms not touching that position; the per-position spread must match the
    # finite difference, already covered above. Here: H reconstructs from any
    # single position's candidate energy plus the (position-independent) rest.
    for n in range(seqs.shape[0]):
        cur = cand[n].gather(-1, seqs[n].unsqueeze(-1)).squeeze(-1)  # [L]
        # For each position, H - cur[p] must be identical across positions only
        # if no double counting; instead verify via finite difference proxy:
        for p in range(L):
            for a in range(V):
                m = seqs[n].clone(); m[p] = a
                true = PottsMPNN.calc_potts_eners(etab, E_idx, m[None])[0]
                assert torch.allclose(
                    H[n] + (cand[n, p, a] - cur[p]), true, atol=1e-4
                )


def test_incoming_nonreciprocal_edge_is_represented():
    """A non-reciprocal incoming edge must affect the conditional.

    Directed cycle 0->1->2->0: every edge is non-reciprocal (no reverse), so
    reverse_k would find nothing. The transpose adjacency must still attribute
    each edge to its target.
    """
    L, K, V = 3, 2, 3
    E_idx = torch.tensor([[0, 1], [1, 2], [2, 0]]).unsqueeze(0)  # slot0=self
    in_src, in_slot, in_mask = PottsMPNN._incoming_adjacency(E_idx)
    # incoming to 0 is from 2; to 1 from 0; to 2 from 1 (all at slot 1)
    assert in_src.squeeze(-1).tolist() == [2, 0, 1]
    assert in_slot.squeeze(-1).tolist() == [1, 1, 1]
    assert in_mask.all()

    etab = torch.randn(1, L, K, V, V, generator=torch.Generator().manual_seed(2))
    etab[0, :, 0] = etab[0, :, 0] * torch.eye(V)
    seqs = torch.randint(0, V, (1, L), generator=torch.Generator().manual_seed(3))
    delta = PottsMPNN.potts_mutation_delta(etab, E_idx, seqs)
    base = PottsMPNN.calc_potts_eners(etab, E_idx, seqs)[0]
    for p in range(L):
        for a in range(V):
            m = seqs[0].clone(); m[p] = a
            true = PottsMPNN.calc_potts_eners(etab, E_idx, m[None])[0] - base
            assert torch.allclose(delta[0, p, a], true, atol=1e-4)


def test_greedy_gibbs_never_increases_hamiltonian():
    """At T->0 every accepted move minimises the exact conditional, so H falls."""
    L, K, V = 16, 6, 5
    for seed in range(4):
        etab, E_idx = _make_graph(L, K, V, seed=seed)
        seq0 = torch.randint(0, V, (2, L), generator=torch.Generator().manual_seed(100 + seed))
        free = torch.ones(L, dtype=torch.bool)
        valid = torch.ones(V, dtype=torch.bool)
        E0 = PottsMPNN.calc_potts_eners(etab, E_idx, seq0)
        _, E1 = PottsMPNN.potts_gibbs_optimize(
            etab, E_idx, seq_init=seq0, free_mask=free, valid_aa_mask=valid,
            temperature=1e-4, max_iters=100, convergence_mode=True,
        )
        assert (E1 <= E0 + 1e-3).all(), (seed, E0.tolist(), E1.tolist())


def test_candidate_energies_are_differentiable():
    """Gradients must reach ``etab_out`` through the conditional energy.

    The protonation-state auxiliary loss supervises ``log_softmax(-potts_candidate_energies(...))``, so
    this helper being differentiable is a training-time requirement, not just an inference convenience —
    an in-place rewrite or a stray ``no_grad`` here would silently stop that term from learning.
    """
    L, K, V = 9, 4, 5
    etab, E_idx = _make_graph(L, K, V, seed=31)
    seqs = torch.randint(0, V, (1, L), generator=torch.Generator().manual_seed(32))
    etab = etab.clone().requires_grad_(True)

    cand = PottsMPNN.potts_candidate_energies(etab, E_idx, seqs)
    torch.log_softmax(-cand, dim=-1).gather(-1, seqs.unsqueeze(-1)).sum().backward()

    assert etab.grad is not None
    assert torch.isfinite(etab.grad).all()
    assert etab.grad.abs().sum() > 0


def test_zeroed_edge_contributes_nothing():
    """An edge whose table is zero (as E_mask zeroing does upstream) is inert."""
    L, K, V = 8, 4, 4
    etab, E_idx = _make_graph(L, K, V, seed=5)
    seqs = torch.randint(0, V, (1, L), generator=torch.Generator().manual_seed(6))
    d_full = PottsMPNN.potts_mutation_delta(etab, E_idx, seqs)
    # Zero the edge 0->E_idx[0,1] and its incoming effect on the neighbour.
    etab2 = etab.clone()
    etab2[0, 0, 1] = 0.0
    d_zero = PottsMPNN.potts_mutation_delta(etab2, E_idx, seqs)
    # Positions not touched by that single edge are unchanged.
    j = int(E_idx[0, 0, 1])
    untouched = [p for p in range(L) if p not in (0, j)]
    assert torch.allclose(d_full[0, untouched], d_zero[0, untouched], atol=1e-5)
