"""Unit tests for the protonation-state auxiliary loss (``L_state``).

``L_state`` is the group-renormalised state cross-entropy at titratable positions: the term that isolates
"is this Asp protonated?" from "is this an Asp?". These tests pin the three things that would silently
corrupt a training run:

  1. the value itself, against an independent per-position recomputation,
  2. the supervision mask (``-A``, bare parents and padding must never be supervised),
  3. that the joint loss is bit-for-bit unchanged when the term is switched off.

Pure unit tests on synthetic tensors — no checkpoint, no dataset, no model forward.
"""

import torch

from mpnn.loss.potts_loss import PottsJointLoss, ProtonationStateLoss
from mpnn.model.pottsmpnn import PottsContext
from mpnn.transforms.extended_vocab import get_vocab
from test_potts_energy import _make_graph

VOCAB = "v6"


def _idx():
    """(token_to_idx, V) for the v6 vocabulary."""
    enc = get_vocab(VOCAB)["token_encoding"]
    return enc.token_to_idx, enc.n_tokens


def _fake_batch(S, mask, V, seed=0):
    """A synthetic ``(network_input, network_output)`` pair carrying both heads.

    The decoder log-probs are a random log-softmax; the Potts context is a synthetic energy table from
    ``test_potts_energy._make_graph``, so the 'pot' head exercises the real conditional energy.
    """
    B, L = S.shape
    g = torch.Generator().manual_seed(seed)
    log_probs = torch.log_softmax(torch.randn(B, L, V, generator=g), dim=-1)

    # _make_graph draws K-1 DISTINCT non-self neighbours per residue, so K cannot exceed L.
    K = max(2, min(4, L))
    etabs, eidxs = [], []
    for b in range(B):
        etab, E_idx = _make_graph(L, K, V, seed=seed + 100 + b)
        etabs.append(etab)
        eidxs.append(E_idx)
    ctx = PottsContext(
        etab_out=torch.cat(etabs, dim=0),
        E_mask=torch.ones(B, L, K, dtype=torch.bool),
        E_idx=torch.cat(eidxs, dim=0),
        potts_loss_mask=torch.ones(B, L, K, dtype=torch.bool),
    )

    network_input = {"input_features": {"S": S}}
    network_output = {
        "input_features": {"mask_for_loss": mask},
        "decoder_features": {"log_probs": log_probs},
        "potts_context": ctx,
    }
    return network_input, network_output


def test_state_loss_matches_independent_recomputation():
    """The decoder-head value must equal a plain per-position recomputation of the definition."""
    t2i, V = _idx()
    # position 0: truly protonated Asp; position 1: neutral His; position 2: a non-titratable Ala.
    S = torch.tensor([[t2i["ASP-P"], t2i["HIS-S"], t2i["ALA"]]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    ni, no = _fake_batch(S, mask, V, seed=1)
    lp = no["decoder_features"]["log_probs"][0]

    loss_fn = ProtonationStateLoss(extended_vocab=VOCAB, heads=("dec",))
    got, d = loss_fn(network_input=ni, network_output=no, loss_input={})

    # Independent recomputation: -log( p(true state) / (p(P) + p(D)) ) at the two titratable positions.
    expect = []
    for pos, true_tok, other_tok in ((0, "ASP-P", "ASP-D"), (1, "HIS-S", "HIS-P")):
        l_true, l_other = lp[pos, t2i[true_tok]], lp[pos, t2i[other_tok]]
        expect.append(-(l_true - torch.logaddexp(l_true, l_other)))
    want = torch.stack(expect).mean()

    assert torch.allclose(got, want, atol=1e-6), (got.item(), want.item())
    assert int(d["state_n_supervised"].item()) == 2
    assert int(d["state_n_supervised_ASP"].item()) == 1
    assert int(d["state_n_supervised_HIS"].item()) == 1
    assert int(d["state_n_supervised_GLU"].item()) == 0


def test_ambiguous_bare_parent_and_padding_are_not_supervised():
    """Only the 6 P/D tokens may be supervised: -A, bare parents and masked positions are excluded."""
    t2i, V = _idx()
    S = torch.tensor([[
        t2i["HIS-A"],   # ambiguous  -> excluded by token
        t2i["HIS"],     # bare parent -> excluded by token
        t2i["GLU-D"],   # supervised
        t2i["GLU-P"],   # masked out  -> excluded by mask
    ]])
    mask = torch.tensor([[True, True, True, False]])
    ni, no = _fake_batch(S, mask, V, seed=2)

    loss_fn = ProtonationStateLoss(extended_vocab=VOCAB, heads=("dec",))
    got, d = loss_fn(network_input=ni, network_output=no, loss_input={})

    assert int(d["state_n_supervised"].item()) == 1
    assert int(d["state_n_supervised_GLU"].item()) == 1
    assert int(d["state_n_supervised_HIS"].item()) == 0
    assert torch.isfinite(got)


def test_no_supervised_position_returns_grad_carrying_zero():
    """A batch with no titratable residue must give a finite 0.0 that still has a graph (DDP safety)."""
    t2i, V = _idx()
    S = torch.tensor([[t2i["ALA"], t2i["GLY"]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    ni, no = _fake_batch(S, mask, V, seed=3)
    no["decoder_features"]["log_probs"].requires_grad_(True)

    loss_fn = ProtonationStateLoss(extended_vocab=VOCAB, heads=("dec",))
    got, d = loss_fn(network_input=ni, network_output=no, loss_input={})

    assert torch.isfinite(got) and got.item() == 0.0
    assert got.requires_grad
    got.backward()
    assert no["decoder_features"]["log_probs"].grad is not None
    assert int(d["state_n_supervised"].item()) == 0


def test_both_heads_average_and_potts_head_flows_gradient():
    """With both heads on, the loss is their mean, and gradients reach the Potts energy table."""
    t2i, V = _idx()
    S = torch.tensor([[t2i["ASP-P"], t2i["HIS-S"], t2i["GLU-D"]]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    ni, no = _fake_batch(S, mask, V, seed=4)
    no["potts_context"].etab_out.requires_grad_(True)

    both = ProtonationStateLoss(extended_vocab=VOCAB, heads=("pot", "dec"))
    v_both, d = both(network_input=ni, network_output=no, loss_input={})
    v_pot = d["state_nll_pot"]
    v_dec = d["state_nll_dec"]

    assert torch.allclose(v_both, (v_pot + v_dec) / 2, atol=1e-6)

    v_both.backward()
    grad = no["potts_context"].etab_out.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_joint_loss_is_unchanged_when_state_term_is_off():
    """state_loss_weight=0 must reproduce the pre-existing total exactly (bit-for-bit)."""
    t2i, V = _idx()
    S = torch.tensor([[t2i["ASP-P"], t2i["HIS-S"], t2i["ALA"], t2i["GLU-D"]]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    ni, no = _fake_batch(S, mask, V, seed=5)

    baseline = PottsJointLoss()                                     # as it was before this change
    with_off = PottsJointLoss(state_loss_weight=0.0, extended_vocab=VOCAB)
    with_on = PottsJointLoss(state_loss_weight=2.0, extended_vocab=VOCAB)

    t_base, _ = baseline(network_input=ni, network_output=no, loss_input={})
    t_off, d_off = with_off(network_input=ni, network_output=no, loss_input={})
    t_on, d_on = with_on(network_input=ni, network_output=no, loss_input={})

    assert with_off.state_loss is None, "weight 0 must not even construct the state term"
    assert torch.equal(t_base, t_off), (t_base.item(), t_off.item())
    assert "state_loss_agg" not in d_off
    # Switching it on must change the total by exactly weight * state_loss.
    assert torch.allclose(t_on, t_off + 2.0 * d_on["state_loss_agg"], atol=1e-6)


def test_missing_vocab_is_rejected():
    """Asking for the term without naming a vocabulary must fail loudly, not silently no-op."""
    try:
        PottsJointLoss(state_loss_weight=1.0, extended_vocab=None)
    except ValueError:
        return
    raise AssertionError("PottsJointLoss must reject state_loss_weight > 0 without extended_vocab")
