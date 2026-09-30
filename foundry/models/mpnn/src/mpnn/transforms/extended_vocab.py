"""Registry of the extended protonation vocabularies.

``extended_vocab`` is a NAME, not a bool. ``None`` means the standard 21-token vocabulary (no
protonation states); a name selects the 32-token protonation vocabulary AND the labeller that defines
it. Each vocabulary is a frozen, self-contained module so a checkpoint can always be scored with
exactly the labels it was trained on:

    "v3"  extended_vocab_v3.py   ev3's original salt-bridge labeller (commit aacb7e4).
                                 Trained: mpnn_output_potts_mpnn_SB_0_2_lcomplex_fixed_ev3
    "v4"  extended_vocab_v4.py   the rewritten atom-level labeller + salt-bridge fallback.
                                 Trained: mpnn_output_potts_sb_* (self_edge / nodefield / mlp arms)

A vocabulary is more than its classifier: the H-bond cutoff and the capability filter change which
bonds the classifier ever sees, so they are part of the vocabulary and travel with it here.

RETIRED: the charge-network vocabulary (mpnn_output_..._fixed_ev4). Its field-based labelling is no
longer reachable by name -- it produced only 7 HIS-P per 80 on PKAD and its decoder lost the His pKa
signal entirely. `charge_network.py` itself stays: `capability_ok` is still used by the bond filter.

THE VOCAB MUST MATCH THE CHECKPOINT. The encoder sees the protonation tokens in ``S``, so scoring a
v3 checkpoint with the v4 vocabulary puts the model out of distribution. Same class of hazard as
``field_source``, but silent -- there is no state_dict key to catch it.
"""
from __future__ import annotations

from mpnn.transforms import extended_vocab_v3, extended_vocab_v4, extended_vocab_v6

_MODULES = {
    "v3": extended_vocab_v3,
    "v4": extended_vocab_v4,
    "v6": extended_vocab_v6,     # AutoML labeller; neutral His = HIS-S, ambiguous -A from ensemble SD
}


def get_vocab(name: str) -> dict:
    """Everything that defines the vocabulary ``name``.

    Returns ``classify`` (the labeller) plus the pipeline parameters that belong to it:
    ``cutoff_HA_dist`` / ``cutoff_DA_dist`` / ``filter_capability`` (which bonds reach the labeller) and
    ``train_deterministic`` (whether ambiguous roles are resampled each epoch during training).

    Also returns the vocabulary's TOKEN SET and the residue -> token maps that travel with it:
    ``token_encoding`` (what the model's ``S`` is built against; sizes the encoder) and
    ``aa_protonated`` / ``aa_deprotonated`` / ``aa_ambiguous`` (the tokens each state maps to, tuple-valued
    so v3/v4's His tautomers and v6's single HIS-S share one interface). Downstream scoring reads these off
    the vocab name so a model is always compared in the vocabulary it was trained on.
    """
    if name not in _MODULES:
        raise ValueError(f"Unknown extended_vocab: {name!r}. Expected one of {sorted(_MODULES)}.")
    m = _MODULES[name]
    return {
        "classify": m.classify_titratable_residues,
        # Per-residue FLAML scores (token, p, sd, metal) for score PERSISTENCE — v6 only; v3/v4 have no
        # continuous score, so this is None and AnnotateProtonationStates(persist_scores=True) rejects them.
        "classify_scores": getattr(m, "classify_titratable_scores", None),
        "cutoff_HA_dist": m.CUTOFF_HA_DIST,
        "cutoff_DA_dist": m.CUTOFF_DA_DIST,
        "filter_capability": m.FILTER_CAPABILITY,
        "train_deterministic": m.TRAIN_DETERMINISTIC,
        "token_encoding": m.TOKEN_ENCODING,
        "aa_protonated": m.AA_PROTONATED,
        "aa_deprotonated": m.AA_DEPROTONATED,
        "aa_ambiguous": m.AA_AMBIGUOUS,
    }


VOCAB_NAMES = tuple(_MODULES)
