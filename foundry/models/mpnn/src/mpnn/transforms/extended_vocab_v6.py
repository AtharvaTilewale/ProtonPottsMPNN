"""EV6 — the AutoML protonation vocabulary (registry module for ``extended_vocab="v6"``).

Unlike v3/v4 (hand rules over the H-bond annotations), v6 is a LEARNED labeller: for every HIS/ASP/GLU it
computes the full geometric feature vector and runs the trained AutoML fold ensemble (see
``transforms/ev6/``). The tokens it emits:

    HIS-P / HIS-S / HIS-A          charged / neutral / ambiguous   (neutral is HIS-S, NOT HID/HIE)
    ASP-P / ASP-D / ASP-A          protonated / deprotonated / ambiguous
    GLU-P / GLU-D / GLU-A

``-A`` (ambiguous) is assigned from ENSEMBLE DISAGREEMENT — the SD across the five fold models — not from
a probability band; that is the deployable form of ``sandbox/ev5/3_automl/figs/uncertainty.png``.

This module only DISPATCHES, like v3/v4: it exposes ``classify_titratable_residues`` plus the pipeline
constants the vocabulary owns. The predictor is loaded lazily (10 pickles) on first use, so importing this
module is cheap.

NOTE — token encoding. ``protonation_label`` is a string annotation, so this labeller runs today. To
TRAIN a PottsMPNN on v6 labels, ``HIS-S`` must also be added to the token encoding
(``feature_aggregation/token_encodings.py``) and ``chemistry.py::BOND_CHEMISTRY`` — a fresh, deliberately
checkpoint-incompatible v6 token set. That wiring + a new checkpoint is the downstream step; it is not
needed to produce the labels.
"""
from __future__ import annotations

from mpnn.transforms.feature_aggregation.token_encodings import POTTS_MPNN_V6_TOKEN_ENCODING

# The vocabulary owns the pipeline's H-bond settings, and for v6 they ARE the EV6 feature pool: the
# upstream CalculateHbondsPlus runs default + combined-override at exactly these cutoffs, stashes the
# merged mode-tagged bonds on ``data["hbond_records"]``, and EV6 consumes them instead of re-running
# HBPLUS (5 subprocess calls per structure -> 2). So these constants are not merely "kept consistent"
# with EV6's cutoffs -- they must EQUAL them (ev6/features.py HA_MAX/DA_MAX), and the predictor asserts
# it rather than trusting the wiring: a pool built at v4's tighter 2.5/3.5 would silently starve the
# models of the bonds they were trained on. Permissive (no capability filter) because the models
# re-threshold internally. Deterministic: EV6 is a model, not a sampler.
CUTOFF_HA_DIST = 3.2
CUTOFF_DA_DIST = 4.0
FILTER_CAPABILITY = False
TRAIN_DETERMINISTIC = True

# The v6 token set the model trains on: neutral His is HIS-S (no HID/HIE). 30 tokens.
TOKEN_ENCODING = POTTS_MPNN_V6_TOKEN_ENCODING

# The residue -> token maps that TRAVEL WITH THE VOCAB. Anything scoring a v6 model (PKAD, sequence
# recovery) converts through THESE so it can never pick a token the model was not trained on. Tuple-valued
# to share the interface with v3/v4, whose neutral His spans the tautomers ("HID","HIE","HIS-D").
AA_PROTONATED = {"ASP": ("ASP-P",), "GLU": ("GLU-P",), "HIS": ("HIS-P",)}
AA_DEPROTONATED = {"ASP": ("ASP-D",), "GLU": ("GLU-D",), "HIS": ("HIS-S",)}   # neutral His = HIS-S
AA_AMBIGUOUS = {"ASP": ("ASP-A",), "GLU": ("GLU-A",), "HIS": ("HIS-A",)}

_PREDICTOR = None          # live labeller (callbacks / inference) — operating point may be swept via env
_PREDICTOR_DEFAULT = None   # score persistence — ALWAYS thresholds.json defaults, env-independent


def _env_prob_thr() -> tuple[float | None, float | None]:
    """The (his, acid) prob_thr overrides from the environment, or (None, None) for thresholds.json.

    ``EV6_HIS_PROB_THR`` / ``EV6_ACID_PROB_THR`` are the SAME two knobs train.py reads to drive
    ``ApplyProtonationThreshold`` on the precomputed scores. Reading them here too keeps the live FLAML in
    the benchmark callbacks at the identical operating point the model trains on (so a residue's NEIGHBOUR
    tokens in the teacher-forced S match training). Unset -> None -> the JSON default. Only ``prob_thr`` is
    env-driven; ``sd_cut`` (the ambiguous -A gate) stays fixed by design.
    """
    import os
    h = os.environ.get("EV6_HIS_PROB_THR", "").strip()
    a = os.environ.get("EV6_ACID_PROB_THR", "").strip()
    return (float(h) if h else None, float(a) if a else None)


def _predictor():
    """Lazy module-singleton for LIVE labelling — loads the fold ensembles once, on first classify call.

    Its prob_thr operating point is taken from the environment (see ``_env_prob_thr``) so the callbacks'
    live FLAML matches the training threshold. The env is fixed at process launch, so caching is safe."""
    global _PREDICTOR
    if _PREDICTOR is None:
        from mpnn.transforms.ev6 import EV6Predictor
        his_thr, acid_thr = _env_prob_thr()
        _PREDICTOR = EV6Predictor(his_thr=his_thr, acid_thr=acid_thr)
    return _PREDICTOR


def _predictor_default():
    """Lazy singleton pinned to thresholds.json — used ONLY to persist scores into snapshots.

    The persisted p/sd/metal are raw FLAML outputs (threshold-independent), and the token written alongside
    them is the DEFAULT-threshold token, so the snapshot's baked label never depends on a stray env var.
    The sweep happens later, at train time, in ApplyProtonationThreshold."""
    global _PREDICTOR_DEFAULT
    if _PREDICTOR_DEFAULT is None:
        from mpnn.transforms.ev6 import EV6Predictor
        _PREDICTOR_DEFAULT = EV6Predictor()
    return _PREDICTOR_DEFAULT


def classify_titratable_residues(atom_array, hbond_records=None, **_ignored) -> dict:
    """Label every HIS/ASP/GLU. Returns ``{(chain_id, res_id, res_name): token}``.

    ``hbond_records`` is the pool CalculateHbondsPlus already computed (``data["hbond_records"]``), passed
    through by AnnotateProtonationStates. None makes EV6 run HBPLUS itself — correct, but two extra
    subprocess calls per structure.

    The keyword arguments of the v3/v4 labellers (use_salt_bridge, deterministic, margin, temperature,
    beta, rng, dyad_cutoff, …) are accepted and ignored — v6's decision comes from the trained model."""
    return _predictor().predict_tokens(atom_array, hbond_records=hbond_records)


def classify_titratable_scores(atom_array, hbond_records=None, **_ignored) -> dict:
    """Per-residue FLAML scores for score PERSISTENCE (the augment/precompute path).

    Returns ``{(chain_id, res_id, res_name): (token, p_protonated, sd, metal_adjacent)}`` at the DEFAULT
    thresholds.json operating point. ``p`` / ``sd`` / ``metal`` are the raw ensemble outputs and are
    threshold-independent; ``token`` is the default-threshold call (what a fresh build would bake). Only v6
    exposes this — v3/v4 have no continuous score, so their ``get_vocab`` entry is None."""
    df = _predictor_default().predict(atom_array, hbond_records=hbond_records)
    return {
        (str(r.chain), int(r.res_id), str(r.res_name)):
            (str(r.token), float(r.p_protonated), float(r.sd), bool(r.metal_adjacent))
        for r in df.itertuples()
    }
