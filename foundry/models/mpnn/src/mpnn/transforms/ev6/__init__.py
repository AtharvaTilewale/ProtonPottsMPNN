"""EV6 — a convenience wrapper that turns a protein structure into protonation-state calls.

    from mpnn.transforms.ev6 import EV6Predictor
    df = EV6Predictor().predict(atom_array)   # per-residue: token, p_protonated, sd, ...

Load a structure the way the model does (`MPNNInferenceInput.from_atom_array_and_dict`) and pass
`inf.atom_array` to `predict`. Feature models only — biotite + HBPLUS + the pickled AutoML models, no
torch/PottsMPNN. See predictor.py for the token policy. `mpnn.transforms.extended_vocab_v6` exposes the
same predictor as the "v6" labelling vocab for the Potts pipeline.
"""
from mpnn.transforms.ev6.predictor import EV6Predictor, discretize_protonation

__all__ = ["EV6Predictor", "discretize_protonation"]
