from typing import Any

import numpy as np
from atomworks.common import KeyToIntMapper
from atomworks.constants import ELEMENT_NAME_TO_ATOMIC_NUMBER
from atomworks.ml.encoding_definitions import TokenEncoding
from atomworks.ml.transforms._checks import check_atom_array_annotation
from atomworks.ml.transforms.base import Transform
from atomworks.ml.transforms.encoding import atom_array_to_encoding
from atomworks.ml.utils.token import get_token_count, get_token_starts, token_iter

from mpnn.transforms.feature_aggregation.token_encodings import POTTS_MPNN_TOKEN_ENCODING


def _build_protonation_aware_seq(
    atom_array,
    encoding: TokenEncoding,
    protonation_label_rate: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Build an integer sequence array from atom_array.

    For residue-level tokens, ``protonation_label`` takes priority over
    ``res_name`` with probability ``protonation_label_rate``. With the
    complementary probability the canonical ``res_name`` token is used
    instead. This stochastic dropout prevents the model from over-relying
    on potentially noisy protonation annotations and ensures it also learns
    from the canonical token representation.

    At inference set ``protonation_label_rate=1.0`` to always use the label.

    Args:
        atom_array: Non-atomized AtomArray. ``protonation_label`` is optional;
            when absent the function behaves identically to the standard encoder.
        encoding: TokenEncoding to use for the integer lookup.
        protonation_label_rate: Probability in [0, 1] of using the protonation
            label instead of the canonical res_name for titratable residues.
        rng: NumPy random Generator. If None, a fresh default-seeded generator
            is created. Pass an explicit generator for reproducibility.

    Returns:
        Integer sequence array of shape ``[n_tokens]``.
    """
    if rng is None:
        rng = np.random.default_rng()

    has_protonation = "protonation_label" in atom_array.get_annotation_categories()
    has_atomize = "atomize" in atom_array.get_annotation_categories()

    n_tokens = get_token_count(atom_array)
    seq = np.empty(n_tokens, dtype=np.int64)

    for i, token in enumerate(token_iter(atom_array)):
        # Atom-level tokens (ligands, ions) are unaffected by protonation state.
        if has_atomize and token.atomize[0]:
            token_name = (
                token.atomic_number[0]
                if "atomic_number" in token.get_annotation_categories()
                else ELEMENT_NAME_TO_ATOMIC_NUMBER[token.element[0].upper()]
            )
            token_is_atom = True
        else:
            token_is_atom = False
            # For residue tokens: use protonation_label with probability
            # protonation_label_rate; fall back to res_name otherwise.
            use_label = (
                has_protonation
                and token.protonation_label[0]  # non-empty label
                and rng.random() < protonation_label_rate
            )
            token_name = token.protonation_label[0] if use_label else token.res_name[0]

        if token_name not in encoding.token_to_idx:
            token_name = encoding.resolve_unknown_token_name(token_name, token_is_atom)

        seq[i] = encoding.token_to_idx[token_name]

    return seq


class EncodePottsMPNNNonAtomizedTokens(Transform):
    """Encode non-atomized tokens for PottsMPNN with X, X_m, and S features.

    Identical to ``EncodeMPNNNonAtomizedTokens`` except:
    - Uses ``POTTS_MPNN_TOKEN_ENCODING`` (29 tokens).
    - Builds the integer sequence ``S`` from ``protonation_label`` when present,
      with stochastic fallback to the canonical ``res_name`` token controlled by
      ``protonation_label_rate``.

    Creates:
        X:   (L, 37, 3)  float32  heavy-atom coordinates.
        X_m: (L, 37)     bool     atom existence / occupancy mask.
        S:   (L,)        int64    token indices into POTTS_MPNN_TOKEN_ENCODING.

    Args:
        occupancy_threshold: Minimum occupancy to consider an atom present.
        protonation_label_rate: Probability of using the protonation label
            for titratable residues. Set to 1.0 at inference (always use label),
            and to a value < 1.0 during training for label-dropout regularisation.
        seed: Optional integer seed for the internal RNG. Useful for tests;
            leave as None during training for independent draws each call.
    """

    def __init__(
        self,
        occupancy_threshold: float = 0.5,
        protonation_label_rate: float = 1.0,
        seed: int | None = None,
        encoding=None,
    ):
        self.occupancy_threshold = occupancy_threshold
        self.protonation_label_rate = protonation_label_rate
        # The vocabulary's token set: v3/v4 -> the 32-token POTTS_MPNN_TOKEN_ENCODING (default), v6 -> its
        # 30-token encoding. Passed from the pipeline via get_vocab(extended_vocab)["token_encoding"] so S
        # is built in the same vocabulary the model is sized to.
        self.encoding = encoding if encoding is not None else POTTS_MPNN_TOKEN_ENCODING
        self._rng = np.random.default_rng(seed)

    def check_input(self, data: dict[str, Any]) -> None:
        check_atom_array_annotation(data, ["atomize", "res_name", "occupancy"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]

        assert len(atom_array) > 0, "atom_array cannot be empty"

        non_atomized_array = atom_array[~atom_array.atomize]

        assert len(non_atomized_array) > 0, "No non-atomized atoms found"

        if len(non_atomized_array) == 0:
            data["input_features"].update(
                {
                    "X": np.zeros((0, 37, 3), dtype=np.float32),
                    "X_m": np.zeros((0, 37), dtype=np.bool_),
                    "S": np.zeros((0,), dtype=np.int64),
                }
            )
            return data

        # Coordinates and mask via the standard encoder. Heavy-atom positions
        # are identical across all protonation variants of the same residue so
        # res_name-based coordinate lookup is correct here.
        encoded = atom_array_to_encoding(
            non_atomized_array,
            encoding=self.encoding,
            default_coord=0.0,
            occupancy_threshold=self.occupancy_threshold,
        )

        X = encoded["xyz"].astype(np.float32)
        X_m = encoded["mask"].astype(np.bool_)

        # Sequence: stochastic protonation-label-aware lookup.
        S = _build_protonation_aware_seq(
            non_atomized_array,
            self.encoding,
            protonation_label_rate=self.protonation_label_rate,
            rng=self._rng,
        )

        data["input_features"].update({"X": X, "X_m": X_m, "S": S})

        assert X.shape[0] > 0, "At least one non-atomized token should be present"
        return data
