from atomworks.constants import AA_LIKE_CHEM_TYPES, STANDARD_AA, UNKNOWN_AA
from atomworks.ml.encoding_definitions import TokenEncoding

# Token ordering for MPNN.
token_order = STANDARD_AA + (UNKNOWN_AA,)

# Token ordering for old versions of MPNN.
legacy_token_order = (
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
    "UNK",
)

# Atom ordering for new versions of MPNN.
atom_order = (
    "N",
    "CA",
    "C",
    "O",
    "CB",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
)

# Atom ordering for old versions of MPNN.
legacy_atom_order = (
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
)

# Token encoding for MPNN.
MPNN_TOKEN_ENCODING = TokenEncoding(
    token_atoms={token: atom_order for token in token_order},
    chemcomp_type_to_unknown={chem_type: "UNK" for chem_type in AA_LIKE_CHEM_TYPES},
)

# Extended token ordering for PottsMPNN: standard 21 tokens + 8 protonation-state
# variants for HIS, ASP, and GLU. Each new token shares the same 37-atom layout
# as its canonical parent (same heavy-atom skeleton; protonation differs only in
# H positions which are absent from crystallographic structures).
PROTONATION_TOKENS = (
    # HIS tautomers / charge states
    "HID",    # delta tautomer (neutral, ND1 protonated)
    "HIE",    # epsilon tautomer (neutral, NE2 protonated)
    "HIS-P",  # doubly protonated (+1, imidazolium)
    "HIS-D",  # deprotonated (-1, imidazolate; rare, enzyme active sites)
    "HIS-A",  # ambiguous / insufficient evidence
    # ASP charge states
    "ASP-P",  # protonated (neutral carboxyl)
    "ASP-D",  # deprotonated (-1, both O acceptors or salt-bridge evidence)
    "ASP-A",  # ambiguous
    # GLU charge states
    "GLU-P",  # protonated (neutral carboxyl)
    "GLU-D",  # deprotonated (-1, both O acceptors or salt-bridge evidence)
    "GLU-A",  # ambiguous
)
potts_token_order = token_order + PROTONATION_TOKENS

# Token encoding for PottsMPNN (32 tokens: 21 standard + 11 protonation states).
POTTS_MPNN_TOKEN_ENCODING = TokenEncoding(
    token_atoms={token: atom_order for token in potts_token_order},
    chemcomp_type_to_unknown={chem_type: "UNK" for chem_type in AA_LIKE_CHEM_TYPES},
)

# The v6 (AutoML) vocabulary. Neutral His is a SINGLE token, HIS-S — v6 predicts CHARGE STATE, not the
# HID/HIE tautomer — so the two neutral tautomers and the rare imidazolate HIS-D are dropped. Each residue
# gets exactly protonated / deprotonated / ambiguous. 30 tokens = 21 standard + 9 protonation states.
# Kept SEPARATE from POTTS_MPNN_TOKEN_ENCODING so v3/v4 checkpoints (32 tokens) are untouched; a v6 model
# is 30-token and deliberately weight-incompatible with them.
POTTS_MPNN_V6_PROTONATION_TOKENS = (
    "HIS-P", "HIS-S", "HIS-A",
    "ASP-P", "ASP-D", "ASP-A",
    "GLU-P", "GLU-D", "GLU-A",
)
potts_v6_token_order = token_order + POTTS_MPNN_V6_PROTONATION_TOKENS
POTTS_MPNN_V6_TOKEN_ENCODING = TokenEncoding(
    token_atoms={token: atom_order for token in potts_v6_token_order},
    chemcomp_type_to_unknown={chem_type: "UNK" for chem_type in AA_LIKE_CHEM_TYPES},
)

# Token encoding for versions of MPNN using the legacy token order and
# new atom order.
MPNN_LEGACY_TOKEN_ENCODING = TokenEncoding(
    token_atoms={token: atom_order for token in legacy_token_order},
    chemcomp_type_to_unknown={chem_type: "UNK" for chem_type in AA_LIKE_CHEM_TYPES},
)

# Token encoding for versions of MPNN using the legacy token order and
# legacy atom order.
MPNN_LEGACY_TOKEN_LEGACY_ATOM_ENCODING = TokenEncoding(
    token_atoms={token: legacy_atom_order for token in legacy_token_order},
    chemcomp_type_to_unknown={chem_type: "UNK" for chem_type in AA_LIKE_CHEM_TYPES},
)
