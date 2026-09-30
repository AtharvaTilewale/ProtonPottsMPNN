"""Per-token bond chemistry: formal charge + H-bond donor/acceptor / salt-bridge capability.

Pure data (NO torch), shared by the model (``mpnn.model.hbond_model``) and the annotation transforms
(``mpnn.transforms.charge_network``) so neither imports the other. Keyed by TOKEN NAME
(``encoding.idx_to_token``, e.g. ``"ASP-D"``, ``"HIS-P"``, ``"LYS"``):
    charge          formal charge (salt-bridge sign matching; a salt bridge needs +/−)
    can_donate      can donate an H-bond (has a polar H)
    can_accept      can accept an H-bond (has a lone pair)
    can_saltbridge  is a charged salt-bridge participant
Also keyed by the bare residue names ``HIS``/``ASP``/``GLU`` and their ambiguous tokens
``HIS-A``/``ASP-A``/``GLU-A``, both meaning "protonation state not (yet) resolved". Anything still
absent (UNK, ligands, non-standard residues) → :data:`DEFAULT_CAPABILITY`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BondCapability:
    charge: int
    can_donate: bool
    can_accept: bool
    can_saltbridge: bool


DEFAULT_CAPABILITY = BondCapability(0, False, False, False)

BOND_CHEMISTRY: dict[str, BondCapability] = {
    # --- titratable / pH-sensitive microstates ---
    "ASP-D": BondCapability(-1, False, True,  True),   # deprotonated COO− : acceptor-only, salt-bridge
    "GLU-D": BondCapability(-1, False, True,  True),
    "ASP-P": BondCapability(0,  True,  True,  False),  # protonated COOH : donor+acceptor, no salt-bridge
    "GLU-P": BondCapability(0,  True,  True,  False),
    "HIS-P": BondCapability(1,  True,  False, True),   # imidazolium (+1) : donor, salt-bridge
    "HID":   BondCapability(0,  True,  True,  False),  # neutral Nδ-H tautomer : donor+acceptor
    "HIE":   BondCapability(0,  True,  True,  False),  # neutral Nε-H tautomer : donor+acceptor
    "HIS-S": BondCapability(0,  True,  True,  False),  # neutral His, tautomer-agnostic (v6) : = HID/HIE
    "HIS-D": BondCapability(-1, False, True,  True),   # imidazolate (−1) : acceptor-only, salt-bridge
    # --- titratable groups whose protonation state is NOT resolved -------------------------------
    # "Unknown state" is the UNION over the states the group can occupy, not a failure of chemistry.
    # A carboxyl that turns out to be ASP-P donates AND accepts; as ASP-D it only accepts. Since the
    # state is exactly what the classifier is trying to infer, raw HBPLUS bonds must be filtered with
    # BOTH directions allowed -- that is the whole point of the carboxyl-as-donor override pass, which
    # deliberately proposes the -P geometry so the classifier can test it. Committing to acceptor-only
    # here would silently delete the evidence for ASP-P/GLU-P before it is ever weighed.
    # charge = 0: the SIGN is unknown, so nothing may infer a cation/anion from these tokens. The
    # H-bond head additionally holds q = NaN for the *-A tokens (see hbond_head.AMBIGUOUS_TOKENS).
    # Stated explicitly rather than left to a dict-lookup default, because the two lookups disagreed:
    # `BOND_CHEMISTRY.get(name)` -> None was read as permissive, while `capability(name)` fell back to
    # DEFAULT_CAPABILITY, which fails every test -- so the head could never emit an H-bond on a *-A.
    "ASP":   BondCapability(0,  True,  True,  True),   # COOH or COO− : may donate, always accepts
    "GLU":   BondCapability(0,  True,  True,  True),
    "HIS":   BondCapability(0,  True,  True,  True),   # imidazolium / HID / HIE / imidazolate
    "ASP-A": BondCapability(0,  True,  True,  True),
    "GLU-A": BondCapability(0,  True,  True,  True),
    "HIS-A": BondCapability(0,  True,  True,  True),
    # --- fixed-charge groups ---
    "LYS":   BondCapability(1,  True,  False, True),   # NH3+ : donor, salt-bridge
    "ARG":   BondCapability(1,  True,  False, True),   # guanidinium+ : donor, salt-bridge
    # --- neutral polar side chains (side-chain donor/acceptor) ---
    "SER":   BondCapability(0,  True,  True,  False),
    "THR":   BondCapability(0,  True,  True,  False),
    "TYR":   BondCapability(0,  True,  True,  False),
    "ASN":   BondCapability(0,  True,  True,  False),
    "GLN":   BondCapability(0,  True,  True,  False),
    "CYS":   BondCapability(0,  True,  True,  False),
    "TRP":   BondCapability(0,  True,  False, False),  # indole N-H : donor only
    "MET":   BondCapability(0,  False, True,  False),  # thioether S(delta) : weak acceptor, no H
    # --- apolar side chains: no side-chain H-bond ---
    "ALA":   BondCapability(0,  False, False, False),
    "GLY":   BondCapability(0,  False, False, False),
    "VAL":   BondCapability(0,  False, False, False),
    "LEU":   BondCapability(0,  False, False, False),
    "ILE":   BondCapability(0,  False, False, False),
    "PHE":   BondCapability(0,  False, False, False),
    "PRO":   BondCapability(0,  False, False, False),
}

# pH-sensitive states counts are grouped by (ambiguous -A tokens intentionally excluded).
# HIS-S is the v6 neutral-His state (the tautomer-agnostic counterpart of HID/HIE).
TITRATABLE_STATES = ("ASP-P", "ASP-D", "GLU-P", "GLU-D", "HIS-P", "HID", "HIE", "HIS-S", "HIS-D")


def capability(name: str) -> BondCapability:
    return BOND_CHEMISTRY.get(name, DEFAULT_CAPABILITY)
