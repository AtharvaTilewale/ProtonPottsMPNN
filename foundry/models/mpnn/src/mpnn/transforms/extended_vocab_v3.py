"""Extended protonation vocabulary **v3** — FROZEN.

The vocabulary that trained ``mpnn_output_potts_mpnn_SB_0_2_lcomplex_fixed_ev3``. A verbatim lift of
``classify_titratable_residues`` and its helpers from commit ``aacb7e4``
(``transforms/vocab_annotation.py``) — the last state of the labeller before the charge-network rewrite.

**Do not "improve" this file.** It exists so an ev3 checkpoint can be scored, and an ev3-vocab model
retrained, with *exactly* the labels it was trained on. It carries its own copies of ``_hb_quality`` /
``_resolve_atom_role`` deliberately: a later edit to v4's helpers must never silently change v3.

Defining characteristics (all of which differ from v4):
  * the salt bridge is consulted **inside** the decision tree, whenever geometry is not *fully* clear
    (a large trigger set), not as a post-pass over leftover ``*-A``;
  * ``_resolve_acid_role``: an acid is ``-P`` only if one O **donates** AND the other **accepts**; ``-D``
    only if **both** O accept and neither donates. A lone acceptor is not enough;
  * HIS needs **both** ring N resolved for a geometric state. Anything less falls through to the salt
    bridge -> ``HIS-P``. (No ring exclusivity.) Since only ~7% of His have a donating ring N, that
    fallback is where nearly all of ev3's ``HIS-P`` comes from;
  * a donor/acceptor tie on ANY titratable atom poisons the whole residue to ``*-A``;
  * no capability filter, no ``_accepts_from_cation``, no carboxyl-dyad post-pass, no charge field.

Pipeline parameters belonging to this vocabulary (see ``extended_vocab.VOCABS``):
    cutoff_HA_dist=3.0, filter_capability=False, train_deterministic=True.
"""
from __future__ import annotations

import numpy as np
import biotite.structure as struc
from biotite.structure import AtomArray

from mpnn.transforms.feature_aggregation.token_encodings import POTTS_MPNN_TOKEN_ENCODING

CUTOFF_HA_DIST = 3.0        # ev3 ran HBPLUS at -h 3.0
CUTOFF_DA_DIST = 3.5        # ev3 ran HBPLUS at the pipeline's old default -d 3.5
FILTER_CAPABILITY = False   # BOND_CHEMISTRY did not exist at aacb7e4
TRAIN_DETERMINISTIC = True  # ev3's pipeline never passed `deterministic`, so it defaulted to True

# Same 32-token vocabulary and maps as v4 (ev3 also labels neutral His as the HID/HIE tautomers).
# HIS-D (imidazolate) is in neither side: it is a different, far higher-pKa deprotonation.
TOKEN_ENCODING = POTTS_MPNN_TOKEN_ENCODING
AA_PROTONATED = {"ASP": ("ASP-P",), "GLU": ("GLU-P",), "HIS": ("HIS-P",)}
AA_DEPROTONATED = {"ASP": ("ASP-D",), "GLU": ("GLU-D",), "HIS": ("HID", "HIE")}
AA_AMBIGUOUS = {"ASP": ("ASP-A",), "GLU": ("GLU-A",), "HIS": ("HIS-A",)}

_ASP_OXYGENS = {"OD1", "OD2"}
_GLU_OXYGENS = {"OE1", "OE2"}


def _hb_quality(dist: float, angle: float, beta: float) -> float:
    """H-bond quality (higher = better): shorter distance + straighter D-H..A angle.
    Distance in Angstrom; angle penalty is beta*(180-angle)/100 (skipped if angle NaN)."""
    q = -float(dist)
    if not np.isnan(angle):
        q -= beta * (180.0 - float(angle)) / 100.0
    return q


def _resolve_atom_role(atoms, mask, *, deterministic, margin, temperature, beta, rng) -> str:
    """Resolve a SINGLE H-bonding atom (one His ring N, or one carboxyl O) to
    'donor' / 'acceptor' / 'none' / 'ambiguous' from its stored best-bond geometry."""
    if not mask.any():
        return "none"

    def _g(field):
        v = atoms.get_annotation(field)[mask]
        return float(v[0]) if v.size else np.nan

    d_dist = _g("active_donor_dist")
    a_dist = _g("active_acceptor_dist")
    d_present = not np.isnan(d_dist)
    a_present = not np.isnan(a_dist)
    if d_present and not a_present:
        return "donor"
    if a_present and not d_present:
        return "acceptor"
    if not (d_present or a_present):
        return "none"

    q_d = _hb_quality(d_dist, _g("active_donor_angle"), beta)
    q_a = _hb_quality(a_dist, _g("active_acceptor_angle"), beta)
    if abs(q_d - q_a) >= margin:
        return "donor" if q_d > q_a else "acceptor"
    if deterministic:
        return "ambiguous"
    p_donor = 1.0 / (1.0 + np.exp(-(q_d - q_a) / temperature))
    return "donor" if rng.random() < p_donor else "acceptor"


def _resolve_acid_role(atoms, names, o_names, *, deterministic, margin, temperature, beta, rng) -> str:
    """Resolve a carboxylate to 'protonated' (COOH) / 'deprotonated' (COO-) / 'ambiguous' / 'none'.

    THE STRICT COOH PATTERN — the single biggest acid difference from v4:
        protonated  : one O DONATES and the other ACCEPTS (-OH donates, C=O accepts). BOTH required.
        deprotonated: BOTH oxygens accept and neither donates. A lone acceptor is NOT enough.
        ambiguous   : lone acceptor / lone donor / both donating / any unresolved per-O tie.
    """
    roles = []
    for o_name in o_names:
        m = names == o_name
        if m.any():
            roles.append(_resolve_atom_role(
                atoms, m, deterministic=deterministic, margin=margin,
                temperature=temperature, beta=beta, rng=rng))
    if not roles:
        return "none"
    if "ambiguous" in roles:
        return "ambiguous"           # ONE ambiguous O poisons the whole residue
    n_donor = roles.count("donor")
    n_accept = roles.count("acceptor")
    if n_donor >= 1 and n_accept >= 1:
        return "protonated"          # one O donates (-OH) + the other accepts (C=O) -> COOH
    if n_donor == 0 and n_accept >= 2:
        return "deprotonated"        # BOTH oxygens accept, neither donates -> COO-
    if n_donor == 0 and n_accept == 0:
        return "none"
    return "ambiguous"               # lone acceptor / lone donor / both-donor: not strict enough


def classify_titratable_residues(
    atom_array: AtomArray,
    *,
    use_salt_bridge: bool = True,
    deterministic: bool = True,
    margin: float = 0.3,
    temperature: float = 0.2,
    beta: float = 1.0,
    rng=None,
    **_ignored,          # v4-only knobs (use_local_charge, local_charge) are accepted and ignored
) -> dict:
    """Classify HIS/ASP/GLU by protonation state — the ev3 rules. Returns
    ``{(chain_id, res_id, res_name): label}``. Requires the ``CalculateHbondsPlus`` +
    ``AnnotateSaltBridges`` annotations."""
    aa = atom_array
    n_atoms = len(aa)
    res_starts = struc.get_residue_starts(aa)
    classifications = {}

    cats = set(aa.get_annotation_categories())
    has_geom = {"active_donor_dist", "active_acceptor_dist",
                "active_donor_angle", "active_acceptor_angle"} <= cats
    has_partner = "active_donor_partner_resns" in cats
    if rng is None:
        rng = np.random.default_rng()
    geom_kw = dict(deterministic=deterministic, margin=margin,
                   temperature=temperature, beta=beta, rng=rng)

    for k, start in enumerate(res_starts):
        end = res_starts[k + 1] if k + 1 < len(res_starts) else n_atoms
        res_name = aa.res_name[start]
        if res_name not in {"HIS", "ASP", "GLU"}:
            continue

        key = (aa.chain_id[start], int(aa.res_id[start]), res_name)
        atoms = aa[start:end]
        names = atoms.atom_name

        donor = atoms.active_donor
        acceptor = atoms.active_acceptor
        positive = atoms.active_positive
        negative = atoms.active_negative

        if res_name == "HIS":
            nd1 = names == "ND1"
            ne2 = names == "NE2"
            salt_bridge_evidence = use_salt_bridge and bool(positive.sum() > 0)

            if has_geom:
                role_nd1 = _resolve_atom_role(atoms, nd1, **geom_kw)
                role_ne2 = _resolve_atom_role(atoms, ne2, **geom_kw)
                # "Clear" geometry = BOTH ring N have a definite donor/acceptor role. Geometry is the
                # direct proton evidence, so it WINS over salt-bridge proximity here.
                if role_nd1 == "donor" and role_ne2 == "donor":
                    label = "HIS-P"
                elif role_nd1 == "donor" and role_ne2 == "acceptor":
                    label = "HID"
                elif role_ne2 == "donor" and role_nd1 == "acceptor":
                    label = "HIE"
                elif role_nd1 == "acceptor" and role_ne2 == "acceptor":
                    label = "HIS-D"
                elif salt_bridge_evidence:
                    # geometry NOT clear (a ring N is ambiguous or has no bond) -> fall back to the
                    # salt-bridge proximity prior: His near a carboxylate -> His+.
                    label = "HIS-P"
                else:
                    label = "HIS-A"
            elif salt_bridge_evidence:
                label = "HIS-P"
            else:  # legacy: boolean donor/acceptor flags (no geometry stored)
                nd1_donor = nd1.any() and bool(donor[nd1].sum() > 0)
                ne2_donor = ne2.any() and bool(donor[ne2].sum() > 0)
                nd1_acceptor = nd1.any() and bool(acceptor[nd1].sum() > 0)
                ne2_acceptor = ne2.any() and bool(acceptor[ne2].sum() > 0)
                if nd1_donor and ne2_donor:
                    label = "HIS-P"
                elif nd1_donor and ne2_acceptor:
                    label = "HID"
                elif ne2_donor and nd1_acceptor:
                    label = "HIE"
                elif nd1_acceptor and ne2_acceptor:
                    label = "HIS-D"
                else:
                    label = "HIS-A"

        else:  # ASP or GLU
            o_names = _ASP_OXYGENS if res_name == "ASP" else _GLU_OXYGENS
            o_mask = np.isin(names, list(o_names))
            # NB active_negative is set when the carboxylate salt-bridges to ANY cation, incl. His
            # (PLIP's positive set) -> His proximity influences the acid label. Inherited quirk.
            salt_bridge_evidence = (
                use_salt_bridge and o_mask.any() and bool(negative[o_mask].sum() > 0)
            )
            # A carboxyl O donating to ANOTHER carboxylate side-chain O is a shared-proton pair ->
            # this carboxyl carries the proton. One-sided: the partner is NOT touched (v4's dyad
            # post-pass is what changed that).
            _carbox = _ASP_OXYGENS | _GLU_OXYGENS
            donates_to_acid = has_partner and any(
                tok.split(":")[0] in ("ASP", "GLU") and tok.split(":")[-1] in _carbox
                for cell in atoms.active_donor_partner_resns[o_mask]
                for tok in str(cell).split(";") if tok
            )

            if donates_to_acid:
                label = f"{res_name}-P"
            elif has_geom:
                role = _resolve_acid_role(atoms, names, o_names, **geom_kw)
                if role == "protonated":        # clear COOH: one O donates + one accepts
                    label = f"{res_name}-P"
                elif role == "deprotonated":    # clear COO-: both O accept, none donate
                    label = f"{res_name}-D"
                elif salt_bridge_evidence:
                    # geometry NOT clear -> salt-bridge prior: carboxyl near a cation -> COO-.
                    label = f"{res_name}-D"
                else:
                    label = f"{res_name}-A"
            elif salt_bridge_evidence:
                label = f"{res_name}-D"
            else:  # legacy: boolean donor/acceptor flags (no geometry stored)
                o_donor = o_mask.any() and bool(donor[o_mask].sum() > 0)
                both_o_acceptors = all(
                    np.any((names == o_name) & (acceptor > 0)) for o_name in o_names
                )
                if o_donor:
                    label = f"{res_name}-P"
                elif both_o_acceptors:
                    label = f"{res_name}-D"
                else:
                    label = f"{res_name}-A"

        classifications[key] = label

    return classifications
