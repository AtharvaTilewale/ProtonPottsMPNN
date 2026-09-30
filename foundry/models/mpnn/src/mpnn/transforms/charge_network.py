"""Geometry-only local charge field for protonation labeling + the H-bond capability filter.

1. ``capability_ok`` — for ``calculate_hbonds``: drop a directed HBPLUS bond whose acceptor can't
   accept or whose donor can't donate, per :data:`mpnn.chemistry.BOND_CHEMISTRY` (Lys/Arg/Trp are
   donor-only; Met SD acceptor-only; backbone N donor-only, backbone O acceptor-only). Titratable side
   chains carry the "state unknown" capability (donate AND accept), so a carboxyl "donating" via the
   override pass survives here — it is a hypothesis about ASP-P, and only the resolved protonation
   label can reject it. That happens downstream in ``BuildBondEdgeLabels`` (ASP-D cannot donate).

2. ``_charge_sites`` + ``local_potential`` — the local electrostatic potential at EVERY titratable
   atom (His ND1/NE2, Asp OD1/OD2, Glu OE1/OE2), used by ``vocab_annotation`` to decide the atoms that
   H-bond geometry leaves undecided:

       Phi(a) = sum_r  q_r * (1/d_r - 1/cutoff)    over residues r != own(a),  d_r <= cutoff
       d_r    = distance from atom a to residue r's CLOSEST functional atom

   Each neighbouring residue contributes EXACTLY ONCE. Phi > 0 means cations dominate the pocket, so
   the anionic / lone-pair form is stabilised there and a proton is disfavoured.

   a-priori charges: Lys/Arg +1, acid -1, His +0.1. Residues already resolved by geometry contribute
   their TRUE charge instead (a protonated acid contributes 0 — it has nothing to donate). NO pKa.

Prototype + calibration: ``scripts/13_charge_network_prototype.py``.
"""
from __future__ import annotations

import numpy as np
import biotite.structure as struc

from mpnn.chemistry import BOND_CHEMISTRY
from mpnn.transforms.bond_annotation import (
    PLIP_POSITIVE_ATOMS, PLIP_NEGATIVE_ATOMS, BACKBONE_DONOR_ATOMS, BACKBONE_ACCEPTOR_ATOMS,
)

# Thresholds on Phi. Set from the CHARGE SCALE, not fitted: one full opposite charge at salt-bridge
# contact (4 A) gives |Phi| = w(4) = 1/4 - 1/8 = 0.125 under the shifted kernel below. Past -0.125 an
# anion is close enough to hold a proton; past +0.125 a cation is close enough to carry the charge
# instead. Between them the field declines and the atom stays ambiguous.
#
# These CANNOT be calibrated against H-bond geometry, and the attempt is circular: HBPLUS's default
# pass never gives a carboxyl oxygen a donor role (it is not in its donor list), so EVERY "known
# protonated" acid oxygen comes from the -E ASP OD1 1 override pass -- the same hypothesis that the
# resolved labels reject 72.6% of the time. Fitting Phi to that class fits it to the artifact. (The
# symptom: the "strict COOH" population has median Phi +0.001 but p75 +0.249 -- a mixture, not a class.)
# His is exempt: ring-N donor/acceptor roles come from the DEFAULT pass, so tautomer_margin below IS
# calibrated.
#
# tautomer_margin is a RELATIVE cut on phi(ND1) - phi(NE2), taken from the UNSHIFTED field (see
# local_potential). The absolute potential barely separates protonated from deprotonated His nitrogens
# (medians -0.158 vs -0.024), but their DIFFERENCE does: the more negative ring N is the protonated one
# 73.3% of the time overall, and 91.4% of the time on the 78% of pairs where |dphi| >= 0.05. A carboxyl
# gets no such rule -- its two oxygens sit 2.2 A apart facing the same pocket (COOH -OH median +0.001 vs
# its C=O +0.037, i.e. no signal) -- which costs nothing, since ASP-P/GLU-P never records which oxygen
# holds the proton.
#
# cutoff 8.0: with the SHIFTED kernel the choice is no longer delicate, because the weight reaches zero
# continuously there. A hard 1/r truncation would instead make a neighbour crossing r_off gain or lose
# 1/r_off in one step -- 0.200 at cutoff 5, i.e. 80% of the decision threshold, enough to flip a label
# on a 0.02 A coordinate change, and training adds structure_noise=0.1 A. Shorter cutoffs make that
# cliff BIGGER, not smaller. NO pKa anywhere.
# phi_corrob is a LOWER, corroborated deprotonation bar: it applies only when one carboxyl oxygen is
# already a geometry-resolved acceptor (a proven lone pair, leaning COO-). The acceptor role does half
# the work, so the field only has to supply the other half -- half of phi_deprot. It reads the EVIDENCE
# field (resolved cations only), so a spent or circular charge cannot corroborate. This is the additive
# "one O accepts AND a Lys/Arg sits nearby -> deprotonated" case; it recovers ~13% of acids that a
# proven acceptor role plus a real contact cation left ambiguous under the standalone bar alone.
# dyad_cutoff: O-O distance under which two carboxylates count as a shared-proton pair (COOH...COO-).
# One proton is shared across the H-bond, so EXACTLY one partner is protonated. See the dyad post-pass
# in vocab_annotation: (A) if one partner is resolved, the other takes the opposite state (deterministic
# inference); (B) if both are ambiguous, sample which holds the proton (training augmentation only).
DEFAULTS = dict(his_w=0.1, acid_amb_w=-1.0, cutoff=8.0, phi_prot=-0.125, phi_deprot=0.125,
                phi_corrob=0.05, tautomer_margin=0.05, dyad_cutoff=3.2, n_iterations=1)


# ── H-bond capability filter ──────────────────────────────────────────────────
# BOND_CHEMISTRY is keyed per RESIDUE, so reading it per ATOM is only sound where the side chain has a
# single H-bonding atom -- true for Lys NZ, Arg NE/NH*, Trp NE1, Met SD, and the titratable groups
# (whose two carboxyl O / two ring N share one capability). Ser/Thr/Tyr/Cys donate and accept on the
# same atom. Asn/Gln are the exception: ND2/NE2 donate only and OD1/OE1 accept only, but the residue
# entry permits both, so those two are left un-gated here (permissive, never deleting a real bond) and
# HBPLUS's own per-atom chemistry already refuses them.
# Unknown residues (ligands, UNK, non-standard) → permissive: we have no chemistry to assert.
def _can_accept(resn: str, atom: str) -> bool:
    if atom in BACKBONE_ACCEPTOR_ATOMS: return True       # backbone O/OXT
    if atom in BACKBONE_DONOR_ATOMS:    return False      # backbone N
    cap = BOND_CHEMISTRY.get(resn)
    return cap.can_accept if cap is not None else True


def _can_donate(resn: str, atom: str) -> bool:
    if atom in BACKBONE_DONOR_ATOMS:    return True
    if atom in BACKBONE_ACCEPTOR_ATOMS: return False
    cap = BOND_CHEMISTRY.get(resn)
    return cap.can_donate if cap is not None else True


def capability_ok(item: dict) -> bool:
    """True iff the directed HBPLUS bond (donor ``d_*`` → acceptor ``a_*``) is a-priori possible."""
    return _can_accept(item["a_resn"], item["a_atom"]) and _can_donate(item["d_resn"], item["d_atom"])


# ── local charge field ────────────────────────────────────────────────────────
# Charge a residue contributes to its neighbourhood, once its label is known. A protonated acid and a
# neutral His tautomer are electrically silent; an unresolved residue falls back to the a-priori.
#
# `acid_amb_w` is what an AMBIGUOUS acid contributes. Default -1.0: an acid is deprotonated at almost any
# relevant pH, so "we found nothing" is safest read as "still a carboxylate". Note this makes *-A and -D
# indistinguishable to Phi, so an ambiguous acid cannot change its own charge and the iteration has
# nothing to propagate through it (measured: 11 of 499 such acids moved across 10 passes, mean q
# -1.0000 -> -0.9980). Softening it to -0.5 is defensible as an uncertainty discount and settles the
# iteration faster (12/30 -> 7/30 structures unsettled at n=5), but its real effect is on HIS: it halves
# HIS-P (10.9% -> 5.9%), because a charged His needs negative potential and acids supply it. Unlike
# his_w=+0.1, which is His's EXPECTED charge at neutral pH, -1.0 here is the acid's expected charge.
_LABEL_CHARGE = {
    "ASP-D": -1.0, "GLU-D": -1.0, "HIS-D": -1.0,
    "ASP-P":  0.0, "GLU-P":  0.0, "HID":  0.0, "HIE": 0.0,
    "HIS-P":  1.0,
}


def residue_charge(resn: str, label: str, his_w: float, acid_amb_w: float = -1.0) -> float:
    """Source charge of one node. Lys/Arg are +1 unconditionally. A resolved titratable uses its label's
    true charge; anything unresolved (``*-A``, or no label yet) falls back to the a-priori: acid
    ``acid_amb_w``, His ``+his_w``. Both a-priori charges are deliberately weak — they are what the field
    is trying to infer, so an unknown His must not act as a full cation (pinning an acid to -D while the
    acid pins it to HIS-P), and an unknown acid must not act as a full carboxylate."""
    if resn in ("LYS", "ARG"):
        return 1.0
    q = _LABEL_CHARGE.get(label)
    if q is not None:
        return q
    return his_w if resn == "HIS" else acid_amb_w


def _charge_sites(aa):
    """Charge sites at the FUNCTIONAL ATOMS, not the residue centroid: Lys NZ, Arg NE/NH1/NH2,
    His ND1/NE2, Asp OD1/OD2, Glu OE1/OE2.

    Resolving to the atom is what lets the field distinguish the two His ring N -- Phi at the nitrogen
    facing a Lys is raised, pushing the proton onto the other one, so the tautomer falls out of the
    electrostatics instead of needing a donating N (which only 7.4% of His have). A residue centroid
    cannot see that, and it also smears a contact salt bridge by 1-2 A (Arg B234 -> Glu B136 is 3.11 A
    atom-to-atom but 3.55 A centroid-to-centroid).

    Returns ``(nodes, site_node, site_atom, site_xyz)`` where ``nodes[i] = (chain, resid, resn)`` and
    ``site_node`` maps each site back to its node."""
    plip = {**PLIP_POSITIVE_ATOMS, **PLIP_NEGATIVE_ATOMS}
    starts = struc.get_residue_starts(aa)
    ends = np.append(starts[1:], len(aa))
    coord, an = np.asarray(aa.coord), np.asarray(aa.atom_name)
    nodes, site_node, site_atom, site_xyz = [], [], [], []
    for s, e in zip(starts, ends):
        rn = str(aa.res_name[s])
        atoms = plip.get(rn)
        if atoms is None:
            continue
        idx = np.where(np.isin(an[s:e], list(atoms)))[0]
        if not len(idx):
            continue
        ni = len(nodes)
        nodes.append((str(aa.chain_id[s]), int(aa.res_id[s]), rn))
        for j in idx:
            site_node.append(ni)
            site_atom.append(str(an[s:e][j]))
            site_xyz.append(coord[s:e][j])
    return (nodes, np.asarray(site_node, dtype=int), np.asarray(site_atom, dtype=object),
            np.asarray(site_xyz, dtype=float).reshape(-1, 3))


def local_potential(site_node, site_xyz, q_node, cutoff: float, shift: bool = True) -> np.ndarray:
    """``Phi`` at each charge site. ``shift=True`` (default) uses the SHIFTED Coulomb kernel
    ``sum_r q_r * (1/d_r - 1/cutoff)``; ``shift=False`` uses bare ``sum_r q_r / d_r``. No screening.

    **Which kernel, and why both.** Bare ``1/r`` decays far too slowly to isolate the first shell -- a
    charge at 8 A still carries 39% of a 3.1 A contact's weight, at 12 A still 26% -- so the cutoff, not
    the kernel, does the suppressing, and chopping there leaves a step of ``1/cutoff`` (0.125 at 8 A,
    0.200 at 5 A) that a sub-Angstrom coordinate change walks across. Subtracting ``1/cutoff`` sends the
    weight continuously to zero at ``r_off``: a contact keeps most of its strength (0.322 -> 0.197 at
    3.11 A) while the second shell is damped by construction (0.150 -> 0.025 at 6.65 A).

    That is right for the ABSOLUTE question ("is this atom protonated?"), where the far tail is noise --
    it is what let two Glu at 4.5/6.7 A cancel Arg B234's 3.11 A salt bridge on Glu B136.

    It is wrong for the RELATIVE question (which His ring N holds the proton). The two ring nitrogens sit
    2.2 A apart and SHARE their near field, so every contact cancels in ``phi(ND1) - phi(NE2)``; the
    discriminating signal IS the asymmetric second shell. Measured on 45 pairs where geometry pins both
    nitrogens to opposite states, the more-negative N is the protonated one:
        bare 1/r  -- |dphi| >= 0.05 keeps 78% of pairs, 91.4% correct
        shifted   -- |dphi| >= 0.05 keeps 42% of pairs, 89.5% correct
    Same accuracy, half the coverage. So the tautomer uses ``shift=False``.

    Two reductions, both load-bearing:

    * **Each neighbouring RESIDUE contributes once**, at ``d_r`` = the distance to *its* closest
      functional atom. Summing per site instead would count a guanidinium three times (NE, NH1, NH2)
      and a carboxylate twice, inflating exactly the groups that matter most.
    * **Self-exclusion is by residue, not by site.** The two oxygens of a carboxylate (or the two ring
      N of a His) are ONE charge group: their mutual term is intramolecular, identical for COOH and
      COO-, so it says nothing about protonation -- and at ~2.2 A apart, 1/r would make it the LARGEST
      term in the sum (-0.23 at every Asp/Glu, against +0.28 for a real Arg contact). It would be a
      constant bias toward neutral, not a signal.
    """
    n_sites, n_res = len(site_node), len(q_node)
    if n_sites == 0 or n_res == 0:
        return np.zeros(n_sites)
    d = np.sqrt(((site_xyz[:, None, :] - site_xyz[None, :, :]) ** 2).sum(-1))   # [S, S]

    # [S, R]: distance from each site to each residue's CLOSEST functional atom
    d_res = np.full((n_sites, n_res), np.inf)
    np.minimum.at(d_res, (np.arange(n_sites)[:, None], site_node[None, :]), d)
    d_res[np.arange(n_sites), site_node] = np.inf                              # mask own residue

    w = np.zeros_like(d_res)
    m = (d_res > 0) & (d_res <= cutoff)
    w[m] = 1.0 / d_res[m] - (1.0 / cutoff if shift else 0.0)   # shifted -> continuous, exactly 0 at r_off
    return w @ q_node
