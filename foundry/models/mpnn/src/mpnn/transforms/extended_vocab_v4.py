"""Extended protonation vocabulary **v4** — FROZEN.

The vocabulary the ``potts_sb_*`` runs train on: the rewritten, atom-level labeller with the PLIP
salt-bridge fallback. Frozen alongside :mod:`extended_vocab_v3` so each vocabulary is immutable and a
checkpoint can always be scored with exactly the labels it was trained on. It carries its own copies of
``_hb_quality`` / ``_resolve_atom_role``: editing v4 must never silently change v3, and vice versa.

Order of operations (the charge-network field is NOT part of this vocabulary):
  1. per-ATOM state from H-bond geometry. A donor/acceptor tie makes the atom **silent**, it does not
     poison the residue (v3 sent the whole residue to ``*-A``).
  2. ``_accepts_from_cation``: a Lys/Arg donating into a titratable atom **forces** it to acceptor
     (no proton), overriding geometry. Chemistry-derived, needs no distance sphere.
  3. residue label from the pair of atom states (:func:`_label_from_atoms`), including **His ring
     exclusivity** — one confidently unprotonated ring N puts the proton on the other. This is the
     biggest divergence from v3, which required BOTH ring N resolved.
  4. salt-bridge fallback over whatever is still ``*-A`` (v3 consulted the sphere *inside* the tree,
     on a far larger trigger set).
  5. carboxyl-dyad post-pass: exactly one proton per shared-proton pair, run to a fixpoint.

Pipeline parameters belonging to this vocabulary (see ``extended_vocab.VOCABS``):
    cutoff_HA_dist=2.5, filter_capability=True, train_deterministic=False.
"""
from __future__ import annotations

import numpy as np
import biotite.structure as struc
from biotite.structure import AtomArray

from mpnn.chemistry import BOND_CHEMISTRY
from mpnn.transforms.bond_annotation import BACKBONE_DONOR_ATOMS
from mpnn.transforms.feature_aggregation.token_encodings import POTTS_MPNN_TOKEN_ENCODING

CUTOFF_HA_DIST = 2.5         # -h 3.0 admitted ~18% near-perpendicular contacts (median DHA 116 deg)
CUTOFF_DA_DIST = 3.5         # tightened from HBPLUS's own 3.9 default
FILTER_CAPABILITY = True     # drop a-priori impossible directed bonds (see chemistry.BOND_CHEMISTRY)
TRAIN_DETERMINISTIC = False  # sample ambiguous roles + symmetric dyads each epoch (augmentation)
DYAD_CUTOFF = 3.2            # O-O distance under which two carboxylates share a proton

# The 32-token vocabulary and the residue -> token maps that travel with it. These ARE the token groups
# downstream scoring uses (PKAD numerator / denominator, sequence recovery) — read off get_vocab, never
# hardcoded at the call site. Tuple-valued because v4's neutral His spans the two tautomers; v6 says HIS-S.
# The two sides are the pKa equilibrium itself: protonated <-> deprotonated. The imidazolate HIS-D is
# deliberately in NEITHER — it is a separate, far higher-pKa deprotonation, not the other side of this one.
TOKEN_ENCODING = POTTS_MPNN_TOKEN_ENCODING
AA_PROTONATED = {"ASP": ("ASP-P",), "GLU": ("GLU-P",), "HIS": ("HIS-P",)}
AA_DEPROTONATED = {"ASP": ("ASP-D",), "GLU": ("GLU-D",), "HIS": ("HID", "HIE")}
AA_AMBIGUOUS = {"ASP": ("ASP-A",), "GLU": ("GLU-A",), "HIS": ("HIS-A",)}

_ASP_OXYGENS = {"OD1", "OD2"}
_GLU_OXYGENS = {"OE1", "OE2"}
_CARBOXYL_OXYGENS = _ASP_OXYGENS | _GLU_OXYGENS

# The two titratable atoms of each family, in CANONICAL ORDER — the residue label is read off the pair,
# and HID/HIE differ only by which ring nitrogen carries the proton.
_FUNCTIONAL_ATOMS = {"HIS": ("ND1", "NE2"), "ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}


def _hb_quality(dist: float, angle: float, beta: float) -> float:
    """H-bond quality (higher = better): shorter distance + straighter D-H..A angle."""
    q = -float(dist)
    if not np.isnan(angle):
        q -= beta * (180.0 - float(angle)) / 100.0
    return q


def _resolve_atom_role(atoms, mask, *, deterministic, margin, temperature, beta, rng) -> str:
    """Resolve a SINGLE titratable atom to 'donor' / 'acceptor' / 'none' / 'ambiguous' from its stored
    best-bond geometry. A near-tie (gap < margin) is 'ambiguous' when deterministic, else sampled."""
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


def _accepts_from_cation(atoms, mask) -> bool:
    """True if this titratable atom accepts an H-bond from a SIDE-CHAIN group that carries a positive
    charge and cannot itself accept (Lys NZ, Arg NE/NH1/NH2 — BOND_CHEMISTRY charge>0, can_accept=False).

    Two facts follow, and neither needs a distance sphere: the donor cannot accept, so the direction is
    chemically forced (this atom really is the acceptor, not an artifact of the carboxyl-as-donor
    override pass); and the donor carries a full +1, so this is a salt-bridging H-bond, which only a
    deprotonated group forms. Backbone N is skipped (an Arg's backbone amide says nothing about its
    guanidinium). A bare HIS is charge 0 in BOND_CHEMISTRY, so a His donor never qualifies — its charge
    is what we are solving for."""
    if "active_acceptor_partner_resns" not in atoms.get_annotation_categories():
        return False   # legacy arrays predate the annotation; no evidence rather than a crash
    for cell in atoms.active_acceptor_partner_resns[mask]:
        for tok in str(cell).split(";"):
            if not tok:
                continue
            resn, _, atom = tok.partition(":")
            if atom in BACKBONE_DONOR_ATOMS:
                continue
            cap = BOND_CHEMISTRY.get(resn)
            if cap is not None and cap.charge > 0 and not cap.can_accept:
                return True
    return False


def _label_from_atoms(res_name: str, sa: str, ga, sb: str, gb) -> str:
    """Residue label from its two titratable atoms (canonical order: His ND1/NE2, acid OD1/OD2 or
    OE1/OE2). ``s*`` is the resolved state, ``g*`` the pinned geometry ("deprotonated" for a
    capability-forced acceptor). The two atoms are NOT independent — the residue's valence couples them.

    ACID — a carboxyl holds 0 or 1 proton. One O protonated -> ``-P`` (COOH2+ cannot exist); both -> a
    contradiction, ``-A``. ``-D`` needs both O to be resolved acceptors. A lone acceptor stays ``-A``:
    the proton could still sit on the undecided oxygen.

    HIS — an imidazole holds 1 or 2 protons. A donor role on a ring N means that N is protonated; both
    -> HIS-P, one -> HID/HIE. ``HIS-D`` only when BOTH ring N are pinned acceptors. **Ring exclusivity**:
    exactly one N confidently unprotonated puts the neutral ring's single proton on the other N."""
    if res_name == "HIS":
        if sa == "protonated" and sb == "protonated":
            return "HIS-P"                       # imidazolium (+1): a proton on each ring N
        if sa == "protonated":
            return "HID"                         # proton on ND1
        if sb == "protonated":
            return "HIE"                         # proton on NE2
        if ga == "deprotonated" and gb == "deprotonated":
            return "HIS-D"                       # imidazolate: both ring N pinned acceptors
        # Ring exclusivity — exactly one ring N is confidently UNprotonated, so the neutral ring's
        # single proton must sit on the OTHER one. (v3 had no such rule: it needed BOTH N resolved,
        # and fell through to the salt bridge -> HIS-P. This is the biggest v3/v4 divergence.)
        if sa == "deprotonated":
            return "HIE"                         # ND1 has no proton -> NE2 holds it
        if sb == "deprotonated":
            return "HID"                         # NE2 has no proton -> ND1 holds it
        return "HIS-A"

    if sa == "protonated" and sb == "protonated":
        return f"{res_name}-A"                   # COOH2+ is impossible -> contradictory evidence
    if sa == "protonated" or sb == "protonated":
        return f"{res_name}-P"                   # COOH; the other O cannot also hold a proton
    if sa == "deprotonated" and sb == "deprotonated":
        return f"{res_name}-D"                   # both oxygens resolved acceptors -> COO-
    return f"{res_name}-A"                       # the proton could still sit on the undecided oxygen


def _carboxyl_dyads(aa, cutoff: float):
    """Shared-proton carboxyl dyads: two carboxyl side-chain oxygens from DIFFERENT Asp/Glu residues
    within ``cutoff`` (O-O), where at least one donates an H-bond to a carboxyl oxygen. Two free COO-
    cannot sit that close, so exactly one partner is protonated.

    Pairing must be GEOMETRIC: ``active_donor_partner_resns`` records ``GLU:OE1`` (resn+atom) with no
    res_id, so it cannot say WHICH Glu. (v3's rule was token-name based and therefore one-sided — it
    labelled the donor ``-P`` and never touched the partner.) Returns ``[(keyA, keyB, qgap), ...]`` with
    keyA < keyB and ``qgap`` = donor-quality(A) - donor-quality(B): positive means A is the better donor
    and the likelier COOH."""
    cats = aa.get_annotation_categories()
    if not ({"active_donor_partner_resns", "active_donor_dist", "active_donor_angle"} <= set(cats)):
        return []
    starts = struc.get_residue_starts(aa)
    ends = np.append(starts[1:], len(aa))
    names = np.asarray(aa.atom_name)
    sites = []   # (key, coord, donates_to_carboxyl, donor_quality)
    for s, e in zip(starts, ends):
        rn = str(aa.res_name[s])
        if rn not in ("ASP", "GLU"):
            continue
        key = (str(aa.chain_id[s]), int(aa.res_id[s]), rn)
        for an in (_ASP_OXYGENS if rn == "ASP" else _GLU_OXYGENS):
            m = np.zeros(len(aa), dtype=bool)
            m[s:e] = names[s:e] == an
            if not m.any():
                continue
            partners = str(aa.active_donor_partner_resns[m][0])
            donates = any(t.split(":")[0] in ("ASP", "GLU") and t.split(":")[-1] in _CARBOXYL_OXYGENS
                          for t in partners.split(";") if t)
            dd = float(aa.active_donor_dist[m][0])
            dang = float(aa.active_donor_angle[m][0])
            dq = _hb_quality(dd, dang, 1.0) if not np.isnan(dd) else float("-inf")
            sites.append((key, aa.coord[m][0], donates, dq))
    best = {}   # sorted residue-pair -> (dist, qgap oriented to pair[0])
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            if sites[i][0] == sites[j][0]:
                continue
            d = float(np.linalg.norm(sites[i][1] - sites[j][1]))
            if d > cutoff or not (sites[i][2] or sites[j][2]):
                continue
            pair = tuple(sorted((sites[i][0], sites[j][0])))
            qgap = (sites[i][3] - sites[j][3]) if sites[i][0] == pair[0] else (sites[j][3] - sites[i][3])
            if pair not in best or d < best[pair][0]:
                best[pair] = (d, qgap)
    return [(a, b, g) for (a, b), (d, g) in best.items()]


def _dyad_holder(pA: str, pB: str):
    """Which partner holds the shared proton, from the two per-residue states ('P'/'D'/'A'). Returns
    'A', 'B', or None (both ambiguous, or a both-P / both-D conflict). ``-D`` ("no proton here") is the
    stronger signal, so it is consulted first."""
    if pA == "D" and pB != "D":
        return "B"                        # A carries no proton -> B holds it
    if pB == "D" and pA != "D":
        return "A"
    if pA == "P" and pB == "A":
        return "A"                        # A holds the proton, B unknown -> B is COO-
    if pB == "P" and pA == "A":
        return "B"
    return None


def _apply_carboxyl_dyads(aa, labels: dict, *, deterministic: bool, temperature: float, rng,
                          cutoff: float) -> dict:
    """Enforce "exactly one proton per shared-proton carboxyl dyad". A dyad shares ONE proton, so the
    only valid pair is one ``-P`` + one ``-D``.

    (A) EXCLUSIVITY (always): a definite ``-P``/``-D`` partner makes an ambiguous one take the opposite
        state — pure inference, no randomness. Run to a fixpoint so acid-acid-acid chains settle.
    (B) TIE-BREAK the rest — both ``*-A``, or a both-``-P`` / both-``-D`` conflict. NOTE a mutual-donor
        pair produces both-``-P`` even when deterministic, so the ``qgap`` safeguard below is load-bearing
        and not merely defensive.
          * deterministic: a genuine both-``*-A`` stays ``*-A`` (honest — geometry cannot say which);
            a conflict is broken by the donor gap.
          * stochastic: a SINGLE Bernoulli for the PAIR places the proton (never independent, which is
            what allowed the invalid both-``-P``), weighted toward the better donor by ``qgap``.
    """
    dyads = [(a, b, g) for a, b, g in _carboxyl_dyads(aa, cutoff) if a in labels and b in labels]
    st = lambda lab: "P" if lab.endswith("-P") else "D" if lab.endswith("-D") else "A"

    def _set(kp, kd):
        labels[kp], labels[kd] = f"{kp[2]}-P", f"{kd[2]}-D"

    # (A) exclusivity to a fixpoint. Each step only turns an ambiguous partner definite -> monotone.
    for _ in range(len(dyads) + 1):
        changed = False
        for kA, kB, _g in dyads:
            pA, pB = st(labels[kA]), st(labels[kB])
            if {pA, pB} == {"P", "D"}:
                continue
            holder = _dyad_holder(pA, pB)
            if holder is not None:
                _set(*((kA, kB) if holder == "A" else (kB, kA)))
                changed = True
        if not changed:
            break

    # (B) tie-break the rest.
    locked = set()
    for kA, kB, qgap in dyads:
        if kA in locked or kB in locked:
            continue
        pA, pB = st(labels[kA]), st(labels[kB])
        if {pA, pB} == {"P", "D"}:
            continue
        if pA == "A" and pB == "A" and deterministic:
            continue                                    # symmetric + deterministic -> stay *-A
        if deterministic:                               # conflict safeguard: better donor keeps it
            holder = "A" if qgap >= 0 else "B"
        else:                                           # ONE Bernoulli for the pair
            holder = "A" if rng.random() < 1.0 / (1.0 + np.exp(-qgap / max(temperature, 1e-6))) else "B"
        _set(*((kA, kB) if holder == "A" else (kB, kA)))
        locked.update((kA, kB))
    return labels


def _apply_salt_bridge_fallback(aa: AtomArray, labels: dict) -> dict:
    """PLIP salt-bridge fallback for residues still ``*-A``.

    Reads ``active_positive`` / ``active_negative`` from ``AnnotateSaltBridges`` (a 0.5-5.5 A sphere
    between PLIP cationic and anionic atom groups) and applies the proximity prior that H-bond geometry
    could not settle:

        His inside the sphere of a carboxylate    -> HIS-P   (the classic pKa-RAISING environment)
        carboxylate inside the sphere of a cation -> ``-D``  (the anion is stabilised)

    ONLY residues left ``*-A`` are touched. NB this is a much SMALLER trigger set than v3's, where the
    sphere was consulted whenever geometry was not *fully* clear — which, before ring exclusivity and the
    single-donor acid rule, was most residues. Caveat inherited from v3: PLIP's cation set includes His,
    so a carboxylate next to a His can be called ``-D`` on the strength of a residue whose own charge is
    what we are inferring. ``_accepts_from_cation`` does not have that problem and runs first."""
    cats = set(aa.get_annotation_categories())
    if not {"active_positive", "active_negative"} <= cats:
        return labels

    res_starts = struc.get_residue_starts(aa)
    n_atoms = len(aa)
    out = dict(labels)
    for k, start in enumerate(res_starts):
        end = res_starts[k + 1] if k + 1 < len(res_starts) else n_atoms
        res_name = str(aa.res_name[start])
        if res_name not in _FUNCTIONAL_ATOMS:
            continue
        key = (str(aa.chain_id[start]), int(aa.res_id[start]), res_name)
        if not str(out.get(key, "")).endswith("-A"):
            continue                                     # already resolved -> leave it alone

        atoms = aa[start:end]
        if res_name == "HIS":
            if bool(np.asarray(atoms.active_positive).sum() > 0):
                out[key] = "HIS-P"
        else:
            o_mask = np.isin(atoms.atom_name, list(_FUNCTIONAL_ATOMS[res_name]))
            if o_mask.any() and bool(np.asarray(atoms.active_negative)[o_mask].sum() > 0):
                out[key] = f"{res_name}-D"
    return out


def classify_titratable_residues(
    atom_array: AtomArray,
    *,
    use_salt_bridge: bool = True,
    deterministic: bool = True,
    margin: float = 0.3,
    temperature: float = 0.2,
    beta: float = 1.0,
    rng=None,
    dyad_cutoff: float = DYAD_CUTOFF,
    **_ignored,          # v3-only knobs are accepted and ignored
) -> dict:
    """Classify HIS/ASP/GLU by protonation state — the v4 rules. Returns
    ``{(chain_id, res_id, res_name): label}``. Requires the ``CalculateHbondsPlus`` +
    ``AnnotateSaltBridges`` annotations."""
    aa = atom_array
    n_atoms = len(aa)
    res_starts = struc.get_residue_starts(aa)
    if rng is None:
        rng = np.random.default_rng()
    geom_kw = dict(deterministic=deterministic, margin=margin,
                   temperature=temperature, beta=beta, rng=rng)

    cats = set(aa.get_annotation_categories())
    has_geom = {"active_donor_dist", "active_acceptor_dist",
                "active_donor_angle", "active_acceptor_angle"} <= cats

    labels: dict = {}
    for k, start in enumerate(res_starts):
        end = res_starts[k + 1] if k + 1 < len(res_starts) else n_atoms
        res_name = str(aa.res_name[start])
        func_atoms = _FUNCTIONAL_ATOMS.get(res_name)
        if func_atoms is None:
            continue

        key = (str(aa.chain_id[start]), int(aa.res_id[start]), res_name)
        atoms = aa[start:end]
        names = atoms.atom_name

        states, geoms = [], []
        for atom_name in func_atoms:
            m = names == atom_name
            if has_geom:
                role = _resolve_atom_role(atoms, m, **geom_kw)
            else:   # legacy arrays: boolean donor/acceptor flags, no stored geometry
                role = ("donor" if m.any() and bool(atoms.active_donor[m].sum() > 0)
                        else "acceptor" if m.any() and bool(atoms.active_acceptor[m].sum() > 0)
                        else "none")
            # A tie ("ambiguous") or no bond ("none") leaves the atom SILENT -- it does not poison the
            # residue the way it did in v3; the other atom gets to decide.
            geom_state = ("protonated" if role == "donor"
                          else "deprotonated" if role == "acceptor" else None)
            forced = _accepts_from_cation(atoms, m)
            # Capability wins: a Lys/Arg donating in forces the atom to acceptor, overriding geometry.
            states.append("deprotonated" if forced else (geom_state or "ambiguous"))
            geoms.append("deprotonated" if forced else geom_state)

        labels[key] = _label_from_atoms(res_name, states[0], geoms[0], states[1], geoms[1])

    if use_salt_bridge:
        labels = _apply_salt_bridge_fallback(aa, labels)
    return _apply_carboxyl_dyads(aa, labels, deterministic=deterministic,
                                 temperature=temperature, rng=rng, cutoff=dyad_cutoff)
