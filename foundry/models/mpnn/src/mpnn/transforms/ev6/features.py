"""EV6 — the protonation feature engine.

Given a protein structure (a biotite ``AtomArray``), compute the EXACT feature vectors the AutoML models
were trained on: 138 columns for HIS, 141 for ASP/GLU. This is a faithful port of sandbox EV5 stages 47
(``pdb_his_features``) and 49 (``pdb_acid_features``) — the single-pass "structure in → feature vector
out" reference — with one deliberate change: H-bond evidence comes from foundry's
``bond_annotation.calculate_hbonds`` (extended to carry H···A) instead of a private HBPLUS invocation, so
there is ONE HBPLUS code path in the repo. The override pool it produces was measured identical to the
sandbox's three-pass pool (see ``_hbond_pool``).

Inside the pipeline the pool is not computed here at all: CalculateHbondsPlus runs it upstream at v6's
cutoffs and stashes it on ``data["hbond_records"]``, which arrives as the ``hbond_records`` argument.
Both entrypoints still run HBPLUS themselves when it is None, so a bare AtomArray works standalone.

All features are heavy-atom geometry on the 20 canonical amino acids only — no waters, ligands, cofactors
or metals enter a feature. The ground-truth H/D labels that the sandbox read from neutron structures are
NOT computed here: at inference there are no observed hydrogens, which is the whole point.

Public API:
    his_features(atom_array)  -> DataFrame  (one row per HIS,     138 model cols + keys + metal flags)
    acid_features(atom_array) -> DataFrame  (one row per ASP/GLU, 141 model cols + keys + metal flags)

The frames carry `chain, res_id` (+ `res_name` for acids) and `d_metal, metal_adjacent`. The predictor
reindexes to the model's stored `feature_columns` and never feeds it the id/metal columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import biotite.structure as struc

from mpnn.transforms.bond_annotation import calculate_hbonds

# ── constants (verbatim from sandbox stages 47/49) ───────────────────────────────────────────────
METAL_CUT = 3.5                                 # a functional atom this close to a metal is out-of-domain
HA_MAX, DA_MAX = 3.2, 4.0                        # permissive HBPLUS pool; the model re-thresholds
PHI_RC, Q_CAT, Q_HIS, Q_ACID = 6.0, 1.0, 1.0, -0.5     # the fitted a-priori charge field

AA20 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
        "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
METALS = {"ZN", "CU", "FE", "MN", "NI", "CO", "MG", "CA", "CD", "HG"}
CLASS = {**{r: "acidic" for r in ("ASP", "GLU")}, **{r: "basic" for r in ("LYS", "ARG")},
         **{r: "polar" for r in ("SER", "THR", "ASN", "GLN", "TYR", "CYS")},
         **{r: "aromatic" for r in ("PHE", "TRP", "TYR")},
         **{r: "aliphatic" for r in ("ALA", "VAL", "LEU", "ILE", "MET")},
         **{r: "glypro" for r in ("GLY", "PRO")}, "HIS": "his"}
CLASSES = ("acidic", "basic", "polar", "aromatic", "aliphatic", "glypro", "his")
CARBOX = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}
CARBOX_O = {"OD1", "OD2", "OE1", "OE2"}
CATION_N = {"NZ", "NE", "NH1", "NH2"}
HYDROXYL = {("SER", "OG"), ("THR", "OG1"), ("TYR", "OH")}
AMIDE_O = {("ASN", "OD1"), ("GLN", "OE1")}
SULFUR = {("CYS", "SG"), ("MET", "SD")}
RING_N = {("HIS", "ND1"), ("HIS", "NE2")}
AROM_RINGS = {"PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
              "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
              "TRP": ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2")}
CHARGE_SITES = {"HIS": ("ND1", "NE2"), "ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2"),
                "LYS": ("NZ",), "ARG": ("NE", "NH1", "NH2")}
PCLS = ("carboxylate", "bbcarbonyl", "bbamide", "cation", "hydroxyl", "amide", "sulfur", "hisN", "other")


def _partner_class(prn: str, pat: str) -> str:
    """Classify the OTHER end of an H-bond by residue+atom (sandbox 47:62-79, verbatim)."""
    if pat in CARBOX_O:
        return "carboxylate"
    if pat in ("O", "OXT"):
        return "bbcarbonyl"
    if pat == "N":
        return "bbamide"
    if pat in CATION_N:
        return "cation"
    if (prn, pat) in HYDROXYL:
        return "hydroxyl"
    if (prn, pat) in AMIDE_O:
        return "amide"
    if (prn, pat) in SULFUR:
        return "sulfur"
    if (prn, pat) in RING_N:
        return "hisN"
    return "other"


# ── H-bond evidence, via foundry's HBPLUS ────────────────────────────────────────────────────────
# The pool EV6's models were trained on, as `calculate_hbonds` arguments. The pipeline's
# CalculateHbondsPlus is configured from extended_vocab_v6's constants, so for v6 it runs EXACTLY this
# and we can reuse its bonds; `_hbond_pool` asserts the match rather than trusting that wiring.
POOL_PARAMS = dict(cutoff_HA_dist=HA_MAX, cutoff_DA_distance=DA_MAX, motif_interface_only=False,
                   od1_oe1=True, od2_oe2=True, filter_capability=False)


def _chain_key(atom_array) -> np.ndarray:
    """The per-atom chain label the H-bond pool is keyed by.

    `calculate_hbonds` reports every bond under the donor/acceptor's ``chain_iid``, so the feature loop
    must join on the same thing. Pipeline arrays carry chain_iid; a bare structure load may not, and then
    chain_id is what the sandbox keyed on. Getting this wrong does not raise — it just matches zero bonds
    and hands the models an all-empty H-bond block."""
    if "chain_iid" in atom_array.get_annotation_categories():
        return np.array([str(x) for x in atom_array.chain_iid])
    return np.array([(str(x).strip() or "A") for x in atom_array.chain_id])


def _hbond_pool(prot, records: dict | None) -> list[dict]:
    """The merged default + combined-override H-bond pool, at EV6's cutoffs.

    `records` is CalculateHbondsPlus's stashed pool (``data["hbond_records"]``): the pipeline already ran
    these two HBPLUS passes for a v6 structure, so reusing them takes v6 from 5 subprocess calls per
    structure to 2. Its `params` are CHECKED, not assumed — features built from a tighter pool (v4's
    2.5/3.5, say) are silently off-distribution for models trained at 3.2/4.0, which is precisely the
    class of failure that stays invisible until the predictions are quietly wrong.

    `records=None` runs HBPLUS here (standalone use: the sandbox parity check, a bare AtomArray).

    The combined override (`od1_oe1` and `od2_oe2` in one pass) was MEASURED identical to the sandbox's
    three separate override passes (12 structures, 233 carboxyl-O donor bonds, 0 disagreements — HBPLUS's
    `-E` is per-atom-independent).

    Foundry MERGES its passes (one row per bond), so a carboxyl-O ACCEPTOR bond is counted ONCE. The
    training table ``acid_rich.parquet`` (stage 43) instead counts it once PER mode-tagged pass (3x),
    because it aggregates the raw multi-pass dump without cross-mode dedup — so its acceptor-count columns
    (``*_acceptor_n``, ``n_hbonds``, the acceptor partner-class counts) run ~3x higher. This looks alarming
    but is inert at the decision: measured across 776 acids, the deduped-vs-3x-counted features produce
    ZERO token flips (max |Δp| 2.8e-3, never crossing the 0.06 cut; those columns are low-importance).
    EV6 keeps the correct deduped count."""
    if records is None:
        aa = prot.copy()
        aa.set_annotation("chain_iid", _chain_key(aa))
        _, bonds, _ = calculate_hbonds(aa, **POOL_PARAMS)
        return bonds

    got = records["params"]
    bad = {k: (got.get(k), v) for k, v in POOL_PARAMS.items() if got.get(k) != v}
    if bad:
        raise ValueError(
            "EV6 was handed an H-bond pool built with settings its models were not trained on: "
            + ", ".join(f"{k}={g!r} (need {w!r})" for k, (g, w) in bad.items())
            + ". The upstream CalculateHbondsPlus must be configured from extended_vocab_v6's constants."
        )
    return records["bonds"]


def _his_bonds(prot, pool: list[dict]) -> pd.DataFrame:
    """Per-bond table at HIS ring N (default pass). Cols: chain,res_id,slot(nd1/ne2),role,pcls,da,dha,ha.

    HIS was trained on the DEFAULT pass only, so the override-only bonds (carboxyl O acting as a donor)
    are filtered out — a HIS ring N accepting from an ASP/GLU OD/OE exists only in the override pass and
    would be a bond the models never saw. `_merge_hbonds` keeps the default pass's geometry for bonds
    found in both, so this filter reproduces a standalone default-pass run exactly."""
    rows = []
    for b in pool:
        if "default" not in b["hbplus_modes"].split(","):
            continue
        for side, other in (("d", "a"), ("a", "d")):
            at = b[f"{side}_atom"]
            if b[f"{side}_resn"] == "HIS" and at in ("ND1", "NE2"):
                rows.append(dict(
                    chain=b[f"{side}_chain"], res_id=int(b[f"{side}_resi"]),
                    slot="nd1" if at == "ND1" else "ne2",
                    role="donor" if side == "d" else "acceptor",
                    pcls=_partner_class(b[f"{other}_resn"], b[f"{other}_atom"]),
                    da=b["dist"], dha=b["dha_angle"], ha=b["ha_dist"]))
    return pd.DataFrame(rows, columns=["chain", "res_id", "slot", "role", "pcls", "da", "dha", "ha"])


def _acid_bonds(prot, pool: list[dict]) -> pd.DataFrame:
    """Per-bond table at carboxyl O (default+override). Cols: chain,res_id,slot(o1/o2),role,pcls,da,dha,ha.

    Unlike HIS this uses the WHOLE pool: the override pass is the only source of the carboxyl-O donor
    role, which is the entire point of running it."""
    rows = []
    for b in pool:
        for side, other in (("d", "a"), ("a", "d")):
            rr, at = b[f"{side}_resn"], b[f"{side}_atom"]
            if rr in CARBOX and at in CARBOX[rr]:
                rows.append(dict(
                    chain=b[f"{side}_chain"], res_id=int(b[f"{side}_resi"]),
                    slot="o1" if at in ("OD1", "OE1") else "o2",
                    role="donor" if side == "d" else "acceptor",
                    pcls=_partner_class(b[f"{other}_resn"], b[f"{other}_atom"]),
                    da=b["dist"], dha=b["dha_angle"], ha=b["ha_dist"]))
    cols = ["chain", "res_id", "slot", "role", "pcls", "da", "dha", "ha"]
    return pd.DataFrame(rows, columns=cols).drop_duplicates()


def _hbond_features(rec: dict, h: pd.DataFrame, slots: tuple, slot_col: str) -> None:
    """Aggregate a residue's H-bonds into the 77 columns, per functional-atom slot (sandbox 47/49).

    Counted per ATOM without re-thresholding: count, dha MAX, ha MIN, da MIN, and per-partner-class
    COUNTS; then the symmetric `n_donor_atoms / best_* / donor_<class>` summaries."""
    for tag in slots:
        for role in ("donor", "acceptor"):
            s = h[(h[slot_col] == tag) & (h.role == role)] if len(h) else h
            rec[f"{tag}_{role}_n"] = len(s)
            rec[f"{tag}_{role}_dha"] = float(s.dha.max()) if len(s) else 0.0
            rec[f"{tag}_{role}_ha"] = float(s.ha.min()) if len(s) else 9.0
            rec[f"{tag}_{role}_da"] = float(s.da.min()) if len(s) else 9.0
            for pc in PCLS:
                rec[f"{tag}_{role}_{pc}"] = int((s.pcls == pc).sum()) if len(s) else 0
    for role in ("donor", "acceptor"):
        a = int(rec[f"{slots[0]}_{role}_n"] > 0)
        b = int(rec[f"{slots[1]}_{role}_n"] > 0)
        rec[f"n_{role}_atoms"] = a + b
        rec[f"best_{role}_dha"] = max(rec[f"{slots[0]}_{role}_dha"], rec[f"{slots[1]}_{role}_dha"])
        rec[f"best_{role}_ha"] = min(rec[f"{slots[0]}_{role}_ha"], rec[f"{slots[1]}_{role}_ha"])
        for pc in PCLS:
            rec[f"{role}_{pc}"] = rec[f"{slots[0]}_{role}_{pc}"] + rec[f"{slots[1]}_{role}_{pc}"]
    rec["n_hbonds"] = sum(rec[f"{t}_{r}_n"] for t in slots for r in ("donor", "acceptor"))


# ── shared structure prep ────────────────────────────────────────────────────────────────────────
class _Prep:
    """Heavy-atom AA20 protein + the arrays every feature reads. Mirrors sandbox 47:117-164 / 49."""

    def __init__(self, atom_array):
        heavy = atom_array[~np.isin(atom_array.element, ["H", "D"])]
        rn_all = np.array([str(x) for x in heavy.res_name])
        self.prot = heavy[np.isin(rn_all, list(AA20))]
        self.met_xyz = heavy.coord[
            np.isin([str(x).strip().upper() for x in heavy.element], list(METALS))]

        prot = self.prot
        self.nm = np.array([str(x).strip() for x in prot.atom_name])
        self.rn = np.array([str(x) for x in prot.res_name])
        # `ch` identifies a residue (masks + the emitted key, which AnnotateProtonationStates looks up as
        # str(chain_id)); `ch_bond` is what the H-bond pool reports and is used ONLY to join bonds back.
        # They differ in the pipeline, where the pool is keyed by chain_iid rather than chain_id.
        self.ch = np.array([(str(x).strip() or "A") for x in prot.chain_id])
        self.ch_bond = _chain_key(prot)
        self.ri = np.array(prot.res_id)
        self.xyz = prot.coord
        self.is_sc = ~np.isin(self.nm, ["N", "CA", "C", "O", "OXT"])

        # Atom-class masks. These depend only on the structure, never on the residue being scored, so
        # they are built once here rather than per residue inside the feature loops -- the comprehensions
        # cost ~865 us over ~3k atoms each, 14x the full distance pass they feed, and the loops queried
        # them ~4x per titratable residue (measured 3.6 ms -> 0.07 ms per acid; the loop's dominant term).
        self.m_carbox = np.isin(self.nm, list(CARBOX_O))
        self.m_cation = np.isin(self.nm, list(CATION_N))
        self.m_bbO = np.isin(self.nm, ["O", "OXT"])
        self.m_bbN = self.nm == "N"
        self.m_hydroxyl = np.array([(a, b) in HYDROXYL for a, b in zip(self.rn, self.nm)], dtype=bool)
        self.m_amide = np.array([(a, b) in AMIDE_O for a, b in zip(self.rn, self.nm)], dtype=bool)
        self.m_sulfur = np.array([(a, b) in SULFUR for a, b in zip(self.rn, self.nm)], dtype=bool)
        self.m_hisN = np.array([(a, b) in RING_N for a, b in zip(self.rn, self.nm)], dtype=bool)
        try:
            self.sasa = np.nan_to_num(struc.sasa(prot, vdw_radii="Single"))
        except Exception:
            self.sasa = np.zeros(len(prot))
        try:
            self.sse = struc.annotate_sse(prot)
        except Exception:
            self.sse = None
        self.starts = struc.get_residue_starts(prot)

        # aromatic ring centroids (>=5 ring atoms), for the d_arom stacking term
        arom = []
        for (c, i, r) in {(a, int(b), c_) for a, b, c_ in zip(self.ch, self.ri, self.rn)
                          if c_ in AROM_RINGS}:
            m = (self.ch == c) & (self.ri == i) & np.isin(self.nm, list(AROM_RINGS[r]))
            if m.sum() >= 5:
                arom.append(self.xyz[m].mean(0))
        self.arom = np.array(arom) if arom else np.zeros((0, 3))

        # a-priori charge sites, for the fitted field Phi
        site_mask = np.array([(a in CHARGE_SITES and b in CHARGE_SITES[a])
                              for a, b in zip(self.rn, self.nm)])
        self.sk, self.sx, sq = [], [], []
        for c, i, r in {(a, int(b), c_) for a, b, c_
                        in zip(self.ch[site_mask], self.ri[site_mask], self.rn[site_mask])}:
            m = site_mask & (self.ch == c) & (self.ri == i)
            self.sk.append((c, i)); self.sx.append(self.xyz[m])
            sq.append(Q_CAT if r in ("LYS", "ARG") else (Q_HIS if r == "HIS" else Q_ACID))
        self.sq = np.array(sq)

    def common(self, rec: dict, key: tuple, own, cen, atoms_xyz: list, idx: int) -> None:
        """Density, neighbourhood composition, SSE/position, sequence context — identical for both
        residue types (sandbox 47:178-220). `atoms_xyz` = the residue's functional-atom coords."""
        xyz, ch, ri, rn, nm, is_sc = self.xyz, self.ch, self.ri, self.rn, self.nm, self.is_sc
        d_cen = np.linalg.norm(xyz - cen, axis=1)
        for r_ in (4, 6, 8, 10, 12):
            rec[f"dens{r_}"] = int((d_cen < r_).sum())
        near8 = d_cen < 8
        rec["sc_frac8"] = float(is_sc[near8].mean()) if near8.any() else 0.0

        for r_ in (6, 8, 10):
            near = d_cen < r_
            seen = {(c, int(i)): rr for c, i, rr in zip(ch[near], ri[near], rn[near])}
            cnt = {c_: 0 for c_ in CLASSES}
            for (c, i), rr in seen.items():
                if (c, i) == key:
                    continue
                cnt[CLASS.get(rr, "aliphatic")] += 1
            for c_ in CLASSES:
                rec[f"n_{c_}{r_}"] = cnt[c_]
            rec[f"n_res{r_}"] = sum(cnt.values())
            rec[f"net_q{r_}"] = cnt["basic"] - cnt["acidic"]

        rec["sse"] = str(self.sse[idx]) if (self.sse is not None and idx < len(self.sse)) else "c"
        rec["rel_pos"] = float(idx) / max(len(self.starts), 1)
        rec["chain_len"] = int((ch[self.starts] == key[0]).sum())
        prot = self.prot
        for off in (-2, -1, 1, 2):
            j = idx + off
            rec[f"seq{off:+d}"] = (CLASS.get(str(prot.res_name[self.starts[j]]), "aliphatic")
                                   if 0 <= j < len(self.starts) else "none")

    def near_to(self, atoms_xyz: list, mask=None, coords=None) -> float:
        """Min distance from any functional atom to the nearest selected atom (99.0 if none)."""
        src = coords if coords is not None else self.xyz[mask]
        if not len(src):
            return 99.0
        return float(min(np.linalg.norm(src - a, axis=1).min() for a in atoms_xyz))

    def phi(self, rec: dict, key: tuple, tags_xyz: list) -> None:
        """Fitted charge field at each functional atom: each neighbour residue contributes q/d once, at
        its closest charge-site atom, truncated at rc (sandbox 47:222-238)."""
        names = [t[0] for t in tags_xyz]
        for tag, a_xyz in tags_xyz:
            for rc in (6.0, 8.0):
                tot = 0.0
                for (c, i), x_, q_ in zip(self.sk, self.sx, self.sq):
                    if (c, i) == key:
                        continue
                    d = float(np.linalg.norm(x_ - a_xyz, axis=1).min())
                    if d <= rc:
                        tot += q_ / d
                rec[f"phi_{tag}_{rc:.0f}"] = tot
        for rc in (6, 8):
            a, b = rec[f"phi_{names[0]}_{rc}"], rec[f"phi_{names[1]}_{rc}"]
            rec[f"phi_min_{rc}"] = min(a, b)
            rec[f"phi_max_{rc}"] = max(a, b)
            rec[f"phi_diff_{rc}"] = abs(a - b)


# ── HIS ──────────────────────────────────────────────────────────────────────────────────────────
def his_features(atom_array, hbond_records: dict | None = None, prep: "_Prep | None" = None) -> pd.DataFrame:
    """One row of 138 model features per HIS (+ chain,res_id,d_metal,metal_adjacent). Port of stage 47.

    `hbond_records`: the pipeline's stashed H-bond pool (see `_hbond_pool`); None runs HBPLUS here.
    `prep`: a `_Prep` to reuse. It depends only on the structure, so scoring both residue types builds it
    twice unless shared -- and it runs biotite's SASA (~90 ms). The predictor passes one in."""
    P = prep if prep is not None else _Prep(atom_array)
    if not (P.rn == "HIS").any():
        return pd.DataFrame()
    HB = _his_bonds(P.prot, _hbond_pool(P.prot, hbond_records))
    xyz, ch, ri, nm = P.xyz, P.ch, P.ri, P.nm
    rows = []
    for idx, s in enumerate(P.starts):
        if str(P.prot.res_name[s]) != "HIS":
            continue
        key = (ch[s], int(ri[s]))
        own = (ch == key[0]) & (ri == key[1])
        rx = {a: xyz[own & (nm == a)] for a in ("ND1", "NE2")}
        if any(len(v) == 0 for v in rx.values()):
            continue
        nd1, ne2 = rx["ND1"][0], rx["NE2"][0]
        cen = np.vstack([nd1, ne2]).mean(0)

        dmet = 99.0
        if len(P.met_xyz):
            dmet = float(min(np.linalg.norm(P.met_xyz - a, axis=1).min() for a in (nd1, ne2)))
        rec = dict(pdb="", chain=key[0], res_id=key[1], d_metal=dmet, metal_adjacent=int(dmet < METAL_CUT))

        P.common(rec, key, own, cen, [nd1, ne2], idx)
        rec["sasa_sc"] = float(P.sasa[own & P.is_sc].sum())
        rec["sasa_ring"] = float(P.sasa[own & np.isin(nm, ["ND1", "NE2"])].sum())

        rec["d_carbox"] = P.near_to([nd1, ne2], P.m_carbox & ~own)
        rec["d_cation"] = P.near_to([nd1, ne2], P.m_cation & ~own)
        rec["d_hydroxyl"] = P.near_to([nd1, ne2], P.m_hydroxyl & ~own)
        rec["d_amide"] = P.near_to([nd1, ne2], P.m_amide & ~own)
        rec["d_sulfur"] = P.near_to([nd1, ne2], P.m_sulfur & ~own)
        rec["d_bbO"] = P.near_to([nd1, ne2], P.m_bbO & ~own)
        rec["d_bbN"] = P.near_to([nd1, ne2], P.m_bbN & ~own)
        rec["d_hisN"] = P.near_to([nd1, ne2], P.m_hisN & ~own)
        rec["d_arom"] = P.near_to([nd1, ne2], coords=P.arom) if len(P.arom) else 99.0

        P.phi(rec, key, [("nd1", nd1), ("ne2", ne2)])
        h = HB[(HB.chain == P.ch_bond[s]) & (HB.res_id == key[1])] if len(HB) else HB
        _hbond_features(rec, h, ("nd1", "ne2"), "slot")
        rows.append(rec)

    G = pd.DataFrame(rows)
    return G.drop(columns=["pdb"]) if len(G) else G


# ── ASP / GLU ──────────────────────────────────────────────────────────────────────────────────
def acid_features(atom_array, hbond_records: dict | None = None, prep: "_Prep | None" = None) -> pd.DataFrame:
    """One row of 141 model features per ASP/GLU (+ ids, d_metal, metal_adjacent). Port of stage 49.

    `hbond_records`: the pipeline's stashed H-bond pool (see `_hbond_pool`); None runs HBPLUS here.
    `prep`: a `_Prep` to reuse across both residue types (see `his_features`)."""
    P = prep if prep is not None else _Prep(atom_array)
    if not np.isin(P.rn, ["ASP", "GLU"]).any():
        return pd.DataFrame()
    HB = _acid_bonds(P.prot, _hbond_pool(P.prot, hbond_records))
    xyz, ch, ri, nm = P.xyz, P.ch, P.ri, P.nm
    rows = []
    for idx, s in enumerate(P.starts):
        r = str(P.prot.res_name[s])
        if r not in CARBOX:
            continue
        key = (ch[s], int(ri[s]))
        own = (ch == key[0]) & (ri == key[1])
        o_names = CARBOX[r]
        ox = {a: xyz[own & (nm == a)] for a in o_names}
        if any(len(v) == 0 for v in ox.values()):
            continue
        o1, o2 = ox[o_names[0]][0], ox[o_names[1]][0]
        cen = (o1 + o2) / 2

        dmet = 99.0
        if len(P.met_xyz):
            dmet = float(min(np.linalg.norm(P.met_xyz - o1, axis=1).min(),
                             np.linalg.norm(P.met_xyz - o2, axis=1).min()))
        rec = dict(pdb="", chain=key[0], res_id=key[1], res_name=r, d_metal=dmet,
                   metal_adjacent=int(dmet < METAL_CUT), is_glu=int(r == "GLU"))

        P.common(rec, key, own, cen, [o1, o2], idx)
        rec["sasa_sc"] = float(P.sasa[own & P.is_sc].sum())
        rec["sasa_o"] = float(P.sasa[own & np.isin(nm, list(o_names))].sum())

        # `& ~own` on the dyad and backbone terms only -- the acid loop deliberately lets cation/
        # hydroxyl/amide/hisN see the residue's own atoms where the HIS loop excludes them. Faithful to
        # sandbox stage 49; the models are calibrated on it, so do not "fix" the asymmetry.
        rec["d_dyad"] = P.near_to([o1, o2], P.m_carbox & ~own)                      # THE dyad
        rec["d_cation"] = P.near_to([o1, o2], P.m_cation)
        rec["d_hydroxyl"] = P.near_to([o1, o2], P.m_hydroxyl)
        rec["d_amide"] = P.near_to([o1, o2], P.m_amide)
        rec["d_sulfur"] = P.near_to([o1, o2], P.m_sulfur)
        rec["d_hisN"] = P.near_to([o1, o2], P.m_hisN)
        rec["d_bbO"] = P.near_to([o1, o2], P.m_bbO & ~own)
        rec["d_bbN"] = P.near_to([o1, o2], P.m_bbN & ~own)
        rec["d_arom"] = P.near_to([o1, o2], coords=P.arom) if len(P.arom) else 99.0
        rec["d_oo"] = float(np.linalg.norm(o1 - o2))

        P.phi(rec, key, [("o1", o1), ("o2", o2)])
        h = HB[(HB.chain == P.ch_bond[s]) & (HB.res_id == key[1])] if len(HB) else HB
        _hbond_features(rec, h, ("o1", "o2"), "slot")
        rec["strict_cooh"] = int((rec["n_donor_atoms"] >= 1) and (rec["n_acceptor_atoms"] >= 1))
        rows.append(rec)

    G = pd.DataFrame(rows)
    return G.drop(columns=["pdb"]) if len(G) else G
