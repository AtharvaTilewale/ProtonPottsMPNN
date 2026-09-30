"""EV5-strict — THE canonical rule definition. Every other script should import from here.

Earlier stages drifted: some counted H-bond DONOR/ACCEPTOR roles per BOND, others per ATOM, and the
HIS numbers wobbled between 133/64% and 138/63% as a result. This module is the single source of truth.
Import `label_his` / `label_acid`, or run it to print the reference decision tree with live numbers.

Every input is a heavy-atom geometric measurement on PROTEIN atoms only — no metals, no ligands, no
cofactors, no waters, no pKa. HBPLUS supplies H-bond geometry (it places the hydrogens itself).
"""
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["pdb", "chain", "res_id"]

# ── H-bond geometry ──────────────────────────────────────────────────────────────────────
# v4 uses H-A 2.5 / D-A 3.5 / DHA >= 90. Its DHA floor of 90 deg is no filter at all -- it admits
# near-perpendicular contacts that are not H-bonds, and that single number is why v4's HIS rule scores
# at chance (MCC 0.013). Raising it is the whole fix. The DISTANCE cuts, by contrast, turn out not to
# matter inside the rules (stage 24: H-A 2.0 -> 3.2 gives identical results), because the dyad / ion-pair
# terms already force the heavy atoms close. They are kept only to stay inside HBPLUS's sane range.
HA_MAX, DA_MAX = 2.2, 3.2
DHA_MIN = 140.0

# ── charge field (HIS only) ─────────────────────────────────────────────────────────────
# Phi = sum over protein charge sites within PHI_RC of a functional atom, of q / d.
# Fitted against neutron truth (stage 15). ev4 uses q_his = +0.1, which cripples it: AUC 0.542 (chance)
# vs 0.628 at q_his = +1.0. A His in a CATION-RICH pocket cannot be charged -- the pocket already paid.
PHI_RC, Q_CAT, Q_HIS, Q_ACID = 6.0, 1.0, 1.0, -0.5

# ── the thresholds, and what the bootstrap said about each ──────────────────────────────
HIS_DCARB = 2.8      # chosen by 99% of bootstrap resamples
HIS_PHI = 0.0        # 55% chose 0.0; +/-0.05 took the rest
ACID_DYAD = 2.8      # 96% of resamples
ACID_BURIAL = 250    # 60% chose 250, 30% chose 300, 10% chose 200  <- the ONLY unstable threshold
ACID_NETQ = 0        # 100% of resamples

CARBOX_O = {"OD1", "OD2", "OE1", "OE2"}
RING_N = {"ND1", "NE2"}


# ═══════════════════════════════════════════════════════════════════════════════════════
def hb_roles(bonds, res_names, use_override):
    """Per-RESIDUE counts of functional ATOMS that donate / accept an H-bond passing the geometry.

    Counted per ATOM, not per bond: a ring N with three separate H-bonds still donates from ONE
    nitrogen, and the rule asks "do BOTH ring N carry a proton", which is an atom question.
    """
    b = bonds[bonds.res_name.isin(res_names) &
              (bonds.ha <= HA_MAX) & (bonds.da <= DA_MAX) & (bonds.dha >= DHA_MIN)]
    if not use_override:
        b = b[b["mode"] == "default"]        # HBPLUS's own chemistry never lets a carboxyl O donate
    b = b.assign(don=(b.role == "donor").astype(int),
                 acc=(b.role == "acceptor").astype(int),
                 ionic=((b.role == "donor") & b.p_atom.isin(CARBOX_O)).astype(int))
    per_atom = b.groupby(KEY + ["atom"])[["don", "acc", "ionic"]].max()
    return per_atom.groupby(KEY).agg(n_donor=("don", "sum"), n_acceptor=("acc", "sum"),
                                     n_ionic=("ionic", "sum"))


def charge_field(ctx):
    """Phi at each titratable residue = the MINIMUM over its two functional atoms (the atom best able
    to hold a proton decides the residue). Each neighbouring residue contributes ONCE, at its closest
    charge-site atom. A-priori charges only -- no protonation label is ever consulted, so Phi is not
    circular."""
    q = np.where(ctx.n_res_name.isin(("LYS", "ARG")), Q_CAT,
                 np.where(ctx.n_res_name == "HIS", Q_HIS,
                          np.where(ctx.n_res_name.isin(("ASP", "GLU")), Q_ACID, 0.0)))
    v = np.where(ctx.d.values <= PHI_RC, q / ctx.d.values, 0.0)
    return (pd.DataFrame(dict(pdb=ctx.pdb, chain=ctx.chain, res_id=ctx.res_id, atom=ctx.atom, v=v))
            .groupby(KEY + ["atom"]).v.sum().groupby(KEY).min().rename("phi"))


# ═══════════════════════════════ THE RULES ═══════════════════════════════
def label_his(d):
    """HIS-P iff (A or B) and the pocket is net-anionic.

    A  both ring N DONATE and NEITHER accepts     -- no lone pair anywhere, so both N carry a proton.
                                                     A NEUTRAL His has exactly one ring N-H, so one N
                                                     always keeps a lone pair. This is the sharp test.
    B  a carboxylate O within 2.8 A of a ring N   -- a genuine ionic N-H...O(-) contact. NOT v3's
                                                     "a carboxylate somewhere within 5.5 A", and NOT
                                                     v4's "a ring N donates" (a neutral His does too).
    Phi < 0  no cation has already paid for the charge.
    """
    A = (d.n_donor >= 2) & (d.n_acceptor == 0)
    B = d.dcarb_min < HIS_DCARB
    return ((A | B) & (d.phi < HIS_PHI)).values, A.values, B.values


def label_acid(d):
    """*-P iff ALL FOUR hold. The chemistry is a conjunction and the data insists on it: an additive
    log-odds model halves the precision (50% -> 25%) because it gives partial credit for 3-of-4, and
    3-of-4 is only 10% precise (stages 23, 29).

    1 a carboxyl O DONATES an H-bond   -- the proton has to BE somewhere. Only HBPLUS's override pass
                                          can ever produce this; the default pass can never propose *-P.
    2 another carboxylate O < 2.8 A    -- and it needs somewhere to GO (the shared-proton dyad).
    3 >= 250 heavy atoms within 12 A   -- the charge must be EXPENSIVE (desolvated).
    4 net charge within 8 A < 0        -- and nothing must have NEUTRALISED it already.
    """
    c1 = d.n_donor >= 1
    c2 = d.d_dyad < ACID_DYAD
    c3 = d.burial12 > ACID_BURIAL
    c4 = d.net_q8 < ACID_NETQ
    return (c1 & c2 & c3 & c4).values, (c1.values, c2.values, c3.values, c4.values)


# ═══════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from sklearn.metrics import matthews_corrcoef
    HERE = Path(__file__).parent
    DATA = HERE / "data"               # the shared parquet inputs; this file sits ABOVE them
    B = pd.read_parquet(DATA / "hbonds.parquet")
    C = pd.read_parquet(DATA / "charge_ctx.parquet")
    F = pd.read_parquet(DATA / "features.parquet").set_index(KEY)
    F = F.drop(columns=["n_donor", "n_acceptor"])
    phi = charge_field(C)

    H = F[F.res_name == "HIS"].join(hb_roles(B, ("HIS",), False)).join(phi).fillna(
        {"n_donor": 0, "n_acceptor": 0, "n_ionic": 0, "phi": 0.0})
    H["y"] = (H.truth == "HIS-P").astype(int)
    A_ = F[F.res_name.isin(("ASP", "GLU"))].join(hb_roles(B, ("ASP", "GLU"), True)).join(phi).fillna(
        {"n_donor": 0, "n_acceptor": 0, "n_ionic": 0, "phi": 0.0})
    A_["y"] = A_.truth.str.endswith("-P").astype(int)

    his_p, detA, detB = label_his(H)
    acid_p, (k1, k2, k3, k4) = label_acid(A_)

    def node(lbl, m, y, depth, leaf=""):
        n = int(m.sum()); tp = int(y[m].sum())
        pad = "   " * depth
        tag = f"   ==> {leaf}" if leaf else ""
        print(f"  {pad}{lbl:<{54 - 3*depth}} n={n:>5}   protonated {tp:>4} "
              f"({100*tp/max(n,1):>3.0f}%){tag}")

    yH, yA = H.y.values, A_.y.values
    print("=" * 116)
    print(f"EV5-strict  ·  HIS        H-bond: HBPLUS default pass, H-A<={HA_MAX} D-A<={DA_MAX} "
          f"DHA>={DHA_MIN:.0f}deg")
    print(f"                          Phi:    sum q/d within {PHI_RC:.0f}A "
          f"(Lys/Arg {Q_CAT:+.0f}, His {Q_HIS:+.0f}, Asp/Glu {Q_ACID:+.1f})")
    print("=" * 116)
    node("ALL HIS", np.ones(len(H), bool), yH, 0)
    node("A: both ring N DONATE & NEITHER accepts", detA, yH, 1)
    node("B: carboxylate O < 2.8 A of a ring N", detB, yH, 1)
    node("A or B", detA | detB, yH, 1)
    node("...and Phi >= 0   (cation-rich pocket)", (detA | detB) & ~(H.phi < HIS_PHI).values, yH, 2,
         "HIS-S")
    node("...and Phi <  0   (net-anionic pocket)", his_p, yH, 2, "HIS-P")
    node("neither A nor B", ~(detA | detB), yH, 1, "HIS-S")
    print(f"\n  HIS-P: {int(his_p.sum())} labels, {int(yH[his_p].sum())} right "
          f"({100*yH[his_p].sum()/max(his_p.sum(),1):.0f}% precision, "
          f"{100*yH[his_p].sum()/yH.sum():.0f}% recall), MCC {matthews_corrcoef(yH, his_p):.3f}")

    print("\n" + "=" * 116)
    print(f"EV5-strict  ·  ASP / GLU  H-bond: HBPLUS + carboxyl-donor OVERRIDE, same geometry")
    print("=" * 116)
    node("ALL ASP + GLU", np.ones(len(A_), bool), yA, 0)
    node("1. a carboxyl O DONATES an H-bond", k1, yA, 1)
    node("2. + another carboxylate O < 2.8 A", k1 & k2, yA, 2)
    node("3. + >= 250 heavy atoms within 12 A", k1 & k2 & k3, yA, 3)
    node("4. + net charge within 8 A < 0", acid_p, yA, 4, "*-P")
    node("anything else", ~acid_p, yA, 1, "*-D")
    print(f"\n  *-P: {int(acid_p.sum())} labels, {int(yA[acid_p].sum())} right "
          f"({100*yA[acid_p].sum()/max(acid_p.sum(),1):.0f}% precision, "
          f"{100*yA[acid_p].sum()/yA.sum():.0f}% recall), MCC {matthews_corrcoef(yA, acid_p):.3f}")
