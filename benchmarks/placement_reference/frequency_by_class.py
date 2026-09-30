"""Run the placement SCAN on each backbone: which class (interface / core / surface) does the algorithm
choose to place each protonation type (HIS-S, HIS-P, ASP-P, GLU-P) in?

For every design backbone (both campaigns) we (1) classify each binder residue interface/core/surface
(biotite), then (2) run the actual placement scan the pipeline uses — for each state we score placing it
at EVERY free binder position with both scorers (`scan_potts` = Potts energy field, `scan_mpnn` = PottsMPNN
decoder field, binder masked) exactly like `_ranked_candidates`:
    score(pos) = field(S)[pos, STATE] - field(S)[pos, native]  -  min_d(field(S)[pos, DEP])   (lower=better)
and take the algorithm's top choice (rank-0). We tally which class the top choice falls in, over all
backbones, per state and scorer. `random` baseline = uniform over free positions.

Classification (biotite only, on the ORIGINAL binder = origin_pdb_path):
  interface : binder CA within IFACE_DIST A of any target CA
  core      : not interface AND relative RASA < RASA_THR
  surface   : not interface AND relative RASA >= RASA_THR
  relative RASA = residue SASA / max residue SASA in that binder  (biotite struc.sasa; TABLE-FREE, no max-ASA lookup)

Sharded (needs the v6 model for both fields, CPU):
    cd <path-to-internal-project-root>
    SHARD=0 NUM_SHARDS=40 ./dpo_potts/bin/python sandbox/placement/frequency_by_class.py
    AGGREGATE=1 ./dpo_potts/bin/python sandbox/placement/frequency_by_class.py     # -> parquet + plot
Non-destructive.
"""
# %% ------------------------------------------------------------------ config
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("PROJECT_ROOT", "."))
sys.path.insert(0, str(ROOT))
OUTDIR = ROOT / "sandbox/placement"
FIGDIR = OUTDIR / "figures"
SHARD_DIR = ROOT / "data/internal/placement_scan_shards"
OUT = ROOT / "data/internal/placement_scan_by_class.parquet"
POTTS_CKPT = os.environ.get("POTTS_CKPT", str(ROOT / "trained_models/mpnn_output_potts_v6_afdb_edge/ckpt/epoch-0085.ckpt"))

CAMPAIGNS = {
    "internal_campaign":   ROOT / "data/internal/internal_campaign_library/optimized_sweep_scores.parquet",
    "internal_campaign_2": ROOT / "data/internal/affinity_hit_structures/internal_campaign_2_deprot_all.parquet",
}
# state -> deprotonated/contrast alternative(s) (v6 tokens)
STATES = {"HIS-P": ["HIS-S"], "HIS-S": ["HIS-P"], "ASP-P": ["ASP-D"], "GLU-P": ["GLU-D"]}
STATE_ORDER = ["HIS-P", "HIS-S", "ASP-P", "GLU-P"]
IFACE_DIST = 8.0
RASA_THR = 0.2
CLASSES = ["interface", "core", "surface"]
TOPK = 12                                   # candidate-pool size, for the top-K distribution panel
SHARD = int(os.environ.get("SHARD", "0"))
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))


def target_label(campaign, pdb):
    if campaign == "internal_campaign_2":
        return "INTERNAL_2"
    b = os.path.basename(pdb).upper()
    return "EGFR" if "6ARU" in b else ("PD-L1" if "3BIK" in b else "EGFR/PD-L1")


# %% -------------------------------------------------------- residue classification (biotite, table-free)
def classify_binder(structure, bc):
    """{res_id: 'interface'|'core'|'surface'} for binder chain bc."""
    import biotite.structure as struc
    from biotite.structure import CellList
    b = structure[structure.chain_id == bc]
    t = structure[structure.chain_id != bc]
    bca, tca = b[b.atom_name == "CA"], t[t.atom_name == "CA"]
    iface = set()
    if tca.array_length() and bca.array_length():
        cl = CellList(tca, cell_size=IFACE_DIST)
        cont = cl.get_atoms(bca.coord, radius=IFACE_DIST)
        hit = (np.asarray(cont) != -1).reshape(bca.array_length(), -1).any(axis=1)
        iface = {int(r) for r, h in zip(bca.res_id, hit) if h}
    sasa = struc.sasa(b, point_number=200)
    starts = list(struc.get_residue_starts(b)) + [b.array_length()]
    res_sasa, res_ids = [], []
    for i in range(len(starts) - 1):
        s, e = starts[i], starts[i + 1]
        res_sasa.append(np.nansum(sasa[s:e]))
        res_ids.append(int(b.res_id[s]))
    res_sasa = np.array(res_sasa, float)
    mx = np.nanmax(res_sasa) if np.isfinite(res_sasa).any() and np.nanmax(res_sasa) > 0 else 1.0
    rasa = res_sasa / mx                                       # relative RASA, table-free (per-binder max)
    cls = {}
    for rid, ra in zip(res_ids, rasa):
        cls[rid] = "interface" if rid in iface else ("core" if ra < RASA_THR else "surface")
    return cls


# %% ------------------------------------------------------------------------- scan one backbone
def scan_backbone(eng, ctx, cls, rng):
    """Rows: for each (method, state) the class of the algorithm's top placement + top-K class counts."""
    import torch
    t2i = ctx.encoding.token_to_idx
    unk = t2i["UNK"]
    region = ctx.chA_free_idx                                  # all free binder positions ("all")
    pos_cls = np.array([cls.get(int(ctx.token_aa.res_id[int(p)]), "surface") for p in region.tolist()])
    bg = {c: int((pos_cls == c).sum()) for c in CLASSES}
    rows = []
    with torch.no_grad():
        for method, field, mask in [("scan_potts", ctx.field_potts, False),
                                    ("scan_mpnn", ctx.field_mpnn, True)]:
            seq = ctx.S_native.clone()
            if mask:
                seq[region] = unk
            ef = field(seq)                                    # [L, V]
            base = ef[region, seq[region]]
            for st, deps in STATES.items():
                if st not in t2i or any(d not in t2i for d in deps):
                    continue
                sc = ef[region, t2i[st]] - base
                dep = torch.stack([ef[region, t2i[d]] - base for d in deps], 0).min(0).values
                order = torch.argsort(sc - dep).cpu().numpy()  # ascending; rank-0 = algorithm's pick
                topk = pos_cls[order[:TOPK]]
                rows.append(dict(method=method, state=st, top1=pos_cls[order[0]],
                                 **{f"topk_{c}": int((topk == c).sum()) for c in CLASSES}))
    # random baseline: uniform over free positions -> expected class = background composition
    n = len(pos_cls)
    for st in STATES:
        pick = pos_cls[rng.randrange(n)] if n else "surface"
        rows.append(dict(method="random", state=st, top1=pick,
                         **{f"topk_{c}": bg[c] for c in CLASSES}))
    for r in rows:
        r.update(n_iface=bg["interface"], n_core=bg["core"], n_surf=bg["surface"], n_free=n)
    return rows


def compute_shard():
    import random
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    from src.core.io import load_structure_from_pdb
    from src.foundry_wrappers.pottsmpnn_ph import PottsMPNNPHEngine

    bbs = []
    for camp, p in CAMPAIGNS.items():
        d = pd.read_parquet(p, columns=["origin_pdb_path", "binder_chain"]).drop_duplicates()
        for _, r in d.iterrows():
            bbs.append((camp, target_label(camp, r.origin_pdb_path), r.origin_pdb_path, str(r.binder_chain)))
    bbs = sorted(set(bbs))[SHARD::NUM_SHARDS]
    print(f"[scan] shard {SHARD}/{NUM_SHARDS}: {len(bbs)} backbones", flush=True)

    eng = PottsMPNNPHEngine(checkpoint_path=POTTS_CKPT, extended_vocab="v6", device="cpu",
                            write_fasta=False, write_structures=False)
    rng = random.Random(1234 + SHARD)
    out, ok, fail = [], 0, 0
    for i, (camp, tgt, pdb, bc) in enumerate(bbs):
        try:
            st = load_structure_from_pdb(pdb)
            cls = classify_binder(st, bc)
            ctx = eng._build_context(st, bc)
            for r in scan_backbone(eng, ctx, cls, rng):
                r.update(campaign=camp, target=tgt, origin_pdb_path=pdb, binder_chain=bc)
                out.append(r)
            ok += 1
        except Exception as e:
            fail += 1
            if fail <= 8:
                print(f"[scan] FAIL {os.path.basename(pdb)[:44]}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"[scan] {i+1}/{len(bbs)} ok={ok} fail={fail}", flush=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    fp = SHARD_DIR / f"shard{SHARD:03d}_of{NUM_SHARDS:03d}.parquet"
    pd.DataFrame(out).to_parquet(fp, index=False)
    print(f"[scan] shard {SHARD} done ok={ok} fail={fail} -> {fp}", flush=True)


# %% --------------------------------------------------------------------------- aggregate + plot
def aggregate_and_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.concat([pd.read_parquet(s) for s in sorted(glob.glob(str(SHARD_DIR / "shard*.parquet")))],
                   ignore_index=True)
    df.to_parquet(OUT, index=False)
    nbb = df.origin_pdb_path.nunique()
    print(f"[agg] {nbb} backbones -> {OUT} ({len(df):,} rows)")
    FIGDIR.mkdir(parents=True, exist_ok=True)

    # background availability per (campaign,target): mean over backbones of class fraction of free positions
    b1 = df.drop_duplicates(["campaign", "target", "origin_pdb_path", "binder_chain"]).copy()
    for c, col in zip(CLASSES, ["n_iface", "n_core", "n_surf"]):
        b1[c] = b1[col] / b1["n_free"].replace(0, np.nan)
    BG = b1.groupby(["campaign", "target"])[CLASSES].mean()

    import matplotlib as mpl
    from matplotlib.patches import Patch
    mpl.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300})
    # class = colourblind-safe (Okabe-Ito), consistent across all figures; method = solid/hatched.
    CLS_COL = {"interface": "#E69F00", "core": "#0072B2", "surface": "#009E73"}
    METHODS = [("scan_potts", {}), ("scan_mpnn", {"hatch": "////"})]     # Potts solid, MPNN white-hatched

    frac = (df.groupby(["campaign", "target", "state", "method"]).top1
            .value_counts(normalize=True).rename("f").reset_index())
    F = {(r.campaign, r.target, r.state, r.method, r.top1): r.f for r in frac.itertuples(index=False)}

    torder = {"EGFR": 0, "PD-L1": 1, "INTERNAL_2": 2}
    gk = sorted(df.groupby(["campaign", "target"]).groups.keys(), key=lambda ct: torder.get(ct[1], 9))
    fig, axes = plt.subplots(1, len(gk), figsize=(4.2 * len(gk), 4.6), squeeze=False, sharey=True)
    bw, off = 0.36, 0.20
    for gi, (camp, tgt) in enumerate(gk):
        ax = axes[0, gi]
        for ti, st in enumerate(STATE_ORDER):
            for mi, (m, kw) in enumerate(METHODS):
                xpos = ti + (mi - 0.5) * 2 * off
                bottom = 0.0
                for c in CLASSES:                                   # stack interface/core/surface
                    h = F.get((camp, tgt, st, m, c), 0.0)
                    ax.bar(xpos, h, bw, bottom=bottom, color=CLS_COL[c],
                           edgecolor="white", linewidth=1.2, **kw)
                    bottom += h
        ax.set_xticks(range(len(STATE_ORDER))); ax.set_xticklabels(STATE_ORDER)
        ax.tick_params(axis="x", length=0)
        ax.set_title(tgt, pad=6); ax.set_ylim(0, 1); ax.set_xlim(-0.55, len(STATE_ORDER) - 0.45)
        ax.grid(axis="y", color="0.9", lw=0.8); ax.set_axisbelow(True)
        if gi == 0:
            ax.set_ylabel("Fraction of backbones (top-1 placement)")
    handles = ([Patch(facecolor=CLS_COL[c], label=c.capitalize()) for c in CLASSES]
               + [Patch(facecolor="0.75", edgecolor="white", label="Potts"),
                  Patch(facecolor="0.75", edgecolor="white", hatch="////", label="MPNN")])
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=5, frameon=False, columnspacing=1.6)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIGDIR / "placement_scan_by_class.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {FIGDIR / 'placement_scan_by_class.png'}")

    print("\n=== background class availability (mean over backbones) ===")
    print((BG * 100).round(1).to_string())
    print("\n=== top-1 placement class frequency (fraction of backbones) ===")
    piv = (df.groupby(["campaign", "target", "state", "method"]).top1
           .value_counts(normalize=True).rename("frac").reset_index()
           .pivot_table(index=["campaign", "target", "state", "method"], columns="top1", values="frac")
           .reindex(columns=CLASSES).round(3))
    print(piv.to_string())


if __name__ == "__main__":
    aggregate_and_plot() if os.environ.get("AGGREGATE") == "1" else compute_shard()
