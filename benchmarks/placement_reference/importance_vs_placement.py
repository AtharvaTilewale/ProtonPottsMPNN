"""Does the pipeline place each protonation state where the model's own per-residue signal points?

Two per-residue quantities are read off ONE Potts conditional-energy forward per backbone (scan_potts
field on the native sequence), for every free binder position of every library backbone:

  selective placement score(pos, state) = ef[pos, STATE] - ef[pos, native] - min_d(ef[pos, DEP])
        the exact scan the pipeline ranks placements by (lower = the model prefers this state here).

  substitution dE importance(pos)       = mean over the 20 canonical AAs of ( ef[pos, aa] - ef[pos, native] )
        the full L x 20 single-residue scan: how much worse, on average, every other residue is at
        this position -> HIGHER = the native residue is costly to substitute / the position is important.
        (State-independent; the "importance by dG" axis.)

Each free residue is classified interface / core / surface with the pipeline's own rule
(classify_binder in frequency_by_class). The OBSERVATIONS are the library's actual pinned centres
(center_res_ids x center_protonation_types in optimized_sweep_scores.parquet); we tally, per state,
which class they most often land in, and overlay them on the importance-vs-placement plane.

    cd <path-to-internal-project-root>
    SHARD=0 NUM_SHARDS=8 ./dpo_potts/bin/python sandbox/placement/importance_vs_placement.py   # compute a shard
    AGGREGATE=1 ./dpo_potts/bin/python sandbox/placement/importance_vs_placement.py            # -> parquet + plots
Non-destructive; per-residue rows only.
"""
# %% ------------------------------------------------------------------------- config
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from frequency_by_class import (classify_binder, STATES, STATE_ORDER, POTTS_CKPT, CAMPAIGNS,  # noqa: E402
                                target_label, CLASSES)

ROOT = Path(os.environ.get("PROJECT_ROOT", "."))
FIGDIR = HERE / "figures"
SHARD_DIR = ROOT / "data/internal/importance_placement_shards"
OUT = ROOT / "data/internal/importance_vs_placement.parquet"
LIB = ROOT / "data/internal/internal_campaign_library/optimized_sweep_scores.parquet"
SHARD = int(os.environ.get("SHARD", "0"))
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))
CLS_COL = {"interface": "#E69F00", "core": "#0072B2", "surface": "#009E73"}   # Okabe-Ito, consistent

# the 20 canonical AAs -> v6 base token (His/Asp/Glu use their neutral/deprotonated representative)
CANON = (["ALA", "ARG", "ASN", "CYS", "GLN", "GLY", "ILE", "LEU", "LYS", "MET",
          "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "HIS-S", "ASP-D", "GLU-D"])


# %% ------------------------------------------------------------------------- compute one shard
def compute_shard():
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    from src.core.io import load_structure_from_pdb
    from src.foundry_wrappers.pottsmpnn_ph import PottsMPNNPHEngine

    bbs = []
    for camp, p in CAMPAIGNS.items():
        if camp != "internal_campaign":                         # library = the internal-campaign designs
            continue
        d = pd.read_parquet(p, columns=["origin_pdb_path", "binder_chain"]).drop_duplicates()
        for _, r in d.iterrows():
            bbs.append((camp, target_label(camp, r.origin_pdb_path), r.origin_pdb_path, str(r.binder_chain)))
    bbs = sorted(set(bbs))[SHARD::NUM_SHARDS]
    print(f"[imp] shard {SHARD}/{NUM_SHARDS}: {len(bbs)} backbones", flush=True)

    eng = PottsMPNNPHEngine(checkpoint_path=POTTS_CKPT, extended_vocab="v6", device="cpu",
                            write_fasta=False, write_structures=False)
    out, ok, fail = [], 0, 0
    for i, (camp, tgt, pdb, bc) in enumerate(bbs):
        try:
            st = load_structure_from_pdb(pdb)
            cls = classify_binder(st, bc)
            ctx = eng._build_context(st, bc)
            t2i = ctx.encoding.token_to_idx
            region = ctx.chA_free_idx
            res_ids = np.array([int(ctx.token_aa.res_id[int(p)]) for p in region.tolist()])
            pos_cls = np.array([cls.get(int(r), "surface") for r in res_ids])
            canon_idx = [t2i[a] for a in CANON if a in t2i]
            with torch.no_grad():
                ef = ctx.field_potts(ctx.S_native.clone())     # [L, V] conditional energies (scan_potts)
            base = ef[region, ctx.S_native[region]]            # native residue energy per position
            imp = (ef[region][:, canon_idx] - base[:, None]).mean(1).cpu().numpy()   # mean dE over 20 AAs
            row = dict(campaign=camp, target=tgt, origin_pdb_path=pdb, binder_chain=bc)
            per = {"res_id": res_ids, "cls": pos_cls, "imp_de": imp}
            for stt, deps in STATES.items():
                if stt not in t2i or any(d not in t2i for d in deps):
                    continue
                sc = ef[region, t2i[stt]] - base
                dep = torch.stack([ef[region, t2i[d]] - base for d in deps], 0).min(0).values
                per[f"score_{stt}"] = (sc - dep).cpu().numpy()
            df = pd.DataFrame(per)
            for k, v in row.items():
                df[k] = v
            out.append(df)
            ok += 1
        except Exception as e:
            fail += 1
            if fail <= 8:
                print(f"[imp] FAIL {os.path.basename(pdb)[:44]}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"[imp] {i+1}/{len(bbs)} ok={ok} fail={fail}", flush=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    fp = SHARD_DIR / f"shard{SHARD:03d}_of{NUM_SHARDS:03d}.parquet"
    pd.concat(out, ignore_index=True).to_parquet(fp, index=False)
    print(f"[imp] shard {SHARD} done ok={ok} fail={fail} -> {fp}", flush=True)


# %% ------------------------------------------------------------------------- observed centres
def observed_centres():
    """Long table (origin_pdb_path, res_id, state) of the library's pinned centres, scan_potts only
    (the placements the potts scan actually drove; random is a separate baseline)."""
    d = pd.read_parquet(LIB, columns=["origin_pdb_path", "center_res_ids",
                                       "center_protonation_types", "placement_by"])
    d = d[d.placement_by == "scan_potts"]
    rows = []
    for pdb, ids, ts in zip(d.origin_pdb_path, d.center_res_ids, d.center_protonation_types):
        if ids is None or ts is None:
            continue
        for i, t in zip(list(ids), list(ts)):
            rows.append((pdb, int(i), str(t)))
    return pd.DataFrame(rows, columns=["origin_pdb_path", "res_id", "state"])


# %% ------------------------------------------------------------------------- aggregate + plot
def aggregate_and_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from scipy.stats import pearsonr
    mpl.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300})

    df = pd.concat([pd.read_parquet(s) for s in sorted(glob.glob(str(SHARD_DIR / "shard*.parquet")))],
                   ignore_index=True)
    df.to_parquet(OUT, index=False)
    print(f"[agg] {df.origin_pdb_path.nunique()} backbones, {len(df):,} residue rows -> {OUT}")

    import seaborn as sns
    obs = observed_centres()
    OBS_STATES = ["HIS-P", "ASP-P", "GLU-P"]          # the library only ever pins PROTONATED centres
    # mark, per state, which residues were placed as a centre of THAT state (>=1 design)
    placed = {st: set(map(tuple, obs[obs.state == st][["origin_pdb_path", "res_id"]].values))
              for st in OBS_STATES}
    key = np.array(list(zip(df.origin_pdb_path.astype(str), df.res_id.astype(int))), dtype=object)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    # ---- FIG 1: importance vs selective placement, per state, coloured by class + placement DENSITY ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), sharex=True)
    for ax, st in zip(axes, OBS_STATES):
        col = f"score_{st}"
        sub = df.dropna(subset=[col, "imp_de"]).reset_index(drop=True)
        for c in CLASSES:
            s = sub[sub.cls == c]
            ax.scatter(s.imp_de, s[col], s=4, color=CLS_COL[c], alpha=0.07, linewidths=0, rasterized=True)
        pm = np.array([(str(a), int(b)) in placed[st] for a, b in zip(sub.origin_pdb_path, sub.res_id)])
        o = sub[pm]                                                    # observed centres of this state
        if len(o) > 30:                                                # placement DENSITY as contours (not a blob)
            sns.kdeplot(x=o.imp_de, y=o[col], levels=[0.25, 0.5, 0.75, 0.95], color="black",
                        linewidths=1.3, ax=ax, zorder=6)
        r, _ = pearsonr(sub.imp_de, sub[col])
        ax.axhline(0, color="0.8", lw=0.8, zorder=0)
        ax.set_title(f"{st}   r={r:+.2f}   (placed n={len(o):,})", pad=6)
        ax.set_xlabel("Substitution $\\Delta E$\n(higher = substitution is costly)")
    for a in axes:                                            # sns.kdeplot stamps the column name; clear it
        a.set_ylabel("")
    axes[0].set_ylabel("Selective placement score\n(lower = preferred)")
    handles = [Line2D([0], [0], marker="o", ls="", color=CLS_COL[c], label=c.capitalize()) for c in CLASSES]
    handles += [Line2D([0], [0], color="black", lw=1.3, label="placement density (KDE)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("$\\Delta E$ vs selective placement across the library — where each state is placed",
                 y=1.10, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out1 = FIGDIR / "importance_vs_placement.png"
    fig.savefig(out1, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {out1}")

    # ---- FIG 2: 'where they most often place the state' — observed class freq vs what the scan prefers ----
    # scan preference per class = fraction of residues whose best-scoring placement (min score) falls in
    # that class, per state; observed = fraction of that state's centres landing in that class.
    obs_cls = obs.merge(df[["origin_pdb_path", "res_id", "cls"]].drop_duplicates(),
                        on=["origin_pdb_path", "res_id"], how="left").dropna(subset=["cls"])
    bg = df.cls.value_counts(normalize=True)                            # class availability baseline
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    x = np.arange(len(CLASSES)); bw = 0.38
    for ax, st in zip(axes, OBS_STATES):
        col = f"score_{st}"
        obs_f = obs_cls[obs_cls.state == st].cls.value_counts(normalize=True).reindex(CLASSES).fillna(0)
        # scan preference: among residues, class of the min-score position per backbone -> here use
        # the class-mean score instead, shown as a line on a twin axis (lower = preferred)
        mean_sc = df.groupby("cls")[col].mean().reindex(CLASSES)
        ax.bar(x - bw / 2, obs_f.values, bw, color=[CLS_COL[c] for c in CLASSES], label="observed")
        ax.bar(x + bw / 2, bg.reindex(CLASSES).values, bw, color="0.8", label="available (baseline)")
        ax2 = ax.twinx(); ax2.plot(x, mean_sc.values, "k--o", lw=1.6, ms=6, zorder=6)
        ax2.invert_yaxis()                                   # flip so HIGHER on the axis = more preferred
        ax2.set_ylabel("mean placement score\n(higher = preferred)" if st == OBS_STATES[-1] else "", color="0.2")
        ax2.spines["top"].set_visible(False)
        ax.set_xticks(x); ax.set_xticklabels([c.capitalize() for c in CLASSES], rotation=20)
        ax.set_title(st, pad=4); ax.set_ylim(0, 1)
    axes[0].set_ylabel("Fraction of placed centres")
    h = [Patch(facecolor="0.5", label="observed placements"), Patch(facecolor="0.8", label="class availability"),
         Line2D([0], [0], color="k", ls="--", marker="o", label="mean placement score (up = preferred)")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Where the library places each state, by structural class", y=1.16, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out2 = FIGDIR / "placement_frequency_by_class.png"
    fig.savefig(out2, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {out2}")

    # ---- FIG 3: SELECTIVITY scan placement vs IMPORTANCE scan placement, by class ----
    # For each backbone: the class of the top-1 pick of each scan. Selectivity scan = argmin(score_state)
    # (per state). Importance scan = argmax(imp_de) (STATE-INDEPENDENT, so identical across the panels =
    # a fixed reference). Fraction over the 823 backbones.
    imp_top = df.loc[df.groupby("origin_pdb_path").imp_de.idxmax(), "cls"]
    imp_frac = imp_top.value_counts(normalize=True).reindex(CLASSES).fillna(0)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    x = np.arange(len(CLASSES)); bw = 0.38
    for ax, st in zip(axes, OBS_STATES):
        sel_top = df.loc[df.groupby("origin_pdb_path")[f"score_{st}"].idxmin(), "cls"]
        sel_frac = sel_top.value_counts(normalize=True).reindex(CLASSES).fillna(0)
        ax.bar(x - bw / 2, sel_frac.values, bw, color=[CLS_COL[c] for c in CLASSES],
               edgecolor="white", linewidth=1.0)
        ax.bar(x + bw / 2, imp_frac.values, bw, color="0.7", edgecolor="white", linewidth=1.0)
        ax.set_xticks(x); ax.set_xticklabels([c.capitalize() for c in CLASSES], rotation=20)
        ax.set_title(st, pad=4); ax.set_ylim(0, 1)
        ax.grid(axis="y", color="0.92", lw=0.8); ax.set_axisbelow(True)
    axes[0].set_ylabel("Fraction of backbones (top-1 placement)")
    h3 = [Patch(facecolor=CLS_COL[c], label=c.capitalize()) for c in CLASSES]
    h3 += [Patch(facecolor="0.7", label="$\\Delta E$ scan")]
    fig.legend(handles=h3, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Selectivity scan (coloured, per state) vs $\\Delta E$ scan (grey) — top-1 placement by class",
                 y=1.12, fontsize=13.5)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out3 = FIGDIR / "selectivity_vs_importance_scan_by_class.png"
    fig.savefig(out3, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {out3}")
    print("\n[summary] top-1 class fraction — selectivity scan (per state) vs importance scan (agnostic):")
    print("  importance: " + ", ".join(f"{c[:4]} {imp_frac[c]:.2f}" for c in CLASSES))
    sel = {}
    for st in OBS_STATES:
        sel[st] = df.loc[df.groupby("origin_pdb_path")[f"score_{st}"].idxmin(), "cls"].value_counts(
            normalize=True).reindex(CLASSES).fillna(0)
        print(f"  {st} selectivity: " + ", ".join(f"{c[:4]} {sel[st][c]:.2f}" for c in CLASSES))

    # ---- FIG 4: everything on ONE grouped bar chart — 3 selectivity states + importance, per class ----
    STATE_COL = {"HIS-P": "#4153A4", "ASP-P": "#a3002e", "GLU-P": "#189486"}   # manuscript palette
    series = [(st, sel[st], STATE_COL[st]) for st in OBS_STATES] + [("importance", imp_frac, "#828080")]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CLASSES)); bw = 0.20
    for i, (lab, frac, c) in enumerate(series):
        off = (i - (len(series) - 1) / 2) * bw
        ax.bar(x + off, frac.reindex(CLASSES).values, bw, color=c, edgecolor="white", linewidth=0.8,
               label=("$\\Delta E$ scan" if lab == "importance" else f"{lab} (selectivity)"))
    ax.set_xticks(x); ax.set_xticklabels([c.capitalize() for c in CLASSES])
    ax.set_ylabel("Fraction of backbones (top-1 placement)"); ax.set_ylim(0, 1)
    ax.grid(axis="y", color="0.92", lw=0.8); ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=4, frameon=False, columnspacing=1.3)
    fig.tight_layout()
    out4 = FIGDIR / "selectivity_vs_importance_scan_combined.png"
    fig.savefig(out4, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {out4}")

    # ---- console: the by-class correlation summary ----
    print("\n[summary] observed placement class fraction | mean placement score by class:")
    for st in OBS_STATES:
        col = f"score_{st}"
        of = obs_cls[obs_cls.state == st].cls.value_counts(normalize=True).reindex(CLASSES).fillna(0)
        ms = df.groupby("cls")[col].mean().reindex(CLASSES)
        print(f"  {st}: placed " + ", ".join(f"{c[:4]} {of[c]:.2f}" for c in CLASSES)
              + "  |  score " + ", ".join(f"{c[:4]} {ms[c]:+.1f}" for c in CLASSES))


if __name__ == "__main__":
    if os.environ.get("AGGREGATE") == "1":
        aggregate_and_plot()
    else:
        compute_shard()
