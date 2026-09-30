# %% [markdown]
# # Label a PDB with the protonation pipeline (example)
#
# This shows the **first half** of Proton-PottsMPNN: taking a raw structure and assigning a **protonation
# state** to every titratable residue (His / Asp / Glu) — the exact labels the Potts model is trained on and
# the ones the design engine consumes.
#
# It runs the **transformation pipeline** `prepare_potts_input(..., extended_vocab="v6")` — the *same* code
# path the training data and the design engine use (`_build_context` calls it too). The pipeline strips
# hydrogens, runs HBPLUS, applies the FLAML protonation labeller, and attaches a per-residue
# `protonation_label` token to the atom array:  `-P` protonated · `-S` neutral His · `-D` deprotonated acid ·
# `-A` ambiguous.
#
# Needs `HBPLUS_PATH` (the labeller reads H-bond geometry). See `../README.md` §"Label a PDB".

# %%
import os
import json
from pathlib import Path

os.environ.pop("DEBUG", None)                      # a stray non-boolean DEBUG breaks environs/foundry
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU is plenty; the labeller runs no model forward

import numpy as np
import pandas as pd
from mpnn.potts_inference import prepare_potts_input   # resolved from the venv: pip install -e ./foundry

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
PKG = HERE.parent if HERE.name == "labeller" else HERE

if not os.environ.get("HBPLUS_PATH"):
    raise SystemExit("set HBPLUS_PATH=/path/to/hbplus — the protonation labeller reads H-bond geometry from HBPLUS")

# any PDB/CIF works; here the shipped PD-L1 seed binder (its binder chain + PD-L1 target)
META = json.loads((PKG / "inference" / "examples" / "example_meta.json").read_text())
PDB = PKG / "inference" / "examples" / META["pdb"]
BINDER_CHAIN = META["binder_chain"]
print(f"labelling {META['pdb_id']}  ({META['binder_len']} aa binder on chain {BINDER_CHAIN} + target)")

# %%
# --- run the transformation pipeline: raw structure -> featurised atom array with protonation labels ---
# prepare_potts_input takes a path (or an AtomArray) and returns {"network_input": tensors, "atom_array": ...}.
# The atom array carries the per-residue `protonation_label` token attached by the v6 labeller transform.
out = prepare_potts_input(str(PDB), extended_vocab="v6")
arr = out["atom_array"]

ca = arr[arr.atom_name == "CA"]                                   # one CA per residue, in token order
df = pd.DataFrame({
    "chain":       np.asarray(ca.chain_id),
    "res_id":      np.asarray(ca.res_id).astype(int),
    "res_name":    np.asarray(ca.res_name).astype(str),
    "token":       np.asarray(ca.get_annotation("protonation_label")).astype(str),
})
titratable = df[df.res_name.isin(["HIS", "ASP", "GLU"])].reset_index(drop=True)
print(f"\n{len(titratable)} titratable residues (His/Asp/Glu) labelled by the pipeline:\n")
print(titratable.to_string(index=False))

# %%
# --- focus on the binder chain + a compact summary ---
binder = titratable[titratable.chain == BINDER_CHAIN]
print(f"\nbinder chain {BINDER_CHAIN}: {len(binder)} titratable residues")
for stem, grp in binder.groupby(binder.res_name):
    print(f"  {stem}: {grp.token.value_counts().to_dict()}")

# the protonated centres the labeller found at rest (design later PINS -P centres deliberately)
protonated = titratable[titratable.token.str.endswith("-P")]
print(f"\nprotonated centres found by the pipeline: {len(protonated)}")
for r in protonated.itertuples():
    print(f"  chain {r.chain} res {r.res_id:>4} {r.res_name} -> {r.token}")

# %%
# --- write the labels out ---
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
titratable.to_csv(OUT / "protonation_labels.csv", index=False)
print("\nwrote", OUT / "protonation_labels.csv")
print("\nThese `protonation_label` tokens are exactly what the model trains on and what the design engine")
print("consumes (its `_build_context` calls this same `prepare_potts_input`). See §'Pipeline' in ../README.md.")
print("For the raw labeller probabilities (p_protonated, sd), call mpnn.transforms.ev6.EV6Predictor directly")
print("— it is the labeller this pipeline wraps.")
