"""Auxiliary pH-bond / charge-clash SCORING of one structure — a runnable demo.

Given a folded structure + a set of pinned protonation centres, this computes the design read-outs the
manuscript uses to characterise a pH-switch:

  charge_clash / salt_bridge  — pure geometry (biotite): same-sign charged residues ≤4 Å (destabilising)
                                and opposite-sign ≤5 Å (stabilising) around each centre. NO HBPLUS.
                                → scoring/annotate_charge_clash.py::charge_clash
  his/acid_ph_hbonds,          — HBPLUS + PLIP bonds of the actual structure with ≥1 titratable endpoint,
  *_saltbridges, *_bonds         from prepare_potts_input(pdb, extended_vocab="v6", build_bond_labels=True);
  center_gated_bonds,            plus the microstate match at the pinned centres (re-predicted v6 label ==
  microstate_match               pinned type).  → scoring/annotate_folds.py::annotate   [needs HBPLUS_PATH]

This demo scores the example dimer's OWN structure, using its binder His/Asp/Glu residues as protonated
centres. For a real design you would fold it first (e.g. RF3) and score the fold.

Run:  HBPLUS_PATH=/path/to/hbplus python scoring/score_example.py
      (without HBPLUS only the geometric charge_clash/salt_bridge are computed.)
"""
import os
import sys
import json
import tempfile
from pathlib import Path

os.environ.pop("DEBUG", None)                      # a stray non-boolean DEBUG breaks environs/foundry
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HERE = Path(__file__).resolve().parent             # scoring/  — mpnn comes from the venv (pip install -e ./foundry)
PKG = HERE.parent
sys.path.insert(0, str(HERE))                      # so `import annotate_charge_clash / annotate_folds` (siblings) works

import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdb as pdbio
import biotite.structure.io.pdbx as pdbx

from annotate_charge_clash import charge_clash          # geometry (no HBPLUS)

META = json.loads((PKG / "inference" / "examples" / "example_meta.json").read_text())
PDB = PKG / "inference" / "examples" / META["pdb"]
BINDER_CHAIN = META["binder_chain"]
BINDER_LEN = int(META["binder_len"])

# --- read the structure, pick the binder's titratable residues as protonated centres ---
aa = pdbio.PDBFile.read(str(PDB)).get_structure(model=1)
aa = aa[struc.filter_amino_acids(aa)]
ca = aa[aa.atom_name == "CA"]
chain = np.asarray(ca.chain_id); resid = np.asarray(ca.res_id).astype(int); resn = np.asarray(ca.res_name)
TYPE = {"HIS": "HIS-P", "ASP": "ASP-P", "GLU": "GLU-P"}
centers = [(int(r), TYPE[str(n)]) for r, n, c in zip(resid, resn, chain)
           if c == BINDER_CHAIN and str(n) in TYPE]
print(f"example {META['pdb_id']} · binder chain {BINDER_CHAIN} ({BINDER_LEN} aa)")
print(f"pinned centres ({len(centers)}): {centers}")

# --- write a temp CIF (annotate/charge_clash read CIFs) ---
cif = tempfile.NamedTemporaryFile(suffix=".cif", delete=False).name
_f = pdbx.CIFFile(); pdbx.set_structure(_f, aa); _f.write(cif)

# --- geometric clash / salt bridge (always) ---
cc, sb, ncc = charge_clash(cif, BINDER_LEN, centers)
print("\n── geometric (no HBPLUS) ──")
print(f"  charge_clash (same-sign ≤4 Å, mean/centre) : {cc}")
print(f"  salt_bridge  (opp-sign  ≤5 Å, mean/centre) : {sb}   [{ncc} centres scored]")

# --- HBPLUS/PLIP pH-sensitive bonds (needs HBPLUS_PATH) ---
if os.environ.get("HBPLUS_PATH") and Path(os.environ["HBPLUS_PATH"]).exists():
    from annotate_folds import annotate
    a = annotate(cif, BINDER_LEN, centers)
    print("\n── pH-sensitive bonds (HBPLUS + PLIP + v6 labeller) ──")
    for k in ("his_ph_hbonds", "his_ph_saltbridges", "his_ph_bonds",
              "acid_ph_hbonds", "acid_ph_saltbridges", "acid_ph_bonds",
              "center_gated_bonds", "center_gated_bonds_mean", "microstate_match"):
        print(f"  {k:26s}: {a.get(k)}")
else:
    print("\n(set HBPLUS_PATH to also compute the pH-sensitive H-bond / salt-bridge counts)")
print("\ndone")
