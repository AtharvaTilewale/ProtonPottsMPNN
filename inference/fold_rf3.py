"""Fold a designed binder with the **packaged RF3 engine** (from_target templating).

The RF3 CODE ships in this package (`foundry/models/rf3/`), so folding is wired to the real engine —
`RF3InferenceEngine(...).run(atom_array, template_selection=[target_chains])`. But two things are NOT in
the package and MUST be supplied by you:

  * the RF3 **weights** (~3 GB) — set ``RF3_CKPT=/abs/path/to/rf3_*.ckpt``  (see ../README.md), and
  * a **GPU** *and* the RF3 atom-embedding cache (MACE-OMOL). RF3 will fall back to CPU / zero-embeddings,
    but only produces a PHYSICAL structure with the GPU + the embedding cache present.

✅  This path WAS exercised end-to-end (input build → 3 GB checkpoint load → RF3 forward → a 2-chain
    binder+target `.cif`) with an internal checkpoint. On a CPU-only box without the embedding cache the
    forward still runs but the coordinates are unphysical — so fold on a properly configured GPU node.
    Faithful port of the reference from_target fold (diffusion `foundry_wrappers/rf3_engine.py`).

from_target = the TARGET chain(s) are templated (coordinates held); the binder is folded fresh from its
(designed) sequence. So we thread the designed canonical sequence onto the binder chain as a backbone-only
chain, keep the target coordinates, write a PDB, and let RF3 rebuild side chains from the CCD.

n_recycles / num_steps default to production values; drop them (e.g. 3 / 20) for a faster, coarser fold.
"""
import os
from pathlib import Path

import numpy as np
import biotite.structure as struc
from biotite.sequence import ProteinSequence
from biotite.structure.io.pdb import PDBFile


def rf3_available() -> bool:
    """True only when an RF3 checkpoint is configured AND a CUDA device is visible."""
    if not os.environ.get("RF3_CKPT"):
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


_BB = ("N", "CA", "C", "O")


def _binder_backbone_with_sequence(binder, canonical_seq: str):
    """Backbone-only copy of `binder` with each residue's res_name set to the designed canonical AA.
    RF3 folds the binder from this sequence (add_missing_atoms rebuilds side chains from the CCD)."""
    starts = struc.get_residue_starts(binder)
    if len(starts) != len(canonical_seq):
        raise ValueError(f"binder has {len(starts)} residues but the sequence is {len(canonical_seq)} long")
    keep = np.isin(binder.atom_name, _BB)
    bb = binder[keep].copy()
    three = {i: ProteinSequence.convert_letter_1to3(c) for i, c in enumerate(canonical_seq)}
    bb_starts = struc.get_residue_starts(bb)
    for ri, s in enumerate(bb_starts):
        e = bb_starts[ri + 1] if ri + 1 < len(bb_starts) else len(bb)
        bb.res_name[s:e] = three[ri]
    return bb


def fold_from_target(structure_path, canonical_seq: str, binder_chain: str, target_chains,
                     out_dir, ckpt: str | None = None, n_recycles: int = 6, num_steps: int = 50) -> Path:
    """Fold `canonical_seq` on `binder_chain`, templating `target_chains`, with the packaged RF3 engine.

    Returns the path to the predicted `.cif`. Raises RuntimeError if RF3 weights / GPU are unavailable —
    call `rf3_available()` first to branch cleanly."""
    ckpt = ckpt or os.environ.get("RF3_CKPT")
    if not rf3_available():
        raise RuntimeError("RF3 weights (RF3_CKPT) + a GPU are required to fold. See fold_rf3.py header.")
    from rf3.inference_engines.rf3 import RF3InferenceEngine

    target_chains = [target_chains] if isinstance(target_chains, str) else list(target_chains)
    aa = PDBFile.read(str(structure_path)).get_structure(model=1)
    target = aa[np.isin(aa.chain_id, target_chains)]                       # templated (coords held)
    binder = aa[aa.chain_id == binder_chain]                               # folded from the designed sequence
    fold_input = target + _binder_backbone_with_sequence(binder, canonical_seq)

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    inp_pdb = out_dir / "rf3_input.pdb"
    f = PDBFile(); f.set_structure(fold_input); f.write(str(inp_pdb))       # path input → add_missing_atoms rebuilds side chains

    engine = RF3InferenceEngine(ckpt_path=ckpt, n_recycles=n_recycles, diffusion_batch_size=1, num_steps=num_steps)
    engine.run(inputs=str(inp_pdb), template_selection=target_chains, out_dir=str(out_dir),
               dump_predictions=True, annotate_b_factor_with_plddt=True)

    cifs = sorted(out_dir.glob("**/*.cif")) + sorted(out_dir.glob("**/*.cif.gz"))
    if not cifs:
        raise RuntimeError(f"RF3 produced no .cif under {out_dir}")
    return cifs[0]
