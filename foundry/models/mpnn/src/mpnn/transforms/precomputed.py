"""Precomputed annotation snapshots for the PottsMPNN training pipeline.

Protonation labelling (HBPLUS + biotite geometry + the EV6 fold ensemble) is the dominant per-structure
cost in the data pipeline, and it is DETERMINISTIC and epoch-invariant: the label is a function of the
un-noised heavy-atom geometry only (structure noise is applied last, in FeaturizeUserSettings), and v6 is a
trained model. So it is annotated ONCE offline and reused every epoch, instead of recomputed in the
DataLoader workers on every ``__getitem__``.

A snapshot is the pipeline ``data`` dict captured AFTER cleanup + annotation but BEFORE the chain crop, on
the FULL (all-chains) structure. The crop is the only per-epoch randomness upstream of annotation and it
merely removes chains, so cropping the annotated full structure at train time is equivalent to (for
interface residues, a full-context version of) annotating the crop. The snapshot is the whole ``data`` dict
rather than just the atom_array because ``CropChainsAroundSpecifiedPNUnits`` also reads
``data["extra_info"]`` (chain contacts) and the query-pn-unit key.

Serialization is plain pickle: it round-trips a biotite ``AtomArray`` with all custom annotations
(``protonation_label``, ``pn_unit_iid``, ...) plus the ``extra_info`` dict exactly, and this is an internal
same-environment cache, not an interchange format.
"""
from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any, Callable

from atomworks.ml.transforms.base import Transform

logger = logging.getLogger(__name__)

# data keys that are re-established by the training pipeline itself (AddData) or are transient bookkeeping,
# so they must NOT be frozen into a snapshot -- they would go stale or shadow the live values.
_EXCLUDE_KEYS = ("model_type", "is_inference", "transform_history", "_transform_history")


def snapshot_path(precomputed_dir: str | Path, example_id: str) -> Path:
    """Deterministic, collision-free path for a structure's snapshot.

    ``example_id`` carries pipeline-unsafe characters (e.g. ``{['ds']}{ex}{1}{[A_1,B_1]}``), so the file is
    named by its SHA1 and sharded two levels deep to keep any one directory small across ~460k structures.
    """
    h = hashlib.sha1(str(example_id).encode()).hexdigest()
    return Path(precomputed_dir) / h[:2] / h[2:4] / f"{h}.pkl"


def save_snapshot(data: dict[str, Any], precomputed_dir: str | Path) -> Path:
    """Write ``data`` (minus the re-established keys) to its snapshot path. Returns the path."""
    example_id = data["example_id"]
    path = snapshot_path(precomputed_dir, example_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in data.items() if k not in _EXCLUDE_KEYS}
    # atomic write: a crashed/killed shard must never leave a truncated pickle that later loads as a hit.
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return path


def load_snapshot(precomputed_dir: str | Path, example_id: str) -> dict[str, Any] | None:
    """Return the stored ``data`` dict for ``example_id``, or ``None`` if there is no snapshot."""
    path = snapshot_path(precomputed_dir, example_id)
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _row_example_id(row) -> str:
    """The example_id for a metadata row. PandasDataset indexes by example_id (id_column), and the
    GenericDFParser also carries it as a column, so accept either."""
    try:
        return row["example_id"]
    except (KeyError, TypeError):
        return row.name


def make_snapshot_loader(precomputed_dir: str | Path) -> Callable:
    """Dataset ``loader(row) -> data`` that returns the precomputed snapshot, or RAISES on a miss.

    Training / validation are PRECOMPUTED-ONLY: an example with no snapshot is not annotated live -- it is
    skipped (the dataset's fallback draws another cached example). This is deliberate. EV6 is a
    FLAML/tree ensemble that uses OpenMP; running it inside a forked DataLoader worker deadlocks (fork
    poisons libgomp), which hung training. The snapshots are the model's only annotation source here; EV6
    runs live ONLY in the PKAD / MegaScale benchmark callbacks, which execute in the main process.

    Pair with ``cached_mask`` to filter the manifest to cached examples up front, so this raise is a safety
    net (e.g. a snapshot deleted mid-run) rather than a routine event.
    """
    def loader(row):
        example_id = _row_example_id(row)
        snap = load_snapshot(precomputed_dir, example_id)
        if snap is None:
            raise KeyError(f"no precomputed snapshot for example {example_id!r} (training is precomputed-only)")
        return snap

    return loader


class SaveAnnotationSnapshot(Transform):
    """Persist the current ``data`` as this structure's annotation snapshot, then pass it through unchanged.

    Placed at the end of the offline build pipeline (after annotation, before the crop). Requires
    ``example_id`` in ``data`` (the loader/parser sets it)."""

    def __init__(self, precomputed_dir: str | Path):
        self.precomputed_dir = precomputed_dir

    def check_input(self, data: dict) -> None:
        if "example_id" not in data:
            raise KeyError("SaveAnnotationSnapshot requires 'example_id' in data")
        if "atom_array" not in data:
            raise KeyError("SaveAnnotationSnapshot requires 'atom_array' in data")

    def forward(self, data: dict) -> dict:
        save_snapshot(data, self.precomputed_dir)
        return data
