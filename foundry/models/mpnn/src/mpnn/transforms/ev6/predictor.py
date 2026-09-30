"""EV6 — the protonation-state predictor.

Give it a protein ``AtomArray``; it computes the geometric features (``features.py``), runs the trained
AutoML models, and returns a per-residue protonation call with a probability and a confidence.

Two design points, both decided with the user:

  * ENSEMBLE, not a single model. Each residue is scored by all FIVE fold models (each was trained on 4/5
    of the neutron data; on a NEW structure they are all valid). Their MEAN is the point estimate
    ``p_protonated``; their SD is the disagreement signal.
  * The AMBIGUOUS class comes from that DISAGREEMENT, not from a probability band. A residue the folds
    argue about (SD > sd_cut) is labelled ``-A`` — the deployable analogue of
    ``sandbox/ev5/3_automl/figs/uncertainty.png``. Within the confident set, ``-P`` if the mean p clears
    ``prob_thr`` (prevalence-matched by default), else ``-S`` (His) / ``-D`` (acid).

Feature models only — no torch, no PottsMPNN. Just biotite + HBPLUS + the pickled sklearn/FLAML models.
"""
from __future__ import annotations

import json
import pickle
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from mpnn.transforms.ev6 import features as _F

_WEIGHTS = Path(__file__).parent / "weights"


def discretize_protonation(p: float, sd: float, metal: bool, stem: str,
                           sd_cut: float, prob_thr: float) -> str:
    """The two-cut rule that turns a FLAML score into a protonation token — the SINGLE source of truth.

    ``metal`` (out-of-domain) or ``sd > sd_cut`` (the folds disagree) -> ``-A`` (ambiguous); else
    ``p >= prob_thr`` -> ``-P``; else the neutral token (``-S`` for His, ``-D`` for the acids). Pure
    scalar arithmetic (no FLAML, no OpenMP), so it is safe to call live in a forked DataLoader worker —
    which is what ``ApplyProtonationThreshold`` does to re-threshold a stored score at train time.
    """
    if metal or sd > sd_cut:
        return f"{stem}-A"
    if p >= prob_thr:
        return f"{stem}-P"
    return f"{stem}-{'S' if stem == 'HIS' else 'D'}"


def _load_folds(sub: str) -> list[dict]:
    """The five fold models for one residue type — each a self-contained FLAML predictor."""
    d = _WEIGHTS / sub
    return [pickle.load(open(d / f"fold{k}.pkl", "rb")) for k in range(1, 6)]


class EV6Predictor:
    """Predict HIS-P/HIS-S/HIS-A and ASP/GLU-P/-D/-A from a protein structure.

    Parameters override the operating points in ``weights/thresholds.json``:
      his_sd_cut / acid_sd_cut  — SD above which a residue is ambiguous (-A).
      his_thr / acid_thr        — mean-probability cut for -P vs -S/-D (prevalence-matched by default).
    """

    def __init__(self, his_sd_cut=None, acid_sd_cut=None, his_thr=None, acid_thr=None,
                 weights_dir: Path | None = None):
        self._wd = Path(weights_dir) if weights_dir else _WEIGHTS
        cfg = json.loads((self._wd / "thresholds.json").read_text())
        self.cfg = {
            "his": dict(sd_cut=his_sd_cut if his_sd_cut is not None else cfg["his"]["sd_cut"],
                        thr=his_thr if his_thr is not None else cfg["his"]["prob_thr"]),
            "acid": dict(sd_cut=acid_sd_cut if acid_sd_cut is not None else cfg["acid"]["sd_cut"],
                         thr=acid_thr if acid_thr is not None else cfg["acid"]["prob_thr"]),
        }

    @cached_property
    def _his_folds(self):
        return [pickle.load(open(self._wd / "his" / f"fold{k}.pkl", "rb")) for k in range(1, 6)]

    @cached_property
    def _acid_folds(self):
        return [pickle.load(open(self._wd / "acid" / f"fold{k}.pkl", "rb")) for k in range(1, 6)]

    # ── prediction ──────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _ensemble(folds: list[dict], G: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Mean and SD of p(protonated) across the fold models. Reindex to each model's own feature
        order and cast its categoricals to `category` (FLAML re-encodes the raw string labels)."""
        preds = []
        for b in folds:
            X = G.reindex(columns=b["feature_columns"]).copy()
            for c in b["categorical_columns"]:
                X[c] = X[c].astype("category")
            preds.append(b["model"].predict_proba(X)[:, 1])
        P = np.vstack(preds)                       # (5, n_residues)
        return P.mean(0), P.std(0)

    def _call(self, G: pd.DataFrame, folds: list[dict], kind: str, neutral: str) -> pd.DataFrame:
        if not len(G):
            return pd.DataFrame(columns=["chain", "res_id", "res_name", "p_protonated", "sd",
                                         "token", "ambiguous", "metal_adjacent"])
        p, sd = self._ensemble(folds, G)
        sd_cut, thr = self.cfg[kind]["sd_cut"], self.cfg[kind]["thr"]
        res_name = G["res_name"].values if "res_name" in G.columns else np.full(len(G), "HIS")
        metal = G["metal_adjacent"].values.astype(bool)

        token = np.empty(len(G), dtype=object)
        for i in range(len(G)):
            token[i] = discretize_protonation(p[i], sd[i], bool(metal[i]), res_name[i], sd_cut, thr)
        return pd.DataFrame(dict(
            chain=G["chain"].values, res_id=G["res_id"].values, res_name=res_name,
            p_protonated=p, sd=sd, token=token,
            ambiguous=np.array([t.endswith("-A") for t in token]), metal_adjacent=metal))

    def predict(self, atom_array, hbond_records: dict | None = None) -> pd.DataFrame:
        """Per-residue calls for every HIS/ASP/GLU in the structure, sorted by chain then res_id.

        `hbond_records` is the H-bond pool the pipeline's CalculateHbondsPlus already built
        (``data["hbond_records"]``). Passing it skips two HBPLUS subprocess calls per structure; its
        settings are validated against EV6's own (see `features._hbond_pool`). None -> compute it here."""
        prep = _F._Prep(atom_array)          # structure-only; shared so biotite's SASA runs once, not twice
        his = self._call(_F.his_features(atom_array, hbond_records, prep), self._his_folds, "his", "S")
        acid = self._call(_F.acid_features(atom_array, hbond_records, prep), self._acid_folds, "acid", "D")
        out = pd.concat([his, acid], ignore_index=True)
        return out.sort_values(["chain", "res_id"]).reset_index(drop=True)

    def predict_tokens(self, atom_array, hbond_records: dict | None = None) -> dict:
        """`{(chain_id, res_id, res_name): token}` — the form the vocab adapter consumes."""
        df = self.predict(atom_array, hbond_records)
        return {(str(r.chain), int(r.res_id), str(r.res_name)): r.token for r in df.itertuples()}
