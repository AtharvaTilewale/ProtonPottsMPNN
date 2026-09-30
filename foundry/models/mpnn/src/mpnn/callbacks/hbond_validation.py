"""Validation metrics for the H-bond / salt-bridge head, per epoch.

Accumulates the per-edge predictions/labels emitted by
``HBondHeadTrainer.validation_step`` (``outputs["edge_eval"]``), gathers them across DDP
ranks, and on rank 0 at each validation-epoch end computes & saves:

  * PR curve for H-bond presence (AP),
  * ROC curve for donor-vs-acceptor direction (AUC),
  * PR curve for salt-bridge presence (AP),
  * a per-H-bond-type bar plot: mean BCE loss + relative frequency (+ accuracy) for the
    most common donor->acceptor residue-type pairs,
  * a running summary.csv + learning-curve plot of AP/AUC vs epoch.
"""

from pathlib import Path

import numpy as np
import torch

from foundry.callbacks.callback import BaseCallback
from foundry.utils.ddp import RankedLogger
from mpnn.transforms.feature_aggregation.token_encodings import POTTS_MPNN_TOKEN_ENCODING as ENC

ranked_logger = RankedLogger(__name__, rank_zero_only=True)

_EDGE_KEYS = ("hbond_p", "hbond_y", "salt_p", "salt_y", "dir_p", "dir_y",
              "type_donor", "type_acceptor", "type_phbond")


class HBondValidationCallback(BaseCallback):
    """Compute & persist H-bond head validation curves and per-type stats each epoch.

    Args:
        save_dir: directory for plots / CSVs (created if missing).
        top_n_types: number of most-frequent donor->acceptor pairs shown in the bar plot.
    """

    def __init__(self, save_dir: Path | str, top_n_types: int = 25, encoding=ENC):
        self.save_dir = Path(save_dir) / "hbond_val"
        self.top_n_types = top_n_types
        # Vocab for the per-type donor->acceptor token-pair breakdown. Must match the encoder
        # (v6 = 30-token); the default 32-token ENC would mislabel v6 token indices. The overall
        # AP/AUC metrics are label-based and vocab-agnostic -- only this breakdown needs it.
        self.encoding = encoding
        self._buf = None

    def on_validation_epoch_start(self, trainer):
        self._buf = {k: [] for k in _EDGE_KEYS}

    def on_validation_batch_end(self, trainer, outputs, batch, batch_idx, num_batches, dataset_name=None):
        ee = outputs.get("edge_eval") if isinstance(outputs, dict) else None
        if ee is None:
            return
        for k in _EDGE_KEYS:
            v = ee.get(k)
            if v is not None and len(v):
                self._buf[k].append(np.asarray(v))

    def _gather(self, trainer) -> dict:
        """Concatenate this rank's buffers, then all_gather across ranks (rank 0 full)."""
        local = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in self._buf.items()}
        world = getattr(trainer.fabric, "world_size", 1)
        if world <= 1:
            return local
        gathered = [None] * world
        torch.distributed.all_gather_object(gathered, local)
        if trainer.fabric.global_rank != 0:
            return local
        return {k: np.concatenate([g[k] for g in gathered if len(g[k])]) if any(len(g[k]) for g in gathered)
                else np.zeros(0) for k in _EDGE_KEYS}

    def on_validation_epoch_end(self, trainer):
        if self._buf is None:
            return
        data = self._gather(trainer)
        if trainer.fabric.global_rank != 0:
            self._buf = None
            return

        self.save_dir.mkdir(parents=True, exist_ok=True)
        epoch = trainer.state.get("current_epoch", 0)
        try:
            from sklearn.metrics import (precision_recall_curve, average_precision_score,
                                         roc_curve, roc_auc_score)
        except Exception as e:  # noqa: BLE001
            ranked_logger.warning(f"sklearn unavailable; skipping hbond val plots: {e}")
            self._buf = None
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        hb_p, hb_y = data["hbond_p"], data["hbond_y"]
        sb_p, sb_y = data["salt_p"], data["salt_y"]
        dir_p, dir_y = data["dir_p"], data["dir_y"]

        ap_hb = float(average_precision_score(hb_y, hb_p)) if hb_y.sum() and len(np.unique(hb_y)) == 2 else float("nan")
        ap_sb = float(average_precision_score(sb_y, sb_p)) if sb_y.sum() and len(np.unique(sb_y)) == 2 else float("nan")
        auc_dir = float(roc_auc_score(dir_y, dir_p)) if len(np.unique(dir_y)) == 2 else float("nan")

        # ── curves ───────────────────────────────────────────────────────────
        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        if len(np.unique(hb_y)) == 2:
            pr, rc, _ = precision_recall_curve(hb_y, hb_p)
            ax[0].plot(rc, pr, lw=2, color="#2a9d8f")
            ax[0].axhline(hb_y.mean(), ls="--", color="gray", label=f"prev={hb_y.mean():.3f}")
        ax[0].set_title(f"H-bond PR (AP={ap_hb:.3f})"); ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision"); ax[0].legend()
        if len(np.unique(dir_y)) == 2:
            fpr, tpr, _ = roc_curve(dir_y, dir_p)
            ax[1].plot(fpr, tpr, lw=2, color="#e76f51"); ax[1].plot([0, 1], [0, 1], "--", color="gray")
        ax[1].set_title(f"Donor/Acceptor ROC (AUC={auc_dir:.3f})"); ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR")
        if len(np.unique(sb_y)) == 2:
            prs, rcs, _ = precision_recall_curve(sb_y, sb_p)
            ax[2].plot(rcs, prs, lw=2, color="#264653")
            ax[2].axhline(sb_y.mean(), ls="--", color="gray", label=f"prev={sb_y.mean():.4f}")
        ax[2].set_title(f"Salt-bridge PR (AP={ap_sb:.3f})"); ax[2].set_xlabel("recall"); ax[2].set_ylabel("precision"); ax[2].legend()
        for a in ax:
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
        fig.suptitle(f"H-bond head validation — epoch {epoch}")
        fig.tight_layout(); fig.savefig(self.save_dir / f"epoch_{epoch:04d}_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ── per-type accuracy / frequency / loss bar plot ─────────────────────
        self._per_type_plot(data, epoch, plt)

        # ── running summary + learning curve ─────────────────────────────────
        self._update_summary(epoch, ap_hb, auc_dir, ap_sb,
                             n_hb=int(len(hb_y)), n_hb_pos=int(hb_y.sum()),
                             n_salt_pos=int(sb_y.sum()), n_dir=int(len(dir_y)), plt=plt)

        ranked_logger.info(
            f"[hbond val] epoch {epoch}: AP_hbond={ap_hb:.3f}  AUC_dir={auc_dir:.3f}  AP_salt={ap_sb:.3f}"
        )
        print(f"[hbond val] epoch {epoch}  AP_hbond={ap_hb:.3f}  AUC_dir={auc_dir:.3f}  "
              f"AP_salt={ap_sb:.3f}  (hbond_pos={int(hb_y.sum())}/{len(hb_y)})", flush=True)
        self._buf = None

    def _per_type_plot(self, data, epoch, plt):
        import pandas as pd
        td, ta, pp = data["type_donor"], data["type_acceptor"], data["type_phbond"]
        if len(td) == 0:
            return
        idx2tok = self.encoding.idx_to_token
        names = np.array([f"{idx2tok[int(d)]}->{idx2tok[int(a)]}" for d, a in zip(td, ta)])
        bce = -np.log(np.clip(pp, 1e-6, 1.0))            # per-edge BCE vs the positive label
        acc = (pp >= 0.5).astype(float)
        df = pd.DataFrame({"pair": names, "bce": bce, "acc": acc})
        g = df.groupby("pair").agg(n=("bce", "size"), mean_loss=("bce", "mean"),
                                   accuracy=("acc", "mean")).reset_index()
        g["freq"] = g["n"] / g["n"].sum()
        g = g.sort_values("n", ascending=False)
        g.to_csv(self.save_dir / f"epoch_{epoch:04d}_per_type.csv", index=False)

        top = g.head(self.top_n_types)
        x = np.arange(len(top))
        fig, ax1 = plt.subplots(figsize=(max(10, len(top) * 0.5), 5))
        ax1.bar(x - 0.2, top["mean_loss"], 0.4, color="#e76f51", label="mean BCE loss")
        ax1.set_ylabel("mean BCE loss (− log p_hbond)", color="#e76f51")
        ax1.tick_params(axis="y", labelcolor="#e76f51")
        ax2 = ax1.twinx()
        ax2.bar(x + 0.2, top["freq"], 0.4, color="#2a9d8f", label="relative frequency")
        ax2.set_ylabel("relative frequency", color="#2a9d8f")
        ax2.tick_params(axis="y", labelcolor="#2a9d8f")
        for xi, a in zip(x, top["accuracy"]):
            ax1.text(xi, 0, f"{a:.2f}", ha="center", va="bottom", fontsize=7, rotation=90, color="black")
        ax1.set_xticks(x); ax1.set_xticklabels(top["pair"], rotation=90, fontsize=8)
        ax1.set_title(f"Per H-bond-type loss & frequency (acc annotated) — epoch {epoch}")
        for sp in ("top",):
            ax1.spines[sp].set_visible(False); ax2.spines[sp].set_visible(False)
        fig.tight_layout(); fig.savefig(self.save_dir / f"epoch_{epoch:04d}_per_type.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _update_summary(self, epoch, ap_hb, auc_dir, ap_sb, *, n_hb, n_hb_pos, n_salt_pos, n_dir, plt):
        import pandas as pd
        path = self.save_dir / "summary.csv"
        row = dict(epoch=epoch, ap_hbond=ap_hb, auc_dir=auc_dir, ap_salt=ap_sb,
                   n_hbond_edges=n_hb, n_hbond_pos=n_hb_pos, n_salt_pos=n_salt_pos, n_dir_edges=n_dir)
        df = pd.concat([pd.read_csv(path), pd.DataFrame([row])], ignore_index=True) if path.exists() \
            else pd.DataFrame([row])
        df = df.drop_duplicates(subset="epoch", keep="last").sort_values("epoch")
        df.to_csv(path, index=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df["epoch"], df["ap_hbond"], "-o", label="AP H-bond", color="#2a9d8f")
        ax.plot(df["epoch"], df["auc_dir"], "-o", label="AUC donor/acceptor", color="#e76f51")
        ax.plot(df["epoch"], df["ap_salt"], "-o", label="AP salt", color="#264653")
        ax.set_xlabel("epoch"); ax.set_ylabel("score"); ax.set_ylim(0, 1); ax.legend()
        ax.set_title("H-bond head — validation metrics vs epoch")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout(); fig.savefig(self.save_dir / "learning_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
