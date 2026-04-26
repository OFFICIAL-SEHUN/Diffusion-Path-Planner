"""
Visual Backbone Ablation Study — Training Logger
=================================================

CSV columns
───────────
Primary  (printed to stdout + logged at every val_interval):
  1. val_loss            — held-out noise-prediction MSE
  2. isr                 — mean Instruction Success Rate [0, 1]
  3. mean_cot            — mean cumulative CoT (J)
  4. mean_risk           — mean risk_integral (deg·steps, threshold=15°)
  5. inference_latency_s — mean DDPM sampling time per path (s)
  6. param_count         — trainable parameter count  (populated once at epoch 1)

Secondary (log-only, for Appendix; every epoch or val_interval):
  7. train_loss          — epoch-average training MSE
  8. lr                  — current learning rate
  9. best_val_epoch      — epoch of the lowest val_loss seen so far
 10. max_risk            — max risk_integral across val samples
 11. isr_{intent}        — per-intent ISR (one column per known intent type)
 12. snapshot_path       — path to qualitative path PNG

Log files written to  <log_dir>/
  backbone_ablation_<backbone>_<timestamp>.csv   ← primary + secondary tabular log
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional


# All known intent types (+ composite ones).  Column names use sanitised keys.
KNOWN_INTENTS: list[str] = [
    "baseline",
    "left_bias",
    "right_bias",
    "center_bias",
    "avoid_steep",
    "prefer_flat",
    "minimize_elevation_change",
    "short_path",
    "energy_efficient",
    "left_bias+avoid_steep",
    "right_bias+prefer_flat",
    "center_bias+prefer_flat",
    "short_path+avoid_steep",
    "energy_efficient+minimize_elevation_change",
]

# Primary columns (printed to stdout)
_PRIMARY = [
    "val_loss",
    "isr",
    "mean_cot",
    "mean_risk",
    "inference_latency_s",
    "param_count",
]

# Secondary columns (log-only)
_SECONDARY = [
    "train_loss",
    "lr",
    "best_val_epoch",
    "max_risk",
    "epoch_time_s",
    "val_time_s",
    "total_time_s",
] + [f"isr_{k.replace('+', '_')}" for k in KNOWN_INTENTS] + [
    "snapshot_path",
]

_ALL_COLUMNS = ["epoch"] + _PRIMARY + _SECONDARY


def _intent_col(intent_type: str) -> str:
    return f"isr_{intent_type.replace('+', '_')}"


class AblationLogger:
    """Structured CSV logger for the Visual Backbone Ablation Study.

    Usage pattern inside the training loop::

        logger = AblationLogger(backbone="convnext", log_dir="logs/ablation", param_count=n)
        for epoch in range(1, epochs + 1):
            ...train...
            logger.accumulate(train_loss=avg_loss, lr=current_lr)

            if epoch % val_loss_interval == 0:
                logger.accumulate(val_loss=vl)

            if epoch % val_interval == 0:
                logger.accumulate(
                    isr=0.82, mean_cot=12.3, mean_risk=4.5,
                    inference_latency_s=0.34, max_risk=9.1,
                    isr_per_intent={"baseline": 0.9, "avoid_steep": 0.75},
                    snapshot_path="results/train_vis/sample_epoch_02000.png",
                )
                logger.print_primary_summary(epoch)

            logger.flush(epoch)
        logger.close()
    """

    def __init__(
        self,
        backbone: str,
        log_dir: str,
        param_count: int,
    ) -> None:
        self._backbone = backbone
        self._param_count = param_count
        self._best_val_loss = float("inf")
        self._best_val_epoch: int = -1
        self._buf: dict = {}
        self._train_start_time: float = time.time()

        ts = time.strftime("%Y%m%d_%H%M%S")
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self._csv_path = os.path.join(
            log_dir, f"backbone_ablation_{backbone}_{ts}.csv"
        )
        self._fh = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._fh,
            fieldnames=_ALL_COLUMNS,
            extrasaction="ignore",
            restval="",
        )
        self._writer.writeheader()
        self._fh.flush()

        print(f"[AblationLogger] Log → {self._csv_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def accumulate(
        self,
        *,
        train_loss: Optional[float] = None,
        lr: Optional[float] = None,
        val_loss: Optional[float] = None,
        isr: Optional[float] = None,
        mean_cot: Optional[float] = None,
        mean_risk: Optional[float] = None,
        inference_latency_s: Optional[float] = None,
        max_risk: Optional[float] = None,
        isr_per_intent: Optional[Dict[str, float]] = None,
        snapshot_path: Optional[str] = None,
        epoch_time_s: Optional[float] = None,
        val_time_s: Optional[float] = None,
    ) -> None:
        """Buffer metric values for the current epoch.  Call multiple times
        per epoch; all values are merged and written together at :meth:`flush`.
        """
        if train_loss is not None:
            self._buf["train_loss"] = train_loss
        if lr is not None:
            self._buf["lr"] = lr
        if val_loss is not None:
            self._buf["val_loss"] = val_loss
        if isr is not None:
            self._buf["isr"] = isr
        if mean_cot is not None:
            self._buf["mean_cot"] = mean_cot
        if mean_risk is not None:
            self._buf["mean_risk"] = mean_risk
        if inference_latency_s is not None:
            self._buf["inference_latency_s"] = inference_latency_s
        if max_risk is not None:
            self._buf["max_risk"] = max_risk
        if isr_per_intent is not None:
            for intent, val in isr_per_intent.items():
                self._buf[_intent_col(intent)] = val
        if snapshot_path is not None:
            self._buf["snapshot_path"] = snapshot_path
        if epoch_time_s is not None:
            self._buf["epoch_time_s"] = epoch_time_s
        if val_time_s is not None:
            self._buf["val_time_s"] = val_time_s

    def flush(self, epoch: int) -> None:
        """Write the buffered metrics for *epoch* to the CSV and reset the buffer."""
        # Keep best_val_epoch up to date before writing
        val_loss = self._buf.get("val_loss")
        if val_loss is not None and val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._best_val_epoch = epoch

        row: dict = {col: "" for col in _ALL_COLUMNS}
        row["epoch"] = epoch
        row["param_count"] = self._param_count
        row["best_val_epoch"] = self._best_val_epoch if self._best_val_epoch >= 0 else ""
        row["total_time_s"] = time.time() - self._train_start_time
        row.update(self._buf)

        self._writer.writerow(row)
        self._fh.flush()
        self._buf = {}

    def print_summary(self, epoch: int, total_epochs: int = 0) -> None:
        """Print all logged metrics (primary + secondary) for the current epoch."""
        ep_str = f"{epoch}" + (f"/{total_epochs}" if total_epochs else "")

        def _fmt(key: str, fmt: str = ".4f") -> str:
            v = self._buf.get(key, "")
            if v == "":
                return "—"
            try:
                return format(float(v), fmt)
            except (TypeError, ValueError):
                return str(v)

        # Collect per-intent ISR rows (only intents that have data)
        intent_rows = []
        for col in _ALL_COLUMNS:
            if col.startswith("isr_") and col != "isr":
                v = self._buf.get(col, "")
                if v != "":
                    label = col[4:].replace("_", " ")   # strip "isr_", humanise
                    try:
                        intent_rows.append((label, format(float(v), ".4f")))
                    except (TypeError, ValueError):
                        pass

        best_ep = self._best_val_epoch if self._best_val_epoch >= 0 else "—"
        total_elapsed = time.time() - self._train_start_time

        def _fmt_time(seconds: float) -> str:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = seconds % 60
            if h > 0:
                return f"{h}h {m:02d}m {s:04.1f}s"
            if m > 0:
                return f"{m}m {s:04.1f}s"
            return f"{s:.2f}s"

        W = 64
        lines = [
            f"\n{'═'*W}",
            f"  Backbone Ablation Log  [Epoch {ep_str}]  backbone={self._backbone}",
            f"{'─'*W}",
            f"  {'【 Primary 】':<30}",
            f"  {'─'*(W//2)}",
            f"  {'Val Loss':<28}  {_fmt('val_loss'):>10}",
            f"  {'ISR':<28}  {_fmt('isr'):>10}",
            f"  {'Mean CoT':<28}  {_fmt('mean_cot', '.3f'):>10}",
            f"  {'Mean Risk':<28}  {_fmt('mean_risk', '.3f'):>10}",
            f"  {'Inference Latency (s)':<28}  {_fmt('inference_latency_s', '.4f'):>10}",
            f"  {'Param Count':<28}  {self._param_count:>10,}",
            f"{'─'*W}",
            f"  {'【 Secondary 】':<30}",
            f"  {'─'*(W//2)}",
            f"  {'Train Loss':<28}  {_fmt('train_loss'):>10}",
            f"  {'Learning Rate':<28}  {_fmt('lr', '.2e'):>10}",
            f"  {'Best Val Epoch':<28}  {str(best_ep):>10}",
            f"  {'Max Risk':<28}  {_fmt('max_risk', '.3f'):>10}",
            f"{'─'*W}",
            f"  {'【 Timing 】':<30}",
            f"  {'─'*(W//2)}",
            f"  {'Epoch Time':<28}  {_fmt_time(self._buf['epoch_time_s']) if 'epoch_time_s' in self._buf else '—':>10}",
            f"  {'Val Time':<28}  {_fmt_time(self._buf['val_time_s']) if 'val_time_s' in self._buf else '—':>10}",
            f"  {'Total Elapsed':<28}  {_fmt_time(total_elapsed):>10}",
        ]

        if intent_rows:
            lines.append(f"  {'─'*(W//2)}")
            lines.append(f"  {'Intent-wise ISR':<28}")
            for label, val in intent_rows:
                lines.append(f"    {label:<26}  {val:>10}")

        snap = self._buf.get("snapshot_path", "")
        if snap:
            lines.append(f"  {'─'*(W//2)}")
            lines.append(f"  Snapshot → {snap}")

        lines.append(f"{'═'*W}")
        print("\n".join(lines))

    # Keep old name as alias so any external callers don't break
    def print_primary_summary(self, epoch: int, total_epochs: int = 0) -> None:
        self.print_summary(epoch, total_epochs)

    def close(self) -> None:
        """Flush and close the log file."""
        self._fh.flush()
        self._fh.close()
        print(f"[AblationLogger] Closed log → {self._csv_path}")

    @property
    def csv_path(self) -> str:
        return self._csv_path

    @property
    def best_val_epoch(self) -> int:
        return self._best_val_epoch
