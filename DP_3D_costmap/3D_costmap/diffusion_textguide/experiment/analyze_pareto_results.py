#!/usr/bin/env python3
"""
Paper-ready analysis of Pareto sweep results (sweep_results.json).

Usage:
  python -m experiment.analyze_pareto_results \\
      --input results/pareto/sweep_results.json \\
      --output_dir results/analyze_pareto

Outputs are grouped under *output_dir*:
  - tables/     CSV exports
  - markdown/   Markdown tables + narrative summaries
  - figures/    PNG plots (2-D scatters + summaries)
  - figures/3d/   ``pareto_3d_weight_frontier.png`` — Pareto front in
                **(α,β,γ,δ) space** using objectives **mean**-aggregated across intents.
  - figures/      ``pareto_weights_*.png`` — 2-D projections of that same weight-level front.
                Optional per-intent figures: ``--per-intent-pareto-plots``.

Rows with **non-finite** ``cot_mean`` / ``risk_mean`` / ``isr_mean`` are dropped before
aggregation.

**Primary Pareto front** (figures + ``weight_pareto_front`` table): decision variables are
**(α, β, γ, δ)**. Each tuple is mapped to objectives by **averaging** ``cot_mean``,
``risk_mean``, ``isr_mean`` over all intents in the sweep; then non-dominated points are
taken w.r.t. mean cot ↓, mean risk ↓, mean isr ↑.

**Optional** per-intent fronts (same objectives, fixed intent): ``--per-intent-pareto-plots``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, TriAnalyzer


OBJECTIVES = ["cot_mean", "risk_mean", "isr_mean"]
DIRECTIONS = ["min", "min", "max"]


def _safe_pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation; returns nan if undefined (too few samples or zero variance).

    Avoids ``np.corrcoef`` internal divisions that emit RuntimeWarning when std is 0.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape[0] < 2 or y.shape[0] != x.shape[0]:
        return float("nan")
    sx = float(np.std(x, ddof=1))
    sy = float(np.std(y, ddof=1))
    if not (np.isfinite(sx) and np.isfinite(sy)) or sx <= 0.0 or sy <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _paper_rcparams() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def load_results(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    for c in OBJECTIVES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("feasibility", "alpha", "beta", "gamma", "delta", "n_success", "n_total"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def filter_finite_objectives(df: pd.DataFrame, cols: Optional[list[str]] = None) -> tuple[pd.DataFrame, int]:
    """Drop rows where any listed objective is non-finite (needed for Pareto math)."""
    use_cols = cols or OBJECTIVES
    if df.empty:
        return df.copy(), 0
    n_before = len(df)
    mask = pd.Series(True, index=df.index)
    for c in use_cols:
        if c not in df.columns:
            return df.iloc[0:0].copy(), n_before
        v = pd.to_numeric(df[c], errors="coerce")
        mask &= np.isfinite(v.to_numpy(dtype=float))
    out = df.loc[mask].copy()
    return out, n_before - len(out)


def format_weights(a: float, b: float, g: float, d: float) -> str:
    return f"({a:g}, {b:g}, {g:g}, {d:g})"


def is_dominated(
    values: np.ndarray,
    other: np.ndarray,
    directions: list[str],
) -> bool:
    """Return True if *values* is dominated by *other* (other is strictly better on at least one, not worse on any)."""
    if np.any(np.isnan(values)) or np.any(np.isnan(other)):
        return False
    strictly_better = False
    for k, direction in enumerate(directions):
        vr, vo = values[k], other[k]
        if direction == "min":
            if vo > vr:
                return False
            if vo < vr:
                strictly_better = True
        else:  # max
            if vo < vr:
                return False
            if vo > vr:
                strictly_better = True
    return strictly_better


def pareto_front(
    df: pd.DataFrame,
    objective_cols: Optional[list[str]] = None,
    directions: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Non-dominated rows (default: cot↓ risk↓ isr↑). Non-finite objectives excluded."""
    cols = objective_cols or OBJECTIVES
    dirs = directions or DIRECTIONS
    if len(cols) != len(dirs):
        raise ValueError("objective_cols and directions must have the same length")
    if df.empty:
        return df.copy()
    sub = df.copy()
    for c in cols:
        if c not in sub.columns:
            return sub.iloc[0:0]
        v = pd.to_numeric(sub[c], errors="coerce")
        sub = sub[np.isfinite(v.to_numpy(dtype=float))]
    if sub.empty:
        return sub
    mat = sub[cols].to_numpy(dtype=np.float64)
    idx_keep = []
    for i in range(len(sub)):
        vi = mat[i]
        dominated = False
        for j in range(len(sub)):
            if i == j:
                continue
            if is_dominated(vi, mat[j], dirs):
                dominated = True
                break
        if not dominated:
            idx_keep.append(sub.index[i])
    return sub.loc[idx_keep]


CROSS_OBJECTIVES = ["mean_cot", "mean_risk", "mean_isr"]


def save_markdown_table(df: pd.DataFrame, path: Path, float_fmt: str = "%.4f") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(str(c) for c in df.columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in df.columns) + " |")
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, (float, np.floating)) and np.isfinite(v):
                if float_fmt and abs(v) < 1e6:
                    cells.append(float_fmt % v)
                else:
                    cells.append(str(v))
            elif pd.isna(v):
                cells.append("")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_intent_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for intent in sorted(df["intent"].dropna().unique()):
        sub = df[df["intent"] == intent]
        n_tup = len(sub)
        front = pareto_front(sub)
        pf_sz = len(front)
        feas = sub["feasibility"] if "feasibility" in sub.columns else pd.Series(dtype=float)
        rows.append(
            {
                "intent": intent,
                "number_of_weight_tuples": int(n_tup),
                "pareto_front_size": int(pf_sz),
                "best_cot": float(sub["cot_mean"].min(skipna=True)) if n_tup else np.nan,
                "best_risk": float(sub["risk_mean"].min(skipna=True)) if n_tup else np.nan,
                "best_isr": float(sub["isr_mean"].max(skipna=True)) if n_tup else np.nan,
                "best_feasibility": float(feas.max(skipna=True)) if len(feas) else np.nan,
                "mean_cot_across_all_tuples": float(sub["cot_mean"].mean(skipna=True)) if n_tup else np.nan,
                "mean_risk_across_all_tuples": float(sub["risk_mean"].mean(skipna=True)) if n_tup else np.nan,
                "mean_isr_across_all_tuples": float(sub["isr_mean"].mean(skipna=True)) if n_tup else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_best_candidate_table(df: pd.DataFrame) -> pd.DataFrame:
    criteria = [
        ("minimum cot_mean", "cot_mean", "idxmin"),
        ("minimum risk_mean", "risk_mean", "idxmin"),
        ("maximum isr_mean", "isr_mean", "idxmax"),
        ("maximum feasibility", "feasibility", "idxmax"),
    ]
    rows = []
    for intent in sorted(df["intent"].dropna().unique()):
        sub = df[df["intent"] == intent].copy()
        if sub.empty:
            continue
        for name, col, how in criteria:
            if col not in sub.columns:
                continue
            s = sub[col].dropna()
            if s.empty:
                continue
            if how == "idxmin":
                idx = s.idxmin()
            else:
                idx = s.idxmax()
            r = sub.loc[idx]
            rows.append(
                {
                    "intent": intent,
                    "criterion_selected_by": name,
                    "alpha": r["alpha"],
                    "beta": r["beta"],
                    "gamma": r["gamma"],
                    "delta": r["delta"],
                    "cot_mean": r.get("cot_mean", np.nan),
                    "risk_mean": r.get("risk_mean", np.nan),
                    "isr_mean": r.get("isr_mean", np.nan),
                    "feasibility": r.get("feasibility", np.nan),
                }
            )
    return pd.DataFrame(rows)


def build_weight_pareto_front_table(cross: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated (α,β,γ,δ) tuples in (mean_cot, mean_risk, mean_isr) space."""
    cc, _ = filter_finite_objectives(cross, cols=CROSS_OBJECTIVES)
    if cc.empty:
        return pd.DataFrame()
    return pareto_front(cc, objective_cols=CROSS_OBJECTIVES).reset_index(drop=True)


def build_pareto_fronts_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for intent in sorted(df["intent"].dropna().unique()):
        sub = df[df["intent"] == intent]
        front = pareto_front(sub)
        for _, r in front.iterrows():
            rows.append(
                {
                    "intent": intent,
                    "alpha": r["alpha"],
                    "beta": r["beta"],
                    "gamma": r["gamma"],
                    "delta": r["delta"],
                    "cot_mean": r["cot_mean"],
                    "risk_mean": r["risk_mean"],
                    "isr_mean": r["isr_mean"],
                    "feasibility": r.get("feasibility", np.nan),
                }
            )
    return pd.DataFrame(rows)


def build_cross_intent_selection(df: pd.DataFrame) -> pd.DataFrame:
    gcols = ["alpha", "beta", "gamma", "delta"]
    agg = (
        df.groupby(gcols, dropna=False)
        .agg(
            mean_feasibility=("feasibility", "mean"),
            min_feasibility=("feasibility", "min"),
            mean_cot=("cot_mean", "mean"),
            mean_risk=("risk_mean", "mean"),
            mean_isr=("isr_mean", "mean"),
            min_isr=("isr_mean", "min"),
            number_of_intents_covered=("intent", "nunique"),
        )
        .reset_index()
    )
    agg["pass_mean_feasibility_0_90"] = agg["mean_feasibility"] >= 0.90
    agg["pass_min_feasibility_0_80"] = agg["min_feasibility"] >= 0.80
    agg["pass_mean_isr_0_70"] = agg["mean_isr"] >= 0.70
    agg["pass_min_isr_0_50"] = agg["min_isr"] >= 0.50
    agg = agg.sort_values(
        by=["mean_isr", "mean_cot", "mean_risk"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    return agg


def _normalize_for_utopia(
    arr: np.ndarray, direction: str
) -> np.ndarray:
    """Map to [0,1] where 1 is best (min objective → invert)."""
    a = np.asarray(arr, dtype=np.float64)
    if np.all(~np.isfinite(a)):
        return np.zeros_like(a)
    lo = np.nanmin(a)
    hi = np.nanmax(a)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-15:
        return np.zeros_like(a)
    n = (a - lo) / (hi - lo)
    if direction == "min":
        return 1.0 - n
    return n


def select_recommended_tuples(cross: pd.DataFrame) -> dict[str, Optional[pd.Series]]:
    out: dict[str, Optional[pd.Series]] = {
        "balanced": None,
        "energy_favoring": None,
        "instruction_favoring": None,
    }
    if cross.empty:
        return out

    mask_all = (
        cross["pass_mean_feasibility_0_90"]
        & cross["pass_min_feasibility_0_80"]
        & cross["pass_mean_isr_0_70"]
        & cross["pass_min_isr_0_50"]
    )
    pool = cross[mask_all].copy()
    if not pool.empty:
        c = pool["mean_cot"].to_numpy()
        r = pool["mean_risk"].to_numpy()
        i = pool["mean_isr"].to_numpy()
        cn = _normalize_for_utopia(c, "min")
        rn = _normalize_for_utopia(r, "min")
        inn = _normalize_for_utopia(i, "max")
        dist = np.sqrt((1.0 - cn) ** 2 + (1.0 - rn) ** 2 + (1.0 - inn) ** 2)
        pool = pool.assign(_utopia_dist=dist)
        out["balanced"] = pool.loc[pool["_utopia_dist"].idxmin()].drop(labels=["_utopia_dist"])

    en_mask = (cross["mean_feasibility"] >= 0.90) & (cross["mean_isr"] >= 0.60)
    pool_e = cross[en_mask]
    if not pool_e.empty:
        out["energy_favoring"] = pool_e.loc[pool_e["mean_cot"].idxmin()]

    ins_mask = cross["mean_feasibility"] >= 0.90
    pool_i = cross[ins_mask]
    if not pool_i.empty:
        out["instruction_favoring"] = pool_i.loc[pool_i["mean_isr"].idxmax()]

    return out


def _safe_filename_intent(intent: str) -> str:
    s = intent.replace("+", "_plus_")
    for ch in ("/", "\\"):
        s = s.replace(ch, "_")
    return s


def plot_pareto_cot_risk(df: pd.DataFrame, intent: str, out_base: Path) -> None:
    sub = df[df["intent"] == intent]
    front = pareto_front(sub)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    if front.empty:
        ax.text(0.5, 0.5, "No Pareto front (no finite objectives)", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.scatter(
            front["cot_mean"],
            front["risk_mean"],
            s=48,
            alpha=0.85,
            edgecolors="none",
            label="Pareto front",
        )
        top = front.nlargest(5, "isr_mean")
        for _, r in top.iterrows():
            lbl = format_weights(r["alpha"], r["beta"], r["gamma"], r["delta"])
            ax.annotate(
                lbl,
                (r["cot_mean"], r["risk_mean"]),
                fontsize=6,
                alpha=0.85,
                xytext=(4, 4),
                textcoords="offset points",
            )
    ax.set_xlabel("cot_mean (↓)")
    ax.set_ylabel("risk_mean (↓)")
    ax.set_title(f"Pareto front — CoT vs risk — {intent}")
    if not front.empty:
        ax.legend(loc="best")
    fig.tight_layout()
    stem = out_base / f"pareto_cot_risk_{_safe_filename_intent(intent)}"
    fig.savefig(f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def plot_pareto_3d_weight_frontier(cross: pd.DataFrame, out_base: Path) -> None:
    """3-D Pareto front over weight tuples (α,β,γ,δ); mesh = Delaunay in (mean_cot, mean_risk)."""
    req = ["mean_cot", "mean_risk", "mean_isr", "mean_feasibility"]
    if cross.empty or not all(c in cross.columns for c in req):
        return
    cc, _ = filter_finite_objectives(cross, cols=CROSS_OBJECTIVES)
    if cc.empty:
        return
    front = pareto_front(cc, objective_cols=CROSS_OBJECTIVES)
    if front.empty:
        return
    xc = front["mean_cot"].to_numpy(dtype=float)
    yc = front["mean_risk"].to_numpy(dtype=float)
    zc = front["mean_isr"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    if len(xc) >= 3 and np.ptp(xc) > 1e-15 and np.ptp(yc) > 1e-15:
        try:
            tri = Triangulation(xc, yc)
            try:
                mask = TriAnalyzer(tri).get_flat_tri_mask(min_circle_ratio=0.06)
                tri.set_mask(mask)
            except Exception:
                pass
            ax.plot_trisurf(
                tri,
                zc,
                alpha=0.42,
                cmap="coolwarm",
                linewidth=0.25,
                edgecolor="0.35",
                antialiased=True,
                shade=True,
            )
        except (RuntimeError, ValueError):
            pass

    feas = front["mean_feasibility"].fillna(0).to_numpy(dtype=float)
    if np.ptp(feas) > 1e-9:
        fe_n = (feas - feas.min()) / (feas.max() - feas.min())
        sm = np.clip(6 + 22 * fe_n, 8, 32)
    else:
        sm = np.full_like(xc, 14.0, dtype=float)

    ax.scatter(
        xc,
        yc,
        zc,
        s=sm,
        alpha=0.95,
        depthshade=True,
        c=zc,
        cmap="coolwarm",
        edgecolors="0.15",
        linewidths=0.2,
    )

    top = front.nlargest(5, "mean_isr")
    for _, r in top.iterrows():
        lbl = format_weights(r["alpha"], r["beta"], r["gamma"], r["delta"])
        ax.text(
            float(r["mean_cot"]),
            float(r["mean_risk"]),
            float(r["mean_isr"]),
            lbl,
            fontsize=5,
        )
    ax.set_xlabel("mean_cot (↓, over intents)")
    ax.set_ylabel("mean_risk (↓, over intents)")
    ax.set_zlabel("mean_isr (↑, over intents)")
    ax.set_title(
        "Pareto front over (α, β, γ, δ)\n"
        "(Delaunay mesh in mean_cot–mean_risk; height = mean_isr)"
    )
    ax.view_init(elev=22, azim=42)
    fig.tight_layout()
    fig.savefig(out_base / "pareto_3d_weight_frontier.png", bbox_inches="tight")
    plt.close(fig)


def plot_pareto_3d_objectives(df: pd.DataFrame, intent: str, out_base: Path) -> None:
    """Optional: Pareto front in raw (cot,risk,isr) for one intent only (--per-intent-pareto-plots)."""
    front = pareto_front(df[df["intent"] == intent])
    if front.empty:
        return
    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        front["cot_mean"],
        front["risk_mean"],
        front["isr_mean"],
        s=52,
        alpha=0.88,
        depthshade=True,
        label="Pareto front",
        edgecolors="black",
        linewidths=0.25,
    )
    for _, r in front.nlargest(3, "isr_mean").iterrows():
        lbl = format_weights(r["alpha"], r["beta"], r["gamma"], r["delta"])
        ax.text(
            float(r["cot_mean"]),
            float(r["risk_mean"]),
            float(r["isr_mean"]),
            lbl,
            fontsize=6,
            alpha=0.9,
        )
    ax.set_xlabel("cot_mean (↓)")
    ax.set_ylabel("risk_mean (↓)")
    ax.set_zlabel("isr_mean (↑)")
    ax.set_title(f"Pareto front (per intent) — {intent}")
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=22, azim=42)
    fig.tight_layout()
    stem = out_base / f"pareto_3d_{_safe_filename_intent(intent)}"
    fig.savefig(f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def plot_pareto_cot_isr(df: pd.DataFrame, intent: str, out_base: Path) -> None:
    sub = df[df["intent"] == intent]
    front = pareto_front(sub)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    if front.empty:
        ax.text(0.5, 0.5, "No Pareto front (no finite objectives)", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.scatter(
            front["cot_mean"],
            front["isr_mean"],
            s=48,
            alpha=0.85,
            edgecolors="none",
            label="Pareto front",
        )
        top = front.nsmallest(5, "risk_mean")
        for _, r in top.iterrows():
            lbl = format_weights(r["alpha"], r["beta"], r["gamma"], r["delta"])
            ax.annotate(
                lbl,
                (r["cot_mean"], r["isr_mean"]),
                fontsize=6,
                alpha=0.85,
                xytext=(4, 4),
                textcoords="offset points",
            )
    ax.set_xlabel("cot_mean (↓)")
    ax.set_ylabel("isr_mean (↑)")
    ax.set_title(f"Pareto front — CoT vs ISR — {intent}")
    if not front.empty:
        ax.legend(loc="best")
    fig.tight_layout()
    stem = out_base / f"pareto_cot_isr_{_safe_filename_intent(intent)}"
    fig.savefig(f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def plot_pareto_front_size_summary(summary: pd.DataFrame, out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(summary))
    ax.bar(x, summary["pareto_front_size"].to_numpy())
    ax.set_xticks(x)
    ax.set_xticklabels(summary["intent"], rotation=45, ha="right")
    ax.set_ylabel("Pareto front size")
    ax.set_title("Pareto front size by intent")
    fig.tight_layout()
    fig.savefig(out_base / "pareto_front_size_summary.png", bbox_inches="tight")
    plt.close(fig)


def plot_best_metric_by_intent(summary: pd.DataFrame, metric_col: str, ylabel: str, fname: str, out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(summary)), summary[metric_col].to_numpy(), marker="o", markersize=4)
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(summary["intent"], rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} (best per intent across sweep)")
    fig.tight_layout()
    fig.savefig(out_base / f"{fname}.png", bbox_inches="tight")
    plt.close(fig)


def plot_cross_intent_tradeoff_scatter(cross: pd.DataFrame, out_base: Path) -> None:
    cc, _ = filter_finite_objectives(cross, cols=CROSS_OBJECTIVES)
    front = pareto_front(cc, objective_cols=CROSS_OBJECTIVES)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if front.empty:
        ax.text(0.5, 0.5, "No cross-intent Pareto front", ha="center", va="center", transform=ax.transAxes)
    else:
        sizes = np.clip(40 + 200 * front["mean_feasibility"].fillna(0).to_numpy(), 15, 350)
        ax.scatter(front["mean_cot"], front["mean_isr"], s=sizes, alpha=0.72, edgecolors="none")
        top10 = front.nlargest(10, "mean_isr")
        for _, r in top10.iterrows():
            lbl = format_weights(r["alpha"], r["beta"], r["gamma"], r["delta"])
            ax.annotate(lbl, (r["mean_cot"], r["mean_isr"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("mean_cot (↓, mean over intents)")
    ax.set_ylabel("mean_isr (↑, mean over intents)")
    ax.set_title("Pareto front over (α,β,γ,δ): mean CoT vs mean ISR")
    fig.tight_layout()
    fig.savefig(out_base / "pareto_weights_mean_cot_mean_isr.png", bbox_inches="tight")
    plt.close(fig)


def plot_cross_intent_risk_isr_scatter(cross: pd.DataFrame, out_base: Path) -> None:
    cc, _ = filter_finite_objectives(cross, cols=CROSS_OBJECTIVES)
    front = pareto_front(cc, objective_cols=CROSS_OBJECTIVES)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if front.empty:
        ax.text(0.5, 0.5, "No cross-intent Pareto front", ha="center", va="center", transform=ax.transAxes)
    else:
        sizes = np.clip(40 + 200 * front["mean_feasibility"].fillna(0).to_numpy(), 15, 350)
        ax.scatter(front["mean_risk"], front["mean_isr"], s=sizes, alpha=0.72, edgecolors="none")
        top10 = front.nlargest(10, "mean_isr")
        for _, r in top10.iterrows():
            lbl = format_weights(r["alpha"], r["beta"], r["gamma"], r["delta"])
            ax.annotate(lbl, (r["mean_risk"], r["mean_isr"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("mean_risk (↓, mean over intents)")
    ax.set_ylabel("mean_isr (↑, mean over intents)")
    ax.set_title("Pareto front over (α,β,γ,δ): mean risk vs mean ISR")
    fig.tight_layout()
    fig.savefig(out_base / "pareto_weights_mean_risk_mean_isr.png", bbox_inches="tight")
    plt.close(fig)


def _tradeoff_strength_phrase(sub: pd.DataFrame, col_x: str, col_y: str, _dir_x: str = "", _dir_y: str = "") -> str:
    front = pareto_front(sub)
    if len(front) < 3:
        return "unclear (too few Pareto points)."
    x = front[col_x].to_numpy(dtype=np.float64)
    y = front[col_y].to_numpy(dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return "unclear (insufficient valid Pareto samples)."
    c = _safe_pearson_r(x, y)
    if not np.isfinite(c):
        return "unclear (correlation undefined)."
    xr = float(np.nanmax(x) - np.nanmin(x))
    yr = float(np.nanmax(y) - np.nanmin(y))
    bx = sub[col_x].to_numpy(dtype=np.float64)
    med_x = float(np.nanmedian(bx[np.isfinite(bx)])) if np.any(np.isfinite(bx)) else 0.0
    spread_note = (xr > med_x * 0.05) or (yr > 0.05)
    # Expected: cot vs isr often negative
    if abs(c) < 0.25:
        return f"possibly weak (|ρ|≈{abs(c):.2f}); objectives may be loosely coupled on this front."
    if spread_note:
        return f"moderate (|ρ|≈{abs(c):.2f}); spread across the front suggests non-trivial trade-offs."
    return f"notably structured (|ρ|≈{abs(c):.2f}); interpret with terrain and sampling variability in mind."


def _balanced_tuple_for_intent(sub: pd.DataFrame) -> str:
    front = pareto_front(sub)
    if front.empty:
        return "n/a (no valid Pareto front)."
    c = front["cot_mean"].to_numpy()
    r = front["risk_mean"].to_numpy()
    i = front["isr_mean"].to_numpy()
    cn = _normalize_for_utopia(c, "min")
    rn = _normalize_for_utopia(r, "min")
    inn = _normalize_for_utopia(i, "max")
    dist = np.sqrt((1.0 - cn) ** 2 + (1.0 - rn) ** 2 + (1.0 - inn) ** 2)
    j = int(np.nanargmin(dist))
    row = front.iloc[j]
    return format_weights(row["alpha"], row["beta"], row["gamma"], row["delta"])


def write_recommended_tuples_md(path: Path, recs: dict[str, Optional[pd.Series]]) -> None:
    lines = ["# Recommended weight tuples", ""]
    blurbs = {
        "balanced": "Balances mean CoT, mean risk, and mean ISR (after crossing gates on feasibility/ISR floors).",
        "energy_favoring": "Favors low mean CoT while keeping mean feasibility and ISR moderately high.",
        "instruction_favoring": "Favors high mean ISR with a feasibility floor; may trade higher CoT.",
    }
    for key, title in [
        ("balanced", "Balanced (multi-gate utopia distance)"),
        ("energy_favoring", "Energy-favoring"),
        ("instruction_favoring", "Instruction-favoring"),
    ]:
        lines.append(f"## {title}")
        lines.append("")
        s = recs.get(key)
        if s is None:
            lines.append("*No tuple satisfied the selection criteria.*")
            lines.append("")
            continue
        lines.append(f"- **Weights:** `{format_weights(s['alpha'], s['beta'], s['gamma'], s['delta'])}`")
        lines.append(f"- **mean_feasibility:** {s.get('mean_feasibility', float('nan')):.4f}")
        lines.append(f"- **mean_cot:** {s.get('mean_cot', float('nan')):.4f}")
        lines.append(f"- **mean_risk:** {s.get('mean_risk', float('nan')):.4f}")
        lines.append(f"- **mean_isr:** {s.get('mean_isr', float('nan')):.4f}")
        lines.append(f"- **min_isr:** {s.get('min_isr', float('nan')):.4f}")
        lines.append(f"- **Interpretation:** {blurbs[key]}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_analysis_summary(
    path: Path,
    df: pd.DataFrame,
    intent_summary: pd.DataFrame,
    cross: pd.DataFrame,
    recs: dict[str, Optional[pd.Series]],
    n_source_rows: Optional[int] = None,
    weight_pareto_count: Optional[int] = None,
) -> None:
    n_intents = df["intent"].nunique(dropna=True)
    n_tuples = df.groupby(["alpha", "beta", "gamma", "delta"], dropna=False).ngroups

    lines = [
        "# Pareto sweep analysis summary",
        "",
        "## Data loaded",
    ]
    if n_source_rows is not None:
        lines.append(
            f"- Raw JSON rows: **{n_source_rows}**; **{len(df)}** rows used after dropping non-finite "
            "`cot_mean` / `risk_mean` / `isr_mean`."
        )
    else:
        lines.append(f"- Source rows: **{len(df)}** (one per weight tuple × intent).")
    lines.extend(
        [
            f"- Unique intents: **{n_intents}**.",
            f"- Unique weight tuples `(α,β,γ,δ)`: **{n_tuples}**.",
            "",
        ]
    )
    if weight_pareto_count is not None:
        lines.extend(
            [
                "## Pareto front over weight tuples `(α, β, γ, δ)`",
                "",
                f"- Non-dominated weight tuples (mean cot↓, mean risk↓, mean isr↑ over intents): **{weight_pareto_count}**.",
                "- See `tables/weight_pareto_front.csv` and `figures/3d/pareto_3d_weight_frontier.png`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Per-intent notes (same weights, fixed intent; use `--per-intent-pareto-plots` for figures)",
            "",
        ]
    )

    for _, row in intent_summary.iterrows():
        it = row["intent"]
        sub = df[df["intent"] == it]
        lines.append(f"### {it}")
        lines.append(f"- Pareto front size: **{int(row['pareto_front_size'])}**.")
        lines.append(
            f"- CoT–ISR trade-off (on Pareto front): {_tradeoff_strength_phrase(sub, 'cot_mean', 'isr_mean', 'min', 'max')}"
        )
        lines.append(
            f"- CoT–Risk trade-off (on Pareto front): {_tradeoff_strength_phrase(sub, 'cot_mean', 'risk_mean', 'min', 'min')}"
        )
        lines.append(f"- Heuristic balanced tuple (within intent Pareto front): `{_balanced_tuple_for_intent(sub)}`.")
        lines.append("")

    lines.extend(
        [
            "## Cross-intent summary",
            "",
        ]
    )

    if not cross.empty and n_tuples > 0:
        top = cross.iloc[0]
        lines.append(
            f"- Top row after sorting (high mean ISR, then low mean CoT / risk): `{format_weights(top['alpha'], top['beta'], top['gamma'], top['delta'])}`."
        )
        fv = cross["mean_feasibility"].to_numpy()
        iv = cross["mean_isr"].to_numpy()
        cv = cross["mean_cot"].to_numpy()
        mask = np.isfinite(fv) & np.isfinite(iv) & np.isfinite(cv)
        if mask.sum() >= 3:
            rho = _safe_pearson_r(cv[mask], iv[mask])
            if np.isfinite(rho):
                lines.append(
                    f"- Rough correlation across grouped tuples between mean CoT and mean ISR: ρ ≈ **{rho:.2f}** (cautious: not causal)."
                )
    lines.append(
        "- A single `(α,β,γ,δ)` that is near-optimal for every intent is **unlikely**; see `cross_intent_tuple_selection` for `min_isr` / `min_feasibility` spreads."
    )
    lines.append(
        "- Intent-dependent geometry and penalties typically make trade-offs **context-specific** rather than universal."
    )
    lines.append(
        "- Instruction scores (ISR) and energy proxies (CoT) **can** move in opposite directions when grouped; verify on your sweep rather than assuming conflict."
    )
    lines.append("")

    lines.append("## Recommended tuples (cross-intent)")
    for key, label in [
        ("balanced", "Balanced"),
        ("energy_favoring", "Energy-favoring"),
        ("instruction_favoring", "Instruction-favoring"),
    ]:
        s = recs.get(key)
        if s is not None:
            lines.append(
                f"- **{label}:** `{format_weights(s['alpha'], s['beta'], s['gamma'], s['delta'])}` "
                f"(mean ISR={s.get('mean_isr', float('nan')):.3f}, mean CoT={s.get('mean_cot', float('nan')):.3f})."
            )
        else:
            lines.append(f"- **{label}:** *none found under stated gates.*")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Pareto sweep_results.json for paper figures/tables.")
    parser.add_argument("--input", type=str, required=True, help="Path to sweep_results.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for tables and figures")
    parser.add_argument(
        "--per-intent-pareto-plots",
        action="store_true",
        help="Also write per-intent Pareto figures (2-D and optional 3-D per intent). Default is weight-level only.",
    )
    parser.add_argument(
        "--per-intent-3d",
        action="store_true",
        help="With --per-intent-pareto-plots, also write pareto_3d_<intent>.png per intent.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    tables_dir = out_dir / "tables"
    markdown_dir = out_dir / "markdown"
    figures_dir = out_dir / "figures"
    figures_3d_dir = figures_dir / "3d"
    for d in (out_dir, tables_dir, markdown_dir, figures_dir, figures_3d_dir):
        d.mkdir(parents=True, exist_ok=True)

    _paper_rcparams()
    df_raw = load_results(in_path)
    df, n_dropped = filter_finite_objectives(df_raw)
    if n_dropped:
        print(f"Dropped {n_dropped} / {len(df_raw)} rows (non-finite cot_mean, risk_mean, or isr_mean).")
    if df.empty:
        print("No rows with finite objectives; exiting.")
        return

    intent_summary = build_intent_summary(df)
    best_candidates = build_best_candidate_table(df)
    pareto_tbl = build_pareto_fronts_table(df)
    cross = build_cross_intent_selection(df)
    recs = select_recommended_tuples(cross)
    weight_front = build_weight_pareto_front_table(cross)
    weight_front.to_csv(tables_dir / "weight_pareto_front.csv", index=False)
    save_markdown_table(weight_front, markdown_dir / "weight_pareto_front.md")

    intent_summary.to_csv(tables_dir / "intent_summary.csv", index=False)
    save_markdown_table(intent_summary, markdown_dir / "intent_summary.md")

    best_candidates.to_csv(tables_dir / "intent_best_candidates.csv", index=False)
    save_markdown_table(best_candidates, markdown_dir / "intent_best_candidates.md")

    if args.per_intent_pareto_plots:
        pareto_tbl.to_csv(tables_dir / "intent_pareto_fronts.csv", index=False)
        save_markdown_table(pareto_tbl, markdown_dir / "intent_pareto_fronts.md")

    cross.to_csv(tables_dir / "cross_intent_tuple_selection.csv", index=False)
    save_markdown_table(cross, markdown_dir / "cross_intent_tuple_selection.md")

    write_recommended_tuples_md(markdown_dir / "recommended_tuples.md", recs)
    write_analysis_summary(
        markdown_dir / "analysis_summary.md", df, intent_summary, cross, recs,
        n_source_rows=len(df_raw),
        weight_pareto_count=len(weight_front) if not weight_front.empty else 0,
    )

    if not cross.empty:
        plot_pareto_3d_weight_frontier(cross, figures_3d_dir)
        plot_cross_intent_tradeoff_scatter(cross, figures_dir)
        plot_cross_intent_risk_isr_scatter(cross, figures_dir)

    if args.per_intent_pareto_plots:
        for intent in sorted(df["intent"].dropna().unique()):
            plot_pareto_cot_risk(df, intent, figures_dir)
            plot_pareto_cot_isr(df, intent, figures_dir)
            if args.per_intent_3d:
                plot_pareto_3d_objectives(df, intent, figures_3d_dir)
        plot_pareto_front_size_summary(intent_summary, figures_dir)
    plot_best_metric_by_intent(intent_summary, "best_cot", "best_cot (min)", "best_cot_by_intent", figures_dir)
    plot_best_metric_by_intent(intent_summary, "best_risk", "best_risk (min)", "best_risk_by_intent", figures_dir)
    plot_best_metric_by_intent(intent_summary, "best_isr", "best_isr (max)", "best_isr_by_intent", figures_dir)

    print(f"Wrote analysis artifacts to {out_dir.resolve()}")
    print(f"  tables/    → {tables_dir.resolve()}")
    print(f"  markdown/  → {markdown_dir.resolve()}")
    print(f"  figures/   → {figures_dir.resolve()}")
    print(f"  figures/3d → {figures_3d_dir.resolve()} (see pareto_3d_weight_frontier.png)")


if __name__ == "__main__":
    main()
