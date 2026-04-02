"""
δ=0 vs δ=2 비교 그래프 생성 (논문용).

sweep_results.json을 읽고, α=2.0, β=1.5, γ=0.5 에서
δ=0 (intent penalty 없음) vs δ=2 (balanced pick)의 per-intent 지표를 비교.

Usage (from diffusion_textguide/):
  python -m experiment.plot_delta_comparison \
      --sweep results/pareto/sweep_results.json \
      --output-dir results/pareto/figures
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def load_sweep(path):
    with open(path) as f:
        return json.load(f)


def get_row(results, a, b, g, d, intent):
    for r in results:
        if (r["alpha"] == a and r["beta"] == b and r["gamma"] == g
                and r["delta"] == d and r["intent"] == intent):
            return r
    return None


INTENTS = ["baseline", "left_bias", "right_bias",
           "avoid_steep", "prefer_flat", "via_flat_region"]
INTENT_LABELS = ["Baseline", "Left bias", "Right bias",
                 "Avoid steep", "Prefer flat", "Via flat"]


def make_figures(results, out_dir,
                 alpha=2.0, beta=1.5, gamma=0.5,
                 delta_off=0.0, delta_on=2.0):
    os.makedirs(out_dir, exist_ok=True)
    x = np.arange(len(INTENTS))
    w = 0.35

    d0 = {it: get_row(results, alpha, beta, gamma, delta_off, it) for it in INTENTS}
    d2 = {it: get_row(results, alpha, beta, gamma, delta_on, it) for it in INTENTS}

    for it in INTENTS:
        if d0[it] is None or d2[it] is None:
            raise ValueError(f"Missing row for intent={it}. Check alpha/beta/gamma/delta values.")

    # ── Fig 1: ISR bar chart ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    isr0 = [d0[it]["isr_mean"] for it in INTENTS]
    isr2 = [d2[it]["isr_mean"] for it in INTENTS]

    bars0 = ax.bar(x - w / 2, isr0, w,
                   label=rf"$\delta={delta_off:.0f}$ (no intent penalty)",
                   color="#E57373", edgecolor="black", linewidth=0.8, alpha=0.9)
    bars2 = ax.bar(x + w / 2, isr2, w,
                   label=rf"$\delta={delta_on:.0f}$ (balanced, ours)",
                   color="#4CAF50", edgecolor="black", linewidth=0.8, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(INTENT_LABELS, fontsize=10)
    ax.set_ylabel("Instruction Success Rate (ISR)", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title(r"Effect of $\delta$ (Intent Penalty Weight) on ISR",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.axhline(0.8, color="gray", ls="--", lw=1, alpha=0.5)
    ax.text(len(INTENTS) - 0.7, 0.81, "ISR = 0.8", fontsize=8, color="gray")

    for bar, val in zip(bars0, isr0):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color="#B71C1C" if val < 0.8 else "black")
    for bar, val in zip(bars2, isr2):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color="#1B5E20")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"delta_isr_comparison.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved: delta_isr_comparison")

    # ── Fig 2: Multi-metric (ISR, CoT, Risk, Path length) ────────────
    metrics = ["isr_mean", "cot_mean", "risk_mean", "length_m_mean"]
    metric_labels = ["ISR", "Cum. CoT", "Risk Integral", "Path Length (m)"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, mk, ml in zip(axes, metrics, metric_labels):
        v0 = [d0[it][mk] for it in INTENTS]
        v2 = [d2[it][mk] for it in INTENTS]
        ax.bar(x - w / 2, v0, w, label=rf"$\delta={delta_off:.0f}$",
               color="#E57373", edgecolor="black", linewidth=0.5, alpha=0.9)
        ax.bar(x + w / 2, v2, w, label=rf"$\delta={delta_on:.0f}$",
               color="#4CAF50", edgecolor="black", linewidth=0.5, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(INTENT_LABELS, fontsize=8, rotation=30, ha="right")
        ax.set_title(ml, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle(
        rf"$\delta={delta_off:.0f}$ vs $\delta={delta_on:.0f}$: Per-Intent Metric Comparison  "
        rf"($\alpha\!=\!{alpha},\,\beta\!=\!{beta},\,\gamma\!=\!{gamma}$)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"delta_multi_metric.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved: delta_multi_metric")

    # ── Fig 3: Pareto scatter for left_bias, colored by δ ────────────
    subset_lb = [r for r in results if r["intent"] == "left_bias"]
    fig, ax = plt.subplots(figsize=(8, 6))
    cots = [r["cot_mean"] for r in subset_lb]
    isrs = [r["isr_mean"] for r in subset_lb]
    deltas = [r["delta"] for r in subset_lb]

    sc = ax.scatter(cots, isrs, c=deltas, cmap="RdYlGn", s=30, alpha=0.6,
                    edgecolors="gray", linewidth=0.3, vmin=0, vmax=2)
    fig.colorbar(sc, ax=ax, label=r"$\delta$")

    pt0 = d0["left_bias"]
    pt2 = d2["left_bias"]
    ax.scatter([pt0["cot_mean"]], [pt0["isr_mean"]], s=200, marker="X",
               c="red", edgecolors="black", linewidth=1.5, zorder=10,
               label=rf"$\delta\!=\!{delta_off:.0f}$ (ISR={pt0['isr_mean']:.2f})")
    ax.scatter([pt2["cot_mean"]], [pt2["isr_mean"]], s=200, marker="*",
               c="lime", edgecolors="black", linewidth=1.5, zorder=10,
               label=rf"$\delta\!=\!{delta_on:.0f}$ (ISR={pt2['isr_mean']:.2f})")

    ax.set_xlabel("Cumulative CoT (energy)", fontsize=11)
    ax.set_ylabel("ISR (instruction success rate)", fontsize=11)
    ax.set_title(r"Left Bias: CoT vs ISR colored by $\delta$",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"pareto_left_bias_delta.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved: pareto_left_bias_delta")

    # ── Fig 4: Summary table image ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axis("off")

    col_labels = ["Intent",
                  f"ISR\n(δ={delta_off:.0f})", f"ISR\n(δ={delta_on:.0f})",
                  f"CoT\n(δ={delta_off:.0f})", f"CoT\n(δ={delta_on:.0f})",
                  f"Risk\n(δ={delta_off:.0f})", f"Risk\n(δ={delta_on:.0f})",
                  f"Len(m)\n(δ={delta_off:.0f})", f"Len(m)\n(δ={delta_on:.0f})"]

    table_data = []
    for it, il in zip(INTENTS, INTENT_LABELS):
        table_data.append([
            il,
            f"{d0[it]['isr_mean']:.2f}", f"{d2[it]['isr_mean']:.2f}",
            f"{d0[it]['cot_mean']:.1f}", f"{d2[it]['cot_mean']:.1f}",
            f"{d0[it]['risk_mean']:.1f}", f"{d2[it]['risk_mean']:.1f}",
            f"{d0[it]['length_m_mean']:.1f}", f"{d2[it]['length_m_mean']:.1f}",
        ])

    table_data.append([
        "Average",
        f"{np.mean([d0[it]['isr_mean'] for it in INTENTS]):.2f}",
        f"{np.mean([d2[it]['isr_mean'] for it in INTENTS]):.2f}",
        f"{np.mean([d0[it]['cot_mean'] for it in INTENTS]):.1f}",
        f"{np.mean([d2[it]['cot_mean'] for it in INTENTS]):.1f}",
        f"{np.mean([d0[it]['risk_mean'] for it in INTENTS]):.1f}",
        f"{np.mean([d2[it]['risk_mean'] for it in INTENTS]):.1f}",
        f"{np.mean([d0[it]['length_m_mean'] for it in INTENTS]):.1f}",
        f"{np.mean([d2[it]['length_m_mean'] for it in INTENTS]):.1f}",
    ])

    table = ax.table(cellText=table_data, colLabels=col_labels, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    for i in range(len(table_data)):
        isr_val = float(table_data[i][1])
        if isr_val < 0.8:
            table[i + 1, 1].set_facecolor("#FFCDD2")
        isr_val2 = float(table_data[i][2])
        if isr_val2 >= 0.9:
            table[i + 1, 2].set_facecolor("#C8E6C9")

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#E0E0E0")
        table[0, j].set_text_props(fontweight="bold")
    table[len(table_data), 0].set_text_props(fontweight="bold")

    ax.set_title(
        rf"Effect of intent penalty weight $\delta$ on path quality  "
        rf"($\alpha\!=\!{alpha},\,\beta\!=\!{beta},\,\gamma\!=\!{gamma}$)",
        fontsize=11, fontweight="bold", pad=20,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"delta_comparison_table.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved: delta_comparison_table")

    print(f"\nAll figures → {out_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=str,
                    default="results/pareto/sweep_results.json")
    ap.add_argument("--output-dir", type=str,
                    default="results/pareto/figures")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=1.5)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--delta-off", type=float, default=0.0)
    ap.add_argument("--delta-on", type=float, default=2.0)
    args = ap.parse_args()

    results = load_sweep(args.sweep)
    make_figures(results, args.output_dir,
                 alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                 delta_off=args.delta_off, delta_on=args.delta_on)


if __name__ == "__main__":
    main()
