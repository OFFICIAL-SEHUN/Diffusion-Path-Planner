"""
Plot evaluation metrics (Success Rate, Mean Cost %) from metrics/eval_results.csv.
Run after evaluate_metrics.py with --save_csv for one or more scales.

Scale = Cost Guidance Scale (cost_guidance_scale):
  - Sampling 시 costmap gradient로 경로를 벽에서 밀어내는 힘의 세기.
  - 0 = 가이던스 없음 (vanilla), 클수록 장애물 회피가 강해짐.
  - guidance_diffusion.py: x = x - cost_guidance_scale * grad

Usage:
  python evaluate_metrics.py --scale 0.5 --save_csv metrics/eval_results.csv
  python evaluate_metrics.py --scale 1.0 --save_csv metrics/eval_results.csv
  python evaluate_metrics.py --scale 2.0 --save_csv metrics/eval_results.csv
  python plot_eval_metrics.py
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def load_csv(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row:
                continue
            rows.append(row)
    return header, rows


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "metrics", "eval_results.csv")
    if not os.path.isfile(csv_path):
        print(f"CSV not found: {csv_path}")
        print("Run: python evaluate_metrics.py --scale <s> --save_csv metrics/eval_results.csv (for each scale)")
        return

    _, rows = load_csv(csv_path)
    if not rows:
        print("CSV is empty.")
        return

    # Parse: scale, success_rate_pct, mean_cost_pct
    # scale = cost guidance scale (벽 회피 가이던스 세기)
    col_labels = ["Cost Guidance Scale", "Success Rate (%)", "Mean Cost (Diffusion/A*)"]
    cell_text = []
    scales = []
    success_rates = []
    mean_costs = []
    for row in rows:
        scale = row[0]
        sr = float(row[1])
        mc = float(row[2])
        scales.append(float(scale))
        success_rates.append(sr)
        mean_costs.append(mc)
        cell_text.append([scale, f"{sr:.2f}", f"{mc:.2f}"])

    scales = np.array(scales)
    success_rates = np.array(success_rates)
    mean_costs = np.array(mean_costs)
    out_dir = os.path.join(base_dir, "metrics")
    os.makedirs(out_dir, exist_ok=True)

    # --- Table ---
    fig1, ax1 = plt.subplots(figsize=(8, 1.5 + 0.4 * len(cell_text)))
    ax1.axis("off")
    table = ax1.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colColours=["#e0e0e0"] * len(col_labels),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.2)

    def set_cell_bold(cell_key):
        cell = table[cell_key]
        cell.get_text().set_fontweight("bold")

    for j in range(len(col_labels)):
        set_cell_bold((0, j))
    for i in range(len(cell_text)):
        set_cell_bold((i + 1, 0))
    # Bold best: max success rate, min mean cost
    best_sr_idx = int(np.argmax(success_rates))
    best_mc_idx = int(np.argmin(mean_costs))
    for i in range(len(cell_text)):
        if i == best_sr_idx:
            set_cell_bold((i + 1, 1))
        if i == best_mc_idx:
            set_cell_bold((i + 1, 2))
    out_table = os.path.join(out_dir, "eval_metrics_table.png")
    fig1.savefig(out_table, bbox_inches="tight", dpi=150)
    plt.close(fig1)
    print(f"Saved: {out_table}")

    # --- Graph (two subplots or dual y-axis) ---
    fig2, ax1 = plt.subplots(figsize=(7, 5))
    x = np.arange(len(scales))
    width = 0.35
    bars1 = ax1.bar(x - width / 2, success_rates, width, label="Success Rate (%)", color="C0", alpha=0.8)
    ax1.bar_label(bars1, labels=[f"{v:.1f}%" for v in success_rates], color="C0")
    ax1.set_ylabel("Success Rate (%)", color="C0")
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis="y", labelcolor="C0")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, mean_costs, width, label="Mean Cost (Diffusion/A*, GT=100%)", color="C1", alpha=0.8)
    ax2.axhline(y=100, color="gray", linestyle="--", linewidth=1, label="GT (A*=100%)")
    ax2.set_ylabel("Mean Cost (%)", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")

    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in scales])
    ax1.set_xlabel("Cost Guidance Scale")
    ax1.set_title("Diffusion Evaluation: Success Rate & Mean Cost (A* = GT = 100%)")
    fig2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    fig2.tight_layout()
    out_graph = os.path.join(out_dir, "eval_metrics_graph.png")
    fig2.savefig(out_graph, bbox_inches="tight", dpi=150)
    plt.close(fig2)
    print(f"Saved: {out_graph}")


if __name__ == "__main__":
    main()
