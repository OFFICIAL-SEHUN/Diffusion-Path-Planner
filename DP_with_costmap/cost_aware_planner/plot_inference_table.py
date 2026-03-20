"""
Plot inference time table and graph from metrics/inference_time_by_map_size.csv using matplotlib.
Run from cost_aware_planner: python plot_inference_table.py
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


def parse_rows(header, rows):
    """Return map_sizes + dict of planner_name -> (means, stds). N/A -> np.nan."""
    map_sizes = []
    data = {}  # e.g. "a_star": (means, stds), "dijkstra": (means, stds), ...
    col_map = {h.strip(): i for i, h in enumerate(header)}
    for row in rows:
        map_sizes.append(int(row[0]))
    map_sizes = np.array(map_sizes)
    for name, mean_key, std_key in [
        ("a_star", "a_star_mean_ms", "a_star_std_ms"),
        ("dijkstra", "dijkstra_mean_ms", "dijkstra_std_ms"),
        ("rrt_star", "rrt_star_mean_ms", "rrt_star_std_ms"),
        ("diffusion", "diffusion_mean_ms", "diffusion_std_ms"),
    ]:
        if mean_key not in col_map or std_key not in col_map:
            continue
        mi, si = col_map[mean_key], col_map[std_key]
        means, stds = [], []
        for row in rows:
            if row[mi] == "N/A" or row[si] == "N/A":
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(float(row[mi]))
                stds.append(float(row[si]))
        data[name] = (np.array(means), np.array(stds))
    return map_sizes, data


def plot_table(ax, header, rows):
    """Build table from header; supports a_star, dijkstra, rrt_star, diffusion."""
    col_map = {h.strip(): i for i, h in enumerate(header)}
    col_labels_final = ["Map Size"]
    if "a_star_mean_ms" in col_map:
        col_labels_final.extend(["A* Mean (ms)", "A* Std (ms)"])
    if "dijkstra_mean_ms" in col_map:
        col_labels_final.extend(["Dijkstra Mean (ms)", "Dijkstra Std (ms)"])
    if "rrt_star_mean_ms" in col_map:
        col_labels_final.extend(["RRT* Mean (ms)", "RRT* Std (ms)"])
    if "diffusion_mean_ms" in col_map:
        col_labels_final.extend(["Diffusion Mean (ms)", "Diffusion Std (ms)"])
    if "ratio_astar_diff" in col_map:
        col_labels_final.append("A*/Diff")
    cell_text = []
    for row in rows:
        r = [row[0]]
        if "a_star_mean_ms" in col_map:
            r.append(f"{float(row[col_map['a_star_mean_ms']]):.2f}")
            r.append(f"{float(row[col_map['a_star_std_ms']]):.2f}")
        if "dijkstra_mean_ms" in col_map:
            v = row[col_map["dijkstra_mean_ms"]]
            r.append(f"{float(v):.2f}" if v != "N/A" else "N/A")
            v = row[col_map["dijkstra_std_ms"]]
            r.append(f"{float(v):.2f}" if v != "N/A" else "N/A")
        if "rrt_star_mean_ms" in col_map:
            v = row[col_map["rrt_star_mean_ms"]]
            r.append(f"{float(v):.2f}" if v != "N/A" else "N/A")
            v = row[col_map["rrt_star_std_ms"]]
            r.append(f"{float(v):.2f}" if v != "N/A" else "N/A")
        if "diffusion_mean_ms" in col_map:
            v = row[col_map["diffusion_mean_ms"]]
            r.append(f"{float(v):.2f}" if v != "N/A" else "N/A")
            v = row[col_map["diffusion_std_ms"]]
            r.append(f"{float(v):.2f}" if v != "N/A" else "N/A")
        if "ratio_astar_diff" in col_map:
            a_mean = float(row[col_map["a_star_mean_ms"]])
            d_mean = row[col_map["diffusion_mean_ms"]]
            if d_mean != "N/A" and float(d_mean) > 0:
                r.append(f"{a_mean / float(d_mean):.2f}×")
            else:
                r.append("—")
        cell_text.append(r)
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels_final,
        loc="center",
        cellLoc="center",
        colColours=["#e0e0e0"] * len(col_labels_final),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 2.2)
    for j in range(len(col_labels_final)):
        table[(0, j)].set_text_props(weight="bold")
    for i in range(len(cell_text)):
        table[(i + 1, 0)].set_text_props(weight="bold")


def plot_graph(ax, map_sizes, data):
    """data: dict planner_name -> (means, stds). Plot all with different colors."""
    ax.set_xlabel("Map Size")
    ax.set_ylabel("Inference Time (ms)")
    ax.set_title("Planners vs Diffusion Inference Time by Map Size")
    ax.set_xticks(map_sizes)
    ax.set_xticklabels(map_sizes)
    markers = ["o", "s", "^", "D"]
    for idx, (name, (means, stds)) in enumerate(data.items()):
        valid = ~np.isnan(means)
        if not np.any(valid):
            continue
        xs = map_sizes[valid]
        m, s = means[valid], stds[valid]
        label = {"a_star": "A*", "dijkstra": "Dijkstra", "rrt_star": "RRT*", "diffusion": "Diffusion"}.get(name, name)
        ax.plot(xs, m, f"{markers[idx % len(markers)]}-", color=f"C{idx}", linewidth=2, markersize=8, label=label)
        ax.fill_between(xs, m - s, m + s, color=f"C{idx}", alpha=0.2)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_yscale("log")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "metrics", "inference_time_by_map_size.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    header, rows = load_csv(csv_path)
    out_dir = os.path.join(base_dir, "metrics")
    os.makedirs(out_dir, exist_ok=True)

    # --- Table ---
    fig1, ax1 = plt.subplots(figsize=(min(14, 2 + len(header) * 1.2), 2 + 0.35 * len(rows)))
    plot_table(ax1, header, rows)
    out_table = os.path.join(out_dir, "inference_time_by_map_size_table.png")
    fig1.savefig(out_table, bbox_inches="tight", dpi=150)
    plt.close(fig1)
    print(f"Saved: {out_table}")

    # --- Graph ---
    map_sizes, data = parse_rows(header, rows)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    plot_graph(ax2, map_sizes, data)
    out_graph = os.path.join(out_dir, "inference_time_by_map_size_graph.png")
    fig2.savefig(out_graph, bbox_inches="tight", dpi=150)
    plt.close(fig2)
    print(f"Saved: {out_graph}")


if __name__ == "__main__":
    main()
