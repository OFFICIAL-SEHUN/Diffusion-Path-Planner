"""
Compare A* paths on the same terrain for multiple cost-weight tuples.

This script:
1) loads one existing .pt sample (terrain + start/goal),
2) re-runs A* with different (alpha, beta, gamma, delta),
3) overlays all paths in one figure.

python scripts/compare_weights_same_terrain.py   --pt data/raw_new/terrain_00016.pt   --intent left_bias   --weights 0.5,1.8,0.4,2 1,1,1,1  --output results/path_vis/terrain_00016_weight_compare.png   --show-base-paths
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from generate_data import INTENT_CATALOG, SlopeCotGenerator

FIXED_PATH_COLORS = [
    "#E63946",  # red
    "#355070",  # dark blue
    "#1D4ED8",  # blue
    "#16A34A",  # green
    "#F59E0B",  # amber
    "#7C3AED",  # violet
    "#0EA5E9",  # sky
    "#F97316",  # orange
    "#EC4899",  # pink
    "#8D6E63",  # brown
    "#2A9D8F",  # teal
    "#6D597A",  # purple
    "#43AA8B",  # turquoise
    "#00695C",  # emerald
    "#1565C0",  # sky blue
    "#003366",  # navy blue
    "#999999",  # gray
]


def _parse_weight_tuple(text: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Invalid weight tuple '{text}'. Use format: a,b,g,d")
    try:
        a, b, g, d = [float(x) for x in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid number in '{text}'. Use float values.") from exc
    return a, b, g, d


def _get_intent_params(intent: str) -> dict:
    for x in INTENT_CATALOG:
        if x["type"] == intent:
            return dict(x["params"])
    raise ValueError(f"Unknown intent '{intent}'.")


def _norm_to_pixels(path_norm_xy: np.ndarray, img_size: int) -> np.ndarray:
    p = np.asarray(path_norm_xy, dtype=np.float64)
    if p.size == 0:
        return p.reshape(0, 2)
    px = (p[:, 0] + 1.0) / 2.0 * img_size
    py = (p[:, 1] + 1.0) / 2.0 * img_size
    return np.stack([px, py], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare weight tuples on same terrain")
    ap.add_argument("--pt", type=str, required=True, help="Path to terrain_XXXXX.pt")
    ap.add_argument("--intent", type=str, default="avoid_steep")
    ap.add_argument(
        "--weights",
        type=str,
        nargs="+",
        required=True,
        help="One or more tuples: a,b,g,d  (e.g. 1,0.8,0.1,1  0.5,1.8,0.4,2)",
    )
    ap.add_argument("--risk-threshold-deg", type=float, default=None)
    ap.add_argument("--output", type=str, default=None, help="Output png path")
    ap.add_argument("--bg-cmap", type=str, default="terrain",
                    help="Background colormap for slope map (e.g., terrain, turbo, viridis, gray)")
    ap.add_argument("--show-base-paths", action="store_true",
                    help="Also overlay paths already stored in the .pt file")
    args = ap.parse_args()

    pt_path = Path(args.pt)
    if not pt_path.exists():
        raise FileNotFoundError(f"Not found: {pt_path}")

    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    for key in ("height_map", "slope_map", "start_position", "goal_position", "img_size"):
        if key not in data:
            raise KeyError(f"Missing key '{key}' in {pt_path}")

    height_map = data["height_map"].numpy()
    slope_deg_map = data["slope_map"].numpy()
    start = tuple(int(v) for v in data["start_position"])
    goal = tuple(int(v) for v in data["goal_position"])
    img_size = int(data["img_size"])
    pixel_resolution = float(data.get("pixel_resolution", 0.5))

    risk_threshold_deg = (
        float(args.risk_threshold_deg)
        if args.risk_threshold_deg is not None
        else float(data.get("risk_threshold_deg", 15.0))
    )

    # Generator instance re-used with fixed terrain.
    gen = SlopeCotGenerator(
        img_size=img_size,
        height_range=(0.0, 5.0),
        mass=10.0,
        gravity=9.8,
        limit_angle_deg=25.0,
        max_iterations=20000,
        pixel_resolution=pixel_resolution,
    )
    gen.height_map = height_map.astype(np.float32)
    gen.slope_map = np.radians(slope_deg_map).astype(np.float32)

    intent_params = _get_intent_params(args.intent)
    weight_list = [_parse_weight_tuple(w) for w in args.weights]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(slope_deg_map, origin="lower", cmap=args.bg_cmap)
    palette = FIXED_PATH_COLORS

    if args.show_base_paths and "paths" in data:
        base_paths = data["paths"].numpy()
        for i, p in enumerate(base_paths):
            px = _norm_to_pixels(p, img_size)
            if px.size == 0:
                continue
            ax.plot(px[:, 0], px[:, 1], color="#999999", lw=1.0, alpha=0.35,
                    label="stored .pt path" if i == 0 else None)

    for idx, (a, b, g, d) in enumerate(weight_list):
        path = gen.find_path_with_intent(
            start=start,
            goal=goal,
            alpha=a,
            beta=b,
            gamma=g,
            delta=d,
            risk_threshold_deg=risk_threshold_deg,
            intent_type=args.intent,
            intent_params=intent_params,
        )
        if path is None:
            print(f"[{idx}] no path for weights ({a}, {b}, {g}, {d})")
            continue

        arr = np.asarray(path, dtype=np.float64)
        ax.plot(
            arr[:, 1],
            arr[:, 0],
            lw=2.2,
            alpha=0.95,
            color=palette[idx % len(palette)],
            label=f"alpha={a:g}, beta={b:g}, gamma={g:g}, delta={d:g}",
        )

    ax.scatter([start[1]], [start[0]], c="lime", s=90, marker="o",
               edgecolors="black", linewidths=0.7, zorder=5, label="start")
    ax.scatter([goal[1]], [goal[0]], c="red", s=120, marker="*",
               edgecolors="black", linewidths=0.7, zorder=5, label="goal")

    ax.set_title(f"{args.intent}")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    out_path = Path(args.output) if args.output else (
        pt_path.parent.parent / "results" / "path_vis" / f"{pt_path.stem}_weights_compare.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
