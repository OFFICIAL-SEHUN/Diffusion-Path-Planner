"""
지형 + intent 기반 경로 시각화

data/raw/*.pt 로드 → Slope/Height 맵 + intent별 경로 오버레이 → results/path_vis/ 저장
새 포맷(intent_types, instructions)과 레거시 포맷(weights) 모두 지원.
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/raw"
RESULT_DIR = ROOT / "results/path_vis"

INTENT_COLORS = {
    "baseline":               "#888888",
    "left_bias":              "#E63946",
    "right_bias":             "#457B9D",
    "center_bias":            "#BC6C25",
    "avoid_steep":            "#2A9D8F",
    "prefer_flat":            "#E9C46A",
    "minimize_elevation_change": "#6D597A",
    "short_path":             "#355070",
    "energy_efficient":       "#43AA8B",
    "left_bias+avoid_steep":  "#9B2226",
    "right_bias+prefer_flat": "#264653",
    "center_bias+prefer_flat": "#8D6E63",
    "short_path+avoid_steep": "#1565C0",
    "energy_efficient+minimize_elevation_change": "#00695C",
}

WEIGHT_COLORS = {
    0.1: "#FFB6C1",
    0.3: "#FF69B4",
    0.5: "#FF1493",
    0.7: "#FF00FF",
    0.9: "#C71585",
}


def _norm_to_pixels(path_norm: np.ndarray, img_size: int) -> np.ndarray:
    """정규화 경로 [N,2] (x,y) → 픽셀 [N,2] (col, row)."""
    p = np.asarray(path_norm, dtype=np.float64)
    if p.size == 0:
        return p.reshape(0, 2)
    px = (p[:, 0] + 1) / 2 * img_size
    py = (p[:, 1] + 1) / 2 * img_size
    return np.stack([px, py], axis=1)


def _get_intent_color(intent_type: str) -> str:
    if intent_type in INTENT_COLORS:
        return INTENT_COLORS[intent_type]
    fallback = list(INTENT_COLORS.values())
    return fallback[hash(intent_type) % len(fallback)]


def visualize_one(
    pt_path: str | Path,
    out_path: str | Path,
    dpi: int = 120,
) -> None:
    """단일 .pt 파일 시각화. intent 포맷과 레거시 weight 포맷 모두 지원."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    height_map = data["height_map"].numpy()
    slope_map = data["slope_map"].numpy()
    paths = data["paths"].numpy()
    img_size = int(data["img_size"])

    is_intent_format = "intent_types" in data
    intent_types = data.get("intent_types", [])
    instructions = data.get("instructions", [])
    weights = data["weights"].numpy() if "weights" in data else None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Slope map
    ax = axes[0]
    im0 = ax.imshow(slope_map, cmap="jet", origin="lower", vmin=0, vmax=35)
    ax.set_title("Slope (deg)")
    plt.colorbar(im0, ax=ax, shrink=0.6)

    # Height map
    ax = axes[1]
    im1 = ax.imshow(height_map, cmap="terrain", origin="lower")
    ax.set_title("Height (m)")
    plt.colorbar(im1, ax=ax, shrink=0.6)

    # 경로 그리기
    for i, path in enumerate(paths):
        px = _norm_to_pixels(path, img_size)
        col, row = px[:, 0], px[:, 1]

        if is_intent_format and i < len(intent_types):
            c = _get_intent_color(intent_types[i])
            label = intent_types[i]
        elif weights is not None:
            w = float(weights[i])
            c = WEIGHT_COLORS.get(round(w, 1), "#FF1493")
            label = f"w={w:.2f}"
        else:
            c = "#FF1493"
            label = f"path_{i}"

        for a in axes:
            a.plot(col, row, color=c, lw=2, alpha=0.85,
                   label=label if a == axes[0] else None)

    # Start / Goal 마커
    if is_intent_format:
        sp = data.get("start_position")
        gp = data.get("goal_position")
        if sp is not None:
            for a in axes:
                a.scatter([sp[1]], [sp[0]], c="black", s=80, marker="o",
                          edgecolors="white", zorder=10,
                          label="Start" if a == axes[0] else None)
        if gp is not None:
            for a in axes:
                a.scatter([gp[1]], [gp[0]], c="black", s=80, marker="*",
                          edgecolors="white", zorder=10,
                          label="Goal" if a == axes[0] else None)
    else:
        starts = data.get("start_positions", [])
        goals = data.get("goal_positions", [])
        if starts:
            sr, sc = starts[0]
            for a in axes:
                a.scatter([sc], [sr], c="black", s=80, marker="o",
                          edgecolors="white", zorder=10,
                          label="Start" if a == axes[0] else None)
        if goals:
            gr, gc = goals[0]
            for a in axes:
                a.scatter([gc], [gr], c="black", s=80, marker="*",
                          edgecolors="white", zorder=10,
                          label="Goal" if a == axes[0] else None)

    axes[0].legend(loc="upper right", fontsize=7, framealpha=0.9,
                   handlelength=1.5, columnspacing=0.5)

    for a in axes:
        a.set_aspect("equal")
        a.axis("off")

    map_id = data.get("map_id", Path(pt_path).stem)
    subtitle_parts = [f"{map_id}", f"{len(paths)} paths"]
    if is_intent_format and instructions:
        subtitle_parts.append(f"intents: {len(set(intent_types))}")
    fig.suptitle("  |  ".join(subtitle_parts), fontsize=12)
    fig.tight_layout()
    os.makedirs(Path(out_path).parent, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize terrain + intent paths")
    ap.add_argument("--data-dir", type=str, default=None,
                    help=f"Raw .pt directory (default: {DATA_DIR})")
    ap.add_argument("--output-dir", type=str, default=None,
                    help=f"Output images directory (default: {RESULT_DIR})")
    ap.add_argument("--limit", type=int, default=0,
                    help="Max .pt files to process (0 = all)")
    ap.add_argument("--dpi", type=int, default=120)
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    out_dir = Path(args.output_dir) if args.output_dir else RESULT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(data_dir.glob("*.pt"))
    if not pt_files:
        print(f"No .pt files in {data_dir}")
        return
    if args.limit > 0:
        pt_files = pt_files[:args.limit]
    print(f"Visualizing {len(pt_files)} files: {data_dir} -> {out_dir}")

    for p in tqdm(pt_files, desc="Generate visuals"):
        out_path = out_dir / f"{p.stem}.png"
        visualize_one(str(p), str(out_path), dpi=args.dpi)

    print(f"Saved {len(pt_files)} images to {out_dir}")


if __name__ == "__main__":
    main()
