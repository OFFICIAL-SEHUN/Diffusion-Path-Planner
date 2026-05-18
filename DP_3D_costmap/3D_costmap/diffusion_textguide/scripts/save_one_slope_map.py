"""
Save slope/height map images (generate new terrain OR load from dataset .pt).

Usage:
  # 새 지형 생성
  python3 scripts/save_one_slope_map.py

  # 데이터셋 .pt에서 불러오기
  python3 scripts/save_one_slope_map.py --pt data/raw/terrain_00001.pt
  python3 scripts/save_one_slope_map.py --pt data/raw/terrain_00001.pt \
      --output results/slope_map_single.png \
      --height-output results/height_map_single.png
"""

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from generate_data import SlopeCotGenerator, load_config


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = ROOT / "configs" / "default_config.yaml"
DEFAULT_OUTPUT = ROOT / "results" / "slope_map_single.png"


def _default_height_output_from_slope(slope_output: Path) -> Path:
    name = slope_output.name
    if "slope" in name:
        return slope_output.with_name(name.replace("slope", "height"))
    return slope_output.with_name(f"{slope_output.stem}_height{slope_output.suffix}")


def _load_maps_from_pt(pt_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """데이터셋 .pt에서 height_map, slope_map(deg) 로드."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    height_map = data["height_map"].numpy()
    slope_map = data["slope_map"].numpy()

    # slope_map: 저장 포맷이 deg(0~35) 또는 rad(<2)일 수 있음
    if np.nanmax(slope_map) < 2.0:
        slope_map_deg = np.degrees(slope_map)
    else:
        slope_map_deg = slope_map

    map_id = data.get("map_id", pt_path.stem)
    return height_map, slope_map_deg, str(map_id)


def main():
    ap = argparse.ArgumentParser(description="Save slope and height map images")
    ap.add_argument("--pt", type=str, default=None,
                    help="데이터셋 .pt 경로 (지정 시 생성 대신 로드)")
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                    help="Config YAML path (--pt 미지정 시 새 지형 생성에 사용)")
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output image path")
    ap.add_argument("--height-output", type=str, default=None, help="Height map output image path")
    ap.add_argument("--seed", type=int, default=None, help="Random seed (generate 모드만)")
    args = ap.parse_args()

    if args.pt:
        pt_path = Path(args.pt)
        if not pt_path.exists():
            raise FileNotFoundError(f".pt not found: {pt_path}")
        height_map, slope_map_deg, map_id = _load_maps_from_pt(pt_path)
        title_suffix = f"dataset | {map_id}"
    else:
        # generate_data import sets fixed seeds; override here for non-deterministic runs.
        if args.seed is None:
            np.random.seed(None)
            random.seed(None)
        else:
            np.random.seed(args.seed)
            random.seed(args.seed)

        cfg = load_config(args.config)
        data_cfg = cfg.get("data", {})
        grad_cfg = cfg.get("gradient", {})

        img_size = data_cfg.get("img_size", 100)
        height_range = tuple(grad_cfg.get("height_range", [0, 5]))
        terrain_scales = grad_cfg.get("terrain_scales", [[20, 10], [10, 5], [5, 2]])
        mass = grad_cfg.get("mass", 10.0)
        gravity = grad_cfg.get("gravity", 9.8)
        limit_angle_deg = grad_cfg.get("limit_angle_deg", 25)
        pixel_resolution = grad_cfg.get("pixel_resolution", 0.5)
        max_iterations = grad_cfg.get("max_iterations", 20000)

        generator = SlopeCotGenerator(
            img_size=img_size,
            height_range=height_range,
            mass=mass,
            gravity=gravity,
            limit_angle_deg=limit_angle_deg,
            max_iterations=max_iterations,
            pixel_resolution=pixel_resolution,
        )
        height_map, slope_map_rad = generator.generate(terrain_scales=terrain_scales)
        slope_map_deg = np.degrees(slope_map_rad)
        title_suffix = "generated"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(slope_map_deg, cmap="jet", origin="lower", vmin=0, vmax=35)
    plt.title(f"Slope Map (deg) | {title_suffix}")
    plt.colorbar(label="Slope (deg)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    if args.height_output is None:
        height_output_path = _default_height_output_from_slope(output_path)
    else:
        height_output_path = Path(args.height_output)
    height_output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(height_map, cmap="terrain", origin="lower")
    plt.title(f"Height Map (m) | {title_suffix}")
    plt.colorbar(label="Height (m)")
    plt.tight_layout()
    plt.savefig(height_output_path, dpi=150)
    plt.close()

    print(f"Saved slope map: {output_path}")
    print(f"Saved height map: {height_output_path}")


if __name__ == "__main__":
    main()
