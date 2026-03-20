"""
지형만 생성하는 스크립트 (경로 생성 없음)

각 terrain마다:
- height_map, slope_map 생성
- diffusion_textguide/data/raw/ 에 개별 .pt 파일로 저장
"""

import os
import argparse
import numpy as np
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

np.random.seed(42)

# diffusion_textguide 루트 / 데이터 저장 기준
_SCRIPT_DIR = Path(__file__).resolve().parents[0]
_ROOT = _SCRIPT_DIR.parent
_DATA_RAW = _ROOT / "data" / "raw"


# -----------------------------------------------------------------------------
# SlopeCotGenerator
# -----------------------------------------------------------------------------


class SlopeCotGenerator:
    """Slope + CoT 지형 생성."""

    def __init__(self, img_size, height_range, mass, gravity, limit_angle_deg, max_iterations,
                 pixel_resolution=0.5):
        self.img_size = img_size
        self.height_range = tuple(height_range)
        self.mass = mass
        self.gravity = gravity
        self.limit_angle = np.radians(limit_angle_deg)
        self.pixel_resolution = pixel_resolution
        self.max_iterations = max_iterations
        self.height_map = None
        self.slope_map = None

    def generate(self, terrain_scales):
        """지형 생성. terrain_scales: [(scale, weight), ...]. 반환: (height_map, slope_map)."""
        if terrain_scales is None:
            raise ValueError("terrain_scales must be provided")
        self.height_map = self._height_map(terrain_scales)
        self.slope_map = self._slope_map(self.height_map)
        return self.height_map, self.slope_map

    def _height_map(self, terrain_scales):
        h = np.zeros((self.img_size, self.img_size))
        for scale, weight in terrain_scales:
            noise = np.random.rand(self.img_size, self.img_size)
            h += gaussian_filter(noise, sigma=scale) * weight
        h = (h - h.min()) / (h.max() - h.min() + 1e-12)
        h = h * (self.height_range[1] - self.height_range[0]) + self.height_range[0]
        return h.astype(np.float32)

    def _slope_map(self, height_map):
        gy, gx = np.gradient(height_map, self.pixel_resolution)
        mag = np.sqrt(gx**2 + gy**2)
        return np.arctan(mag).astype(np.float32)


# -----------------------------------------------------------------------------
# 데이터 생성 및 저장
# -----------------------------------------------------------------------------


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_terrain_only(
    num_terrains=100,
    img_size=100,
    height_range=(0, 5),
    mass=10.0,
    gravity=9.8,
    limit_angle_deg=30,
    pixel_resolution=0.5,
    terrain_scales=None,
    max_iterations=10000,
    output_dir="data/raw",
    debug=False,
    filter_terrain=False,  # 필터링 활성화 여부
):
    """지형만 생성 후 개별 .pt로 저장. 반환: 생성된 파일 경로 리스트."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Terrain Generation Only (No Paths)")
    print("=" * 60)
    print(f"Image: {img_size}x{img_size}  Terrains: {num_terrains}")
    print(f"Output: {output_dir}")
    
    # terrain_scales 확인
    if terrain_scales:
        total_weight = sum(w for _, w in terrain_scales)
        if total_weight == 0:
            print("WARNING: All terrain_scales weights are 0! Terrain will be flat.")
        else:
            print(f"Terrain scales: {terrain_scales} (total weight: {total_weight})")
    else:
        print("WARNING: terrain_scales is None!")
    
    print(f"Filter terrain: {filter_terrain}")
    if debug:
        print("DEBUG: on")
    print("=" * 60)

    generated = []
    stats = {"mean_slope": [], "max_slope": []}
    pbar = tqdm(total=num_terrains, desc="Generating terrains")
    n_done = 0
    attempts = 0
    max_attempts = num_terrains * 20

    while n_done < num_terrains and attempts < max_attempts:
        attempts += 1
        gen = SlopeCotGenerator(
            img_size=img_size,
            height_range=height_range,
            mass=mass,
            gravity=gravity,
            limit_angle_deg=limit_angle_deg,
            max_iterations=max_iterations,
            pixel_resolution=pixel_resolution,
        )
        h_map, s_map = gen.generate(terrain_scales=terrain_scales)
        slope_deg = np.degrees(s_map)
        mean_slope = float(np.mean(slope_deg))
        max_slope = float(np.max(slope_deg))
        steep = np.sum(slope_deg > 30.0) / slope_deg.size
        
        # 지형 필터링 (선택적)
        if filter_terrain:
            if mean_slope < 8.0 or mean_slope > 32.0 or max_slope > 55.0 or steep > 0.55:
                if debug:
                    print(f"  [debug] attempt {attempts}: terrain skip (mean={mean_slope:.1f} max={max_slope:.1f} steep={steep:.2f})")
                continue
        else:
            # 필터링 비활성화 시 디버그 정보만 출력
            if debug and attempts <= 5:
                print(f"  [debug] attempt {attempts}: terrain OK (mean={mean_slope:.1f}° max={max_slope:.1f}° steep={steep:.2f})")

        n_done += 1
        slope_norm = slope_deg / 90.0
        height_norm = (h_map - h_map.min()) / (h_map.max() - h_map.min() + 1e-8)
        costmap = np.stack([slope_norm, height_norm], axis=0)

        map_id = f"terrain_{n_done:05d}"
        save_path = os.path.join(output_dir, f"{map_id}.pt")

        torch.save({
            "map_id": map_id,
            "costmap": torch.from_numpy(costmap).float(),
            "height_map": torch.from_numpy(h_map).float(),
            "slope_map": torch.from_numpy(slope_deg).float(),  # 도 단위로 저장
            "img_size": img_size,
            "pixel_resolution": pixel_resolution,
            "limit_angle_deg": limit_angle_deg,
            "max_iterations": max_iterations,
        }, save_path)

        generated.append(save_path)
        stats["mean_slope"].append(mean_slope)
        stats["max_slope"].append(max_slope)
        pbar.update(1)
        pbar.set_postfix(
            done=f"{n_done}/{num_terrains}",
            attempts=attempts,
            mean_slope=f"{mean_slope:.1f}°",
            max_slope=f"{max_slope:.1f}°"
        )

    pbar.close()
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Terrains: {n_done}/{num_terrains}  Files: {len(generated)}")
    if stats["mean_slope"]:
        print(f"  Mean slope: {np.mean(stats['mean_slope']):.2f}°  "
              f"Max slope: {np.mean(stats['max_slope']):.2f}°")
    print("=" * 60)
    return generated


def main():
    ap = argparse.ArgumentParser(description="Generate terrain only (no paths)")
    ap.add_argument("--config", type=str, default=None, help="Config YAML path")
    ap.add_argument("--num-terrains", type=int, default=100)
    ap.add_argument("--output-dir", type=str, default=None,
                    help=f"Output directory (default: {_DATA_RAW})")
    ap.add_argument("--debug", action="store_true", help="Print skip reasons")
    ap.add_argument("--filter", action="store_true", help="Enable terrain filtering")
    args = ap.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(_DATA_RAW)

    if args.config:
        config = load_config(args.config)
    else:
        cfg_path = _ROOT / "configs" / "default_config.yaml"
        if not cfg_path.exists():
            cfg_path = Path(__file__).resolve().parents[2] / "diffusion_patch" / "configs" / "default_config.yaml"
        if cfg_path.exists():
            config = load_config(str(cfg_path))
            print(f"Using config: {cfg_path}")
        else:
            config = {
                "data": {
                    "num_samples": args.num_terrains,
                    "img_size": 100,
                },
                "gradient": {
                    "height_range": [0, 5],
                    "terrain_scales": [[80, 50], [8, 10], [10, 5], [5, 2]],
                    "mass": 10.0,
                    "gravity": 9.8,
                    "limit_angle_deg": 20,
                    "pixel_resolution": 0.5,
                    "max_iterations": 10000,
                },
            }

    num_terrains = args.num_terrains
    img_size = config["data"]["img_size"]
    height_range = tuple(config["gradient"]["height_range"])
    mass = config["gradient"]["mass"]
    gravity = config["gradient"]["gravity"]
    limit_angle_deg = config["gradient"]["limit_angle_deg"]
    pixel_resolution = config["gradient"].get("pixel_resolution", 0.5)
    terrain_scales = config["gradient"].get("terrain_scales")
    max_iterations = config["gradient"].get("max_iterations", 10000)

    files = generate_terrain_only(
        num_terrains=num_terrains,
        img_size=img_size,
        height_range=height_range,
        mass=mass,
        gravity=gravity,
        limit_angle_deg=limit_angle_deg,
        pixel_resolution=pixel_resolution,
        terrain_scales=terrain_scales,
        max_iterations=max_iterations,
        output_dir=output_dir,
        debug=args.debug,
        filter_terrain=args.filter,
    )
    print(f"✓ Saved {len(files)} terrain files to {output_dir}")


if __name__ == "__main__":
    main()
