"""
단일 지형을 생성한 뒤, 여러 epoch 체크포인트 × 6 intent inference 결과를
그리드(행=epoch, 위에서부터 오래된 순; --epoch-step 배수만 사용, 열=intent)로 저장합니다.

Usage (from diffusion_textguide/):
python experiment/epoch_intent_grid.py \
  --config configs/convnext.yaml \
  --checkpoint-dir checkpoints/convnext \
  --output experiment/epoch_intent_grid_convnext_2k.png
"""

from __future__ import annotations

import argparse
import secrets
import sys
import warnings
from pathlib import Path

# PyTorch Transformer nested-tensor prototype noise (harmless for inference).
warnings.filterwarnings(
    "ignore",
    message=".*nested tensors is in prototype.*",
    category=UserWarning,
    module=r"torch\.nn\.modules\.transformer",
)

import numpy as np
import torch
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import only_generate_terrain as _ogt  # noqa: E402
from data_loader import text_to_tokens  # noqa: E402
from inference_6intent import (  # noqa: E402
    INTENT_LABELS,
    INTENTS,
    load_model,
    run_inference,
)

SlopeCotGenerator = _ogt.SlopeCotGenerator

PATH_COLOR = "#E63946"


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def sample_start_goal(
    s_map: np.ndarray,
    limit_angle: float,
    img_size: int,
    margin: int,
    rng: np.random.Generator,
    max_tries: int = 400,
):
    """generate_data.py와 동일한 코너 대각 스타일 start/goal 샘플 (s_map: rad)."""
    edge_lo = margin
    edge_hi = margin * 3
    far_lo = img_size - margin * 3
    far_hi = img_size - margin
    for _ in range(max_tries):
        side = int(rng.integers(4))
        if side == 0:
            s = (int(rng.integers(edge_lo, edge_hi)), int(rng.integers(edge_lo, edge_hi)))
            g = (int(rng.integers(far_lo, far_hi)), int(rng.integers(far_lo, far_hi)))
        elif side == 1:
            s = (int(rng.integers(edge_lo, edge_hi)), int(rng.integers(far_lo, far_hi)))
            g = (int(rng.integers(far_lo, far_hi)), int(rng.integers(edge_lo, edge_hi)))
        elif side == 2:
            s = (int(rng.integers(far_lo, far_hi)), int(rng.integers(edge_lo, edge_hi)))
            g = (int(rng.integers(edge_lo, edge_hi)), int(rng.integers(far_lo, far_hi)))
        else:
            s = (int(rng.integers(far_lo, far_hi)), int(rng.integers(far_lo, far_hi)))
            g = (int(rng.integers(edge_lo, edge_hi)), int(rng.integers(edge_lo, edge_hi)))
        d = float(np.sqrt((g[0] - s[0]) ** 2 + (g[1] - s[1]) ** 2))
        if d >= img_size * 0.6 and s_map[s] < limit_angle and s_map[g] < limit_angle:
            return s, g
    return None, None


def make_terrain_from_config(cfg: dict):
    """매 실행마다 새 랜덤 terrain(start/goal 포함) 생성."""
    d = cfg["data"]
    g = cfg["gradient"]
    img_size = int(d["img_size"])
    height_range = tuple(g["height_range"])
    terrain_scales = g.get("terrain_scales")
    run_seed = secrets.randbits(32)
    rng = np.random.default_rng(run_seed)
    np.random.seed(run_seed)
    print(f"Generating random terrain (seed={run_seed})")

    margin = img_size // 10
    attempts = 0
    max_attempts = 200
    while attempts < max_attempts:
        attempts += 1
        gen = SlopeCotGenerator(
            img_size=img_size,
            height_range=height_range,
            mass=g["mass"],
            gravity=g["gravity"],
            limit_angle_deg=g["limit_angle_deg"],
            max_iterations=g.get("max_iterations", 20000),
            pixel_resolution=g.get("pixel_resolution", 0.5),
        )
        h_map, s_map = gen.generate(terrain_scales=terrain_scales)
        slope_deg = np.degrees(s_map)
        mean_slope = float(np.mean(slope_deg))
        max_slope = float(np.max(slope_deg))
        steep_ratio = float(np.sum(slope_deg > 30.0) / slope_deg.size)
        if mean_slope < 8.0 or mean_slope > 32.0 or max_slope > 55.0 or steep_ratio > 0.55:
            continue
        start, goal = sample_start_goal(s_map, gen.limit_angle, img_size, margin, rng)
        if start is None:
            continue
        slope_norm = slope_deg / 90.0
        height_norm = (h_map - h_map.min()) / (h_map.max() - h_map.min() + 1e-8)
        costmap = np.stack([slope_norm, height_norm], axis=0)
        blob = {
            "map_id": "epoch_grid_eval",
            "costmap": torch.from_numpy(costmap).float(),
            "height_map": torch.from_numpy(h_map).float(),
            "slope_map": torch.from_numpy(slope_deg).float(),
            "img_size": img_size,
            "start_position": start,
            "goal_position": goal,
            "pixel_resolution": g.get("pixel_resolution", 0.5),
            "limit_angle_deg": g["limit_angle_deg"],
        }
        return blob

    raise RuntimeError("Could not sample a valid terrain + start/goal; retry or loosen filters.")


def checkpoint_paths(ckpt_dir: Path):
    files = sorted(ckpt_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return files


def terrain_tensors(terrain: dict):
    costmap = terrain["costmap"]
    height_map = terrain["height_map"].numpy()
    img_size = int(terrain["img_size"])
    if "start_position" in terrain:
        s = terrain["start_position"]
        g = terrain["goal_position"]
        start_pos = torch.tensor(
            [(s[1] / img_size) * 2 - 1, (s[0] / img_size) * 2 - 1], dtype=torch.float32
        )
        goal_pos = torch.tensor(
            [(g[1] / img_size) * 2 - 1, (g[0] / img_size) * 2 - 1], dtype=torch.float32
        )
    else:
        raise ValueError("Terrain needs start_position / goal_position")
    return costmap, height_map, start_pos, goal_pos, img_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/convnext.yaml")
    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints/convnext")
    ap.add_argument(
        "--terrain",
        type=str,
        default=None,
        help="Deprecated: ignored (terrain .pt is no longer loaded)",
    )
    ap.add_argument(
        "--terrain-out",
        type=str,
        default="experiment/terrain_epoch_grid_eval.pt",
        help="Deprecated: ignored (terrain .pt is no longer saved)",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="experiment/epoch_intent_grid_convnext.png",
    )
    ap.add_argument(
        "--epoch-step",
        type=int,
        default=2000,
        help="epoch_N.pt 중 N %% 이 값 == 0 인 체크포인트만 사용 (기본 2000 = 2k 간격)",
    )
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        cfg_path = _ROOT / args.config
    cfg = load_yaml(str(cfg_path))
    if args.terrain or args.terrain_out:
        print("Note: --terrain / --terrain-out are ignored; always generating a new random terrain.")
    terrain = make_terrain_from_config(cfg)

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = _ROOT / ckpt_dir
    all_ckpts = checkpoint_paths(ckpt_dir)
    step = max(1, int(args.epoch_step))
    ckpts = [p for p in all_ckpts if int(p.stem.split("_")[1]) % step == 0]
    if not ckpts:
        raise RuntimeError(
            f"No checkpoints in {ckpt_dir} with epoch %% {step} == 0 "
            f"(have {len(all_ckpts)} epoch_*.pt; try --epoch-step 1000)"
        )
    print(f"Using {len(ckpts)} checkpoints (epoch %% {step} == 0), skipped {len(all_ckpts) - len(ckpts)}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    costmap, height_map, start_pos, goal_pos, img_size = terrain_tensors(terrain)
    horizon = cfg.get("data", {}).get("horizon", 120)

    n_rows = len(ckpts)
    n_cols = len(INTENTS)

    def to_px(p):
        return (p + 1) / 2 * img_size

    paths_grid: list[list[np.ndarray]] = []
    epoch_labels: list[str] = []

    for ck in ckpts:
        ep = int(ck.stem.split("_")[1])
        epoch_labels.append(str(ep))
        model, scheduler, vocab, _ = load_model(str(ck), device)
        row_paths = []
        for _name, instruction in INTENTS:
            tokens = text_to_tokens(instruction, vocab, max_seq_len=16)
            path = run_inference(
                model, scheduler, costmap, start_pos, goal_pos, tokens, horizon, device
            )
            row_paths.append(path)
        paths_grid.append(row_paths)
        del model, scheduler
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fig_w = 3.0 * n_cols
    fig_h = 2.85 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.imshow(height_map, cmap="terrain", origin="lower")
            gen_px = to_px(paths_grid[r][c])
            ax.plot(gen_px[:, 0], gen_px[:, 1], color=PATH_COLOR, lw=1.8, alpha=0.95)
            ax.scatter(
                [gen_px[0, 0]],
                [gen_px[0, 1]],
                c="lime",
                s=28,
                zorder=10,
                marker="o",
                edgecolors="black",
                linewidths=0.5,
            )
            ax.scatter(
                [gen_px[-1, 0]],
                [gen_px[-1, 1]],
                c="orange",
                s=30,
                zorder=10,
                marker="*",
                edgecolors="black",
                linewidths=0.5,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(INTENT_LABELS[c], fontsize=11, fontweight="bold")
        axes[r, 0].set_ylabel(f"{epoch_labels[r]} ep.", fontsize=10, fontweight="bold")

    if step >= 1000 and step % 1000 == 0:
        step_disp = f"{step // 1000}k"
    else:
        step_disp = str(step)
    fig.suptitle(
        f"ConvNeXt-Tiny · Height map · rows=epoch (step {step_disp}, top=earliest), columns=intent",
        fontsize=14,
        fontweight="bold",
        y=1.002,
    )
    out = Path(args.output)
    if not out.is_absolute():
        out = _ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid: {out}")


if __name__ == "__main__":
    main()
