"""
Validation helper:
- Generate one terrain automatically
- Sample start/goal automatically
- Run text-guided inference with a user-provided instruction
- Save visualization image (and optional terrain .pt)

Usage example:
  python3 scripts/valid.py --checkpoint checkpoints/final_model.pt --instruction "Stay to the left side"
  
  python3 scripts/valid.py \
  --checkpoint checkpoints/final_model.pt \
  --instruction "Stay to the left side" \
  --terrain-out data/valid/terrain_valid.pt \
  --terrain-with-paths-out data/valid/terrain_valid_with_paths.pt # 경로 포함 파일 저장
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from data_loader import text_to_tokens
from generate_data import (
    SlopeCotGenerator,
    load_config,
    INTENT_CATALOG,
    _path_pixels_to_normalized,
    _resample_path,
    _sample_instruction,
)
from inference import load_model, run_inference, visualize_result


def sample_start_goal(
    slope_map_rad: np.ndarray,
    img_size: int,
    limit_angle_rad: float,
    min_distance_factor: float = 1.5,
    max_tries: int = 400,
):
    """Sample start/goal near opposite corners, like generate_data.py."""
    margin = img_size // 10
    edge_lo = margin
    edge_hi = margin * 3
    far_lo = img_size - margin * 3
    far_hi = img_size - margin
    min_dist = img_size / max(min_distance_factor, 1e-6)

    for _ in range(max_tries):
        side = np.random.randint(4)
        if side == 0:  # left-bottom -> right-top
            s = (np.random.randint(edge_lo, edge_hi), np.random.randint(edge_lo, edge_hi))
            g = (np.random.randint(far_lo, far_hi), np.random.randint(far_lo, far_hi))
        elif side == 1:  # right-bottom -> left-top
            s = (np.random.randint(edge_lo, edge_hi), np.random.randint(far_lo, far_hi))
            g = (np.random.randint(far_lo, far_hi), np.random.randint(edge_lo, edge_hi))
        elif side == 2:  # left-top -> right-bottom
            s = (np.random.randint(far_lo, far_hi), np.random.randint(edge_lo, edge_hi))
            g = (np.random.randint(edge_lo, edge_hi), np.random.randint(far_lo, far_hi))
        else:  # right-top -> left-bottom
            s = (np.random.randint(far_lo, far_hi), np.random.randint(far_lo, far_hi))
            g = (np.random.randint(edge_lo, edge_hi), np.random.randint(edge_lo, edge_hi))

        dist = float(np.hypot(g[0] - s[0], g[1] - s[1]))
        if dist < min_dist:
            continue
        if slope_map_rad[s] >= limit_angle_rad or slope_map_rad[g] >= limit_angle_rad:
            continue
        return s, g

    raise RuntimeError("Failed to sample valid start/goal. Try another seed.")


def to_normalized_xy(rc, img_size: int) -> torch.Tensor:
    """(row, col) -> normalized (x, y) in [-1, 1]."""
    r, c = rc
    x = (c / img_size) * 2 - 1
    y = (r / img_size) * 2 - 1
    return torch.tensor([x, y], dtype=torch.float32)


def build_intent_paths(
    gen: SlopeCotGenerator,
    start_rc,
    goal_rc,
    img_size: int,
    horizon: int,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    risk_threshold_deg: float,
):
    """현재 terrain/start-goal에서 intent별 pseudo paths 생성."""
    paths = []
    intent_types = []
    intent_params = []
    instructions = []
    pseudo_labels = []

    for intent_def in INTENT_CATALOG:
        itype = intent_def["type"]
        iparams = dict(intent_def["params"])
        path_pixels = gen.find_path_with_intent(
            start_rc,
            goal_rc,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            risk_threshold_deg=risk_threshold_deg,
            intent_type=itype,
            intent_params=iparams,
        )
        if path_pixels is None or len(path_pixels) < 2:
            continue

        norm = _path_pixels_to_normalized(path_pixels, img_size)
        fixed = _resample_path(norm, horizon)
        paths.append(fixed)
        intent_types.append(itype)
        intent_params.append(iparams)
        instructions.append(_sample_instruction(itype))
        pseudo_labels.append(
            {
                "intent_type": itype,
                "intent_params": iparams,
                "cost_weights": {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta},
                "risk_threshold_deg": risk_threshold_deg,
            }
        )

    return paths, intent_types, intent_params, instructions, pseudo_labels


def main():
    ap = argparse.ArgumentParser(description="Generate one validation terrain and run inference")
    ap.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    ap.add_argument("--instruction", type=str, required=True, help='Text instruction, e.g. "Stay left"')
    ap.add_argument("--config", type=str, default=str(ROOT / "configs" / "default_config.yaml"))
    ap.add_argument("--output", type=str, default=str(ROOT / "results" / "valid_inference.png"))
    ap.add_argument("--terrain-out", type=str, default=str(ROOT / "data" / "valid" / "terrain_valid.pt"))
    ap.add_argument(
        "--terrain-with-paths-out",
        type=str,
        default=str(ROOT / "data" / "valid" / "terrain_valid_with_paths.pt"),
        help="Path to save terrain .pt with intent-based pseudo paths",
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=None, help="Optional fixed seed")
    ap.add_argument("--no-gt", action="store_true",
                    help="Do not draw reference path (pseudo label) even if present")
    ap.add_argument("--terrain-note", type=str, default=None,
                    help="Title suffix, e.g. 'Unseen terrain'. Default: 'Unseen terrain (no reference path)'")
    args = ap.parse_args()

    if args.seed is None:
        np.random.seed(None)
        random.seed(None)
    else:
        np.random.seed(args.seed)
        random.seed(args.seed)

    cfg = load_config(args.config)
    d_cfg = cfg.get("data", {})
    g_cfg = cfg.get("gradient", {})

    img_size = int(d_cfg.get("img_size", 100))
    min_distance_factor = float(d_cfg.get("min_distance_factor", 1.5))
    height_range = tuple(g_cfg.get("height_range", [0, 5]))
    terrain_scales = g_cfg.get("terrain_scales", [[20, 10], [10, 5], [5, 2]])
    mass = float(g_cfg.get("mass", 10.0))
    gravity = float(g_cfg.get("gravity", 9.8))
    limit_angle_deg = float(g_cfg.get("limit_angle_deg", 25))
    max_iterations = int(g_cfg.get("max_iterations", 20000))
    pixel_resolution = float(g_cfg.get("pixel_resolution", 0.5))
    i_cfg = cfg.get("intent", {})
    cw = i_cfg.get("cost_weights", {})
    alpha = float(cw.get("alpha", 1.0))
    beta = float(cw.get("beta", 0.8))
    gamma = float(cw.get("gamma", 0.1))
    delta = float(cw.get("delta", 1.0))
    risk_threshold_deg = float(i_cfg.get("risk_threshold_deg", 15.0))

    gen = SlopeCotGenerator(
        img_size=img_size,
        height_range=height_range,
        mass=mass,
        gravity=gravity,
        limit_angle_deg=limit_angle_deg,
        max_iterations=max_iterations,
        pixel_resolution=pixel_resolution,
    )
    height_map, slope_map_rad = gen.generate(terrain_scales=terrain_scales)
    slope_map_deg = np.degrees(slope_map_rad)

    start_rc, goal_rc = sample_start_goal(
        slope_map_rad=slope_map_rad,
        img_size=img_size,
        limit_angle_rad=gen.limit_angle,
        min_distance_factor=min_distance_factor,
    )

    slope_norm = slope_map_deg / 90.0
    height_norm = (height_map - height_map.min()) / (height_map.max() - height_map.min() + 1e-8)
    costmap = torch.from_numpy(np.stack([slope_norm, height_norm], axis=0)).float()

    terrain_path = Path(args.terrain_out)
    terrain_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "map_id": "terrain_valid",
            "costmap": costmap,
            "height_map": torch.from_numpy(height_map).float(),
            "slope_map": torch.from_numpy(slope_map_deg).float(),
            "start_position": start_rc,
            "goal_position": goal_rc,
            "img_size": img_size,
            "pixel_resolution": pixel_resolution,
            "limit_angle_deg": limit_angle_deg,
        },
        terrain_path,
    )

    # 2) intent별 pseudo path 포함 파일 저장
    horizon = int(cfg.get("data", {}).get("horizon", 120))
    paths, intent_types, intent_params, instructions, pseudo_labels = build_intent_paths(
        gen=gen,
        start_rc=start_rc,
        goal_rc=goal_rc,
        img_size=img_size,
        horizon=horizon,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        risk_threshold_deg=risk_threshold_deg,
    )

    terrain_with_paths_out = Path(args.terrain_with_paths_out)
    terrain_with_paths_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "map_id": "terrain_valid",
        "costmap": costmap,
        "height_map": torch.from_numpy(height_map).float(),
        "slope_map": torch.from_numpy(slope_map_deg).float(),
        "start_position": start_rc,
        "goal_position": goal_rc,
        "img_size": img_size,
        "pixel_resolution": pixel_resolution,
        "limit_angle_deg": limit_angle_deg,
        "horizon": horizon,
        "cost_weights": {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta},
        "risk_threshold_deg": risk_threshold_deg,
    }
    if paths:
        payload.update(
            {
                "paths": torch.from_numpy(np.array(paths, dtype=np.float32)).float(),
                "intent_types": intent_types,
                "intent_params": intent_params,
                "instructions": instructions,
                "pseudo_labels": pseudo_labels,
            }
        )
    torch.save(payload, terrain_with_paths_out)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, scheduler, vocab, model_cfg = load_model(args.checkpoint, device)
    horizon = int(model_cfg.get("data", {}).get("horizon", 120))
    tokens = text_to_tokens(args.instruction, vocab, max_seq_len=16)

    start_pos = to_normalized_xy(start_rc, img_size)
    goal_pos = to_normalized_xy(goal_rc, img_size)
    gen_path = run_inference(
        model=model,
        scheduler=scheduler,
        costmap=costmap,
        start_pos=start_pos,
        goal_pos=goal_pos,
        text_tokens=tokens,
        horizon=horizon,
        device=device,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    terrain_note = args.terrain_note if args.terrain_note is not None else "Unseen terrain (no reference path)"
    visualize_result(
        slope_map=slope_map_deg,
        height_map=height_map,
        gt_paths=None,
        gen_path=gen_path,
        instruction=args.instruction,
        img_size=img_size,
        out_path=str(output_path),
        show_gt=not args.no_gt,
        terrain_note=terrain_note,
    )

    print(f"Saved terrain pt : {terrain_path}")
    print(f"Saved terrain+paths pt : {terrain_with_paths_out}")
    print(f"Saved output png : {output_path}")
    print(f"Start (row,col)  : {start_rc}")
    print(f"Goal  (row,col)  : {goal_rc}")


if __name__ == "__main__":
    main()
