"""
Text-conditioned Diffusion Path Planner — Inference Script

Usage:
  python inference.py --checkpoint checkpoints/final_model.pt \
                      --terrain data/raw/terrain_00001.pt \
                      --instruction "Stay to the left side"
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import sys
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from model.network import ConditionalPathModel
from model.diffusion import DiffusionScheduler
from data_loader import text_to_tokens


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    vocab = ckpt["vocab"]

    d_cfg = config.get("data", {})
    m_cfg = config.get("model", {})
    diff_cfg = config.get("diffusion", {})

    state_dict = ckpt["model_state_dict"]
    visual_backbone = m_cfg.get("visual_backbone")
    if visual_backbone is None:
        visual_backbone = (
            "convnext"
            if any(k.startswith("visual_encoder.backbone.") for k in state_dict)
            else "resnet"
        )

    model = ConditionalPathModel(
        transition_dim=2,
        dim=m_cfg.get("base_dim", 64),
        horizon=d_cfg.get("horizon", 120),
        visual_dim=m_cfg.get("image_feat_dim", 256),
        text_dim=256,
        vocab_size=len(vocab),
        max_seq_len=16,
        visual_backbone=visual_backbone,
        convnext_pretrained=False,
        timm_model_name=m_cfg.get("timm_model_name"),
        timm_pretrained=False,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    scheduler = DiffusionScheduler(
        timesteps=diff_cfg.get("timesteps", 200),
        beta_start=diff_cfg.get("beta_start", 0.0001),
        beta_end=diff_cfg.get("beta_end", 0.02),
        device=device,
    )

    return model, scheduler, vocab, config


def run_inference(model, scheduler, costmap, start_pos, goal_pos,
                  text_tokens, horizon, device):
    """단일 instruction에 대해 경로 샘플링."""
    costmap_t = costmap.unsqueeze(0).to(device)
    start_t = start_pos.unsqueeze(0).to(device)
    goal_t = goal_pos.unsqueeze(0).to(device)
    tokens_t = text_tokens.unsqueeze(0).to(device)

    path = scheduler.sample(
        model, costmap_t, shape=(1, horizon, 2),
        start_pos=start_t, end_pos=goal_t,
        text_tokens=tokens_t, show_progress=True,
    )
    return path[0].cpu().numpy()


def visualize_result(slope_map, height_map, gt_paths, gen_path, instruction,
                     img_size, out_path=None, show_gt=True, terrain_note=None):
    """슬로프맵 + 하이트맵 둘 다에 경로를 오버레이해서 보여줌.
    show_gt=False면 학습/데이터에 있던 reference path(pseudo label)는 그리지 않음.
    terrain_note: e.g. 'Unseen terrain' → 제목에 표시해 어떤 조건인지 구분.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    def to_px(p):
        return (p + 1) / 2 * img_size

    draw_gt = show_gt and gt_paths is not None

    # Slope map (deg)
    ax0 = axes[0]
    im0 = ax0.imshow(slope_map, cmap="jet", origin="lower", vmin=0, vmax=35)
    if draw_gt:
        for i, gp in enumerate(gt_paths):
            ax0.plot(*to_px(gp).T, "g-", lw=1.5, alpha=0.4,
                     label="Reference (pseudo)" if i == 0 else None)
    gen_px = to_px(gen_path)
    ax0.plot(gen_px[:, 0], gen_px[:, 1], "r-", lw=2.0, alpha=0.9, label="Generated")
    ax0.scatter([gen_px[0, 0]], [gen_px[0, 1]], c="lime", s=60, zorder=10, marker="o",
                edgecolors="black", label="Start")
    ax0.scatter([gen_px[-1, 0]], [gen_px[-1, 1]], c="orange", s=60, zorder=10, marker="*",
                edgecolors="black", label="Goal")
    ax0.set_title("Slope map (deg)", fontsize=15)
    ax0.axis("off")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02)

    # Height map
    ax1 = axes[1]
    im1 = ax1.imshow(height_map, cmap="terrain", origin="lower")
    if draw_gt:
        for gp in gt_paths:
            px = to_px(gp)
            ax1.plot(px[:, 0], px[:, 1], "g-", lw=1.5, alpha=0.4)
    ax1.plot(gen_px[:, 0], gen_px[:, 1], "r-", lw=2.0, alpha=0.9, label="Generated")
    ax1.scatter([gen_px[0, 0]], [gen_px[0, 1]], c="lime", s=60, zorder=10, marker="o",
                edgecolors="black", label="Start")
    ax1.scatter([gen_px[-1, 0]], [gen_px[-1, 1]], c="orange", s=60, zorder=10, marker="*",
                edgecolors="black", label="Goal")
    ax1.set_title("Height map", fontsize=15)
    ax1.axis("off")

    handles, labels = ax0.get_legend_handles_labels()
    if handles:
        ax0.legend(handles, labels, loc="lower right", fontsize=10, framealpha=0.8)
    title = f'Instruction: "{instruction}"'
    if terrain_note:
        title += f"  ·  {terrain_note}"
    fig.suptitle(title, fontsize=20)

    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--terrain", type=str, required=True,
                    help=".pt terrain file")
    ap.add_argument("--instruction", type=str, required=True,
                    help='e.g. "Stay to the left side"')
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--no-gt", action="store_true",
                    help="Do not draw reference path (pseudo label) even if present in terrain")
    ap.add_argument("--terrain-note", type=str, default=None,
                    help="e.g. 'Unseen terrain' — shown in plot title")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, scheduler, vocab, config = load_model(args.checkpoint, device)
    horizon = config.get("data", {}).get("horizon", 120)
    print("[Config] Model, diffusion, horizon from checkpoint (training config)")

    terrain = torch.load(args.terrain, map_location="cpu", weights_only=False)
    costmap = terrain["costmap"]
    slope_map = terrain["slope_map"].numpy()
    height_map = terrain["height_map"].numpy()
    img_size = int(terrain["img_size"])
    gt_paths = terrain["paths"].numpy() if "paths" in terrain else None

    if "start_position" in terrain:
        s = terrain["start_position"]
        g = terrain["goal_position"]
        start_pos = torch.tensor([(s[1] / img_size) * 2 - 1,
                                   (s[0] / img_size) * 2 - 1], dtype=torch.float32)
        goal_pos = torch.tensor([(g[1] / img_size) * 2 - 1,
                                  (g[0] / img_size) * 2 - 1], dtype=torch.float32)
    elif gt_paths is not None:
        start_pos = torch.tensor(gt_paths[0, 0], dtype=torch.float32)
        goal_pos = torch.tensor(gt_paths[0, -1], dtype=torch.float32)
    else:
        raise ValueError("No start/goal found in terrain file")

    tokens = text_to_tokens(args.instruction, vocab, max_seq_len=16)

    print(f"Instruction: \"{args.instruction}\"")
    print(f"Start: {start_pos.numpy()}, Goal: {goal_pos.numpy()}")

    gen_path = run_inference(model, scheduler, costmap, start_pos, goal_pos,
                             tokens, horizon, device)

    out_path = args.output or str(_ROOT / "results" / "inference_output.png")
    visualize_result(
        slope_map, height_map, gt_paths, gen_path, args.instruction,
        img_size, out_path,
        show_gt=not args.no_gt,
        terrain_note=args.terrain_note,
    )


if __name__ == "__main__":
    main()
