"""
6-Intent Inference — Height map (row 1) + Slope map (row 2), 6 intents per row.

Usage:
  python inference_6intent.py --checkpoint checkpoints/sample40k/final_model.pt \
                              --terrain data/raw/terrain_00087.pt \
                              --output results/inference_6intent.png
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

INTENTS = [
    ("baseline",        "Navigate along the default route"),
    ("left_bias",       "Stay to the left side"),
    ("right_bias",      "Stay to the right side"),
    ("avoid_steep",     "Avoid steep slopes"),
    ("prefer_flat",     "Choose the most level path available"),
    ("via_flat_region", "Pass through a flat midpoint region"),
]

INTENT_LABELS = [
    "Baseline",
    "Left bias",
    "Right bias",
    "Avoid steep",
    "Prefer flat",
    "Via flat region",
]

PATH_COLORS = [
    "#E63946",  # red
    "#E63946",  # steel blue
    "#E63946",  # teal
    "#E63946",  # gold
    "#E63946",  # sandy orange
    "#E63946",  # dark teal
]


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
        visual_pretrained=False,
        timm_model_name=m_cfg.get("timm_model_name"),
        timm_pretrained=False,
        input_img_size=d_cfg.get("img_size"),
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
    costmap_t = costmap.unsqueeze(0).to(device)
    start_t = start_pos.unsqueeze(0).to(device)
    goal_t = goal_pos.unsqueeze(0).to(device)
    tokens_t = text_tokens.unsqueeze(0).to(device)

    path = scheduler.sample(
        model, costmap_t, shape=(1, horizon, 2),
        start_pos=start_t, end_pos=goal_t,
        text_tokens=tokens_t, show_progress=False,
    )
    return path[0].cpu().numpy()


def visualize_6intent(height_map, slope_map, gen_paths, gt_paths, img_size,
                      out_path, show_gt=True, terrain_note=None):
    """Row 1: Height map × 6 intents, Row 2: Slope map × 6 intents."""
    n = len(INTENTS)
    fig, axes = plt.subplots(2, n, figsize=(4.0 * n, 8.5))

    def to_px(p):
        return (p + 1) / 2 * img_size

    draw_gt = show_gt and gt_paths is not None

    row_configs = [
        (height_map, "terrain", None, None, "Height map"),
        (slope_map,  "jet",     0,    35,   "Slope map (deg)"),
    ]

    for row, (bg_map, cmap, vmin, vmax, row_label) in enumerate(row_configs):
        for col in range(n):
            ax = axes[row, col]
            ax.imshow(bg_map, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)

            gen_px = to_px(gen_paths[col])
            ax.plot(gen_px[:, 0], gen_px[:, 1], color=PATH_COLORS[col],
                    lw=2.2, alpha=0.95,
                    label="Generated" if col == 0 and row == 0 else None)
            ax.scatter([gen_px[0, 0]], [gen_px[0, 1]], c="lime", s=50,
                       zorder=10, marker="o", edgecolors="black", linewidths=0.8)
            ax.scatter([gen_px[-1, 0]], [gen_px[-1, 1]], c="orange", s=50,
                       zorder=10, marker="*", edgecolors="black", linewidths=0.8)

            if row == 0:
                ax.set_title(INTENT_LABELS[col], fontsize=13, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        axes[row, 0].set_ylabel(row_label, fontsize=13, fontweight="bold",
                                labelpad=10)

    suptitle = "6-Intent Comparison"
    if terrain_note:
        suptitle += f"  ·  {terrain_note}"
    fig.suptitle(suptitle, fontsize=18, fontweight="bold", y=1.01)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--terrain", type=str, required=True,
                    help=".pt terrain file")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--no-gt", action="store_true",
                    help="Do not draw reference paths")
    ap.add_argument("--terrain-note", type=str, default=None,
                    help="e.g. 'Unseen terrain' — shown in title")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, scheduler, vocab, config = load_model(args.checkpoint, device)
    horizon = config.get("data", {}).get("horizon", 120)
    print(f"[Config] horizon={horizon}, device={device}")

    terrain = torch.load(args.terrain, map_location="cpu", weights_only=False)
    costmap = terrain["costmap"]
    height_map = terrain["height_map"].numpy()
    slope_map = terrain["slope_map"].numpy()
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

    gen_paths = []
    for intent_type, instruction in INTENTS:
        tokens = text_to_tokens(instruction, vocab, max_seq_len=16)
        print(f"[{intent_type:20s}] \"{instruction}\"")
        path = run_inference(model, scheduler, costmap, start_pos, goal_pos,
                             tokens, horizon, device)
        gen_paths.append(path)

    out_path = args.output or str(_ROOT / "results" / "inference_6intent.png")
    visualize_6intent(
        height_map, slope_map, gen_paths, gt_paths, img_size, out_path,
        show_gt=not args.no_gt,
        terrain_note=args.terrain_note,
    )


if __name__ == "__main__":
    main()
