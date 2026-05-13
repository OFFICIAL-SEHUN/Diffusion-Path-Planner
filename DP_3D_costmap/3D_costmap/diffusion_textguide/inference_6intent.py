"""
Multi-intent inference — Height map (row 1) + Slope map (row 2), one column per intent.

Intents = every key in ``data/instruction/train/inst_train.json``: ``INTENT_CATALOG`` order
first, then any extra template-only types (e.g. ``short_path``) sorted alphabetically.
Each column uses the first template sentence for that type.

Usage:
  python inference_6intent.py --checkpoint checkpoints/final_model.pt    --terrain data/raw/terrain_05000.pt   --output results/inference_all_intents.png
"""

import argparse
from typing import Optional

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
from instruction_utils import load_instruction_templates
from scripts.generate_data import INTENT_CATALOG
from text_conditioning import (
    DEFAULT_FEATURE_DIMS,
    get_intent_to_id,
    is_frozen_feature_encoder,
    normalize_text_encoder_type,
)


INSTRUCTION_TEMPLATES = load_instruction_templates("train")


def _first_instruction(itype: str) -> str:
    templates = INSTRUCTION_TEMPLATES.get(itype)
    if templates:
        return templates[0]
    combined = []
    for p in itype.split("+"):
        ts = INSTRUCTION_TEMPLATES.get(p, [])
        if ts:
            combined.append(ts[0])
    return " 그리고 ".join(combined) if combined else itype


def _intent_types_in_order() -> list[str]:
    """All train-template intent keys: catalog order, then remaining keys sorted."""
    catalog_order = [e["type"] for e in INTENT_CATALOG]
    seen: set[str] = set()
    out: list[str] = []
    for t in catalog_order:
        if t in INSTRUCTION_TEMPLATES and t not in seen:
            out.append(t)
            seen.add(t)
    for t in sorted(INSTRUCTION_TEMPLATES.keys()):
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


INTENTS = [(t, _first_instruction(t)) for t in _intent_types_in_order()]

INTENT_LABELS = [
    t.replace("_", " ").replace("+", " + ").title() for t, _ in INTENTS
]

PATH_COLORS = ["#E63946"] * len(INTENTS)


def _resolve_text_encoder_type(m_cfg: dict, state_dict: dict) -> str:
    raw = m_cfg.get("text_encoder_type")
    if raw is None and isinstance(m_cfg.get("text_encoder"), dict):
        raw = m_cfg["text_encoder"].get("type")
    if raw is not None:
        return normalize_text_encoder_type(raw)
    keys = set(state_dict.keys())
    if any(k.startswith("text_encoder.") for k in keys):
        return "learnable"
    if any(k.startswith("intent_encoder.") for k in keys):
        return "onehot"
    if any(k.startswith("text_projection.") for k in keys):
        return normalize_text_encoder_type("t5_proj")
    return "learnable"


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    vocab = ckpt["vocab"]

    d_cfg = config.get("data", {})
    m_cfg = config.get("model", {})
    diff_cfg = config.get("diffusion", {})
    text_cfg = m_cfg.get("text_encoder", {}) or {}

    state_dict = ckpt["model_state_dict"]
    visual_backbone = m_cfg.get("visual_backbone")
    if visual_backbone is None:
        visual_backbone = (
            "convnext"
            if any(k.startswith("visual_encoder.backbone.") for k in state_dict)
            else "resnet"
        )

    text_encoder_type = _resolve_text_encoder_type(m_cfg, state_dict)
    num_intents = int(m_cfg.get("num_intents") or len(get_intent_to_id("train")))
    text_feature_dim = int(
        m_cfg.get("text_feature_dim")
        or DEFAULT_FEATURE_DIMS.get(text_encoder_type, 256)
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
        text_encoder_type=text_encoder_type,
        num_intents=num_intents,
        text_feature_dim=text_feature_dim,
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


def run_inference(
    model,
    scheduler,
    costmap,
    start_pos,
    goal_pos,
    horizon,
    device,
    *,
    text_encoder_type: str,
    text_tokens=None,
    text_features=None,
    intent_id: Optional[int] = None,
):
    costmap_t = costmap.unsqueeze(0).to(device)
    start_t = start_pos.unsqueeze(0).to(device)
    goal_t = goal_pos.unsqueeze(0).to(device)

    te = normalize_text_encoder_type(text_encoder_type)
    sample_kw: dict = {"show_progress": False}
    if te == "learnable":
        sample_kw["text_tokens"] = text_tokens.unsqueeze(0).to(device)
    elif te == "onehot":
        sample_kw["intent_ids"] = torch.tensor([intent_id], dtype=torch.long, device=device)
    elif te in {"clip", "clip_proj", "bert_proj", "t5_proj"}:
        sample_kw["text_features"] = text_features.unsqueeze(0).to(device)

    path = scheduler.sample(
        model,
        costmap_t,
        shape=(1, horizon, 2),
        start_pos=start_t,
        end_pos=goal_t,
        **sample_kw,
    )
    return path[0].cpu().numpy()


def visualize_intents(height_map, slope_map, gen_paths, gt_paths, img_size,
                      out_path, show_gt=True, terrain_note=None):
    """Row 1: Height map × N intents, Row 2: Slope map × N intents."""
    n = len(INTENTS)
    fig, axes = plt.subplots(2, n, figsize=(min(4.0 * n, 56), 8.5), squeeze=False)

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
                fs = max(7, min(13, 220 // max(n, 1)))
                ax.set_title(INTENT_LABELS[col], fontsize=fs, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        axes[row, 0].set_ylabel(row_label, fontsize=13, fontweight="bold",
                                labelpad=10)

    suptitle = f"{n}-Intent Comparison"
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
    text_encoder_type = model.text_encoder_type
    m_cfg = config.get("model", {})
    text_cfg = m_cfg.get("text_encoder", {}) or {}
    print(f"[Config] horizon={horizon}, device={device}, text_encoder={text_encoder_type}")

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

    frozen_feats = None
    if is_frozen_feature_encoder(text_encoder_type):
        from experiment.support.text_encoder_ablation import FrozenTextFeatureEncoder

        tname = text_cfg.get("model_name", m_cfg.get("text_model_name"))
        tbatch = int(text_cfg.get("batch_size", 64))
        enc = FrozenTextFeatureEncoder(
            text_encoder_type,
            model_name=tname,
            device=device,
            batch_size=tbatch,
        )
        sentences = [instr for _, instr in INTENTS]
        frozen_feats = enc.encode(sentences)

    intent_to_id = get_intent_to_id("train") if text_encoder_type == "onehot" else None

    gen_paths = []
    for idx, (intent_type, instruction) in enumerate(INTENTS):
        print(f"[{intent_type:20s}] \"{instruction}\"")
        if text_encoder_type == "learnable":
            tokens = text_to_tokens(instruction, vocab, max_seq_len=16)
            path = run_inference(
                model, scheduler, costmap, start_pos, goal_pos, horizon, device,
                text_encoder_type=text_encoder_type,
                text_tokens=tokens,
            )
        elif frozen_feats is not None:
            path = run_inference(
                model, scheduler, costmap, start_pos, goal_pos, horizon, device,
                text_encoder_type=text_encoder_type,
                text_features=frozen_feats[idx],
            )
        elif text_encoder_type == "onehot":
            path = run_inference(
                model, scheduler, costmap, start_pos, goal_pos, horizon, device,
                text_encoder_type=text_encoder_type,
                intent_id=intent_to_id.get(intent_type, 0),
            )
        else:
            path = run_inference(
                model, scheduler, costmap, start_pos, goal_pos, horizon, device,
                text_encoder_type=text_encoder_type,
            )
        gen_paths.append(path)

    out_path = args.output or str(_ROOT / "results" / "inference_all_intents.png")
    visualize_intents(
        height_map, slope_map, gen_paths, gt_paths, img_size, out_path,
        show_gt=not args.no_gt,
        terrain_note=args.terrain_note,
    )


if __name__ == "__main__":
    main()
