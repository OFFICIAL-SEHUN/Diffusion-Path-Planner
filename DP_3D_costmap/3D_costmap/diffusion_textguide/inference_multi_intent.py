"""
Multi-intent inference — 2x4 height-map grid from the 10-intent catalog.

Shows all train-template types except ``baseline`` and ``energy_efficient``
(catalog order). Row 1: intents 1–4, row 2: intents 5–8.

Usage:
  python inference_multi_intent.py --checkpoint checkpoints/backbone_ablation_10intent/final_model.pt \\
      --terrain data/raw/terrain_05000.pt \\
      --output results/inference_all_intents_backbone_ablation_8intent.png
"""

import argparse
from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
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

# Omit from grid (10 train intents → 8 panels in 2×4 layout).
EXCLUDED_INTENTS = frozenset({"baseline", "energy_efficient"})
GRID_COLS = 4


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
        if t in EXCLUDED_INTENTS:
            continue
        if t in INSTRUCTION_TEMPLATES and t not in seen:
            out.append(t)
            seen.add(t)
    for t in sorted(INSTRUCTION_TEMPLATES.keys()):
        if t in EXCLUDED_INTENTS:
            continue
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


INTENTS = [(t, _first_instruction(t)) for t in _intent_types_in_order()]

INTENT_LABELS = [
    t.replace("_", " ").replace("+", " + ").title() for t, _ in INTENTS
]

PATH_COLORS = ["#E63946"] * len(INTENTS)

# Start / goal: small on-map markers; meaning explained in figure legend.
START_GOAL_SCATTER_SIZE = 100
START_COLOR = "lime"
GOAL_COLOR = "orange"


def _configure_plot_font() -> None:
    """Use Times New Roman for all figure text (serif fallback on Linux)."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    if "Times New Roman" in available:
        family = "Times New Roman"
    else:
        family = next(
            (
                name for name in ("Times", "Nimbus Roman", "Liberation Serif")
                if name in available
            ),
            "DejaVu Serif",
        )
        print(f"[Font] 'Times New Roman' not installed; using {family}")

    plt.rcParams.update(
        {
            "font.family": family,
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
        }
    )


_START_GOAL_LEGEND_HANDLES = [
    Line2D(
        [0], [0],
        marker="o",
        color="w",
        markerfacecolor=START_COLOR,
        markeredgecolor="black",
        markeredgewidth=0.6,
        markersize=8,
        linestyle="None",
        label="Start",
    ),
    Line2D(
        [0], [0],
        marker="*",
        color="w",
        markerfacecolor=GOAL_COLOR,
        markeredgecolor="black",
        markeredgewidth=0.6,
        markersize=10,
        linestyle="None",
        label="Goal",
    ),
]


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
                      out_path, show_gt=True):
    """2 rows × 4 columns height-map grid, one panel per intent."""
    n = len(INTENTS)
    ncols = GRID_COLS
    nrows = (n + ncols - 1) // ncols
    _configure_plot_font()
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.0 * ncols + 0.9, 4.25 * nrows), squeeze=False,
    )

    def to_px(p):
        return (p + 1) / 2 * img_size

    legend_fs = max(10, min(12, 144 // max(ncols, 1)))
    title_fs = max(14, min(16, 440 // max(ncols, 1)))

    for idx in range(nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        if idx >= n:
            ax.axis("off")
            continue

        ax.imshow(height_map, cmap="terrain", origin="lower")

        gen_px = to_px(gen_paths[idx])
        ax.plot(gen_px[:, 0], gen_px[:, 1], color=PATH_COLORS[idx],
                lw=2.2, alpha=0.95)
        ax.scatter(
            [gen_px[0, 0]], [gen_px[0, 1]],
            c=START_COLOR,
            s=START_GOAL_SCATTER_SIZE,
            zorder=10,
            marker="o",
            edgecolors="black",
            linewidths=0.5,
        )
        ax.scatter(
            [gen_px[-1, 0]], [gen_px[-1, 1]],
            c=GOAL_COLOR,
            s=START_GOAL_SCATTER_SIZE,
            zorder=10,
            marker="*",
            edgecolors="black",
            linewidths=1.0,
        )

        ax.legend(
            handles=_START_GOAL_LEGEND_HANDLES,
            loc="upper right",
            fontsize=legend_fs,
            framealpha=0.88,
            handlelength=0.9,
            handletextpad=0.25,
            borderpad=0.25,
            labelspacing=0.2,
        )

        ax.set_title(INTENT_LABELS[idx], fontsize=title_fs, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Leave space on the right for the height colorbar.
    fig.tight_layout(rect=[0, 0, 0.92, 1])

    # Vertical colorbar (height range bar) spanning the full figure height on the right.
    vmin, vmax = float(height_map.min()), float(height_map.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap="terrain", norm=norm)
    sm.set_array([])

    # [left, bottom, width, height] in figure-fraction coordinates
    cbar_ax = fig.add_axes([0.935, 0.06, 0.018, 0.88])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Height (m)", fontsize=max(14, title_fs), fontweight="bold", labelpad=10)
    cbar.ax.tick_params(labelsize=max(13, legend_fs + 1))
    for tick_label in cbar.ax.get_yticklabels():
        tick_label.set_fontweight("bold")

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
    )


if __name__ == "__main__":
    main()
