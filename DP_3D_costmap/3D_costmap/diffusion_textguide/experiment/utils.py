"""
Shared utilities for the experiment suite.

- Terrain / checkpoint loading
- Normalized ↔ pixel coordinate conversion
- Path-along-terrain sampling (slope, height at each waypoint)
- Visualization helpers
"""

import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_EXP_DIR = Path(__file__).resolve().parent
_ROOT = _EXP_DIR.parent
sys.path.insert(0, str(_ROOT))

from model.network import ConditionalPathModel
from model.diffusion import DiffusionScheduler
from data_loader import build_vocab, text_to_tokens


# ── coordinate helpers ───────────────────────────────────────────────────────

def normalized_to_pixel(path_norm: np.ndarray, img_size: int) -> np.ndarray:
    """[-1,1] normalised (x=col, y=row) → pixel (row, col)."""
    px = (path_norm + 1.0) / 2.0 * img_size
    px = np.clip(px, 0, img_size - 1)
    return np.stack([px[:, 1], px[:, 0]], axis=1).astype(np.float64)


def pixel_to_normalized(path_px: np.ndarray, img_size: int) -> np.ndarray:
    """pixel (row, col) → [-1,1] normalised (x=col, y=row)."""
    x = (path_px[:, 1] / img_size) * 2 - 1
    y = (path_px[:, 0] / img_size) * 2 - 1
    return np.stack([x, y], axis=1).astype(np.float32)


def resample_path(path: np.ndarray, horizon: int) -> np.ndarray:
    """Resample arbitrary-length path to fixed *horizon* length."""
    n = len(path)
    if n == 0:
        return np.zeros((horizon, path.shape[-1]), dtype=np.float32)
    t_src = np.linspace(0, 1, n)
    t_dst = np.linspace(0, 1, horizon)
    cols = [np.interp(t_dst, t_src, path[:, c]) for c in range(path.shape[1])]
    return np.stack(cols, axis=1).astype(np.float32)


# ── terrain helpers ──────────────────────────────────────────────────────────

def load_terrain(pt_path: str) -> dict:
    """Load a terrain .pt file and return the dict with numpy arrays."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    out = dict(data)
    for k in ("costmap", "slope_map", "height_map", "paths"):
        if k in out and isinstance(out[k], torch.Tensor):
            out[k] = out[k].numpy()
    return out


def sample_terrain_along_path(
    path_px: np.ndarray,
    slope_map_deg: np.ndarray,
    height_map: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample slope (deg) and height at each waypoint of *path_px* (row, col)."""
    rows = np.clip(path_px[:, 0].astype(int), 0, slope_map_deg.shape[0] - 1)
    cols = np.clip(path_px[:, 1].astype(int), 0, slope_map_deg.shape[1] - 1)
    return slope_map_deg[rows, cols], height_map[rows, cols]


# ── model helpers ────────────────────────────────────────────────────────────

def load_model(
    ckpt_path: str,
    device: torch.device,
    override_backbone: Optional[str] = None,
    disable_text: bool = False,
    disable_cross_attn: bool = False,
) -> Tuple[ConditionalPathModel, DiffusionScheduler, dict, dict]:
    """Load model from checkpoint with optional ablation overrides.

    Returns (model, scheduler, vocab, config).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    vocab = ckpt["vocab"]

    d_cfg = config.get("data", {})
    m_cfg = config.get("model", {})
    diff_cfg = config.get("diffusion", {})

    state_dict = ckpt["model_state_dict"]
    backbone = override_backbone or m_cfg.get("visual_backbone")
    if backbone is None:
        backbone = (
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
        visual_backbone=backbone,
        convnext_pretrained=False,
    ).to(device)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)
    model.eval()

    scheduler = DiffusionScheduler(
        timesteps=diff_cfg.get("timesteps", 200),
        beta_start=diff_cfg.get("beta_start", 0.0001),
        beta_end=diff_cfg.get("beta_end", 0.02),
        device=device,
    )
    return model, scheduler, vocab, config


def run_inference_batch(
    model: ConditionalPathModel,
    scheduler: DiffusionScheduler,
    costmaps: torch.Tensor,
    start_positions: torch.Tensor,
    goal_positions: torch.Tensor,
    text_tokens: torch.Tensor,
    horizon: int,
    device: torch.device,
    disable_text: bool = False,
) -> np.ndarray:
    """Run diffusion sampling for a batch; returns [B, horizon, 2] numpy."""
    costmaps = costmaps.to(device)
    start_positions = start_positions.to(device)
    goal_positions = goal_positions.to(device)
    tokens = text_tokens.to(device) if not disable_text else None

    paths = scheduler.sample(
        model, costmaps, shape=(costmaps.shape[0], horizon, 2),
        start_pos=start_positions, end_pos=goal_positions,
        text_tokens=tokens, show_progress=False,
    )
    return paths.cpu().numpy()


# ── visualisation ────────────────────────────────────────────────────────────

def plot_paths_on_terrain(
    slope_map_deg: np.ndarray,
    height_map: np.ndarray,
    img_size: int,
    paths_dict: Dict[str, np.ndarray],
    title: str = "",
    out_path: Optional[str] = None,
):
    """Overlay multiple named paths (normalised coords) on slope + height maps."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10.colors

    for ax, (data, cmap, label) in zip(
        axes,
        [(slope_map_deg, "jet", "Slope (deg)"), (height_map, "terrain", "Height")],
    ):
        ax.imshow(data, cmap=cmap, origin="lower")
        for i, (name, p_norm) in enumerate(paths_dict.items()):
            px = (p_norm + 1) / 2 * img_size
            ax.plot(px[:, 0], px[:, 1], "-", color=colors[i % len(colors)],
                    lw=1.8, alpha=0.85, label=name)
            ax.scatter([px[0, 0]], [px[0, 1]], c="lime", s=50, zorder=10,
                       edgecolors="k", marker="o")
            ax.scatter([px[-1, 0]], [px[-1, 1]], c="orange", s=50, zorder=10,
                       edgecolors="k", marker="*")
        ax.set_title(label, fontsize=10)
        ax.axis("off")
        ax.legend(fontsize=7, loc="lower right", framealpha=0.7)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
