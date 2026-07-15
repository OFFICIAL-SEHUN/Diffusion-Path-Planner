"""
Visualize diffusion reverse-process snapshots (x_T, x_{T-1}, x_t, x_0)
for a single terrain + intent using a frozen-text (e.g. resnet_t5) checkpoint.

Each subplot overlays a **height map** with the noisy/denoised waypoints
of the chosen step. Waypoints are connected by a dotted polyline so the
denoising trajectory through the path's index dimension is visible.

Example:
  python scripts/visualize_diffusion_steps.py \
    --checkpoint checkpoints/final_model.pt \
    --terrain data/raw/terrain_00376.pt \
    --intent avoid_steep \
    --num-inference-steps 10 \
    --dpi 400 \
    --output results/diffusion_steps/terrain00376_avoid_steep.png

By default the plot is **cropped to the terrain** ``[-1,1]^2``; use
``--full-extent`` to show off-map noise again. Fewer steps → DDIM (η with
``--ddim-eta``); else full DDPM.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from data_loader import text_to_tokens  # noqa: E402
from instruction_utils import load_instruction_templates  # noqa: E402
from model.diffusion import DiffusionScheduler  # noqa: E402
from model.network import ConditionalPathModel  # noqa: E402
from text_conditioning import (  # noqa: E402
    DEFAULT_FEATURE_DIMS,
    get_intent_to_id,
    is_frozen_feature_encoder,
    normalize_text_encoder_type,
)


# ---------------------------------------------------------------------------
# Model loading (mirrors inference_multi_intent.load_model with text_features flow)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Reverse sampling with intermediate-state capture
# ---------------------------------------------------------------------------
def _build_ddim_timestep_subset(T: int, num_steps: int) -> List[int]:
    """Indices in [0, T-1], descending (noisy → clean), linearly spaced."""
    if num_steps < 2:
        raise ValueError("num_inference_steps must be >= 2 when < T")
    if num_steps >= T:
        return list(range(T - 1, -1, -1))
    idx = np.linspace(0, T - 1, num_steps).round().astype(np.int64)
    return np.unique(idx)[::-1].tolist()


@torch.no_grad()
def sample_with_snapshots(
    model: torch.nn.Module,
    scheduler: DiffusionScheduler,
    costmap: torch.Tensor,
    shape: Tuple[int, int, int],
    start_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    snapshot_steps: List[int],
    *,
    text_tokens: Optional[torch.Tensor] = None,
    intent_ids: Optional[torch.Tensor] = None,
    text_features: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    num_inference_steps: Optional[int] = None,
    ddim_eta: float = 0.0,
) -> dict:
    """Run reverse sampling and return x at the requested timesteps.

    ``T`` in ``snapshot_steps`` = initial noise before any denoising; ``0 … T-1``
    are usual DDPM time indices for titles.

    If ``num_inference_steps`` is None or ``>= T``, uses the full **DDPM**
    reverse process. Otherwise uses **DDIM** on a linear subsample of
    ``[0, T-1]`` (same ε network, original ``t`` for conditioning).
    """
    device = scheduler.device
    T = scheduler.timesteps
    alphas_cumprod = scheduler.alphas_cumprod

    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    B = shape[0]
    x = torch.randn(shape, device=device)
    if start_pos is not None:
        x[:, 0, :] = start_pos
    if goal_pos is not None:
        x[:, -1, :] = goal_pos

    snapshots: dict = {}
    if T in snapshot_steps:
        snapshots[T] = x.detach().cpu().clone()

    use_ddim = num_inference_steps is not None and num_inference_steps < T
    timesteps = (
        _build_ddim_timestep_subset(T, num_inference_steps)
        if use_ddim
        else list(reversed(range(T)))
    )

    snapshot_set = set(snapshot_steps)

    for i, t_val in enumerate(timesteps):
        t = torch.full((B,), t_val, device=device, dtype=torch.long)

        eps_pred = model(
            x, t, costmap,
            start_pos=start_pos, goal_pos=goal_pos,
            text_tokens=text_tokens,
            intent_ids=intent_ids,
            text_features=text_features,
        )

        if use_ddim:
            ab_t = alphas_cumprod[t_val]
            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
                ab_prev = alphas_cumprod[t_prev]
            else:
                ab_prev = torch.tensor(1.0, device=device)

            x0_pred = (x - (1.0 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()

            if ddim_eta > 0:
                sigma = ddim_eta * (
                    ((1.0 - ab_prev) / (1.0 - ab_t)).sqrt()
                    * (1.0 - ab_t / ab_prev).sqrt()
                )
            else:
                sigma = torch.tensor(0.0, device=device, dtype=x.dtype)

            dir_xt = (
                (1.0 - ab_prev - sigma.pow(2)).clamp(min=0.0).sqrt() * eps_pred
            )
            x = ab_prev.sqrt() * x0_pred + dir_xt
            if ddim_eta > 0 and i + 1 < len(timesteps):
                x = x + sigma * torch.randn_like(x)
        else:
            alpha_t = scheduler.alphas[t_val]
            beta_t = scheduler.betas[t_val]
            ab_t = alphas_cumprod[t_val]
            mean = (1.0 / alpha_t.sqrt()) * (
                x - (beta_t / (1.0 - ab_t).sqrt()) * eps_pred
            )
            if t_val > 0:
                z = torch.randn_like(x)
                x = mean + beta_t.sqrt() * z
            else:
                x = mean

        if start_pos is not None:
            x[:, 0, :] = start_pos
        if goal_pos is not None:
            x[:, -1, :] = goal_pos

        t_next = timesteps[i + 1] if i + 1 < len(timesteps) else 0
        if t_next in snapshot_set:
            snapshots[t_next] = x.detach().cpu().clone()

    for s_key in snapshot_steps:
        if s_key in snapshots:
            continue
        if not snapshots:
            continue
        nearest = min(snapshots.keys(), key=lambda k: abs(k - s_key))
        snapshots[s_key] = snapshots[nearest]

    return snapshots


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def visualize_steps(
    height_map: np.ndarray,
    snapshots: dict,
    snapshot_order: List[int],
    snapshot_labels: List[str],
    img_size: int,
    instruction: str,
    intent_name: str,
    timesteps: int,
    out_path: Path,
    *,
    dpi: int = 300,
    fig_scale: float = 1.25,
    crop_to_map: bool = True,
    map_pad: float = 0.02,
    full_extent_lim: Optional[float] = None,
) -> None:
    """Render snapshots in normalized [-1, 1] coords.

    By default (``crop_to_map=True``) axes are limited to the terrain square
    ``[-1, 1]^2`` (plus ``map_pad``). Path segments outside are clipped at the
    border. Set ``crop_to_map=False`` to expand axes to fit noisy samples
    (``full_extent_lim`` caps the half-width, default 6).
    """
    n = len(snapshot_order)
    # Figure size (inches); ``dpi`` sets rasterized output pixels ≈ width * dpi.
    fig, axes = plt.subplots(
        1, n, figsize=(4.6 * n * fig_scale, 4.9 * fig_scale), squeeze=False,
    )
    axes = axes[0]

    # Scale line/marker sizes with DPI so 300 DPI exports stay readable.
    r = max(dpi / 150.0, 1.0)

    if crop_to_map:
        pad = float(map_pad)
        half = 1.0 + pad
        xa, xb, ya, yb = -half, half, -half, half
    else:
        all_coords = np.concatenate([s[0].numpy() for s in snapshots.values()], axis=0)
        lim_cap = full_extent_lim if full_extent_lim is not None else 6.0
        extent_lim = max(1.05, float(np.abs(all_coords).max()) * 1.05)
        extent_lim = min(extent_lim, lim_cap)
        xa, xb, ya, yb = -extent_lim, extent_lim, -extent_lim, extent_lim

    for ax, t_val, label in zip(axes, snapshot_order, snapshot_labels):
        ax.imshow(
            height_map,
            cmap="terrain",
            origin="lower",
            extent=(-1.0, 1.0, -1.0, 1.0),
            zorder=1,
        )
        # Light border around the terrain (normalized [-1, 1]).
        ax.add_patch(plt.Rectangle(
            (-1.0, -1.0), 2.0, 2.0,
            fill=False, edgecolor="black", lw=0.8 * r, zorder=2,
        ))

        path = snapshots[t_val][0].numpy()  # [H, 2] in normalized [-1, 1] coords

        # Dotted connector between consecutive waypoints (path-index order).
        ax.plot(
            path[:, 0], path[:, 1],
            linestyle=(0, (1.5, 2.0)),  # tight dotted
            color="black", lw=1.3 * r, alpha=0.95, zorder=3,
            path_effects=[pe.Stroke(linewidth=2.4 * r, foreground="white"), pe.Normal()],
        )
        ax.scatter(
            path[:, 0], path[:, 1],
            s=16 * (r ** 2), c="#E63946", edgecolors="black",
            linewidths=0.35 * r, zorder=4,
        )

        # Start / goal markers (inpainted at every step).
        ax.scatter(
            [path[0, 0]], [path[0, 1]],
            c="lime", s=80 * (r ** 2), marker="o", edgecolors="black", linewidths=0.9 * r,
            zorder=5, label="Start" if ax is axes[0] else None,
        )
        ax.scatter(
            [path[-1, 0]], [path[-1, 1]],
            c="orange", s=110 * (r ** 2), marker="*", edgecolors="black", linewidths=0.9 * r,
            zorder=5, label="Goal" if ax is axes[0] else None,
        )

        ax.set_xlim(xa, xb)
        ax.set_ylim(ya, yb)
        ax.set_aspect("equal")
        ax.set_title(
            f"{label}  (t={t_val}/{timesteps})",
            fontsize=int(12 * min(r, 1.4)), fontweight="bold",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    # if handles:
    #     axes[0].legend(
    #         handles, labels, loc="lower right",
    #         fontsize=int(9 * min(r, 1.35)), framealpha=0.85,
    #     )

    fig.suptitle(
        f"Reverse diffusion snapshots (height map) · intent: {intent_name}\n"
        f"instruction: \"{instruction}\"  ·  img_size={img_size}",
        fontsize=int(13 * min(r, 1.35)), fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    print(f"Saved: {out_path}  (dpi={dpi})")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def _first_instruction(intent: str) -> str:
    templates = load_instruction_templates("train").get(intent)
    if not templates:
        raise ValueError(f"No train instruction templates for intent={intent!r}")
    return templates[0]


def _encode_intent(
    text_encoder_type: str,
    instruction: str,
    intent: str,
    vocab,
    config,
    device,
):
    """Return (text_tokens, intent_ids, text_features) for the configured encoder."""
    te = normalize_text_encoder_type(text_encoder_type)
    if te == "learnable":
        tokens = text_to_tokens(instruction, vocab, max_seq_len=16).unsqueeze(0).to(device)
        return tokens, None, None
    if te == "onehot":
        intent_to_id = get_intent_to_id("train")
        if intent not in intent_to_id:
            raise ValueError(f"intent {intent!r} not in onehot intent table")
        return None, torch.tensor([intent_to_id[intent]], dtype=torch.long, device=device), None
    if is_frozen_feature_encoder(te):
        from experiment.support.text_encoder_ablation import FrozenTextFeatureEncoder
        m_cfg = config.get("model", {})
        text_cfg = m_cfg.get("text_encoder", {}) or {}
        tname = text_cfg.get("model_name", m_cfg.get("text_model_name"))
        tbatch = int(text_cfg.get("batch_size", 64))
        enc = FrozenTextFeatureEncoder(
            te, model_name=tname, device=device, batch_size=tbatch,
        )
        feats = enc.encode([instruction]).to(device)
        return None, None, feats
    if te == "no_text":
        return None, None, None
    raise ValueError(f"Unsupported text_encoder_type={te!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--terrain", type=str, required=True)
    ap.add_argument("--intent", type=str, default="avoid_steep")
    ap.add_argument("--instruction", type=str, default=None,
                    help="Override instruction text (default: first template of intent)")
    ap.add_argument("--mid-step", type=int, default=None,
                    help="Intermediate timestep t to snapshot (default: T // 2)")
    ap.add_argument("--num-inference-steps", type=int, default=None,
                    help="Fewer reverse steps via DDIM (e.g. 10). "
                         "Default: full DDPM (T from checkpoint).")
    ap.add_argument("--ddim-eta", type=float, default=0.0,
                    help="DDIM stochasticity (0 = deterministic, 1 ≈ DDPM-like).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--dpi", type=int, default=300,
                    help="PNG resolution (default 300). Use 400–600 for posters.")
    ap.add_argument("--fig-scale", type=float, default=1.25,
                    help="Multiply default figure width/height in inches (default 1.25).")
    ap.add_argument("--full-extent", action="store_true",
                    help="Show axes wide enough for noise far outside [-1,1] (default: crop to map).")
    ap.add_argument("--map-pad", type=float, default=0.02,
                    help="Extra margin around the terrain when cropped (normalized coords).")
    ap.add_argument("--extent-max", type=float, default=6.0,
                    help="With --full-extent, cap axis half-range (default 6).")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, scheduler, vocab, config = load_model(args.checkpoint, device)
    horizon = config.get("data", {}).get("horizon", 120)
    text_encoder_type = model.text_encoder_type
    T = scheduler.timesteps

    instruction = args.instruction or _first_instruction(args.intent)
    print(f"[Config] text_encoder={text_encoder_type}  T={T}  device={device}")
    print(f"[Intent] {args.intent} | \"{instruction}\"")

    terrain = torch.load(args.terrain, map_location="cpu", weights_only=False)
    costmap = terrain["costmap"].unsqueeze(0).to(device)
    height_map = terrain["height_map"].numpy()
    img_size = int(terrain["img_size"])

    if "start_position" in terrain:
        s = terrain["start_position"]
        g = terrain["goal_position"]
        start_pos = torch.tensor(
            [(s[1] / img_size) * 2 - 1, (s[0] / img_size) * 2 - 1],
            dtype=torch.float32,
        ).unsqueeze(0).to(device)
        goal_pos = torch.tensor(
            [(g[1] / img_size) * 2 - 1, (g[0] / img_size) * 2 - 1],
            dtype=torch.float32,
        ).unsqueeze(0).to(device)
    elif "paths" in terrain:
        gt = terrain["paths"].numpy()
        start_pos = torch.tensor(gt[0, 0], dtype=torch.float32).unsqueeze(0).to(device)
        goal_pos = torch.tensor(gt[0, -1], dtype=torch.float32).unsqueeze(0).to(device)
    else:
        raise ValueError("No start/goal in terrain file")

    text_tokens, intent_ids, text_features = _encode_intent(
        text_encoder_type, instruction, args.intent, vocab, config, device,
    )

    # Full DDPM vs DDIM (fewer steps).
    n_steps = args.num_inference_steps
    if n_steps is not None and n_steps < T:
        sched = _build_ddim_timestep_subset(T, n_steps)
        first_after = sched[1] if len(sched) > 1 else 0
        mid_t = sched[len(sched) // 2]
        snapshot_steps = [T, first_after, mid_t, 0]
        snapshot_labels = ["x_T", "x (DDIM 1st)", "x (DDIM mid)", "x_0"]
        print(
            f"[Sampler] DDIM  S={len(sched)}  η={args.ddim_eta}  "
            f"visited_t (noisy→clean): {sched}"
        )
    else:
        mid_t = args.mid_step if args.mid_step is not None else T // 2
        if not 0 < mid_t < T - 1:
            raise ValueError(
                f"--mid-step must satisfy 0 < t < T-1 (got {mid_t}, T={T})"
            )
        snapshot_steps = [T, T - 1, mid_t, 0]
        snapshot_labels = ["x_T", "x_{T-1}", "x_t", "x_0"]
        print(f"[Sampler] full DDPM  T={T}")

    snapshots = sample_with_snapshots(
        model, scheduler, costmap,
        shape=(1, horizon, 2),
        start_pos=start_pos, goal_pos=goal_pos,
        snapshot_steps=snapshot_steps,
        text_tokens=text_tokens,
        intent_ids=intent_ids,
        text_features=text_features,
        seed=args.seed,
        num_inference_steps=n_steps,
        ddim_eta=args.ddim_eta,
    )

    out_path = Path(args.output) if args.output else (
        _ROOT / "results" / "diffusion_steps"
        / f"{Path(args.terrain).stem}_{args.intent}_steps.png"
    )

    visualize_steps(
        height_map, snapshots, snapshot_steps, snapshot_labels,
        img_size, instruction, args.intent, T, out_path,
        dpi=args.dpi, fig_scale=args.fig_scale,
        crop_to_map=not args.full_extent,
        map_pad=args.map_pad,
        full_extent_lim=args.extent_max,
    )


if __name__ == "__main__":
    main()
