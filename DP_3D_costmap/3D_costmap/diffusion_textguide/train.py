"""
Text-conditioned Diffusion Path Planner — Training Script

Usage:
  python train.py --config configs/default.yaml --data-dir data/raw
  python train.py --config configs/convnext.yaml --data-dir data/raw
  python train.py --config configs/resnet.yaml --data-dir data/raw
  python train.py --config configs/swin.yaml --data-dir data/raw
  # python train.py --config configs/convnext.yaml --data-dir data/raw --resume checkpoints/convnext/epoch_16000.pt
"""

import os
import sys
import argparse
import time
from typing import Optional
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from model.network import ConditionalPathModel
from model.diffusion import DiffusionScheduler
from data_loader import TextGuideDataset, build_vocab, text_to_tokens
from experiment.ablation_logger import AblationLogger
from experiment.utils import load_terrain
from experiment.metrics import compute_all_metrics


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_checkpoint(model, optimizer, epoch, vocab, config, path, scaler=None):
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocab": vocab,
        "config": config,
    }
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(payload, path)


@torch.no_grad()
def visualize_sample(model, scheduler, dataset, epoch, device, out_dir):
    """학습 중 샘플링하여 시각화."""
    model.eval()
    idx = np.random.randint(len(dataset))
    costmap, gt_path, tokens = dataset[idx]
    costmap = costmap.unsqueeze(0).to(device)
    tokens = tokens.unsqueeze(0).to(device)
    gt_path = gt_path.numpy()

    start_pos = torch.tensor(gt_path[0], device=device).unsqueeze(0)
    goal_pos = torch.tensor(gt_path[-1], device=device).unsqueeze(0)

    horizon = gt_path.shape[0]
    generated = scheduler.sample(
        model, costmap, shape=(1, horizon, 2),
        start_pos=start_pos, end_pos=goal_pos,
        text_tokens=tokens, show_progress=False,
    )
    gen_path = generated[0].cpu().numpy()

    slope_norm = costmap[0, 0].cpu().numpy()   # [0,1]
    height_norm = costmap[0, 1].cpu().numpy()  # [0,1]
    img_size = slope_norm.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    def to_pixels(p):
        return (p + 1) / 2 * img_size

    gt_px = to_pixels(gt_path)
    gen_px = to_pixels(gen_path)

    ax0 = axes[0]
    im0 = ax0.imshow(slope_norm * 90.0, cmap="jet", origin="lower", vmin=0, vmax=35)
    ax0.plot(gt_px[:, 0], gt_px[:, 1], "g-", lw=2, alpha=0.7, label="GT")
    ax0.plot(gen_px[:, 0], gen_px[:, 1], "r-", lw=2, alpha=0.7, label="Generated")
    ax0.scatter([gt_px[0, 0]], [gt_px[0, 1]], c="lime", s=50, zorder=10, marker="o")
    ax0.scatter([gt_px[-1, 0]], [gt_px[-1, 1]], c="lime", s=50, zorder=10, marker="*")
    ax0.set_title("Slope map (deg)", fontsize=9)
    ax0.axis("off")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02)

    ax1 = axes[1]
    im1 = ax1.imshow(height_norm, cmap="terrain", origin="lower")
    ax1.plot(gt_px[:, 0], gt_px[:, 1], "g-", lw=2, alpha=0.7, label="GT")
    ax1.plot(gen_px[:, 0], gen_px[:, 1], "r-", lw=2, alpha=0.7, label="Generated")
    ax1.scatter([gt_px[0, 0]], [gt_px[0, 1]], c="lime", s=50, zorder=10, marker="o")
    ax1.scatter([gt_px[-1, 0]], [gt_px[-1, 1]], c="lime", s=50, zorder=10, marker="*")
    ax1.set_title("Height map (norm.)", fontsize=9)
    ax1.axis("off")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)

    fig.suptitle(f"Epoch {epoch}", fontsize=11)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"sample_epoch_{epoch:05d}.png")
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    model.train()
    return out_path


# ── Validation helpers ────────────────────────────────────────────────────────

@torch.no_grad()
def compute_val_loss(
    model: ConditionalPathModel,
    scheduler: DiffusionScheduler,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> float:
    """Compute held-out noise-prediction MSE (cheap — no DDPM sampling)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for costmaps, paths, tokens in val_loader:
        costmaps = costmaps.to(device)
        paths = paths.to(device)
        tokens = tokens.to(device)
        start_pos = paths[:, 0, :]
        goal_pos = paths[:, -1, :]
        B = paths.shape[0]
        t = torch.randint(0, scheduler.timesteps, (B,), device=device)
        noisy_paths, noise = scheduler.forward_process(paths, t)
        with autocast(enabled=use_amp):
            pred_noise = model(
                noisy_paths, t, costmaps,
                start_pos=start_pos, goal_pos=goal_pos,
                text_tokens=tokens,
            )
            loss = F.mse_loss(pred_noise, noise)
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def compute_full_val_metrics(
    model: ConditionalPathModel,
    scheduler: DiffusionScheduler,
    val_pt_files: list,
    config: dict,
    device: torch.device,
    vocab: dict,
    max_samples: int = 50,
) -> dict:
    """Run DDPM sampling on held-out terrain files and compute all metrics.

    Returns a dict with keys:
      isr, mean_cot, mean_risk, inference_latency_s, max_risk,
      isr_per_intent (dict[str, float])
    """
    model.eval()

    d_cfg = config.get("data", {})
    diff_cfg = config.get("diffusion", {})
    cw = config.get("intent", {}).get("cost_weights", {})
    horizon = d_cfg.get("horizon", 120)
    img_size = d_cfg.get("img_size", 100)
    pixel_res = config.get("gradient", {}).get("pixel_resolution", 0.5)
    limit_deg = config.get("gradient", {}).get("limit_angle_deg", 25.0)
    risk_thresh = config.get("intent", {}).get("risk_threshold_deg", 15.0)

    isr_list: list[float] = []
    cot_list: list[float] = []
    risk_list: list[float] = []
    latency_list: list[float] = []
    per_intent: dict[str, list[float]] = {}

    n_collected = 0
    for pt_path in val_pt_files:
        if n_collected >= max_samples:
            break
        try:
            t_data = load_terrain(str(pt_path))
        except Exception:
            continue

        paths_arr = t_data["paths"]          # [N, horizon, 2] numpy
        instructions = t_data.get("instructions", [])
        intent_types = t_data.get("intent_types", [])
        intent_params_list = t_data.get("intent_params", [])
        t_img_size = int(t_data.get("img_size", img_size))
        start = t_data.get("start_position", (0, 0))
        goal = t_data.get("goal_position", (t_img_size - 1, t_img_size - 1))
        slope_map_deg = t_data["slope_map"]
        height_map = t_data["height_map"]
        costmap_t = torch.from_numpy(t_data["costmap"]).float().unsqueeze(0).to(device)

        s_norm = torch.tensor(
            [(start[1] / t_img_size) * 2 - 1, (start[0] / t_img_size) * 2 - 1],
            dtype=torch.float32,
        ).unsqueeze(0).to(device)
        g_norm = torch.tensor(
            [(goal[1] / t_img_size) * 2 - 1, (goal[0] / t_img_size) * 2 - 1],
            dtype=torch.float32,
        ).unsqueeze(0).to(device)

        for i in range(min(paths_arr.shape[0], max_samples - n_collected)):
            instr = instructions[i] if i < len(instructions) else ""
            intent_type = intent_types[i] if i < len(intent_types) else "baseline"
            intent_params = intent_params_list[i] if i < len(intent_params_list) else {}

            tokens = text_to_tokens(instr, vocab, max_seq_len=16).unsqueeze(0).to(device)

            t0 = time.perf_counter()
            gen_path_t = scheduler.sample(
                model, costmap_t, shape=(1, horizon, 2),
                start_pos=s_norm, end_pos=g_norm,
                text_tokens=tokens, show_progress=False,
            )
            elapsed = time.perf_counter() - t0

            gen_path = gen_path_t[0].cpu().numpy()
            goal_norm_np = g_norm[0].cpu().numpy()

            try:
                m = compute_all_metrics(
                    path_norm=gen_path,
                    goal_norm=goal_norm_np,
                    slope_map_deg=slope_map_deg,
                    height_map=height_map,
                    img_size=t_img_size,
                    intent_type=intent_type,
                    intent_params=intent_params,
                    start_pos=start,
                    goal_pos=goal,
                    pixel_resolution=float(t_data.get("pixel_resolution", pixel_res)),
                    limit_angle_deg=float(t_data.get("limit_angle_deg", limit_deg)),
                    risk_threshold_deg=float(t_data.get("risk_threshold_deg", risk_thresh)),
                    alpha=cw.get("alpha", 1.0),
                    beta=cw.get("beta", 0.8),
                    gamma=cw.get("gamma", 0.1),
                    delta=cw.get("delta", 1.0),
                )
            except Exception:
                continue

            isr_list.append(m.get("isr", float("nan")))
            cot_list.append(m.get("cumulative_cot", float("nan")))
            risk_list.append(m.get("risk_integral", float("nan")))
            latency_list.append(elapsed)

            per_intent.setdefault(intent_type, []).append(m.get("isr", float("nan")))
            n_collected += 1

    def _safe_mean(lst: list) -> float:
        clean = [v for v in lst if not (v != v)]  # filter NaN
        return float(np.mean(clean)) if clean else float("nan")

    def _safe_max(lst: list) -> float:
        clean = [v for v in lst if not (v != v)]
        return float(np.max(clean)) if clean else float("nan")

    isr_per_intent_mean = {k: _safe_mean(v) for k, v in per_intent.items()}

    model.train()
    return {
        "isr": _safe_mean(isr_list),
        "mean_cot": _safe_mean(cot_list),
        "mean_risk": _safe_mean(risk_list),
        "inference_latency_s": _safe_mean(latency_list),
        "max_risk": _safe_max(risk_list),
        "isr_per_intent": isr_per_intent_mean,
    }


# ── Main training function ────────────────────────────────────────────────────

def train(
    config: dict,
    data_dir: str,
    val_dir: str,
    device_str: str = "cuda",
    resume_from: Optional[str] = None,
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    d_cfg = config.get("data", {})
    g_cfg = config.get("gradient", {})
    diff_cfg = config.get("diffusion", {})
    m_cfg = config.get("model", {})
    t_cfg = config.get("training", {})
    l_cfg = config.get("logging", {})

    horizon = d_cfg.get("horizon", 120)
    max_seq_len = 16

    # ── Logging config ────────────────────────────────────────────────────────
    val_loss_interval = int(l_cfg.get("val_loss_interval", 200))
    val_interval = int(l_cfg.get("val_interval", 2000))
    val_max_samples = int(l_cfg.get("val_max_samples", 50))
    log_dir = str(_ROOT / l_cfg.get("log_dir", "logs/ablation"))
    backbone_name = m_cfg.get("visual_backbone", "unknown")

    # ── Datasets (train / val fully separated) ────────────────────────────────
    vocab = build_vocab()
    train_dataset = TextGuideDataset(data_dir, max_seq_len=max_seq_len, vocab=vocab)
    val_dataset   = TextGuideDataset(val_dir,  max_seq_len=max_seq_len, vocab=vocab)

    # val .pt files are used for full metric computation (ISR/CoT/Risk/Latency)
    val_pt_files = sorted(Path(val_dir).glob("*.pt"))
    if not val_pt_files:
        raise FileNotFoundError(f"No .pt files found in val_dir: {val_dir}")

    print(f"Train: {len(train_dataset)} samples from {data_dir}")
    print(f"Val  : {len(val_dataset)} samples from {val_dir} "
          f"({len(val_pt_files)} terrain files)")

    batch_size = t_cfg.get("batch_size", 64)
    loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ConditionalPathModel(
        transition_dim=2,
        dim=m_cfg.get("base_dim", 64),
        horizon=horizon,
        visual_dim=m_cfg.get("image_feat_dim", 256),
        text_dim=256,
        vocab_size=train_dataset.vocab_size,
        max_seq_len=max_seq_len,
        visual_backbone=m_cfg.get("visual_backbone", "convnext"),
        visual_pretrained=m_cfg.get("timm_pretrained", m_cfg.get("convnext_pretrained", True)),
        timm_model_name=m_cfg.get("timm_model_name"),
        timm_pretrained=m_cfg.get("timm_pretrained"),
        input_img_size=d_cfg.get("img_size"),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Diffusion ─────────────────────────────────────────────────────────────
    scheduler = DiffusionScheduler(
        timesteps=diff_cfg.get("timesteps", 200),
        beta_start=diff_cfg.get("beta_start", 0.0001),
        beta_end=diff_cfg.get("beta_end", 0.02),
        device=device,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    lr = t_cfg.get("learning_rate", 1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = t_cfg.get("epochs", 5000)
    log_interval = t_cfg.get("log_interval", 1000)
    use_amp = t_cfg.get("use_amp", True) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    ckpt_dir = str(_ROOT / t_cfg.get("checkpoint_dir", "checkpoints"))
    vis_dir = str(_ROOT / "results" / "train_vis")
    os.makedirs(ckpt_dir, exist_ok=True)

    start_epoch = 1
    global_step = 0
    if resume_from:
        ckpt_path = os.path.abspath(resume_from)
        if not os.path.isfile(ckpt_path):
            ckpt_path = str(_ROOT / resume_from)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_from}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if use_amp and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        done = int(ckpt.get("epoch", 0))
        start_epoch = done + 1
        global_step = max(0, (start_epoch - 1) * len(loader))
        sv = ckpt.get("vocab")
        if sv is not None and sv != vocab:
            print("Warning: checkpoint vocab differs from build_vocab(); using checkpoint is OK if keys match.")
        print(f"Resumed from {ckpt_path} (completed epoch {done}, next epoch {start_epoch})")

    # ── Ablation Logger ───────────────────────────────────────────────────────
    logger = AblationLogger(
        backbone=backbone_name,
        log_dir=log_dir,
        param_count=n_params,
    )

    # ── WandB (optional) ──────────────────────────────────────────────────────
    use_wandb = False
    try:
        import wandb
        run = wandb.init(
            project="diffusion-textguide",
            name=f"{backbone_name}_{time.strftime('%m%d_%H%M')}",
            config=config,
        )
        use_wandb = run is not None and hasattr(wandb, "log")
        if use_wandb:
            # Static summary: param_count logged once, visible in run overview
            wandb.summary["param_count"] = n_params
        elif run is not None:
            print("WandB run started but .log unavailable, logging to stdout only")
        else:
            print("WandB not available, logging to stdout only")
    except Exception as e:
        use_wandb = False
        print("WandB not available, logging to stdout only:", e)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining: {epochs} epochs, batch_size={batch_size}, lr={lr}")
    print(f"AMP (fp16): {'on' if use_amp else 'off'}")
    print(f"Checkpoint every {log_interval} epochs → {ckpt_dir}")
    print(f"Val loss every {val_loss_interval} epochs | "
          f"Full metrics every {val_interval} epochs ({val_max_samples} samples)")
    print(f"Dataset: {len(train_dataset)} train / {len(val_dataset)} val  "
          f"(vocab_size={train_dataset.vocab_size})")
    print("=" * 62)

    if start_epoch > epochs:
        print(f"Nothing to do: start_epoch {start_epoch} > training.epochs {epochs}")
        logger.close()
        return

    train_loop_start = time.time()
    _last_val_loss: Optional[float] = None

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        val_time_this_epoch = 0.0

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for costmaps, paths, tokens in loader:
            costmaps = costmaps.to(device)
            paths = paths.to(device)
            tokens = tokens.to(device)

            start_pos = paths[:, 0, :]
            goal_pos = paths[:, -1, :]

            B = paths.shape[0]
            t = torch.randint(0, scheduler.timesteps, (B,), device=device)
            noisy_paths, noise = scheduler.forward_process(paths, t)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                pred_noise = model(
                    noisy_paths, t, costmaps,
                    start_pos=start_pos, goal_pos=goal_pos,
                    text_tokens=tokens,
                )
                loss = F.mse_loss(pred_noise, noise)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Secondary: train_loss + lr (every epoch) ──────────────────────────
        logger.accumulate(train_loss=avg_loss, lr=current_lr)

        if use_wandb:
            try:
                wandb.log({
                    "train/loss": avg_loss,
                    "train/lr": current_lr,
                    "epoch": epoch,
                }, step=global_step)
            except Exception:
                use_wandb = False
                print("WandB logging disabled after error; continuing on stdout only.")

        # ── Primary: val_loss (cheap; every val_loss_interval) ────────────────
        epoch_val_loss: Optional[float] = None
        if epoch % val_loss_interval == 0:
            _val_t = time.time()
            epoch_val_loss = compute_val_loss(model, scheduler, val_loader, device, use_amp)
            val_time_this_epoch += time.time() - _val_t
            _last_val_loss = epoch_val_loss
            logger.accumulate(val_loss=epoch_val_loss)
            if use_wandb:
                try:
                    wandb.log({
                        "val/val_loss": epoch_val_loss,
                        "epoch": epoch,
                    }, step=global_step)
                except Exception:
                    use_wandb = False

        # ── Primary: full metrics (expensive; every val_interval) ─────────────
        if epoch % val_interval == 0:
            _full_val_t = time.time()
            # val_loss is guaranteed here — recompute if val_interval is not a
            # multiple of val_loss_interval (edge case in custom configs).
            if epoch_val_loss is None:
                epoch_val_loss = compute_val_loss(model, scheduler, val_loader, device, use_amp)
                logger.accumulate(val_loss=epoch_val_loss)

            snap_path = visualize_sample(model, scheduler, train_dataset, epoch, device, vis_dir)
            logger.accumulate(snapshot_path=snap_path)

            fm = compute_full_val_metrics(
                model, scheduler, val_pt_files, config,
                device, vocab, max_samples=val_max_samples,
            )
            val_time_this_epoch += time.time() - _full_val_t
            logger.accumulate(
                isr=fm["isr"],
                mean_cot=fm["mean_cot"],
                mean_risk=fm["mean_risk"],
                inference_latency_s=fm["inference_latency_s"],
                max_risk=fm["max_risk"],
                isr_per_intent=fm["isr_per_intent"],
            )
            logger.print_summary(epoch, total_epochs=epochs)

            if use_wandb:
                try:
                    # Build per-intent ISR sub-dict for WandB
                    intent_isr_dict = {
                        f"val/isr_{k.replace('+', '_')}": v
                        for k, v in fm["isr_per_intent"].items()
                    }
                    # Load snapshot as WandB image
                    wb_img = {"val/snapshot": wandb.Image(snap_path, caption=f"epoch {epoch}")}

                    wandb.log({
                        # Primary
                        "val/val_loss": epoch_val_loss,
                        "val/isr": fm["isr"],
                        "val/mean_cot": fm["mean_cot"],
                        "val/mean_risk": fm["mean_risk"],
                        "val/inference_latency_s": fm["inference_latency_s"],
                        "val/param_count": n_params,
                        # Secondary
                        "val/max_risk": fm["max_risk"],
                        "val/best_val_epoch": logger.best_val_epoch,
                        **intent_isr_dict,
                        **wb_img,
                        "epoch": epoch,
                    }, step=global_step)
                except Exception:
                    use_wandb = False

        # ── Timing ────────────────────────────────────────────────────────────
        epoch_time_s = time.time() - epoch_start
        logger.accumulate(
            epoch_time_s=epoch_time_s,
            val_time_s=val_time_this_epoch if val_time_this_epoch > 0 else None,
        )
        if use_wandb:
            try:
                _wb_timing: dict = {"train/epoch_time_s": epoch_time_s, "epoch": epoch}
                if val_time_this_epoch > 0:
                    _wb_timing["val/val_time_s"] = val_time_this_epoch
                wandb.log(_wb_timing, step=global_step)
            except Exception:
                use_wandb = False

        # ── Flush logger row for this epoch ───────────────────────────────────
        logger.flush(epoch)

        if epoch % 50 == 0 or epoch == 1:
            _elapsed = time.time() - train_loop_start
            _done = epoch - start_epoch + 1
            _remaining = epochs - epoch
            _avg_ep = _elapsed / _done
            _eta_s = _avg_ep * _remaining
            def _fmt_dur(s: float) -> str:
                h, s = divmod(int(s), 3600)
                m, s = divmod(s, 60)
                return (f"{h}h {m:02d}m {s:02d}s" if h else
                        f"{m}m {s:02d}s" if m else f"{s}s")
            _pct = epoch / epochs
            _bar_w = 20
            _filled = int(_bar_w * _pct)
            _bar = "█" * _filled + "░" * (_bar_w - _filled)
            _val_str = f"  val={_last_val_loss:.6f}" if _last_val_loss is not None else ""
            print(
                f"[{_bar}] {_pct*100:5.1f}%  "
                f"ep {epoch:>6}/{epochs}  "
                f"loss={avg_loss:.6f}{_val_str}  "
                f"{epoch_time_s:.1f}s/ep  "
                f"ETA {_fmt_dur(_eta_s)}  "
                f"elapsed {_fmt_dur(_elapsed)}"
            )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if epoch % log_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:05d}.pt")
            save_checkpoint(
                model, optimizer, epoch, vocab, config, ckpt_path,
                scaler=scaler if use_amp else None,
            )
            print(f"  → Saved checkpoint: {ckpt_path}")

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = os.path.join(ckpt_dir, "final_model.pt")
    save_checkpoint(
        model, optimizer, epochs, vocab, config, final_path,
        scaler=scaler if use_amp else None,
    )
    print(f"\nTraining complete. Final model: {final_path}")
    print(f"Best val epoch: {logger.best_val_epoch}")

    logger.close()

    if use_wandb:
        try:
            wandb.summary["best_val_epoch"] = logger.best_val_epoch
            wandb.summary["param_count"] = n_params
            wandb.finish()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="Train text-conditioned diffusion path planner")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--data-dir", type=str, default=None,
                    help="Training .pt files directory (default: data/raw)")
    ap.add_argument("--val-dir", type=str, default=None,
                    help="Validation .pt files directory (default: data/valid)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to epoch_*.pt to resume (loads model/optimizer/scaler, continues from next epoch)",
    )
    args = ap.parse_args()

    cfg_path = args.config
    if cfg_path is None:
        cfg_path = str(_ROOT / "configs" / "default_config.yaml")
    config = load_config(cfg_path)
    print(f"Config: {cfg_path}")

    if args.epochs:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size

    data_dir = args.data_dir or str(_ROOT / "data" / "raw")
    val_dir  = args.val_dir  or str(_ROOT / "data" / "valid")

    train(config, data_dir, val_dir, device_str=args.device, resume_from=args.resume)


if __name__ == "__main__":
    main()
