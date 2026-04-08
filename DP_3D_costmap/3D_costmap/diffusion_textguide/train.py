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
from data_loader import TextGuideDataset, build_vocab


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

    # Slope map (deg)
    ax0 = axes[0]
    im0 = ax0.imshow(slope_norm * 90.0, cmap="jet", origin="lower", vmin=0, vmax=35)
    ax0.plot(gt_px[:, 0], gt_px[:, 1], "g-", lw=2, alpha=0.7, label="GT")
    ax0.plot(gen_px[:, 0], gen_px[:, 1], "r-", lw=2, alpha=0.7, label="Generated")
    ax0.scatter([gt_px[0, 0]], [gt_px[0, 1]], c="lime", s=50, zorder=10, marker="o")
    ax0.scatter([gt_px[-1, 0]], [gt_px[-1, 1]], c="lime", s=50, zorder=10, marker="*")
    ax0.set_title("Slope map (deg)", fontsize=9)
    ax0.axis("off")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02)

    # Height map (normalized)
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
    fig.savefig(os.path.join(out_dir, f"sample_epoch_{epoch:05d}.png"),
                dpi=100, bbox_inches="tight")
    plt.close(fig)
    model.train()


def train(
    config: dict,
    data_dir: str,
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

    horizon = d_cfg.get("horizon", 120)
    max_seq_len = 16

    # --- Dataset ---
    vocab = build_vocab()
    dataset = TextGuideDataset(data_dir, max_seq_len=max_seq_len, vocab=vocab)

    batch_size = t_cfg.get("batch_size", 64)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    # --- Model ---
    model = ConditionalPathModel(
        transition_dim=2,
        dim=m_cfg.get("base_dim", 64),
        horizon=horizon,
        visual_dim=m_cfg.get("image_feat_dim", 256),
        text_dim=256,
        vocab_size=dataset.vocab_size,
        max_seq_len=max_seq_len,
        visual_backbone=m_cfg.get("visual_backbone", "convnext"),
        visual_pretrained=m_cfg.get("timm_pretrained", m_cfg.get("convnext_pretrained", True)),
        timm_model_name=m_cfg.get("timm_model_name"),
        timm_pretrained=m_cfg.get("timm_pretrained"),
        input_img_size=d_cfg.get("img_size"),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # --- Diffusion ---
    scheduler = DiffusionScheduler(
        timesteps=diff_cfg.get("timesteps", 200),
        beta_start=diff_cfg.get("beta_start", 0.0001),
        beta_end=diff_cfg.get("beta_end", 0.02),
        device=device,
    )

    # --- Optimizer ---
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

    # --- WandB (optional) ---
    use_wandb = False
    try:
        import wandb
        run = wandb.init(project="diffusion-textguide", config=config)
        use_wandb = run is not None and hasattr(wandb, "log")
        if not use_wandb and run is not None:
            print("WandB run started but .log unavailable, logging to stdout only")
        elif run is None:
            print("WandB not available, logging to stdout only")
    except Exception as e:
        use_wandb = False
        print("WandB not available, logging to stdout only:", e)

    # --- Training loop ---
    print(f"\nTraining: {epochs} epochs, batch_size={batch_size}, lr={lr}")
    print(f"AMP (fp16): {'on' if use_amp else 'off'}")
    print(f"Checkpoint every {log_interval} epochs → {ckpt_dir}")
    print(f"Dataset: {len(dataset)} samples, vocab_size={dataset.vocab_size}")
    print("=" * 60)

    if start_epoch > epochs:
        print(f"Nothing to do: start_epoch {start_epoch} > training.epochs {epochs}")
        return

    for epoch in range(start_epoch, epochs + 1):
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

        if use_wandb:
            try:
                wandb.log({"train_loss": avg_loss, "epoch": epoch}, step=global_step)
            except Exception:
                use_wandb = False
                print("WandB logging disabled after error; continuing training on stdout only.")

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:5d}/{epochs}  loss={avg_loss:.6f}")

        if epoch % log_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:05d}.pt")
            save_checkpoint(
                model, optimizer, epoch, vocab, config, ckpt_path,
                scaler=scaler if use_amp else None,
            )
            print(f"  → Saved checkpoint: {ckpt_path}")
            visualize_sample(model, scheduler, dataset, epoch, device, vis_dir)

    # Final save
    final_path = os.path.join(ckpt_dir, "final_model.pt")
    save_checkpoint(
        model, optimizer, epochs, vocab, config, final_path,
        scaler=scaler if use_amp else None,
    )
    print(f"\nTraining complete. Final model: {final_path}")

    if use_wandb:
        try:
            wandb.finish()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="Train text-conditioned diffusion path planner")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--data-dir", type=str, default=None,
                    help="Directory with .pt files (default: data/raw)")
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

    train(config, data_dir, device_str=args.device, resume_from=args.resume)


if __name__ == "__main__":
    main()
