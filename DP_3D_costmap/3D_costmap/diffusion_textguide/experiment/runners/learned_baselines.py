"""Train and evaluate BC/CVAE learned planner baselines.

This runner uses the ResNet18+T5 conditioning stack from an existing diffusion
checkpoint as a frozen feature extractor, then trains lightweight deterministic
(BC) and latent-variable (CVAE) path decoders on pseudo-label trajectories.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

_EXP_DIR = Path(__file__).resolve().parent
_ROOT = _EXP_DIR.parents[1]
sys.path.insert(0, str(_ROOT))

from experiment.core.metrics import chamfer_distance, compute_all_metrics
from experiment.core.utils import load_terrain
from experiment.evaluators.text_encoder_ablation import _load_model
from experiment.support.text_encoder_ablation import FrozenTextFeatureEncoder


KST = timezone(timedelta(hours=9))


@dataclass
class CacheSpec:
    cond_dim: int
    horizon: int
    visual_dim: int
    text_dim: int


class CachedPathDataset(Dataset):
    def __init__(self, cache: Dict[str, Any]) -> None:
        self.conditions = cache["conditions"].float()
        self.paths = cache["paths"].float()

    def __len__(self) -> int:
        return int(self.paths.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.conditions[idx], self.paths[idx]


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PathMLPDecoder(nn.Module):
    def __init__(self, input_dim: int, horizon: int = 120, hidden_dim: int = 768,
                 depth: int = 5, dropout: float = 0.05) -> None:
        super().__init__()
        self.horizon = horizon
        self.in_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResidualMLPBlock(hidden_dim, dropout) for _ in range(depth)])
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, horizon * 2),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.in_proj(x))
        return self.out(h).view(x.shape[0], self.horizon, 2)


class BCPlanner(nn.Module):
    def __init__(self, cond_dim: int, horizon: int = 120, hidden_dim: int = 768,
                 depth: int = 5, dropout: float = 0.05) -> None:
        super().__init__()
        self.decoder = PathMLPDecoder(cond_dim, horizon, hidden_dim, depth, dropout)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return apply_endpoints(self.decoder(cond), cond)


class CVAEPlanner(nn.Module):
    def __init__(self, cond_dim: int, horizon: int = 120, hidden_dim: int = 768,
                 depth: int = 5, latent_dim: int = 64, dropout: float = 0.05) -> None:
        super().__init__()
        self.horizon = horizon
        self.latent_dim = latent_dim
        enc_in = cond_dim + horizon * 2
        self.encoder = nn.Sequential(
            nn.LayerNorm(enc_in),
            nn.Linear(enc_in, hidden_dim),
            nn.GELU(),
            ResidualMLPBlock(hidden_dim, dropout),
            ResidualMLPBlock(hidden_dim, dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = PathMLPDecoder(cond_dim + latent_dim, horizon, hidden_dim, depth, dropout)

    def encode(self, cond: torch.Tensor, path: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([cond, path.flatten(1)], dim=-1))
        return self.mu(h), self.logvar(h).clamp(min=-10.0, max=8.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, cond: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return apply_endpoints(self.decoder(torch.cat([cond, z], dim=-1)), cond)

    def forward(self, cond: torch.Tensor, path: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(cond, path)
        z = self.reparameterize(mu, logvar)
        return self.decode(cond, z), mu, logvar

    def sample(self, cond: torch.Tensor) -> torch.Tensor:
        z = torch.randn(cond.shape[0], self.latent_dim, device=cond.device)
        return self.decode(cond, z)


def apply_endpoints(path: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    out = path.clone()
    out[:, 0, :] = cond[:, -4:-2]
    out[:, -1, :] = cond[:, -2:]
    return out


def path_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    recon = F.smooth_l1_loss(pred, target)
    vel = F.smooth_l1_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])
    return recon + 0.1 * vel


def kl_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1))


def resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (_ROOT / p).resolve()


def now_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")


def norm_start_goal(terrain: dict) -> tuple[np.ndarray, np.ndarray]:
    img_size = int(terrain["img_size"])
    start = terrain["start_position"]
    goal = terrain["goal_position"]
    s = np.array([(start[1] / img_size) * 2 - 1, (start[0] / img_size) * 2 - 1], dtype=np.float32)
    g = np.array([(goal[1] / img_size) * 2 - 1, (goal[0] / img_size) * 2 - 1], dtype=np.float32)
    return s, g


def gather_unique_instructions(data_dirs: Iterable[Path]) -> list[str]:
    seen: dict[str, None] = {}
    for data_dir in data_dirs:
        for pt in sorted(data_dir.glob("*.pt")):
            terrain = load_terrain(str(pt))
            for instr in terrain.get("instructions", []):
                seen.setdefault(str(instr), None)
    return list(seen.keys())


@torch.inference_mode()
def project_text_features(
    instructions: list[str],
    text_encoder_type: str,
    text_cfg: dict,
    projection: Optional[nn.Module],
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, torch.Tensor]:
    feature_encoder = FrozenTextFeatureEncoder(
        text_encoder_type,
        model_name=text_cfg.get("model_name"),
        device=device,
        batch_size=batch_size,
    )
    out: dict[str, torch.Tensor] = {}
    for i in tqdm(range(0, len(instructions), batch_size), desc="Encoding text"):
        batch = instructions[i:i + batch_size]
        raw = feature_encoder.encode(batch).to(device)
        if projection is not None:
            feat = projection(raw).detach().cpu().float()
        else:
            feat = raw.detach().cpu().float()
        for instr, vec in zip(batch, feat):
            out[instr] = vec
    return out


@torch.inference_mode()
def build_feature_cache(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_path = resolve(args.condition_checkpoint)
    model, _, _, config, text_encoder_type, _ = _load_model(ckpt_path, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    text_cfg = config.get("model", {}).get("text_encoder", {})
    train_dir = resolve(args.train_dir)
    valid_dir = resolve(args.valid_dir)
    cache_dir = resolve(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{now_kst()}] Building frozen condition caches")
    print(f"  checkpoint={ckpt_path}")
    print(f"  train={train_dir}")
    print(f"  valid={valid_dir}")

    unique = gather_unique_instructions([train_dir, valid_dir])
    print(f"  unique instructions={len(unique)}")
    text_by_instr = project_text_features(
        unique, text_encoder_type, text_cfg, model.text_projection, device,
        batch_size=args.text_batch_size,
    )

    for split, data_dir, max_files in (
        ("train", train_dir, args.max_train_files),
        ("valid", valid_dir, args.max_valid_files),
    ):
        out_path = cache_dir / f"{split}_resnet18_t5_features.pt"
        if out_path.exists() and not args.rebuild_cache:
            print(f"  cache exists, skipping: {out_path}")
            continue

        files = sorted(data_dir.glob("*.pt"))
        if max_files is not None:
            files = files[:max_files]
        cond_chunks: list[torch.Tensor] = []
        path_chunks: list[torch.Tensor] = []
        metadata: list[dict[str, Any]] = []

        for start_idx in tqdm(range(0, len(files), args.visual_batch_size), desc=f"Visual cache {split}"):
            batch_files = files[start_idx:start_idx + args.visual_batch_size]
            terrains = [load_terrain(str(p)) for p in batch_files]
            costmaps = torch.stack([
                torch.from_numpy(t["costmap"]).float() for t in terrains
            ]).to(device)
            visual = model.visual_encoder(costmaps).detach().cpu().float()

            for j, (pt_path, terrain) in enumerate(zip(batch_files, terrains)):
                s_norm, g_norm = norm_start_goal(terrain)
                start_goal = torch.from_numpy(np.concatenate([s_norm, g_norm])).float()
                paths = torch.from_numpy(terrain["paths"]).float()
                instructions = terrain.get("instructions", [""] * paths.shape[0])
                intent_types = terrain.get("intent_types", ["baseline"] * paths.shape[0])
                for i in range(paths.shape[0]):
                    instr = str(instructions[i] if i < len(instructions) else "")
                    text_feat = text_by_instr[instr]
                    cond = torch.cat([visual[j], text_feat, start_goal], dim=0)
                    cond_chunks.append(cond)
                    path_chunks.append(paths[i])
                    metadata.append({
                        "terrain_file": pt_path.name,
                        "terrain_path": str(pt_path),
                        "map_id": str(terrain.get("map_id", pt_path.stem)),
                        "path_idx": int(i),
                        "intent_type": str(intent_types[i] if i < len(intent_types) else "baseline"),
                        "instruction": instr,
                    })

        conditions = torch.stack(cond_chunks).float()
        paths = torch.stack(path_chunks).float()
        payload = {
            "conditions": conditions,
            "paths": paths,
            "metadata": metadata,
            "spec": {
                "cond_dim": int(conditions.shape[-1]),
                "horizon": int(paths.shape[1]),
                "visual_dim": int(visual.shape[-1]) if len(files) else 0,
                "text_encoder_type": text_encoder_type,
                "checkpoint": str(ckpt_path),
                "split": split,
                "data_dir": str(data_dir),
                "built_at_kst": now_kst(),
            },
        }
        torch.save(payload, out_path)
        print(f"  saved {split}: {out_path}  n={len(paths)} cond_dim={conditions.shape[-1]}")


def load_cache(cache_dir: Path, split: str) -> Dict[str, Any]:
    p = cache_dir / f"{split}_resnet18_t5_features.pt"
    if not p.exists():
        raise FileNotFoundError(f"Missing feature cache: {p}. Run --mode build-cache first.")
    return torch.load(p, map_location="cpu", weights_only=False)


def run_validation(model: nn.Module, loader: DataLoader, device: torch.device, kind: str) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for cond, target in loader:
            cond = cond.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if kind == "bc":
                pred = model(cond)
            else:
                pred = model.sample(cond)
            losses.append(float(path_loss(pred, target).detach().cpu()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def train_bc(args: argparse.Namespace, train_cache: dict, valid_cache: dict, device: torch.device,
             run_dir: Path) -> dict[str, Any]:
    train_ds = CachedPathDataset(train_cache)
    valid_ds = CachedPathDataset(valid_cache)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"), drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))
    cond_dim = int(train_cache["conditions"].shape[-1])
    horizon = int(train_cache["paths"].shape[1])
    model = BCPlanner(cond_dim, horizon, args.hidden_dim, args.decoder_depth, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.bc_lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    best = {"val_loss": float("inf"), "epoch": 0, "path": str(run_dir / "bc_planner_best.pt")}
    history = []
    t_start = time.perf_counter()

    print(f"[{now_kst()}] Train BC: n={len(train_ds)}, epochs={args.bc_epochs}")
    for epoch in range(1, args.bc_epochs + 1):
        model.train()
        epoch_losses = []
        for cond, target in train_loader:
            cond = cond.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                pred = model(cond)
                loss = path_loss(pred, target)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            epoch_losses.append(float(loss.detach().cpu()))

        do_eval = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.bc_epochs
        val_loss = run_validation(model, valid_loader, device, "bc") if do_eval else float("nan")
        train_loss = float(np.mean(epoch_losses))
        if do_eval:
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            print(f"  BC epoch {epoch:04d}/{args.bc_epochs} train={train_loss:.6f} val={val_loss:.6f}")
            if val_loss < best["val_loss"]:
                best.update({"val_loss": float(val_loss), "epoch": epoch})
                torch.save({
                    "model_type": "bc", "model_state_dict": model.state_dict(),
                    "cond_dim": cond_dim, "horizon": horizon,
                    "args": vars(args), "best": best,
                    "condition_checkpoint": str(resolve(args.condition_checkpoint)),
                    "trained_at_kst": now_kst(),
                }, best["path"])

    last_path = run_dir / "bc_planner_final.pt"
    torch.save({
        "model_type": "bc", "model_state_dict": model.state_dict(),
        "cond_dim": cond_dim, "horizon": horizon,
        "args": vars(args), "best": best,
        "condition_checkpoint": str(resolve(args.condition_checkpoint)),
        "trained_at_kst": now_kst(),
    }, last_path)
    return {"best": best, "final_path": str(last_path), "history": history,
            "elapsed_seconds": time.perf_counter() - t_start}


def train_cvae(args: argparse.Namespace, train_cache: dict, valid_cache: dict, device: torch.device,
               run_dir: Path) -> dict[str, Any]:
    train_ds = CachedPathDataset(train_cache)
    valid_ds = CachedPathDataset(valid_cache)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"), drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))
    cond_dim = int(train_cache["conditions"].shape[-1])
    horizon = int(train_cache["paths"].shape[1])
    model = CVAEPlanner(cond_dim, horizon, args.hidden_dim, args.decoder_depth,
                        args.latent_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.cvae_lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    best = {"val_loss": float("inf"), "epoch": 0, "path": str(run_dir / "cvae_planner_best.pt")}
    history = []
    t_start = time.perf_counter()

    print(f"[{now_kst()}] Train CVAE: n={len(train_ds)}, epochs={args.cvae_epochs}")
    for epoch in range(1, args.cvae_epochs + 1):
        model.train()
        epoch_losses = []
        epoch_recon = []
        epoch_kl = []
        kl_beta = args.kl_beta * min(1.0, epoch / max(1, int(args.cvae_epochs * args.kl_anneal_frac)))
        for cond, target in train_loader:
            cond = cond.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                pred, mu, logvar = model(cond, target)
                recon = path_loss(pred, target)
                kl = kl_normal(mu, logvar)
                loss = recon + kl_beta * kl
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_recon.append(float(recon.detach().cpu()))
            epoch_kl.append(float(kl.detach().cpu()))

        do_eval = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.cvae_epochs
        val_loss = run_validation(model, valid_loader, device, "cvae") if do_eval else float("nan")
        train_loss = float(np.mean(epoch_losses))
        if do_eval:
            history.append({
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "recon": float(np.mean(epoch_recon)), "kl": float(np.mean(epoch_kl)),
                "kl_beta": float(kl_beta),
            })
            print(f"  CVAE epoch {epoch:04d}/{args.cvae_epochs} train={train_loss:.6f} val={val_loss:.6f} recon={np.mean(epoch_recon):.6f} kl={np.mean(epoch_kl):.4f} beta={kl_beta:.5g}")
            if val_loss < best["val_loss"]:
                best.update({"val_loss": float(val_loss), "epoch": epoch})
                torch.save({
                    "model_type": "cvae", "model_state_dict": model.state_dict(),
                    "cond_dim": cond_dim, "horizon": horizon, "latent_dim": args.latent_dim,
                    "args": vars(args), "best": best,
                    "condition_checkpoint": str(resolve(args.condition_checkpoint)),
                    "trained_at_kst": now_kst(),
                }, best["path"])

    last_path = run_dir / "cvae_planner_final.pt"
    torch.save({
        "model_type": "cvae", "model_state_dict": model.state_dict(),
        "cond_dim": cond_dim, "horizon": horizon, "latent_dim": args.latent_dim,
        "args": vars(args), "best": best,
        "condition_checkpoint": str(resolve(args.condition_checkpoint)),
        "trained_at_kst": now_kst(),
    }, last_path)
    return {"best": best, "final_path": str(last_path), "history": history,
            "elapsed_seconds": time.perf_counter() - t_start}


def train(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache_dir = resolve(args.cache_dir)
    run_dir = resolve(args.output_dir) / datetime.now(KST).strftime("learned_baselines_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    train_cache = load_cache(cache_dir, "train")
    valid_cache = load_cache(cache_dir, "valid")
    results = {"run_dir": str(run_dir), "started_kst": now_kst()}
    if args.train_bc:
        results["bc"] = train_bc(args, train_cache, valid_cache, device, run_dir)
    if args.train_cvae:
        results["cvae"] = train_cvae(args, train_cache, valid_cache, device, run_dir)
    results["ended_kst"] = now_kst()
    with (run_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"[{now_kst()}] Training summary: {run_dir / 'training_summary.json'}")
    return results


def load_bc_checkpoint(path: Path, device: torch.device) -> BCPlanner:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = BCPlanner(int(ckpt["cond_dim"]), int(ckpt["horizon"]),
                      int(ckpt.get("args", {}).get("hidden_dim", 768)),
                      int(ckpt.get("args", {}).get("decoder_depth", 5)),
                      float(ckpt.get("args", {}).get("dropout", 0.05))).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_cvae_checkpoint(path: Path, device: torch.device) -> CVAEPlanner:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = CVAEPlanner(int(ckpt["cond_dim"]), int(ckpt["horizon"]),
                        int(ckpt.get("args", {}).get("hidden_dim", 768)),
                        int(ckpt.get("args", {}).get("decoder_depth", 5)),
                        int(ckpt.get("latent_dim", ckpt.get("args", {}).get("latent_dim", 64))),
                        float(ckpt.get("args", {}).get("dropout", 0.05))).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def selected_eval_indices(valid_cache: dict, valid_dir: Path, seed: int, num_terrains: int) -> list[int]:
    all_files = sorted(valid_dir.glob("*.pt"))
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(all_files), size=num_terrains, replace=False).tolist())
    chosen_names = {all_files[i].name for i in chosen}
    return [i for i, m in enumerate(valid_cache["metadata"]) if m["terrain_file"] in chosen_names]


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    def vals(key: str) -> list[float]:
        return [float(r[key]) for r in records if key in r and isinstance(r[key], (int, float, np.floating)) and np.isfinite(r[key])]
    def mean(key: str) -> float:
        xs = vals(key)
        return float(np.mean(xs)) if xs else float("nan")
    def std(key: str) -> float:
        xs = vals(key)
        return float(np.std(xs)) if xs else float("nan")
    keys = sorted({k for r in records for k, v in r.items() if isinstance(v, (int, float, np.floating))})
    per_intent = {}
    for intent in sorted({r["intent_type"] for r in records}):
        subset = [r for r in records if r["intent_type"] == intent]
        per_intent[intent] = {
            "count": len(subset),
            "success": float(np.mean([r["success"] for r in subset])),
            "feasible": float(np.mean([r["feasible"] for r in subset])),
            "isr": float(np.mean([r["isr"] for r in subset])),
            "cot": float(np.mean([r["cumulative_cot"] for r in subset])),
            "risk": float(np.mean([r["risk_integral"] for r in subset])),
        }
    return {
        "count": len(records),
        "by_metric": {k: {"mean": mean(k), "std": std(k), "count": len(vals(k))} for k in keys},
        "intentwise": {
            "left_bias_score": mean("isr_component_left_bias"),
            "right_bias_score": mean("isr_component_right_bias"),
            "avoid_steep_isr": mean("isr_component_avoid_steep"),
            "prefer_flat_isr": mean("isr_component_prefer_flat"),
            "composite_isr": float(np.mean([r["isr"] for r in records if "+" in r["intent_type"]])),
        },
        "per_intent": per_intent,
    }


def path_feasible(path_norm: np.ndarray, slope_map_deg: np.ndarray, img_size: int, limit_angle_deg: float) -> float:
    px = (path_norm + 1.0) / 2.0 * img_size
    px = np.clip(px, 0, img_size - 1)
    rc = np.stack([px[:, 1], px[:, 0]], axis=1).astype(int)
    slopes = slope_map_deg[rc[:, 0], rc[:, 1]]
    return float(np.all(slopes < limit_angle_deg))


def evaluate_prediction(path: np.ndarray, meta: dict[str, Any], terrain_cache: dict[str, dict]) -> dict[str, Any]:
    terrain = terrain_cache.get(meta["terrain_path"])
    if terrain is None:
        terrain = load_terrain(meta["terrain_path"])
        terrain_cache[meta["terrain_path"]] = terrain
    i = int(meta["path_idx"])
    img_size = int(terrain["img_size"])
    start = terrain["start_position"]
    goal = terrain["goal_position"]
    g_norm = np.array([(goal[1] / img_size) * 2 - 1, (goal[0] / img_size) * 2 - 1], dtype=np.float32)
    cw = terrain.get("cost_weights", {})
    intent_params = terrain.get("intent_params", [{}])[i]
    m = compute_all_metrics(
        path_norm=path,
        goal_norm=g_norm,
        slope_map_deg=terrain["slope_map"],
        height_map=terrain["height_map"],
        img_size=img_size,
        intent_type=meta["intent_type"],
        intent_params=intent_params,
        start_pos=start,
        goal_pos=goal,
        ref_path_norm=terrain["paths"][i],
        pixel_resolution=float(terrain.get("pixel_resolution", 0.5)),
        limit_angle_deg=float(terrain.get("limit_angle_deg", 25)),
        risk_threshold_deg=float(terrain.get("risk_threshold_deg", 15.0)),
        alpha=float(cw.get("alpha", 0.5)),
        beta=float(cw.get("beta", 1.75)),
        gamma=float(cw.get("gamma", 1.5)),
        delta=float(cw.get("delta", 1.75)),
    )
    rec = {
        "terrain_file": meta["terrain_file"],
        "map_id": meta["map_id"],
        "path_idx": i,
        "intent_type": meta["intent_type"],
        "instruction": meta.get("instruction", ""),
        "success": float(m.get("goal_error_m", float("inf")) <= float(terrain.get("pixel_resolution", 0.5))),
        "feasible": path_feasible(path, terrain["slope_map"], img_size, float(terrain.get("limit_angle_deg", 25))),
    }
    rec.update({k: float(v) for k, v in m.items() if isinstance(v, (int, float, np.floating)) and np.isfinite(v)})
    return rec


@torch.inference_mode()
def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache_dir = resolve(args.cache_dir)
    valid_cache = load_cache(cache_dir, "valid")
    valid_dir = resolve(args.valid_dir)
    indices = selected_eval_indices(valid_cache, valid_dir, args.eval_seed, args.eval_terrains)
    cond_all = valid_cache["conditions"][indices].float()
    meta_all = [valid_cache["metadata"][i] for i in indices]
    eval_dir = resolve(args.output_dir) / datetime.now(KST).strftime("learned_baselines_eval_%Y%m%d_%H%M%S")
    eval_dir.mkdir(parents=True, exist_ok=True)
    terrain_cache: dict[str, dict] = {}

    print(f"[{now_kst()}] Evaluate learned baselines: refs={len(indices)} K={args.eval_k}")
    results: dict[str, Any] = {"eval_dir": str(eval_dir), "started_kst": now_kst(), "refs": len(indices)}

    def batched_conditions() -> Iterable[tuple[int, torch.Tensor]]:
        for start in range(0, len(cond_all), args.eval_batch_size):
            yield start, cond_all[start:start + args.eval_batch_size].to(device)

    if args.bc_checkpoint:
        bc = load_bc_checkpoint(resolve(args.bc_checkpoint), device)
        records = []
        t0 = time.perf_counter()
        for start, cond in tqdm(list(batched_conditions()), desc="Eval BC"):
            pred = bc(cond).cpu().numpy()
            latency = (time.perf_counter() - t0) / max(start + len(pred), 1)
            for j, path in enumerate(pred):
                rec = evaluate_prediction(path, meta_all[start + j], terrain_cache)
                rec.update({"method": "bc_planner", "method_label": "BC Planner", "sample_index": 0, "inference_time": latency})
                records.append(rec)
        summary = aggregate_records(records)
        summary["best_of_k_isr"] = summary["by_metric"]["isr"]["mean"]
        summary["diversity"] = 0.0
        summary["checkpoint"] = str(resolve(args.bc_checkpoint))
        results["bc"] = summary
        with (eval_dir / "bc_records.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.cvae_checkpoint:
        cvae = load_cvae_checkpoint(resolve(args.cvae_checkpoint), device)
        records = []
        paths_by_ref: dict[int, list[np.ndarray]] = defaultdict(list)
        isr_by_ref: dict[int, list[float]] = defaultdict(list)
        total_paths = 0
        t0 = time.perf_counter()
        for k in range(args.eval_k):
            torch.manual_seed(args.eval_seed + k * 1_000_003)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.eval_seed + k * 1_000_003)
            for start, cond in tqdm(list(batched_conditions()), desc=f"Eval CVAE k={k}"):
                pred = cvae.sample(cond).cpu().numpy()
                for j, path in enumerate(pred):
                    ref_idx = start + j
                    total_paths += 1
                    rec = evaluate_prediction(path, meta_all[ref_idx], terrain_cache)
                    rec.update({
                        "method": "cvae_planner", "method_label": "CVAE Planner",
                        "sample_index": k,
                        "inference_time": (time.perf_counter() - t0) / max(total_paths, 1),
                    })
                    records.append(rec)
                    paths_by_ref[ref_idx].append(path)
                    isr_by_ref[ref_idx].append(float(rec["isr"]))
        summary = aggregate_records(records)
        best_isrs = [float(np.max(v)) for v in isr_by_ref.values()]
        diversities = []
        for plist in paths_by_ref.values():
            if len(plist) >= 2:
                diversities.append(float(np.mean([chamfer_distance(a, b) for a, b in itertools.combinations(plist, 2)])))
        summary["best_of_k_isr"] = float(np.mean(best_isrs)) if best_isrs else float("nan")
        summary["diversity"] = float(np.mean(diversities)) if diversities else float("nan")
        summary["checkpoint"] = str(resolve(args.cvae_checkpoint))
        results["cvae"] = summary
        with (eval_dir / "cvae_records.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    results["ended_kst"] = now_kst()
    results["eval_seed"] = args.eval_seed
    results["eval_terrains"] = args.eval_terrains
    results["eval_k"] = args.eval_k
    summary_path = eval_dir / "learned_baselines_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    write_markdown_report(args, results, summary_path)
    print(f"[{now_kst()}] Eval summary: {summary_path}")
    return results


def fmt(v: Any, digits: int = 3) -> str:
    if not isinstance(v, (int, float, np.floating)) or not np.isfinite(v):
        return "TBD"
    return f"{float(v):.{digits}f}"


def latest_resnet18_t5_summary() -> Optional[Path]:
    root = _ROOT / "results" / "baselines"
    candidates = sorted(root.glob("resnet18_t5_valid_seed42_*/resnet18_t5_summary.json"))
    return candidates[-1] if candidates else None


def write_markdown_report(args: argparse.Namespace, results: dict[str, Any], summary_path: Path) -> None:
    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    md_path = _ROOT / "experiment" / f"external_comparison_learned_baselines_log_{stamp}.md"
    astar_path = resolve(args.astar_summary) if args.astar_summary else _ROOT / "results/baselines/valid_astar_seed42_20260519_003012/astar_baseline_summary.json"
    ours_path = resolve(args.ours_summary) if args.ours_summary else latest_resnet18_t5_summary()
    astar = json.loads(astar_path.read_text()) if astar_path and astar_path.exists() else None
    ours = json.loads(ours_path.read_text()) if ours_path and ours_path.exists() else None

    def overall_from_astar(name: str) -> str:
        d = astar["summary"][name]
        s = d["by_metric"]
        return f"| {d['label']} | {fmt(s['success']['mean'])} | {fmt(s['feasible']['mean'])} | {fmt(s['cumulative_cot']['mean'], 2)} | {fmt(s['risk_integral']['mean'], 2)} | {fmt(s['isr']['mean'])} | {fmt(s['inference_time']['mean'], 4)} |"

    def overall_learned(key: str, label: str) -> str:
        if key not in results:
            return f"| {label} | TBD | TBD | TBD | TBD | TBD | TBD |"
        s = results[key]["by_metric"]
        return f"| {label} | {fmt(s['success']['mean'])} | {fmt(s['feasible']['mean'])} | {fmt(s['cumulative_cot']['mean'], 2)} | {fmt(s['risk_integral']['mean'], 2)} | {fmt(s['isr']['mean'])} | {fmt(s['inference_time']['mean'], 4)} |"

    def overall_ours() -> str:
        if not ours:
            return "| Ours (ResNet18+T5) | TBD | TBD | TBD | TBD | TBD | TBD |"
        s = ours["by_metric"]
        return f"| Ours (ResNet18+T5) | {fmt(s['success']['mean'])} | {fmt(s['feasible']['mean'])} | {fmt(s['cumulative_cot']['mean'], 2)} | {fmt(s['risk_integral']['mean'], 2)} | {fmt(s['isr']['mean'])} | {fmt(s['inference_time']['mean'], 4)} |"

    def intent_row(key: str, label: str) -> str:
        if key == "intent_aware_astar":
            iw = astar["intentwise"][key]
        elif key == "ours":
            iw = ours["intentwise"]
        elif key in results:
            iw = results[key]["intentwise"]
        else:
            return f"| {label} | TBD | TBD | TBD | TBD | TBD |"
        return f"| {label} | {fmt(iw['left_bias_score'])} | {fmt(iw['right_bias_score'])} | {fmt(iw['avoid_steep_isr'])} | {fmt(iw['prefer_flat_isr'])} | {fmt(iw['composite_isr'])} |"

    def fidelity_row(key: str, label: str) -> str:
        if key == "ours":
            if not ours:
                return f"| {label} | TBD | TBD | TBD | TBD | TBD |"
            s = ours["by_metric"]
            return f"| {label} | {fmt(s['cost_gap']['mean'])} | {fmt(s['chamfer']['mean'])} | {fmt(s['frechet']['mean'])} | {fmt(ours['best_of_k_isr'])} | {fmt(ours['diversity'])} |"
        if key not in results:
            return f"| {label} | TBD | TBD | TBD | TBD | TBD |"
        s = results[key]["by_metric"]
        return f"| {label} | {fmt(s['cost_gap']['mean'])} | {fmt(s['chamfer']['mean'])} | {fmt(s['frechet']['mean'])} | {fmt(results[key]['best_of_k_isr'])} | {fmt(results[key]['diversity'])} |"

    lines = [
        f"# Learned Baseline Evaluation Log ({stamp})",
        "",
        "## Run log",
        "",
        f"- Created: {now_kst()}",
        f"- Summary JSON: `{summary_path}`",
        f"- Eval refs: {results.get('refs')} refs, K={results.get('eval_k')}, seed={results.get('eval_seed')}",
        f"- BC checkpoint: `{results.get('bc', {}).get('checkpoint', 'TBD')}`",
        f"- CVAE checkpoint: `{results.get('cvae', {}).get('checkpoint', 'TBD')}`",
        "",
        "## Table. Overall performance comparison",
        "",
        "| Method | Success ↑ | Feasible ↑ | CoT ↓ | Risk ↓ | ISR ↑ | Time ↓ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if astar:
        for name in ["vanilla_astar", "slope_aware_astar", "cot_aware_astar", "intent_aware_astar"]:
            lines.append(overall_from_astar(name))
    lines.extend([
        overall_learned("bc", "BC Planner"),
        overall_learned("cvae", "CVAE Planner"),
        overall_ours(),
        "",
        "## Table. Intent-wise comparison",
        "",
        "| Method | Left-bias score ↑ | Right-bias score ↑ | Avoid-steep ISR ↑ | Prefer-flat ISR ↑ | Composite ISR ↑ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    if astar:
        lines.append(intent_row("intent_aware_astar", "Intent-aware A*"))
    lines.extend([
        intent_row("bc", "BC Planner"),
        intent_row("cvae", "CVAE Planner"),
        intent_row("ours", "Ours (ResNet18+T5)"),
        "",
        "## Table. Student-to-teacher fidelity and generation properties",
        "",
        "| Method | Cost Gap ↓ | Chamfer ↓ | Frechet ↓ | Best-of-K ISR ↑ | Diversity ↑ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        fidelity_row("bc", "BC Planner"),
        fidelity_row("cvae", "CVAE Planner"),
        fidelity_row("ours", "Ours (ResNet18+T5)"),
        "",
        "## Notes",
        "",
        "- BC and CVAE use frozen ResNet18+T5 condition features extracted from `condition_checkpoint`.",
        "- BC is deterministic, so Best-of-K ISR equals single-sample ISR and Diversity is 0.",
        "- CVAE uses K stochastic latent samples per validation reference.",
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[{now_kst()}] Markdown report: {md_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate BC and CVAE learned planner baselines")
    p.add_argument("--mode", choices=["build-cache", "train", "eval", "all"], default="all")
    p.add_argument("--condition-checkpoint", default="checkpoints/final_model.pt")
    p.add_argument("--train-dir", default="data/raw")
    p.add_argument("--valid-dir", default="data/valid")
    p.add_argument("--cache-dir", default="results/baselines/learned_feature_cache")
    p.add_argument("--output-dir", default="results/baselines")
    p.add_argument("--device", default="cuda")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--max-train-files", type=int, default=None)
    p.add_argument("--max-valid-files", type=int, default=None)
    p.add_argument("--visual-batch-size", type=int, default=128)
    p.add_argument("--text-batch-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=768)
    p.add_argument("--decoder-depth", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--bc-lr", type=float, default=3e-4)
    p.add_argument("--cvae-lr", type=float, default=3e-4)
    p.add_argument("--bc-epochs", type=int, default=250)
    p.add_argument("--cvae-epochs", type=int, default=350)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--kl-beta", type=float, default=1e-3)
    p.add_argument("--kl-anneal-frac", type=float, default=0.3)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", action="store_false", dest="amp")
    p.add_argument("--train-bc", action="store_true", default=True)
    p.add_argument("--train-cvae", action="store_true", default=True)
    p.add_argument("--no-train-bc", action="store_false", dest="train_bc")
    p.add_argument("--no-train-cvae", action="store_false", dest="train_cvae")
    p.add_argument("--bc-checkpoint", default=None)
    p.add_argument("--cvae-checkpoint", default=None)
    p.add_argument("--eval-seed", type=int, default=42)
    p.add_argument("--eval-terrains", type=int, default=100)
    p.add_argument("--eval-k", type=int, default=3)
    p.add_argument("--astar-summary", default=None)
    p.add_argument("--ours-summary", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_results = None
    if args.mode in {"build-cache", "all"}:
        build_feature_cache(args)
    if args.mode in {"train", "all"}:
        train_results = train(args)
        if args.mode == "all":
            if args.bc_checkpoint is None and train_results and "bc" in train_results:
                args.bc_checkpoint = train_results["bc"]["best"]["path"]
            if args.cvae_checkpoint is None and train_results and "cvae" in train_results:
                args.cvae_checkpoint = train_results["cvae"]["best"]["path"]
    if args.mode in {"eval", "all"}:
        eval_model(args)


if __name__ == "__main__":
    main()
