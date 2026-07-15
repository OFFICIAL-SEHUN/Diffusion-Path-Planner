"""Train/evaluate independent BC/CVAE planner baselines.

Unlike ``learned_baselines.py``, these baselines do not use the diffusion
checkpoint's visual encoder or text projection. They use raw costmaps, frozen
pretrained T5 sentence features, and train their own condition encoder plus path
decoder. This is the main-paper independent learned-baseline protocol.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_EXP_DIR = Path(__file__).resolve().parent
_ROOT = _EXP_DIR.parents[1]
sys.path.insert(0, str(_ROOT))

from experiment.core.metrics import chamfer_distance
from experiment.core.utils import load_terrain
from experiment.runners.learned_baselines import (
    ResidualMLPBlock,
    PathMLPDecoder,
    aggregate_records,
    evaluate_prediction,
    fmt,
    kl_normal,
    norm_start_goal,
    path_loss,
    selected_eval_indices,
)
from experiment.support.text_encoder_ablation import FrozenTextFeatureEncoder
from instruction_utils import load_instruction_templates

KST = timezone(timedelta(hours=9))


def now_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")


def resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (_ROOT / p).resolve()


def text_bank_lookup(cache: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        str(s): cache["text_bank_features"][i].float()
        for i, s in enumerate(cache["text_bank_instructions"])
    }


def gather_cache_instructions(data_dirs: Iterable[Path]) -> list[str]:
    seen: dict[str, None] = {}
    for data_dir in data_dirs:
        for pt in sorted(data_dir.glob("*.pt")):
            terrain = load_terrain(str(pt))
            for instr in terrain.get("instructions", []):
                seen.setdefault(str(instr), None)
    for split in ("train", "valid"):
        for templates in load_instruction_templates(split).values():
            for instr in templates:
                seen.setdefault(str(instr), None)
    return list(seen.keys())


@torch.inference_mode()
def encode_text_bank(
    instructions: list[str],
    text_encoder_type: str,
    model_name: Optional[str],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    encoder = FrozenTextFeatureEncoder(
        text_encoder_type,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
    features = []
    for start in tqdm(range(0, len(instructions), batch_size), desc="Encoding text bank"):
        features.append(encoder.encode(instructions[start:start + batch_size]).cpu().float())
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(features, dim=0)


@torch.inference_mode()
def build_independent_cache(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_dir = resolve(args.train_dir)
    valid_dir = resolve(args.valid_dir)
    cache_dir = resolve(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    unique_instructions = gather_cache_instructions([train_dir, valid_dir])
    print(f"[{now_kst()}] Build independent raw caches")
    print(f"  train={train_dir}")
    print(f"  valid={valid_dir}")
    print(f"  text_encoder={args.text_encoder_type} model={args.text_model_name}")
    print(f"  text bank instructions={len(unique_instructions)}")
    text_bank_features = encode_text_bank(
        unique_instructions,
        args.text_encoder_type,
        args.text_model_name,
        device,
        args.text_batch_size,
    )
    text_by_instr = {s: text_bank_features[i] for i, s in enumerate(unique_instructions)}

    for split, data_dir, max_files in (
        ("train", train_dir, args.max_train_files),
        ("valid", valid_dir, args.max_valid_files),
    ):
        out_path = cache_dir / f"{split}_independent_samples.pt"
        if out_path.exists() and not args.rebuild_cache:
            print(f"  cache exists, skipping: {out_path}")
            continue

        files = sorted(data_dir.glob("*.pt"))
        if max_files is not None:
            files = files[:max_files]
        terrain_costmaps = []
        terrain_indices = []
        path_chunks = []
        text_chunks = []
        start_goal_chunks = []
        metadata: list[dict[str, Any]] = []

        for terrain_idx, pt_path in enumerate(tqdm(files, desc=f"Cache {split}")):
            terrain = load_terrain(str(pt_path))
            terrain_costmaps.append(torch.from_numpy(terrain["costmap"]).float())
            s_norm, g_norm = norm_start_goal(terrain)
            start_goal = torch.from_numpy(np.concatenate([s_norm, g_norm])).float()
            paths = torch.from_numpy(terrain["paths"]).float()
            instructions = terrain.get("instructions", [""] * paths.shape[0])
            intent_types = terrain.get("intent_types", ["baseline"] * paths.shape[0])
            for path_idx in range(paths.shape[0]):
                instr = str(instructions[path_idx] if path_idx < len(instructions) else "")
                terrain_indices.append(terrain_idx)
                path_chunks.append(paths[path_idx])
                text_chunks.append(text_by_instr[instr])
                start_goal_chunks.append(start_goal)
                metadata.append({
                    "terrain_file": pt_path.name,
                    "terrain_path": str(pt_path.resolve()),
                    "map_id": str(terrain.get("map_id", pt_path.stem)),
                    "path_idx": int(path_idx),
                    "intent_type": str(intent_types[path_idx] if path_idx < len(intent_types) else "baseline"),
                    "instruction": instr,
                })

        payload = {
            "costmaps": torch.stack(terrain_costmaps).float(),
            "terrain_indices": torch.tensor(terrain_indices, dtype=torch.long),
            "paths": torch.stack(path_chunks).float(),
            "text_features": torch.stack(text_chunks).float(),
            "start_goal": torch.stack(start_goal_chunks).float(),
            "metadata": metadata,
            "text_bank_instructions": unique_instructions,
            "text_bank_features": text_bank_features.float(),
            "spec": {
                "split": split,
                "data_dir": str(data_dir),
                "num_terrains": len(files),
                "num_samples": len(path_chunks),
                "text_encoder_type": args.text_encoder_type,
                "text_model_name": args.text_model_name,
                "text_dim": int(text_bank_features.shape[-1]),
                "horizon": int(path_chunks[0].shape[0]) if path_chunks else 0,
                "built_at_kst": now_kst(),
                "protocol": "independent raw costmap + frozen pretrained text feature cache; no diffusion checkpoint encoder",
            },
        }
        torch.save(payload, out_path)
        print(
            f"  saved {split}: {out_path}  "
            f"terrains={len(files)} samples={len(path_chunks)} text_dim={text_bank_features.shape[-1]}"
        )


class IndependentCacheDataset(Dataset):
    def __init__(self, cache: dict[str, Any]) -> None:
        self.costmaps = cache["costmaps"].float()
        self.terrain_indices = cache["terrain_indices"].long()
        self.text_features = cache["text_features"].float()
        self.start_goal = cache["start_goal"].float()
        self.paths = cache["paths"].float()

    def __len__(self) -> int:
        return int(self.paths.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        terrain_idx = self.terrain_indices[idx]
        return self.costmaps[terrain_idx], self.text_features[idx], self.start_goal[idx], self.paths[idx]


class CostmapCNNEncoder(nn.Module):
    def __init__(self, visual_dim: int = 256, dropout: float = 0.05) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 192),
            nn.SiLU(inplace=True),
            nn.Conv2d(192, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 256),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, visual_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, costmap: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(costmap))


class IndependentConditionEncoder(nn.Module):
    def __init__(self, text_dim: int, visual_dim: int = 256, cond_text_dim: int = 256,
                 dropout: float = 0.05) -> None:
        super().__init__()
        self.visual_encoder = CostmapCNNEncoder(visual_dim, dropout)
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, cond_text_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cond_text_dim, cond_text_dim),
            nn.GELU(),
        )
        self.cond_dim = visual_dim + cond_text_dim + 4

    def forward(self, costmap: torch.Tensor, text_feature: torch.Tensor,
                start_goal: torch.Tensor) -> torch.Tensor:
        visual = self.visual_encoder(costmap)
        text = self.text_projection(text_feature)
        return torch.cat([visual, text, start_goal], dim=-1)


def apply_endpoints(path: torch.Tensor, start_goal: torch.Tensor) -> torch.Tensor:
    out = path.clone()
    out[:, 0, :] = start_goal[:, :2]
    out[:, -1, :] = start_goal[:, 2:]
    return out


class IndependentBCPlanner(nn.Module):
    def __init__(self, text_dim: int, horizon: int = 120, visual_dim: int = 256,
                 cond_text_dim: int = 256, hidden_dim: int = 768, depth: int = 5,
                 dropout: float = 0.05) -> None:
        super().__init__()
        self.horizon = horizon
        self.condition_encoder = IndependentConditionEncoder(text_dim, visual_dim, cond_text_dim, dropout)
        self.decoder = PathMLPDecoder(self.condition_encoder.cond_dim, horizon, hidden_dim, depth, dropout)

    def forward(self, costmap: torch.Tensor, text_feature: torch.Tensor,
                start_goal: torch.Tensor) -> torch.Tensor:
        cond = self.condition_encoder(costmap, text_feature, start_goal)
        return apply_endpoints(self.decoder(cond), start_goal)


class IndependentCVAEPlanner(nn.Module):
    def __init__(self, text_dim: int, horizon: int = 120, visual_dim: int = 256,
                 cond_text_dim: int = 256, hidden_dim: int = 768, depth: int = 5,
                 latent_dim: int = 64, dropout: float = 0.05) -> None:
        super().__init__()
        self.horizon = horizon
        self.latent_dim = latent_dim
        self.condition_encoder = IndependentConditionEncoder(text_dim, visual_dim, cond_text_dim, dropout)
        cond_dim = self.condition_encoder.cond_dim
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

    def encode_condition(self, costmap: torch.Tensor, text_feature: torch.Tensor,
                         start_goal: torch.Tensor) -> torch.Tensor:
        return self.condition_encoder(costmap, text_feature, start_goal)

    def encode(self, cond: torch.Tensor, path: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([cond, path.flatten(1)], dim=-1))
        return self.mu(h), self.logvar(h).clamp(min=-10.0, max=8.0)

    def decode(self, cond: torch.Tensor, z: torch.Tensor, start_goal: torch.Tensor) -> torch.Tensor:
        return apply_endpoints(self.decoder(torch.cat([cond, z], dim=-1)), start_goal)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, costmap: torch.Tensor, text_feature: torch.Tensor,
                start_goal: torch.Tensor, path: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cond = self.encode_condition(costmap, text_feature, start_goal)
        mu, logvar = self.encode(cond, path)
        z = self.reparameterize(mu, logvar)
        return self.decode(cond, z, start_goal), mu, logvar

    def sample(self, costmap: torch.Tensor, text_feature: torch.Tensor,
               start_goal: torch.Tensor) -> torch.Tensor:
        cond = self.encode_condition(costmap, text_feature, start_goal)
        z = torch.randn(cond.shape[0], self.latent_dim, device=cond.device)
        return self.decode(cond, z, start_goal)


def load_cache(cache_dir: Path, split: str) -> dict[str, Any]:
    path = cache_dir / f"{split}_independent_samples.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing independent cache: {path}. Run --mode build-cache first.")
    return torch.load(path, map_location="cpu", weights_only=False)


def make_loader(cache: dict[str, Any], batch_size: int, shuffle: bool, args: argparse.Namespace,
                device: torch.device) -> DataLoader:
    return DataLoader(
        IndependentCacheDataset(cache),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


def batch_to_device(batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                    device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    costmap, text, start_goal, path = batch
    return (
        costmap.to(device, non_blocking=True),
        text.to(device, non_blocking=True),
        start_goal.to(device, non_blocking=True),
        path.to(device, non_blocking=True),
    )


def validate(model: nn.Module, loader: DataLoader, device: torch.device, kind: str) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for batch in loader:
            costmap, text, start_goal, target = batch_to_device(batch, device)
            if kind == "bc":
                pred = model(costmap, text, start_goal)
            else:
                pred = model.sample(costmap, text, start_goal)
            losses.append(float(path_loss(pred, target).detach().cpu()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def train_bc(args: argparse.Namespace, train_cache: dict[str, Any], valid_cache: dict[str, Any],
             device: torch.device, run_dir: Path) -> dict[str, Any]:
    train_loader = make_loader(train_cache, args.batch_size, True, args, device)
    valid_loader = make_loader(valid_cache, args.eval_batch_size, False, args, device)
    text_dim = int(train_cache["spec"]["text_dim"])
    horizon = int(train_cache["spec"].get("horizon", train_cache["paths"].shape[1]))
    model = IndependentBCPlanner(
        text_dim, horizon, args.visual_dim, args.cond_text_dim,
        args.hidden_dim, args.decoder_depth, args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.bc_lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    best = {"val_loss": float("inf"), "epoch": 0, "path": str(run_dir / "bc_independent_best.pt")}
    history = []
    t_start = time.perf_counter()
    print(f"[{now_kst()}] Train independent BC: n={len(train_loader.dataset)}, epochs={args.bc_epochs}")

    for epoch in range(1, args.bc_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            costmap, text, start_goal, target = batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                pred = model(costmap, text, start_goal)
                loss = path_loss(pred, target)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        do_eval = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.bc_epochs
        train_loss = float(np.mean(losses))
        if do_eval:
            val_loss = validate(model, valid_loader, device, "bc")
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            print(f"  BC epoch {epoch:04d}/{args.bc_epochs} train={train_loss:.6f} val={val_loss:.6f}")
            if val_loss < best["val_loss"]:
                best.update({"val_loss": float(val_loss), "epoch": epoch})
                torch.save({
                    "model_type": "bc_independent",
                    "model_state_dict": model.state_dict(),
                    "text_dim": text_dim,
                    "horizon": horizon,
                    "args": vars(args),
                    "best": best,
                    "trained_at_kst": now_kst(),
                    "protocol": "independent raw costmap + frozen pretrained T5 features; no diffusion encoder",
                }, best["path"])
    final_path = run_dir / "bc_independent_final.pt"
    torch.save({
        "model_type": "bc_independent",
        "model_state_dict": model.state_dict(),
        "text_dim": text_dim,
        "horizon": horizon,
        "args": vars(args),
        "best": best,
        "trained_at_kst": now_kst(),
        "protocol": "independent raw costmap + frozen pretrained T5 features; no diffusion encoder",
    }, final_path)
    return {"best": best, "final_path": str(final_path), "history": history,
            "elapsed_seconds": time.perf_counter() - t_start}


def train_cvae(args: argparse.Namespace, train_cache: dict[str, Any], valid_cache: dict[str, Any],
               device: torch.device, run_dir: Path) -> dict[str, Any]:
    train_loader = make_loader(train_cache, args.batch_size, True, args, device)
    valid_loader = make_loader(valid_cache, args.eval_batch_size, False, args, device)
    text_dim = int(train_cache["spec"]["text_dim"])
    horizon = int(train_cache["spec"].get("horizon", train_cache["paths"].shape[1]))
    model = IndependentCVAEPlanner(
        text_dim, horizon, args.visual_dim, args.cond_text_dim,
        args.hidden_dim, args.decoder_depth, args.latent_dim, args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.cvae_lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    best = {"val_loss": float("inf"), "epoch": 0, "path": str(run_dir / "cvae_independent_best.pt")}
    history = []
    t_start = time.perf_counter()
    print(f"[{now_kst()}] Train independent CVAE: n={len(train_loader.dataset)}, epochs={args.cvae_epochs}")

    for epoch in range(1, args.cvae_epochs + 1):
        model.train()
        losses = []
        recons = []
        kls = []
        kl_beta = args.kl_beta * min(1.0, epoch / max(1, int(args.cvae_epochs * args.kl_anneal_frac)))
        for batch in train_loader:
            costmap, text, start_goal, target = batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                pred, mu, logvar = model(costmap, text, start_goal, target)
                recon = path_loss(pred, target)
                kl = kl_normal(mu, logvar)
                loss = recon + kl_beta * kl
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            recons.append(float(recon.detach().cpu()))
            kls.append(float(kl.detach().cpu()))
        do_eval = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.cvae_epochs
        train_loss = float(np.mean(losses))
        if do_eval:
            val_loss = validate(model, valid_loader, device, "cvae")
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "recon": float(np.mean(recons)),
                "kl": float(np.mean(kls)),
                "kl_beta": float(kl_beta),
            })
            print(
                f"  CVAE epoch {epoch:04d}/{args.cvae_epochs} train={train_loss:.6f} "
                f"val={val_loss:.6f} recon={np.mean(recons):.6f} kl={np.mean(kls):.4f} beta={kl_beta:.5g}"
            )
            if val_loss < best["val_loss"]:
                best.update({"val_loss": float(val_loss), "epoch": epoch})
                torch.save({
                    "model_type": "cvae_independent",
                    "model_state_dict": model.state_dict(),
                    "text_dim": text_dim,
                    "horizon": horizon,
                    "latent_dim": args.latent_dim,
                    "args": vars(args),
                    "best": best,
                    "trained_at_kst": now_kst(),
                    "protocol": "independent raw costmap + frozen pretrained T5 features; no diffusion encoder",
                }, best["path"])
    final_path = run_dir / "cvae_independent_final.pt"
    torch.save({
        "model_type": "cvae_independent",
        "model_state_dict": model.state_dict(),
        "text_dim": text_dim,
        "horizon": horizon,
        "latent_dim": args.latent_dim,
        "args": vars(args),
        "best": best,
        "trained_at_kst": now_kst(),
        "protocol": "independent raw costmap + frozen pretrained T5 features; no diffusion encoder",
    }, final_path)
    return {"best": best, "final_path": str(final_path), "history": history,
            "elapsed_seconds": time.perf_counter() - t_start}


def train(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache_dir = resolve(args.cache_dir)
    run_dir = resolve(args.output_dir) / datetime.now(KST).strftime("independent_learned_baselines_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    train_cache = load_cache(cache_dir, "train")
    valid_cache = load_cache(cache_dir, "valid")
    results: dict[str, Any] = {
        "run_dir": str(run_dir),
        "started_kst": now_kst(),
        "protocol": "independent BC/CVAE trained from raw costmaps and frozen pretrained T5 features; no diffusion checkpoint encoder",
        "train_cache_spec": train_cache.get("spec", {}),
        "valid_cache_spec": valid_cache.get("spec", {}),
    }
    if args.train_bc:
        results["bc"] = train_bc(args, train_cache, valid_cache, device, run_dir)
    if args.train_cvae:
        results["cvae"] = train_cvae(args, train_cache, valid_cache, device, run_dir)
    results["ended_kst"] = now_kst()
    with (run_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"[{now_kst()}] Training summary: {run_dir / 'training_summary.json'}")
    return results


def load_bc_checkpoint(path: Path, device: torch.device) -> IndependentBCPlanner:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = IndependentBCPlanner(
        int(ckpt["text_dim"]),
        int(ckpt["horizon"]),
        int(args.get("visual_dim", 256)),
        int(args.get("cond_text_dim", 256)),
        int(args.get("hidden_dim", 768)),
        int(args.get("decoder_depth", 5)),
        float(args.get("dropout", 0.05)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_cvae_checkpoint(path: Path, device: torch.device) -> IndependentCVAEPlanner:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = IndependentCVAEPlanner(
        int(ckpt["text_dim"]),
        int(ckpt["horizon"]),
        int(args.get("visual_dim", 256)),
        int(args.get("cond_text_dim", 256)),
        int(args.get("hidden_dim", 768)),
        int(args.get("decoder_depth", 5)),
        int(ckpt.get("latent_dim", args.get("latent_dim", 64))),
        float(args.get("dropout", 0.05)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def valid_language_refs(meta_all: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    templates = load_instruction_templates("valid")
    out = []
    prompts = []
    for ref_idx, meta in enumerate(meta_all):
        m = dict(meta)
        valid_list = templates.get(m["intent_type"], [m.get("instruction", "")])
        instr = str(valid_list[ref_idx % len(valid_list)])
        m["instruction"] = instr
        out.append(m)
        prompts.append(instr)
    return out, prompts


def build_eval_tensors(valid_cache: dict[str, Any], indices: list[int], prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    text_by_instr = text_bank_lookup(valid_cache)
    terrain_indices = valid_cache["terrain_indices"][indices].long()
    costmaps = valid_cache["costmaps"][terrain_indices].float()
    start_goal = valid_cache["start_goal"][indices].float()
    text_features = torch.stack([text_by_instr[p].float() for p in prompts]).float()
    return costmaps, text_features, start_goal


@torch.inference_mode()
def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache_dir = resolve(args.cache_dir)
    valid_cache = load_cache(cache_dir, "valid")
    valid_dir = resolve(args.valid_dir)
    indices = selected_eval_indices(valid_cache, valid_dir, args.eval_seed, args.eval_terrains)
    meta_all = [valid_cache["metadata"][i] for i in indices]
    meta_all, prompts = valid_language_refs(meta_all)
    costmaps, text_features, start_goal = build_eval_tensors(valid_cache, indices, prompts)
    eval_dir = resolve(args.output_dir) / datetime.now(KST).strftime("independent_learned_baselines_eval_%Y%m%d_%H%M%S")
    eval_dir.mkdir(parents=True, exist_ok=True)
    terrain_cache: dict[str, dict] = {}
    results: dict[str, Any] = {
        "eval_dir": str(eval_dir),
        "started_kst": now_kst(),
        "protocol": "independent BC/CVAE, valid terrain plus inst_valid.json language; no diffusion encoder",
        "instruction_split": "data/instruction/valid/inst_valid.json",
        "refs": len(indices),
        "eval_seed": args.eval_seed,
        "eval_terrains": args.eval_terrains,
        "eval_k": args.eval_k,
        "unique_eval_prompts": len(set(prompts)),
    }
    print(
        f"[{now_kst()}] Evaluate independent learned baselines: "
        f"refs={len(indices)} K={args.eval_k} unique_prompts={len(set(prompts))}"
    )

    def batched() -> Iterable[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
        for start in range(0, len(indices), args.eval_batch_size):
            yield (
                start,
                costmaps[start:start + args.eval_batch_size].to(device, non_blocking=True),
                text_features[start:start + args.eval_batch_size].to(device, non_blocking=True),
                start_goal[start:start + args.eval_batch_size].to(device, non_blocking=True),
            )

    if args.bc_checkpoint:
        bc = load_bc_checkpoint(resolve(args.bc_checkpoint), device)
        records = []
        t0 = time.perf_counter()
        for start, cm, txt, sg in tqdm(list(batched()), desc="Eval independent BC"):
            pred = bc(cm, txt, sg).detach().cpu().numpy()
            latency = (time.perf_counter() - t0) / max(start + len(pred), 1)
            for j, path in enumerate(pred):
                rec = evaluate_prediction(path, meta_all[start + j], terrain_cache)
                rec.update({
                    "method": "bc_independent",
                    "method_label": "BC Planner (independent)",
                    "sample_index": 0,
                    "inference_time": latency,
                })
                records.append(rec)
        summary = aggregate_records(records)
        summary["best_of_k_isr"] = summary["by_metric"]["isr"]["mean"]
        summary["diversity"] = 0.0
        summary["checkpoint"] = str(resolve(args.bc_checkpoint))
        results["bc"] = summary
        with (eval_dir / "bc_independent_records.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.cvae_checkpoint:
        cvae = load_cvae_checkpoint(resolve(args.cvae_checkpoint), device)
        records = []
        paths_by_ref: dict[int, list[np.ndarray]] = defaultdict(list)
        isr_by_ref: dict[int, list[float]] = defaultdict(list)
        total = 0
        t0 = time.perf_counter()
        for k in range(args.eval_k):
            torch.manual_seed(args.eval_seed + k * 1_000_003)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.eval_seed + k * 1_000_003)
            for start, cm, txt, sg in tqdm(list(batched()), desc=f"Eval independent CVAE k={k}"):
                pred = cvae.sample(cm, txt, sg).detach().cpu().numpy()
                for j, path in enumerate(pred):
                    ref_idx = start + j
                    total += 1
                    rec = evaluate_prediction(path, meta_all[ref_idx], terrain_cache)
                    rec.update({
                        "method": "cvae_independent",
                        "method_label": "CVAE Planner (independent)",
                        "sample_index": k,
                        "inference_time": (time.perf_counter() - t0) / max(total, 1),
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
        with (eval_dir / "cvae_independent_records.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    results["ended_kst"] = now_kst()
    summary_path = eval_dir / "independent_learned_baselines_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    write_markdown_report(args, results, summary_path)
    print(f"[{now_kst()}] Eval summary: {summary_path}")
    return results


def load_json_if_exists(path: Optional[Path]) -> Optional[dict[str, Any]]:
    if path and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def latest_unseen_language_summary() -> Optional[Path]:
    root = _ROOT / "results" / "baselines"
    candidates = sorted(root.glob("unseen_language_eval_*/unseen_language_summary.json"))
    return candidates[-1] if candidates else None


def write_markdown_report(args: argparse.Namespace, results: dict[str, Any], summary_path: Path) -> None:
    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    md_path = _ROOT / "experiment" / f"external_comparison_independent_learned_baselines_log_{stamp}.md"
    astar_path = resolve(args.astar_summary) if args.astar_summary else _ROOT / "results/baselines/valid_astar_seed42_20260519_003012/astar_baseline_summary.json"
    ours_path = resolve(args.ours_unseen_summary) if args.ours_unseen_summary else latest_unseen_language_summary()
    astar = load_json_if_exists(astar_path)
    ours_payload = load_json_if_exists(ours_path)
    ours = ours_payload.get("summaries", {}).get("ours") if ours_payload else None

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
        if key == "intent_aware_astar" and astar:
            iw = astar["intentwise"][key]
        elif key == "ours" and ours:
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
        f"# Independent Learned Baseline Evaluation Log ({stamp})",
        "",
        "## Run log",
        "",
        f"- Created: {now_kst()}",
        f"- Summary JSON: `{summary_path}`",
        f"- Instruction split: `data/instruction/valid/inst_valid.json`",
        f"- Dataset split: `data/valid`, {results.get('eval_terrains')} terrains sampled with seed {results.get('eval_seed')}",
        f"- Eval refs: {results.get('refs')} refs, K={results.get('eval_k')}, unique prompts={results.get('unique_eval_prompts')}",
        f"- Training protocol: independent raw costmap encoder + trainable text projection; no diffusion checkpoint encoder",
        f"- BC checkpoint: `{results.get('bc', {}).get('checkpoint', 'TBD')}`",
        f"- CVAE checkpoint: `{results.get('cvae', {}).get('checkpoint', 'TBD')}`",
        f"- Ours unseen-language summary: `{ours_path}`",
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
        overall_learned("bc", "BC Planner (independent)"),
        overall_learned("cvae", "CVAE Planner (independent)"),
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
        intent_row("bc", "BC Planner (independent)"),
        intent_row("cvae", "CVAE Planner (independent)"),
        intent_row("ours", "Ours (ResNet18+T5)"),
        "",
        "## Table. Student-to-teacher fidelity and generation properties",
        "",
        "| Method | Cost Gap ↓ | Chamfer ↓ | Frechet ↓ | Best-of-K ISR ↑ | Diversity ↑ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        fidelity_row("bc", "BC Planner (independent)"),
        fidelity_row("cvae", "CVAE Planner (independent)"),
        fidelity_row("ours", "Ours (ResNet18+T5)"),
        "",
        "## Notes",
        "",
        "- BC/CVAE in this report do not load `final_model.pt` or reuse the diffusion visual/text projection encoder.",
        "- They train from raw costmaps and frozen pretrained T5 sentence features only.",
        "- Evaluation uses train-unseen validation terrains and validation instruction templates.",
        "- Classical A* rows are unchanged because A* uses symbolic intent, not language text.",
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[{now_kst()}] Markdown report: {md_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate independent BC and CVAE learned planner baselines")
    p.add_argument("--mode", choices=["build-cache", "train", "eval", "all"], default="all")
    p.add_argument("--train-dir", default="data/raw")
    p.add_argument("--valid-dir", default="data/valid")
    p.add_argument("--cache-dir", default="results/baselines/independent_feature_cache_t5")
    p.add_argument("--output-dir", default="results/baselines")
    p.add_argument("--device", default="cuda")
    p.add_argument("--text-encoder-type", default="t5_proj")
    p.add_argument("--text-model-name", default="t5-base")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--max-train-files", type=int, default=None)
    p.add_argument("--max-valid-files", type=int, default=None)
    p.add_argument("--text-batch-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--visual-dim", type=int, default=256)
    p.add_argument("--cond-text-dim", type=int, default=256)
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
    p.add_argument("--ours-unseen-summary", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_results = None
    if args.mode in {"build-cache", "all"}:
        build_independent_cache(args)
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
