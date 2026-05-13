#!/usr/bin/env python3
"""Run 10-intent backbone ablation jobs for one backbone.

By default this launches both pretrained and scratch initializations for the
requested backbone, so jobs can be distributed by model across servers.  Use
--init pretrained or --init scratch when a server should run only one init.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

PARETO_COST_WEIGHTS = {
    "alpha": 0.5,
    "beta": 1.75,
    "gamma": 1.5,
    "delta": 1.75,
}

MODEL_ALIASES = {
    "convnext": "convnext",
    "convnext_tiny": "convnext",
    "resnet": "resnet18",
    "resnet18": "resnet18",
    "resnet_18": "resnet18",
    "resnet-18": "resnet18",
    "efficientnet": "efficientnet_b0",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet-b0": "efficientnet_b0",
    "swin": "swin_tiny",
    "swin_tiny": "swin_tiny",
    "swin-tiny": "swin_tiny",
    "vit": "vit_tiny",
    "vit_tiny": "vit_tiny",
    "vit-tiny": "vit_tiny",
}

MODEL_SPECS = {
    "convnext": {
        "base_config": "configs/convnext.yaml",
        "visual_backbone": "convnext",
        "timm_model_name": "convnext_tiny",
        "base_dim": 64,
    },
    "resnet18": {
        "base_config": "configs/resnet.yaml",
        "visual_backbone": "resnet",
        "timm_model_name": "resnet18",
        "base_dim": 64,
    },
    "efficientnet_b0": {
        "base_config": "configs/efficientnet_b0.yaml",
        "visual_backbone": "efficientnet_b0",
        "timm_model_name": "efficientnet_b0",
        "base_dim": 64,
    },
    "swin_tiny": {
        "base_config": "configs/swin.yaml",
        "visual_backbone": "swin_tiny",
        "timm_model_name": "swin_tiny_patch4_window7_224",
        "base_dim": 128,
    },
    "vit_tiny": {
        "base_config": "configs/vit_tiny.yaml",
        "visual_backbone": "vit_tiny",
        "timm_model_name": "vit_tiny_patch16_224",
        "base_dim": 128,
    },
}

INIT_CHOICES = ("pretrained", "scratch")


def canonical_model(value: str) -> str:
    key = value.strip().lower().replace(" ", "_")
    try:
        return MODEL_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(sorted(MODEL_ALIASES))
        raise argparse.ArgumentTypeError(f"unknown model {value!r}; valid aliases: {valid}") from exc


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def resolve_run_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def build_config(args: argparse.Namespace, init: str) -> tuple[dict[str, Any], Path, str]:
    model_key = canonical_model(args.model)
    spec = MODEL_SPECS[model_key]
    pretrained = init == "pretrained"
    slug = f"{model_key}_{init}"

    cfg = load_yaml(ROOT / spec["base_config"])
    cfg["project_name"] = f"DiffusionTextGuide_BackboneAblation10Intent_{slug}"
    cfg["seed"] = args.seed

    data_cfg = cfg.setdefault("data", {})
    data_cfg["img_size"] = args.img_size
    data_cfg["horizon"] = args.horizon

    intent_cfg = cfg.setdefault("intent", {})
    intent_cfg["cost_weights"] = dict(PARETO_COST_WEIGHTS)
    intent_cfg["risk_threshold_deg"] = args.risk_threshold_deg

    model_cfg = cfg.setdefault("model", {})
    model_cfg.update({
        "base_dim": spec["base_dim"],
        "time_embed_dim": 256,
        "image_feat_dim": 256,
        "visual_backbone": spec["visual_backbone"],
        "timm_model_name": spec["timm_model_name"],
        "timm_pretrained": pretrained,
    })
    model_cfg.pop("convnext_pretrained", None)

    train_cfg = cfg.setdefault("training", {})
    train_cfg.update({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "checkpoint_dir": str(Path(args.checkpoint_root) / slug),
        "model_name": f"diffusion_textguide_{slug}.pt",
        "log_interval": args.log_interval,
        "use_amp": not args.no_amp,
    })
    if args.max_train_batches is not None:
        train_cfg["max_train_batches"] = args.max_train_batches
    else:
        train_cfg.pop("max_train_batches", None)

    log_cfg = cfg.setdefault("logging", {})
    log_cfg.update({
        "val_loss_interval": args.val_loss_interval,
        "val_interval": args.val_interval,
        "log_dir": str(Path(args.log_root) / slug),
        "val_seed": args.val_seed,
        "val_num_seeds": args.val_num_seeds,
    })
    if args.val_samples_per_intent is not None:
        log_cfg["val_samples_per_intent"] = args.val_samples_per_intent
        log_cfg.pop("val_max_samples", None)
    else:
        log_cfg["val_max_samples"] = args.val_max_samples
        log_cfg.pop("val_samples_per_intent", None)

    wandb_cfg = cfg.setdefault("wandb", {})
    wandb_cfg["group"] = args.wandb_group

    config_path = ROOT / args.config_out_dir / f"{slug}.yaml"
    return cfg, config_path, slug


def ensure_output_dirs(cfg: dict[str, Any]) -> None:
    resolve_run_path(cfg["training"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    resolve_run_path(cfg["logging"]["log_dir"]).mkdir(parents=True, exist_ok=True)


def build_train_command(args: argparse.Namespace, config_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "train.py"),
        "--config",
        str(config_path),
        "--data-dir",
        args.data_dir,
        "--val-dir",
        args.val_dir,
        "--device",
        args.device,
    ]
    if args.resume:
        cmd.extend(["--resume", args.resume])
    return cmd


def selected_inits(args: argparse.Namespace) -> list[str]:
    if args.init == "both":
        return list(INIT_CHOICES)
    return [args.init]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=canonical_model,
                        help="one of convnext, resnet-18, efficientnet_b0, swin, vit_tiny")
    parser.add_argument("--init", choices=["both", *INIT_CHOICES], default="both",
                        help="default: both; run pretrained then scratch for this model")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--val-dir", default="data/valid")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--val-loss-interval", type=int, default=200)
    parser.add_argument("--val-interval", type=int, default=1000)
    parser.add_argument("--val-max-samples", type=int, default=50)
    parser.add_argument("--val-samples-per-intent", type=int, default=None)
    parser.add_argument("--val-seed", type=int, default=42)
    parser.add_argument("--val-num-seeds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--risk-threshold-deg", type=float, default=15.0)
    parser.add_argument("--checkpoint-root", default="checkpoints/backbone_ablation_10intent")
    parser.add_argument("--log-root", default="logs/backbone_ablation_10intent")
    parser.add_argument("--config-out-dir", default="configs/backbone_ablation_10intent")
    parser.add_argument("--wandb-group", default="backbone_ablation_10intent")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="offline")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="debug only; limits batches per epoch in generated config")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="write configs and print train commands without launching training")
    args = parser.parse_args()

    inits = selected_inits(args)
    if args.resume and len(inits) > 1:
        parser.error("--resume can only be used with --init pretrained or --init scratch")

    commands: list[list[str]] = []
    for init in inits:
        cfg, config_path, slug = build_config(args, init)
        write_yaml(config_path, cfg)
        ensure_output_dirs(cfg)
        cmd = build_train_command(args, config_path)
        commands.append(cmd)

        env_prefix = f"WANDB_MODE={args.wandb_mode}"
        printable = " ".join([env_prefix] + [shlex.quote(part) for part in cmd])
        print(f"Run slug: {slug}")
        print(f"Config written: {config_path}")
        print(f"Train command: {printable}")

    if args.dry_run:
        return 0

    env = os.environ.copy()
    env["WANDB_MODE"] = args.wandb_mode
    for cmd in commands:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
