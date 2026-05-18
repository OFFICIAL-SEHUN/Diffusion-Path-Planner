"""Run the text-encoder ablation train/eval pipeline.

Usage:
  python -m experiment.runners.text_encoder_ablation \
      --config experiment/configs/text_encoder_ablation.yaml
"""

from __future__ import annotations

import argparse
import copy
import os
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_SEEDS = [42]


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (ROOT / p).resolve()


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    tmp_path.replace(path)


def _run(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=str(ROOT), check=True)


def _as_int_list(values) -> list[int]:
    if values is None:
        return list(DEFAULT_TRAIN_SEEDS)
    if isinstance(values, int):
        return [int(values)]
    return [int(v) for v in values]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment/configs/text_encoder_ablation.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Training seeds. Defaults to config seeds or 42")
    parser.add_argument(
        "--encoders",
        nargs="*",
        default=None,
        help="Optional encoder names to run, e.g. --encoders no_text frozen_clip_proj",
    )
    args = parser.parse_args()

    sweep_cfg = _load_yaml(_resolve(args.config))
    base_cfg = _load_yaml(_resolve(sweep_cfg["base_config"]))

    os.environ["WANDB_MODE"] = str(sweep_cfg.get("wandb_mode", "disabled"))

    out_dir = _resolve(sweep_cfg.get("output_dir", "results/text_encoder_ablation"))
    gen_cfg_dir = _resolve(sweep_cfg.get("generated_config_dir", "results/text_encoder_ablation/configs"))
    data_dir = sweep_cfg.get("data_dir", "data/raw")
    val_dir = sweep_cfg.get("val_dir", "data/valid")
    device = args.device or sweep_cfg.get("device", "cuda")
    epochs = args.epochs or sweep_cfg.get("epochs")
    batch_size = args.batch_size or sweep_cfg.get("batch_size")
    train_seeds = args.seeds if args.seeds is not None else _as_int_list(sweep_cfg.get("seeds"))

    train_val_max_samples = sweep_cfg.get("train_val_max_samples")
    train_val_max_samples = int(train_val_max_samples) if train_val_max_samples is not None else None
    train_val_samples_per_intent = sweep_cfg.get("train_val_samples_per_intent")
    train_val_samples_per_intent = (
        int(train_val_samples_per_intent) if train_val_samples_per_intent is not None else None
    )
    train_val_seed = int(sweep_cfg.get("train_val_seed", sweep_cfg.get("eval_seed", 42)))
    train_val_num_seeds = int(sweep_cfg.get("train_val_num_seeds", 1))
    eval_max_samples = sweep_cfg.get("eval_max_samples")
    eval_max_samples = int(eval_max_samples) if eval_max_samples is not None else None
    eval_samples_per_intent = sweep_cfg.get("eval_samples_per_intent")
    eval_samples_per_intent = int(eval_samples_per_intent) if eval_samples_per_intent is not None else None
    eval_seed = int(sweep_cfg.get("eval_seed", 42))
    eval_num_seeds = int(sweep_cfg.get("eval_num_seeds", 1))
    val_loss_interval = sweep_cfg.get("val_loss_interval")
    val_interval = sweep_cfg.get("val_interval")
    deterministic = bool(sweep_cfg.get("deterministic", False))

    selected = set(args.encoders) if args.encoders else None
    encoders = [e for e in sweep_cfg["encoders"] if selected is None or e["name"] in selected]
    if selected is not None:
        missing = selected - {e["name"] for e in encoders}
        if missing:
            raise ValueError(f"Unknown encoder names: {sorted(missing)}")

    run_group = sweep_cfg.get("run_group", None)
    if run_group is None:
        run_group = out_dir.name
    run_group = str(run_group).strip()
    checkpoint_root = Path(sweep_cfg.get("checkpoint_root", "checkpoints/text_encoder_ablation"))
    log_root = Path(sweep_cfg.get("log_root", "logs/text_encoder_ablation"))
    train_vis_root = Path(sweep_cfg.get("train_vis_root", "results/text_encoder_ablation/train_vis"))
    summaries = []

    for encoder in encoders:
        name = encoder["name"]
        encoder_type = encoder["type"]
        for seed in train_seeds:
            run_name = name
            cfg = copy.deepcopy(base_cfg)
            cfg["project_name"] = f"{cfg.get('project_name', 'DiffusionTextGuide')}_{run_name}"
            cfg["seed"] = int(seed)
            if sweep_cfg.get("wandb") is not None:
                cfg["wandb"] = copy.deepcopy(sweep_cfg["wandb"])
            if sweep_cfg.get("cost_weights") is not None:
                cfg.setdefault("intent", {})["cost_weights"] = copy.deepcopy(sweep_cfg["cost_weights"])
            if sweep_cfg.get("risk_threshold_deg") is not None:
                cfg.setdefault("intent", {})["risk_threshold_deg"] = float(sweep_cfg["risk_threshold_deg"])
            cfg.setdefault("model", {})["text_encoder_type"] = encoder_type
            cfg["model"]["text_encoder"] = {
                "type": encoder_type,
                "model_name": encoder.get("model_name"),
                "batch_size": int(encoder.get("feature_batch_size", 64)),
            }
            run_ckpt_dir = checkpoint_root / run_name if not run_group else checkpoint_root / run_group / run_name
            cfg.setdefault("training", {})["checkpoint_dir"] = str(run_ckpt_dir)
            cfg["training"]["model_name"] = f"diffusion_textguide_{run_name}.pt"
            cfg["training"]["deterministic"] = deterministic
            if epochs is not None:
                cfg["training"]["epochs"] = int(epochs)
            if batch_size is not None:
                cfg["training"]["batch_size"] = int(batch_size)
            if args.max_train_batches is not None:
                cfg["training"]["max_train_batches"] = int(args.max_train_batches)
            else:
                cfg["training"].pop("max_train_batches", None)

            logging_cfg = cfg.setdefault("logging", {})
            if val_loss_interval is not None:
                logging_cfg["val_loss_interval"] = int(val_loss_interval)
            if val_interval is not None:
                logging_cfg["val_interval"] = int(val_interval)
            if train_val_samples_per_intent is not None:
                logging_cfg["val_samples_per_intent"] = train_val_samples_per_intent
                logging_cfg.pop("val_max_samples", None)
            elif train_val_max_samples is not None:
                logging_cfg["val_max_samples"] = train_val_max_samples
                logging_cfg.pop("val_samples_per_intent", None)
            logging_cfg["val_seed"] = train_val_seed
            logging_cfg["val_num_seeds"] = train_val_num_seeds
            run_log_dir = log_root / run_name if not run_group else log_root / run_group / run_name
            run_vis_dir = train_vis_root / run_name if not run_group else train_vis_root / run_group / run_name
            logging_cfg["log_dir"] = str(run_log_dir)
            logging_cfg["vis_dir"] = str(run_vis_dir)

            cfg_path = gen_cfg_dir / f"{run_name}.yaml"
            _write_yaml(cfg_path, cfg)

            ckpt = ROOT / cfg["training"]["checkpoint_dir"] / "final_model.pt"
            eval_out = out_dir / f"{run_name}_eval.json"

            if not args.skip_train:
                _run([
                    sys.executable, "train.py",
                    "--config", str(cfg_path),
                    "--data-dir", data_dir,
                    "--val-dir", val_dir,
                    "--device", device,
                ], args.dry_run)

            if not args.skip_eval:
                eval_cmd = [
                    sys.executable, "-m", "experiment.evaluators.text_encoder_ablation",
                    "--checkpoint", str(ckpt),
                    "--data-dir", val_dir,
                    "--output", str(eval_out),
                    "--device", device,
                    "--seed", str(eval_seed),
                    "--num-seeds", str(eval_num_seeds),
                ]
                if eval_samples_per_intent is not None:
                    eval_cmd.extend(["--samples-per-intent", str(eval_samples_per_intent)])
                elif eval_max_samples is not None:
                    eval_cmd.extend(["--max-samples", str(eval_max_samples)])
                _run(eval_cmd, args.dry_run)

                if eval_out.exists():
                    with eval_out.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    summaries.append({
                        "run_name": run_name,
                        "text_encoder": name,
                        "training_seed": seed,
                        "encoder_type": encoder_type,
                        "base_config": sweep_cfg["base_config"],
                        "checkpoint": str(ckpt),
                        "mean_isr": data["seen"]["mean_isr"],
                        "mean_isr_std": data["seen"].get("mean_isr_std"),
                        "composite_isr": data["seen"]["composite_isr"],
                        "composite_isr_std": data["seen"].get("composite_isr_std"),
                        "worst_isr": data["seen"]["worst_isr"],
                        "worst_isr_intent": data["seen"].get("worst_isr_intent"),
                        "teacher_worst_isr": data["seen"].get("teacher_worst_isr"),
                        "mean_isr_vs_teacher": data["seen"].get("mean_isr_vs_teacher"),
                        "worst_isr_vs_teacher": data["seen"].get("worst_isr_vs_teacher"),
                        "worst_isr_vs_teacher_intent": data["seen"].get("worst_isr_vs_teacher_intent"),
                        "unseen_isr": data["unseen"]["mean_isr"],
                        "unseen_isr_std": data["unseen"].get("mean_isr_std"),
                        "unseen_isr_vs_teacher": data["unseen"].get("mean_isr_vs_teacher"),
                        "cot": data["seen"]["cot"],
                        "cot_std": data["seen"].get("cot_std"),
                        "risk": data["seen"]["risk"],
                        "risk_std": data["seen"].get("risk_std"),
                        "cost_gap": data["seen"]["cost_gap"],
                        "cost_gap_std": data["seen"].get("cost_gap_std"),
                        "latency_s": data["seen"]["latency_s"],
                        "latency_s_std": data["seen"].get("latency_s_std"),
                        "n_seen": data["seen"]["n"],
                        "n_unseen": data["unseen"]["n"],
                        "n_refs": data.get("eval_sampling", {}).get("n_refs"),
                        "n_terrains": data.get("eval_sampling", {}).get("n_terrains"),
                        "samples_per_intent": data.get("eval_sampling", {}).get("samples_per_intent"),
                        "eval_seed": data.get("eval_sampling", {}).get("seed"),
                        "eval_num_seeds": data.get("eval_sampling", {}).get("num_seeds"),
                    })

    if summaries:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        summary_csv = out_dir / "summary.csv"
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"Saved summary: {summary_path}")
        print(f"Saved summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
