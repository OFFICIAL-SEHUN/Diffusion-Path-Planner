"""Run the text-encoder ablation train/eval pipeline."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (ROOT / p).resolve()


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _run(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=str(ROOT), check=True)


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
    parser.add_argument(
        "--encoders",
        nargs="*",
        default=None,
        help="Optional encoder names to run, e.g. --encoders no_text frozen_clip_proj",
    )
    args = parser.parse_args()

    sweep_cfg = _load_yaml(_resolve(args.config))
    base_cfg = _load_yaml(_resolve(sweep_cfg["base_config"]))

    out_dir = _resolve(sweep_cfg.get("output_dir", "results/text_encoder_ablation"))
    gen_cfg_dir = _resolve(sweep_cfg.get("generated_config_dir", "results/text_encoder_ablation/configs"))
    data_dir = sweep_cfg.get("data_dir", "data/raw")
    val_dir = sweep_cfg.get("val_dir", "data/valid")
    device = args.device or sweep_cfg.get("device", "cuda")
    epochs = args.epochs or sweep_cfg.get("epochs")
    batch_size = args.batch_size or sweep_cfg.get("batch_size")
    eval_max_samples = int(sweep_cfg.get("eval_max_samples", 50))
    val_loss_interval = sweep_cfg.get("val_loss_interval")
    val_interval = sweep_cfg.get("val_interval")

    selected = set(args.encoders) if args.encoders else None
    encoders = [e for e in sweep_cfg["encoders"] if selected is None or e["name"] in selected]
    if selected is not None:
        missing = selected - {e["name"] for e in encoders}
        if missing:
            raise ValueError(f"Unknown encoder names: {sorted(missing)}")

    run_group = out_dir.name
    summaries = []
    for encoder in encoders:
        name = encoder["name"]
        encoder_type = encoder["type"]
        cfg = copy.deepcopy(base_cfg)
        cfg["project_name"] = f"{cfg.get('project_name', 'DiffusionTextGuide')}_{name}"
        cfg.setdefault("model", {})["text_encoder_type"] = encoder_type
        cfg["model"]["text_encoder"] = {
            "type": encoder_type,
            "model_name": encoder.get("model_name"),
            "batch_size": int(encoder.get("feature_batch_size", 64)),
        }
        cfg.setdefault("training", {})["checkpoint_dir"] = str(
            Path("checkpoints") / "text_encoder_ablation" / run_group / name
        )
        cfg["training"]["model_name"] = f"diffusion_textguide_{name}.pt"
        if epochs is not None:
            cfg["training"]["epochs"] = int(epochs)
        if batch_size is not None:
            cfg["training"]["batch_size"] = int(batch_size)
        if args.max_train_batches is not None:
            cfg["training"]["max_train_batches"] = int(args.max_train_batches)
        logging_cfg = cfg.setdefault("logging", {})
        if val_loss_interval is not None:
            logging_cfg["val_loss_interval"] = int(val_loss_interval)
        if val_interval is not None:
            logging_cfg["val_interval"] = int(val_interval)
        logging_cfg["val_max_samples"] = eval_max_samples
        logging_cfg["log_dir"] = str(
            Path("logs") / "text_encoder_ablation" / run_group / name
        )

        cfg_path = gen_cfg_dir / f"{name}.yaml"
        _write_yaml(cfg_path, cfg)

        ckpt = ROOT / cfg["training"]["checkpoint_dir"] / "final_model.pt"
        eval_out = out_dir / f"{name}_eval.json"

        if not args.skip_train:
            _run([
                sys.executable, "train.py",
                "--config", str(cfg_path),
                "--data-dir", data_dir,
                "--val-dir", val_dir,
                "--device", device,
            ], args.dry_run)

        if not args.skip_eval:
            _run([
                sys.executable, "-m", "experiment.evaluators.text_encoder_ablation",
                "--checkpoint", str(ckpt),
                "--data-dir", val_dir,
                "--output", str(eval_out),
                "--device", device,
                "--max-samples", str(eval_max_samples),
            ], args.dry_run)

            if eval_out.exists():
                with eval_out.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                summaries.append({
                    "text_encoder": name,
                    "encoder_type": encoder_type,
                    "base_config": sweep_cfg["base_config"],
                    "checkpoint": str(ckpt),
                    "mean_isr": data["seen"]["mean_isr"],
                    "mean_isr_std": data["seen"].get("mean_isr_std"),
                    "composite_isr": data["seen"]["composite_isr"],
                    "composite_isr_std": data["seen"].get("composite_isr_std"),
                    "worst_isr": data["seen"]["worst_isr"],
                    "unseen_isr": data["unseen"]["mean_isr"],
                    "unseen_isr_std": data["unseen"].get("mean_isr_std"),
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
