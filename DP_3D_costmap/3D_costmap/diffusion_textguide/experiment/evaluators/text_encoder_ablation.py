"""
Evaluate a trained text-encoder ablation checkpoint.

Reports the columns used for encoder selection:
Mean ISR, Composite ISR, Worst ISR, Unseen ISR, CoT, Risk, Cost Gap, Latency.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_EXP_DIR = Path(__file__).resolve().parent
_ROOT = _EXP_DIR.parents[1]
sys.path.insert(0, str(_ROOT))

from data_loader import text_to_tokens
from experiment.core.metrics import compute_all_metrics
from experiment.core.utils import load_terrain
from instruction_utils import load_instruction_templates
from model.diffusion import DiffusionScheduler
from model.network import ConditionalPathModel
from experiment.support.text_encoder_ablation import FrozenTextFeatureEncoder
from text_conditioning import (
    DEFAULT_FEATURE_DIMS,
    get_intent_to_id,
    is_frozen_feature_encoder,
    normalize_text_encoder_type,
)


def _resolve_path(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (_ROOT / p).resolve()


def _load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    vocab = ckpt.get("vocab", {})
    d_cfg = config.get("data", {})
    m_cfg = config.get("model", {})
    diff_cfg = config.get("diffusion", {})

    text_encoder_type = normalize_text_encoder_type(m_cfg.get("text_encoder_type", "learnable"))
    intent_to_id = get_intent_to_id("train")
    text_feature_dim = int(m_cfg.get("text_feature_dim", DEFAULT_FEATURE_DIMS.get(text_encoder_type, 256)))

    model = ConditionalPathModel(
        transition_dim=2,
        dim=m_cfg.get("base_dim", 64),
        horizon=d_cfg.get("horizon", 120),
        visual_dim=m_cfg.get("image_feat_dim", 256),
        text_dim=256,
        vocab_size=len(vocab) if vocab else 200,
        max_seq_len=16,
        visual_backbone=m_cfg.get("visual_backbone", "convnext"),
        visual_pretrained=False,
        timm_model_name=m_cfg.get("timm_model_name"),
        timm_pretrained=False,
        input_img_size=d_cfg.get("img_size"),
        text_encoder_type=text_encoder_type,
        num_intents=int(m_cfg.get("num_intents", len(intent_to_id))),
        text_feature_dim=text_feature_dim,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    scheduler = DiffusionScheduler(
        timesteps=diff_cfg.get("timesteps", 200),
        beta_start=diff_cfg.get("beta_start", 0.0001),
        beta_end=diff_cfg.get("beta_end", 0.02),
        device=device,
    )
    return model, scheduler, vocab, config, text_encoder_type, intent_to_id


def _text_kwargs(
    text_encoder_type: str,
    instruction: str,
    intent_type: str,
    vocab: dict,
    device: torch.device,
    intent_to_id: dict,
    feature_encoder: Optional[FrozenTextFeatureEncoder],
) -> dict:
    if text_encoder_type == "no_text":
        return {}
    if text_encoder_type == "onehot":
        idx = intent_to_id.get(intent_type, intent_to_id.get("baseline", 0))
        return {"intent_ids": torch.tensor([idx], dtype=torch.long, device=device)}
    if is_frozen_feature_encoder(text_encoder_type):
        if feature_encoder is None:
            raise RuntimeError(f"{text_encoder_type} requires a FrozenTextFeatureEncoder")
        return {"text_features": feature_encoder.encode([instruction]).to(device)}
    return {"text_tokens": text_to_tokens(instruction, vocab, max_seq_len=16).unsqueeze(0).to(device)}


def _aggregate(rows: list[dict]) -> dict:
    def values(key: str) -> list[float]:
        return [r[key] for r in rows if key in r and np.isfinite(r[key])]

    def mean(key: str) -> float:
        vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    def std(key: str) -> float:
        vals = values(key)
        return float(np.std(vals)) if vals else float("nan")

    per_intent = defaultdict(list)
    for r in rows:
        per_intent[r["intent_type"]].append(r["isr"])

    intent_means = {
        k: float(np.mean([v for v in vals if np.isfinite(v)]))
        for k, vals in per_intent.items()
        if vals
    }
    composite_vals = [
        r["isr"] for r in rows
        if "+" in r["intent_type"] and np.isfinite(r["isr"])
    ]

    return {
        "mean_isr": mean("isr"),
        "mean_isr_std": std("isr"),
        "composite_isr": float(np.mean(composite_vals)) if composite_vals else float("nan"),
        "composite_isr_std": float(np.std(composite_vals)) if composite_vals else float("nan"),
        "worst_isr": float(min(intent_means.values())) if intent_means else float("nan"),
        "cot": mean("cumulative_cot"),
        "cot_std": std("cumulative_cot"),
        "risk": mean("risk_integral"),
        "risk_std": std("risk_integral"),
        "cost_gap": mean("cost_gap"),
        "cost_gap_std": std("cost_gap"),
        "latency_s": mean("latency_s"),
        "latency_s_std": std("latency_s"),
        "per_intent_isr": intent_means,
        "per_intent_count": {k: len(v) for k, v in per_intent.items()},
        "n": len(rows),
    }


def _build_intent_balanced_refs(pt_files: list[Path], max_samples: int) -> list[tuple[str, int, str]]:
    """Create near-uniform eval refs via round-robin over intent buckets."""
    by_intent = defaultdict(list)
    for pt_path in pt_files:
        terrain = load_terrain(str(pt_path))
        paths = terrain["paths"]
        intent_types = terrain.get("intent_types", [])
        for i in range(paths.shape[0]):
            intent_type = intent_types[i] if i < len(intent_types) else "baseline"
            by_intent[intent_type].append((str(pt_path), i, intent_type))

    if not by_intent:
        return []

    ordered_intents = sorted(by_intent.keys())
    cursors = {intent: 0 for intent in ordered_intents}
    refs: list[tuple[str, int, str]] = []

    while len(refs) < max_samples:
        progressed = False
        for intent in ordered_intents:
            idx = cursors[intent]
            samples = by_intent[intent]
            if idx < len(samples):
                refs.append(samples[idx])
                cursors[intent] = idx + 1
                progressed = True
                if len(refs) >= max_samples:
                    break
        if not progressed:
            break
    return refs


@torch.no_grad()
def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, scheduler, vocab, config, text_encoder_type, intent_to_id = _load_model(
        _resolve_path(args.checkpoint), device,
    )

    feature_encoder = None
    if is_frozen_feature_encoder(text_encoder_type):
        text_cfg = config.get("model", {}).get("text_encoder", {})
        feature_encoder = FrozenTextFeatureEncoder(
            text_encoder_type,
            model_name=text_cfg.get("model_name", config.get("model", {}).get("text_model_name")),
            device=device,
            batch_size=int(text_cfg.get("batch_size", 64)),
        )

    d_cfg = config.get("data", {})
    g_cfg = config.get("gradient", {})
    cw = config.get("intent", {}).get("cost_weights", {})
    horizon = int(d_cfg.get("horizon", 120))
    img_size = int(d_cfg.get("img_size", 100))
    pixel_res = float(g_cfg.get("pixel_resolution", 0.5))
    limit_deg = float(g_cfg.get("limit_angle_deg", 25.0))
    risk_thresh = float(config.get("intent", {}).get("risk_threshold_deg", 15.0))

    unseen_templates = load_instruction_templates("valid")
    rows_seen = []
    rows_unseen = []

    pt_files = sorted(_resolve_path(args.data_dir).glob("*.pt"))
    refs = _build_intent_balanced_refs(pt_files, args.max_samples)
    terrain_cache: dict[str, dict] = {}
    for n_seen, (pt_path_str, i, intent_type) in enumerate(refs, start=1):
        terrain = terrain_cache.get(pt_path_str)
        if terrain is None:
            terrain = load_terrain(pt_path_str)
            terrain_cache[pt_path_str] = terrain
        paths = terrain["paths"]
        instructions = terrain.get("instructions", [])
        intent_types = terrain.get("intent_types", [])
        intent_params_list = terrain.get("intent_params", [])
        t_img_size = int(terrain.get("img_size", img_size))
        start = terrain.get("start_position", (0, 0))
        goal = terrain.get("goal_position", (t_img_size - 1, t_img_size - 1))

        costmap_t = torch.from_numpy(terrain["costmap"]).float().unsqueeze(0).to(device)
        s_norm = torch.tensor(
            [(start[1] / t_img_size) * 2 - 1, (start[0] / t_img_size) * 2 - 1],
            dtype=torch.float32, device=device,
        ).unsqueeze(0)
        g_norm = torch.tensor(
            [(goal[1] / t_img_size) * 2 - 1, (goal[0] / t_img_size) * 2 - 1],
            dtype=torch.float32, device=device,
        ).unsqueeze(0)

        intent_type = intent_types[i] if i < len(intent_types) else intent_type
        intent_params = intent_params_list[i] if i < len(intent_params_list) else {}
        seen_instruction = instructions[i] if i < len(instructions) else intent_type
        valid_list = unseen_templates.get(intent_type, [seen_instruction])
        unseen_instruction = valid_list[i % len(valid_list)]

        for split, instruction, bucket in (
            ("seen", seen_instruction, rows_seen),
            ("unseen", unseen_instruction, rows_unseen),
        ):
            kwargs = _text_kwargs(
                text_encoder_type, instruction, intent_type, vocab,
                device, intent_to_id, feature_encoder,
            )
            t0 = time.perf_counter()
            gen = scheduler.sample(
                model, costmap_t, shape=(1, horizon, 2),
                start_pos=s_norm, end_pos=g_norm,
                show_progress=False,
                **kwargs,
            )[0].cpu().numpy()
            latency = time.perf_counter() - t0

            m = compute_all_metrics(
                path_norm=gen,
                goal_norm=g_norm[0].cpu().numpy(),
                slope_map_deg=terrain["slope_map"],
                height_map=terrain["height_map"],
                img_size=t_img_size,
                intent_type=intent_type,
                intent_params=intent_params,
                start_pos=start,
                goal_pos=goal,
                ref_path_norm=paths[i],
                pixel_resolution=float(terrain.get("pixel_resolution", pixel_res)),
                limit_angle_deg=float(terrain.get("limit_angle_deg", limit_deg)),
                risk_threshold_deg=float(terrain.get("risk_threshold_deg", risk_thresh)),
                alpha=cw.get("alpha", 1.0),
                beta=cw.get("beta", 0.8),
                gamma=cw.get("gamma", 0.1),
                delta=cw.get("delta", 1.0),
            )
            m.update({
                "split": split,
                "intent_type": intent_type,
                "instruction": instruction,
                "latency_s": latency,
            })
            bucket.append(m)

    summary = {
        "checkpoint": str(_resolve_path(args.checkpoint)),
        "text_encoder": text_encoder_type,
        "seen": _aggregate(rows_seen),
        "unseen": _aggregate(rows_unseen),
        "rows": {
            "seen": rows_seen,
            "unseen": rows_unseen,
        },
    }

    out_path = _resolve_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({
        "text_encoder": text_encoder_type,
        "mean_isr": summary["seen"]["mean_isr"],
        "composite_isr": summary["seen"]["composite_isr"],
        "worst_isr": summary["seen"]["worst_isr"],
        "unseen_isr": summary["unseen"]["mean_isr"],
        "cot": summary["seen"]["cot"],
        "risk": summary["seen"]["risk"],
        "cost_gap": summary["seen"]["cost_gap"],
        "latency_s": summary["seen"]["latency_s"],
    }, indent=2))
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/valid")
    parser.add_argument("--output", default="results/text_encoder_ablation/eval_summary.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=50)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
