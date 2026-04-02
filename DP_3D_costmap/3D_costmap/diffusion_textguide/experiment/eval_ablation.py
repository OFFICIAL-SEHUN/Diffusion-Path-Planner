"""
Experiment 2 — Inner Module Ablation

Evaluates how each component of the diffusion model contributes:
  - Full model (baseline)
  - w/o text encoder  (zero-out text tokens)
  - w/o CoT           (data trained without β, but here we test generation quality)
  - w/o structured intent  (uniform random instruction)
  - w/o cross-attention (skip cross-attn at bottleneck)
  - Different visual encoder (resnet vs convnext)

Metrics: goal_error, feasibility, ISR (overall + per-type),
         cumulative_cot, risk, avg/max slope, inference time.

Usage:
  python -m experiment.eval_ablation \
      --checkpoint checkpoints/final_model.pt \
      --data-dir data/raw \
      --output-dir results/ablation
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

_EXP_DIR = Path(__file__).resolve().parent
_ROOT = _EXP_DIR.parent
sys.path.insert(0, str(_ROOT))

from data_loader import build_vocab, text_to_tokens
from experiment.utils import load_model, load_terrain, run_inference_batch
from experiment.metrics import compute_all_metrics, infeasible_rate


# ── ablation wrappers ────────────────────────────────────────────────────────

class AblationRunner:
    """Thin wrapper that runs diffusion inference under various ablation modes."""

    MODES = [
        "full",
        "wo_text",
        "wo_cross_attn",
        "wo_structured_intent",
        "visual_resnet",
    ]

    def __init__(self, ckpt_path: str, device: torch.device):
        self.ckpt_path = ckpt_path
        self.device = device
        self._cache = {}

    def _get_model(self, mode: str):
        """Return (model, scheduler, vocab, config) for the given mode."""
        if mode in self._cache:
            return self._cache[mode]

        override_backbone = None
        disable_text = False
        disable_cross = False

        if mode == "visual_resnet":
            override_backbone = "resnet"
        elif mode == "wo_text":
            disable_text = True
        elif mode == "wo_cross_attn":
            disable_cross = True

        model, scheduler, vocab, config = load_model(
            self.ckpt_path, self.device,
            override_backbone=override_backbone,
            disable_text=disable_text,
            disable_cross_attn=disable_cross,
        )

        if disable_cross:
            def _identity_cross(self_module, x, context):
                return x
            model.cross_attn.forward = lambda x, ctx: x

        self._cache[mode] = (model, scheduler, vocab, config)
        return model, scheduler, vocab, config

    def run(self, mode: str, costmap, start_pos, goal_pos,
            text_tokens, horizon):
        model, scheduler, vocab, config = self._get_model(mode)
        device = self.device

        costmap_t = costmap.unsqueeze(0).to(device)
        start_t = start_pos.unsqueeze(0).to(device)
        goal_t = goal_pos.unsqueeze(0).to(device)

        if mode == "wo_text":
            tokens_t = None
        elif mode == "wo_structured_intent":
            dummy_instr = "Navigate along the default route"
            tokens_t = text_to_tokens(dummy_instr, vocab, max_seq_len=16).unsqueeze(0).to(device)
        else:
            tokens_t = text_tokens.unsqueeze(0).to(device)

        t0 = time.perf_counter()
        path = scheduler.sample(
            model, costmap_t, shape=(1, horizon, 2),
            start_pos=start_t, end_pos=goal_t,
            text_tokens=tokens_t, show_progress=False,
        )
        elapsed = time.perf_counter() - t0
        return path[0].cpu().numpy(), elapsed


def _load_test_samples(data_dir: str, max_samples: int = 200):
    """Load terrain .pt files and flatten into per-intent test samples."""
    data_path = Path(data_dir)
    pt_files = sorted(data_path.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt in {data_dir}")

    samples = []
    for f in pt_files:
        t = load_terrain(str(f))
        img_size = int(t["img_size"])
        paths = t["paths"]
        instructions = t.get("instructions", [])
        intent_types = t.get("intent_types", [])
        intent_params_list = t.get("intent_params", [])
        start = t.get("start_position", (0, 0))
        goal = t.get("goal_position", (img_size - 1, img_size - 1))

        costmap_t = torch.from_numpy(t["costmap"]).float()
        s_norm = torch.tensor([(start[1] / img_size) * 2 - 1,
                               (start[0] / img_size) * 2 - 1], dtype=torch.float32)
        g_norm = torch.tensor([(goal[1] / img_size) * 2 - 1,
                               (goal[0] / img_size) * 2 - 1], dtype=torch.float32)

        for i in range(paths.shape[0]):
            samples.append({
                "costmap": costmap_t,
                "slope_map_deg": t["slope_map"],
                "height_map": t["height_map"],
                "img_size": img_size,
                "start_pos": start,
                "goal_pos": goal,
                "start_norm": s_norm,
                "goal_norm": g_norm,
                "ref_path": paths[i],
                "instruction": instructions[i] if i < len(instructions) else "",
                "intent_type": intent_types[i] if i < len(intent_types) else "baseline",
                "intent_params": intent_params_list[i] if i < len(intent_params_list) else {},
                "pixel_resolution": float(t.get("pixel_resolution", 0.5)),
                "limit_angle_deg": float(t.get("limit_angle_deg", 25)),
                "risk_threshold_deg": float(t.get("risk_threshold_deg", 15.0)),
            })
            if len(samples) >= max_samples:
                return samples
    return samples


def run_ablation(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    runner = AblationRunner(args.checkpoint, device)

    _, _, vocab, config = runner._get_model("full")
    horizon = config.get("data", {}).get("horizon", 120)
    cw = config.get("intent", {}).get("cost_weights", {})

    samples = _load_test_samples(args.data_dir, args.max_samples)
    print(f"Loaded {len(samples)} test samples")

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for mode in AblationRunner.MODES:
        print(f"\n{'='*60}\n  Mode: {mode}\n{'='*60}")
        mode_metrics = defaultdict(list)
        per_intent = defaultdict(lambda: defaultdict(list))

        for sample in tqdm(samples, desc=mode):
            tokens = text_to_tokens(sample["instruction"], vocab, max_seq_len=16)
            gen_path, elapsed = runner.run(
                mode, sample["costmap"], sample["start_norm"],
                sample["goal_norm"], tokens, horizon,
            )

            m = compute_all_metrics(
                path_norm=gen_path,
                goal_norm=sample["goal_norm"].numpy(),
                slope_map_deg=sample["slope_map_deg"],
                height_map=sample["height_map"],
                img_size=sample["img_size"],
                intent_type=sample["intent_type"],
                intent_params=sample["intent_params"],
                start_pos=sample["start_pos"],
                goal_pos=sample["goal_pos"],
                ref_path_norm=sample["ref_path"],
                pixel_resolution=sample["pixel_resolution"],
                limit_angle_deg=sample["limit_angle_deg"],
                risk_threshold_deg=sample["risk_threshold_deg"],
                alpha=cw.get("alpha", 1.0),
                beta=cw.get("beta", 0.8),
                gamma=cw.get("gamma", 0.1),
                delta=cw.get("delta", 1.0),
            )
            m["inference_time"] = elapsed

            for k, v in m.items():
                mode_metrics[k].append(v)
                per_intent[sample["intent_type"]][k].append(v)

        summary = {}
        for k, vals in mode_metrics.items():
            vals_clean = [v for v in vals if not np.isnan(v)]
            summary[k] = {
                "mean": float(np.mean(vals_clean)) if vals_clean else float("nan"),
                "std": float(np.std(vals_clean)) if vals_clean else float("nan"),
            }

        intent_summary = {}
        for itype, idict in per_intent.items():
            intent_summary[itype] = {}
            for k, vals in idict.items():
                vals_clean = [v for v in vals if not np.isnan(v)]
                intent_summary[itype][k] = {
                    "mean": float(np.mean(vals_clean)) if vals_clean else float("nan"),
                    "std": float(np.std(vals_clean)) if vals_clean else float("nan"),
                }

        all_results[mode] = {"overall": summary, "per_intent": intent_summary}

        print(f"  Goal err  = {summary.get('goal_error', {}).get('mean', 'N/A'):.4f}")
        print(f"  ISR       = {summary.get('isr', {}).get('mean', 'N/A'):.4f}")
        print(f"  Cum. CoT  = {summary.get('cumulative_cot', {}).get('mean', 'N/A'):.2f}")
        print(f"  Risk      = {summary.get('risk_integral', {}).get('mean', 'N/A'):.2f}")
        print(f"  Inf. time = {summary.get('inference_time', {}).get('mean', 'N/A'):.4f}s")

    # save
    out_json = os.path.join(args.output_dir, "ablation_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults → {out_json}")

    # comparison bar chart
    _plot_ablation_bars(all_results, args.output_dir)


def _plot_ablation_bars(all_results: dict, output_dir: str):
    """Generate grouped bar charts for key metrics across ablation modes."""
    modes = list(all_results.keys())
    key_metrics = ["goal_error", "isr", "cumulative_cot", "risk_integral",
                   "mean_slope", "inference_time"]
    labels = ["Goal Error", "ISR", "Cum. CoT", "Risk", "Mean Slope", "Inf. Time (s)"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax, metric, label in zip(axes, key_metrics, labels):
        means = [all_results[m]["overall"].get(metric, {}).get("mean", 0) for m in modes]
        stds = [all_results[m]["overall"].get(metric, {}).get("std", 0) for m in modes]
        x = np.arange(len(modes))
        bars = ax.bar(x, means, yerr=stds, capsize=4, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(modes, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(label)

        full_val = means[0] if means else 0
        for i, (b, val) in enumerate(zip(bars, means)):
            if i > 0 and full_val != 0:
                pct = (val - full_val) / abs(full_val) * 100
                ax.annotate(f"{pct:+.1f}%", xy=(b.get_x() + b.get_width() / 2, val),
                            fontsize=7, ha="center", va="bottom")

    fig.suptitle("Inner Module Ablation", fontsize=13)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "ablation_bars.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Bar chart → {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Exp 2: Inner module ablation")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data-dir", type=str, default=str(_ROOT / "data" / "raw"))
    ap.add_argument("--output-dir", type=str, default=str(_ROOT / "results" / "ablation"))
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
