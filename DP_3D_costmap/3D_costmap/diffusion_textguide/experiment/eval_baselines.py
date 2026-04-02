"""
Experiment 3 — External Comparison

Compares our text-conditioned diffusion planner against classical baselines:
  1. Distance-only A*          (β=γ=δ=0)
  2. Slope-aware A*            (β=0, γ>0, δ=0)
  3. CoT-aware A*              (β>0, γ=0, δ=0)
  4. Full intent A* (oracle)   (α,β,γ,δ from config — same as pseudo-label planner)
  5. Text-agnostic diffusion   (diffusion model with zeroed text input)

Metrics: goal_error, path_length, cumulative_cot, mean/max slope,
         risk, ISR, infeasible_rate, inference_time.

Usage:
  python -m experiment.eval_baselines \
      --checkpoint checkpoints/final_model.pt \
      --data-dir data/raw \
      --output-dir results/baselines
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

_EXP_DIR = Path(__file__).resolve().parent
_ROOT = _EXP_DIR.parent
sys.path.insert(0, str(_ROOT))

from scripts.generate_data import (
    SlopeCotGenerator,
    INTENT_CATALOG,
    _path_pixels_to_normalized,
    _resample_path,
)
from data_loader import text_to_tokens
from experiment.utils import load_model, load_terrain
from experiment.metrics import compute_all_metrics


# ── baseline definitions ─────────────────────────────────────────────────────

BASELINES = {
    "distance_only_astar": {
        "alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0,
        "description": "Distance-only A* (no energy/safety/intent)",
    },
    "slope_aware_astar": {
        "alpha": 1.0, "beta": 0.0, "gamma": 0.5, "delta": 0.0,
        "description": "Slope-aware A* (distance + risk, no energy/intent)",
    },
    "cot_aware_astar": {
        "alpha": 1.0, "beta": 0.8, "gamma": 0.0, "delta": 0.0,
        "description": "CoT-aware A* (distance + energy, no safety/intent)",
    },
    "full_intent_astar": {
        "alpha": 1.0, "beta": 0.8, "gamma": 0.1, "delta": 1.0,
        "description": "Full 4-term A* (oracle — same as pseudo-label planner)",
    },
}


def _intent_params_for(itype: str) -> dict:
    for entry in INTENT_CATALOG:
        if entry["type"] == itype:
            return dict(entry["params"])
    return {}


def _run_astar_baseline(
    baseline_name: str,
    terrain: dict,
    intent_type: str,
    intent_params: dict,
    horizon: int,
    risk_threshold_deg: float,
):
    """Run A* with the given baseline's weight config; return (path_norm, elapsed)."""
    bdef = BASELINES[baseline_name]
    img_size = int(terrain["img_size"])
    gen = SlopeCotGenerator(
        img_size=img_size,
        height_range=[0, 5],
        mass=10.0, gravity=9.8,
        limit_angle_deg=float(terrain.get("limit_angle_deg", 25)),
        max_iterations=20000,
        pixel_resolution=float(terrain.get("pixel_resolution", 0.5)),
    )
    gen.height_map = terrain["height_map"]
    gen.slope_map = np.radians(terrain["slope_map"])

    t0 = time.perf_counter()
    path_px = gen.find_path_with_intent(
        terrain["start_position"], terrain["goal_position"],
        alpha=bdef["alpha"], beta=bdef["beta"],
        gamma=bdef["gamma"], delta=bdef["delta"],
        risk_threshold_deg=risk_threshold_deg,
        intent_type=intent_type if baseline_name == "full_intent_astar" else "baseline",
        intent_params=intent_params if baseline_name == "full_intent_astar" else {},
    )
    elapsed = time.perf_counter() - t0

    if path_px is None or len(path_px) < 5:
        return None, elapsed

    norm = _path_pixels_to_normalized(path_px, img_size)
    fixed = _resample_path(norm, horizon)
    return fixed, elapsed


def _run_diffusion(
    model, scheduler, vocab, terrain, instruction, horizon, device,
    disable_text: bool = False,
):
    """Run diffusion inference; return (path_norm, elapsed)."""
    img_size = int(terrain["img_size"])
    start = terrain["start_position"]
    goal = terrain["goal_position"]
    s_norm = torch.tensor([(start[1] / img_size) * 2 - 1,
                           (start[0] / img_size) * 2 - 1], dtype=torch.float32)
    g_norm = torch.tensor([(goal[1] / img_size) * 2 - 1,
                           (goal[0] / img_size) * 2 - 1], dtype=torch.float32)
    costmap_t = torch.from_numpy(terrain["costmap"]).float().unsqueeze(0).to(device)
    s_t = s_norm.unsqueeze(0).to(device)
    g_t = g_norm.unsqueeze(0).to(device)

    if disable_text:
        tokens_t = None
    else:
        tokens = text_to_tokens(instruction, vocab, max_seq_len=16)
        tokens_t = tokens.unsqueeze(0).to(device)

    t0 = time.perf_counter()
    path = scheduler.sample(
        model, costmap_t, shape=(1, horizon, 2),
        start_pos=s_t, end_pos=g_t,
        text_tokens=tokens_t, show_progress=False,
    )
    elapsed = time.perf_counter() - t0
    return path[0].cpu().numpy(), elapsed


def _load_samples(data_dir: str, max_terrains: int = 50):
    """Load terrain files and expand into per-intent samples."""
    pt_files = sorted(Path(data_dir).glob("*.pt"))[:max_terrains]
    samples = []
    for f in pt_files:
        t = load_terrain(str(f))
        n_paths = t["paths"].shape[0]
        for i in range(n_paths):
            samples.append({
                "terrain": t,
                "path_idx": i,
                "ref_path": t["paths"][i],
                "instruction": t.get("instructions", [""])[i] if i < len(t.get("instructions", [])) else "",
                "intent_type": t.get("intent_types", ["baseline"])[i] if i < len(t.get("intent_types", [])) else "baseline",
                "intent_params": t.get("intent_params", [{}])[i] if i < len(t.get("intent_params", [])) else {},
            })
    return samples


def run_comparison(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, scheduler, vocab, config = load_model(args.checkpoint, device)
    horizon = config.get("data", {}).get("horizon", 120)
    cw = config.get("intent", {}).get("cost_weights", {})
    risk_th = config.get("intent", {}).get("risk_threshold_deg", 15.0)

    samples = _load_samples(args.data_dir, args.max_terrains)
    print(f"Loaded {len(samples)} test samples from ≤{args.max_terrains} terrains")
    os.makedirs(args.output_dir, exist_ok=True)

    methods = list(BASELINES.keys()) + ["diffusion_full", "diffusion_no_text"]
    all_results = {m: defaultdict(list) for m in methods}

    for sample in tqdm(samples, desc="Evaluating"):
        ter = sample["terrain"]
        img_size = int(ter["img_size"])
        start = ter["start_position"]
        goal = ter["goal_position"]
        g_norm = np.array([(goal[1] / img_size) * 2 - 1,
                           (goal[0] / img_size) * 2 - 1], dtype=np.float32)
        px_res = float(ter.get("pixel_resolution", 0.5))
        limit_deg = float(ter.get("limit_angle_deg", 25))

        common_kwargs = dict(
            goal_norm=g_norm,
            slope_map_deg=ter["slope_map"],
            height_map=ter["height_map"],
            img_size=img_size,
            intent_type=sample["intent_type"],
            intent_params=sample["intent_params"],
            start_pos=start,
            goal_pos=goal,
            ref_path_norm=sample["ref_path"],
            pixel_resolution=px_res,
            limit_angle_deg=limit_deg,
            risk_threshold_deg=risk_th,
            alpha=cw.get("alpha", 1.0),
            beta=cw.get("beta", 0.8),
            gamma=cw.get("gamma", 0.1),
            delta=cw.get("delta", 1.0),
        )

        # A* baselines
        for bname in BASELINES:
            path, elapsed = _run_astar_baseline(
                bname, ter, sample["intent_type"], sample["intent_params"],
                horizon, risk_th,
            )
            if path is None:
                all_results[bname]["infeasible"].append(1)
                continue
            all_results[bname]["infeasible"].append(0)
            m = compute_all_metrics(path_norm=path, **common_kwargs)
            m["inference_time"] = elapsed
            for k, v in m.items():
                all_results[bname][k].append(v)

        # Diffusion (full)
        path, elapsed = _run_diffusion(
            model, scheduler, vocab, ter, sample["instruction"],
            horizon, device, disable_text=False,
        )
        m = compute_all_metrics(path_norm=path, **common_kwargs)
        m["inference_time"] = elapsed
        for k, v in m.items():
            all_results["diffusion_full"][k].append(v)

        # Diffusion (no text)
        path, elapsed = _run_diffusion(
            model, scheduler, vocab, ter, sample["instruction"],
            horizon, device, disable_text=True,
        )
        m = compute_all_metrics(path_norm=path, **common_kwargs)
        m["inference_time"] = elapsed
        for k, v in m.items():
            all_results["diffusion_no_text"][k].append(v)

    # aggregate
    summary = {}
    for method in methods:
        summary[method] = {}
        for k, vals in all_results[method].items():
            clean = [v for v in vals if not np.isnan(v)]
            summary[method][k] = {
                "mean": float(np.mean(clean)) if clean else float("nan"),
                "std": float(np.std(clean)) if clean else float("nan"),
                "count": len(clean),
            }
        infeas = all_results[method].get("infeasible", [])
        summary[method]["infeasible_rate"] = {
            "mean": float(np.mean(infeas)) if infeas else 0.0,
            "count": len(infeas),
        }

    out_json = os.path.join(args.output_dir, "baseline_comparison.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults → {out_json}")

    _print_comparison_table(summary, methods)
    _plot_comparison_radar(summary, methods, args.output_dir)
    _plot_comparison_bars(summary, methods, args.output_dir)


def _print_comparison_table(summary, methods):
    key_metrics = ["goal_error", "isr", "cumulative_cot", "risk_integral",
                   "mean_slope", "path_length_m", "infeasible_rate", "inference_time"]
    header = f"{'Method':30s}" + "".join(f"{m:>14s}" for m in key_metrics)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for method in methods:
        row = f"{method:30s}"
        for mk in key_metrics:
            val = summary[method].get(mk, {}).get("mean", float("nan"))
            row += f"{val:14.4f}"
        print(row)
    print("=" * len(header))


def _plot_comparison_bars(summary, methods, output_dir):
    key_metrics = ["goal_error", "isr", "cumulative_cot", "risk_integral",
                   "mean_slope", "path_length_m"]
    labels = ["Goal Error", "ISR", "Cum. CoT", "Risk", "Mean Slope (°)",
              "Path Length (m)"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    x = np.arange(len(methods))
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    for ax, metric, label in zip(axes, key_metrics, labels):
        means = [summary[m].get(metric, {}).get("mean", 0) for m in methods]
        stds = [summary[m].get(metric, {}).get("std", 0) for m in methods]
        ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in methods],
                           fontsize=7, rotation=30, ha="right")
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=10)

    fig.suptitle("External Comparison — Baselines vs Diffusion Planner", fontsize=13)
    fig.tight_layout()
    out = os.path.join(output_dir, "baseline_comparison_bars.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Bar chart → {out}")


def _plot_comparison_radar(summary, methods, output_dir):
    """Radar / spider chart normalised to [0,1] across methods."""
    metrics = ["goal_error", "isr", "cumulative_cot", "risk_integral", "mean_slope"]
    directions = ["min", "max", "min", "min", "min"]
    labels = ["Goal Err", "ISR", "Cum. CoT", "Risk", "Slope"]

    raw = np.zeros((len(methods), len(metrics)))
    for i, m in enumerate(methods):
        for j, mk in enumerate(metrics):
            raw[i, j] = summary[m].get(mk, {}).get("mean", 0)

    normed = np.zeros_like(raw)
    for j in range(len(metrics)):
        col = raw[:, j]
        lo, hi = col.min(), col.max()
        if hi - lo < 1e-12:
            normed[:, j] = 1.0
        else:
            normed[:, j] = (col - lo) / (hi - lo)
            if directions[j] == "min":
                normed[:, j] = 1.0 - normed[:, j]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10.colors
    for i, m in enumerate(methods):
        vals = normed[i].tolist() + [normed[i, 0]]
        ax.plot(angles, vals, "-o", color=colors[i % len(colors)],
                lw=2, markersize=5, label=m)
        ax.fill(angles, vals, alpha=0.1, color=colors[i % len(colors)])

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax.set_title("Method Comparison (higher = better)", fontsize=11, pad=20)
    fig.tight_layout()
    out = os.path.join(output_dir, "baseline_comparison_radar.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Radar chart → {out}")


def main():
    ap = argparse.ArgumentParser(description="Exp 3: External comparison")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data-dir", type=str, default=str(_ROOT / "data" / "raw"))
    ap.add_argument("--output-dir", type=str, default=str(_ROOT / "results" / "baselines"))
    ap.add_argument("--max-terrains", type=int, default=50)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    run_comparison(args)


if __name__ == "__main__":
    main()
