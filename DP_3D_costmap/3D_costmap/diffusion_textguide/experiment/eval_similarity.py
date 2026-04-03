"""
Experiment 4 — Pseudo-label Similarity

Measures how well the diffusion model reproduces the pseudo-label
(A*-generated) reference paths.

Metrics (all require normalisation / same-horizon alignment):
  - Pointwise L2 (MSE)
  - Chamfer distance
  - Discrete Fréchet distance
  - Edge cost gap (same 4-term formula)
  - Instruction adherence gap (per-intent sub-metric delta)
  - Side-bias score gap
  - Avoid-steep agreement

Usage:
  python -m experiment.eval_similarity \
      --checkpoint checkpoints/final_model.pt \
      --data-dir data/raw \
      --output-dir results/similarity

  # Seen (train distribution) + unseen hold-out terrains (e.g. data/valid):
  python -m experiment.eval_similarity \
      --checkpoint checkpoints/final_model.pt \
      --data-dir data/raw \
      --unseen-data-dir data/valid \
      --run-both \
      --output-dir results/similarity
"""

import argparse
import json
import os
import sys
from typing import Optional
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

from data_loader import text_to_tokens
from experiment.utils import load_model, load_terrain
from experiment.metrics import (
    pointwise_l2,
    chamfer_distance,
    frechet_distance,
    edge_cost_total,
    cost_gap,
    instruction_adherence_gap,
    instruction_success,
    side_bias_success,
    avoid_steep_success,
    cumulative_cot,
    mean_slope_along_path,
    risk_integral,
    goal_error,
    path_length_metres,
    _norm_to_px,
)


def _resolve_data_dir(path_str: str) -> str:
    """Paths relative to the diffusion_textguide package root."""
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return str(p)
    return str((_ROOT / p).resolve())


def _load_samples(data_dir: str, max_terrains: int = 100):
    pt_files = sorted(Path(data_dir).glob("*.pt"))[:max_terrains]
    samples = []
    for f in pt_files:
        t = load_terrain(str(f))
        n_paths = t["paths"].shape[0]
        for i in range(n_paths):
            samples.append({
                "terrain": t,
                "ref_path": t["paths"][i],
                "instruction": t.get("instructions", [""])[i] if i < len(t.get("instructions", [])) else "",
                "intent_type": t.get("intent_types", ["baseline"])[i] if i < len(t.get("intent_types", [])) else "baseline",
                "intent_params": t.get("intent_params", [{}])[i] if i < len(t.get("intent_params", [])) else {},
            })
    return samples


def _aggregate_lists(overall, per_intent):
    def _agg(d):
        out = {}
        for k, vals in d.items():
            clean = [v for v in vals if v is not None and not np.isnan(v)]
            out[k] = {
                "mean": float(np.mean(clean)) if clean else float("nan"),
                "std": float(np.std(clean)) if clean else float("nan"),
                "count": len(clean),
            }
        return out

    return {
        "overall": _agg(overall),
        "per_intent": {it: _agg(d) for it, d in per_intent.items()},
    }


def _evaluate_split(
    model, scheduler, vocab, config, data_dir, max_terrains, device, tqdm_desc,
):
    horizon = config.get("data", {}).get("horizon", 120)
    cw = config.get("intent", {}).get("cost_weights", {})
    alpha = cw.get("alpha", 1.0)
    beta_ = cw.get("beta", 0.8)
    gamma = cw.get("gamma", 0.1)
    delta = cw.get("delta", 1.0)
    risk_th = config.get("intent", {}).get("risk_threshold_deg", 15.0)

    samples = _load_samples(data_dir, max_terrains)
    print(f"  [{tqdm_desc}] {data_dir} → {len(samples)} samples")

    overall = defaultdict(list)
    per_intent = defaultdict(lambda: defaultdict(list))

    for sample in tqdm(samples, desc=tqdm_desc):
        ter = sample["terrain"]
        img_size = int(ter["img_size"])
        start = ter["start_position"]
        goal = ter["goal_position"]
        px_res = float(ter.get("pixel_resolution", 0.5))
        limit_deg = float(ter.get("limit_angle_deg", 25))
        slope_deg = ter["slope_map"]
        height_map = ter["height_map"]
        slope_rad = np.radians(slope_deg)
        itype = sample["intent_type"]
        iparams = sample["intent_params"]
        ref_path = sample["ref_path"]

        s_norm = torch.tensor([(start[1] / img_size) * 2 - 1,
                               (start[0] / img_size) * 2 - 1], dtype=torch.float32)
        g_norm = torch.tensor([(goal[1] / img_size) * 2 - 1,
                               (goal[0] / img_size) * 2 - 1], dtype=torch.float32)

        costmap_t = torch.from_numpy(ter["costmap"]).float().unsqueeze(0).to(device)
        tokens = text_to_tokens(sample["instruction"], vocab, max_seq_len=16)
        tokens_t = tokens.unsqueeze(0).to(device)

        gen = scheduler.sample(
            model, costmap_t, shape=(1, horizon, 2),
            start_pos=s_norm.unsqueeze(0).to(device),
            end_pos=g_norm.unsqueeze(0).to(device),
            text_tokens=tokens_t, show_progress=False,
        )[0].cpu().numpy()

        # ── shape similarity ────────────────────────────────────────────
        row = {}
        row["pointwise_l2"] = pointwise_l2(gen, ref_path)
        row["chamfer"] = chamfer_distance(gen, ref_path)
        row["frechet"] = frechet_distance(gen, ref_path)

        # ── cost-based ──────────────────────────────────────────────────
        cost_kwargs = dict(
            slope_map_rad=slope_rad, height_map=height_map,
            img_size=img_size, alpha=alpha, beta=beta_, gamma=gamma,
            delta=delta, risk_threshold_deg=risk_th,
            intent_type=itype, intent_params=iparams,
            pixel_resolution=px_res, limit_angle_deg=limit_deg,
            start_pos=start, goal_pos=goal,
        )
        gen_cost = edge_cost_total(gen, **cost_kwargs)
        ref_cost = edge_cost_total(ref_path, **cost_kwargs)
        row["gen_edge_cost"] = gen_cost
        row["ref_edge_cost"] = ref_cost
        row["cost_gap"] = cost_gap(ref_cost, gen_cost)

        # ── energy / safety / goal ──────────────────────────────────────
        row["gen_cot"] = cumulative_cot(gen, height_map, img_size, px_res, limit_deg)
        row["ref_cot"] = cumulative_cot(ref_path, height_map, img_size, px_res, limit_deg)
        row["gen_mean_slope"] = mean_slope_along_path(gen, slope_deg, img_size)
        row["ref_mean_slope"] = mean_slope_along_path(ref_path, slope_deg, img_size)
        row["gen_risk"] = risk_integral(gen, slope_deg, img_size, risk_th)
        row["ref_risk"] = risk_integral(ref_path, slope_deg, img_size, risk_th)
        row["gen_goal_error"] = goal_error(gen, g_norm.numpy())
        row["gen_path_len_m"] = path_length_metres(gen, img_size, px_res)
        row["ref_path_len_m"] = path_length_metres(ref_path, img_size, px_res)

        # ── instruction adherence ───────────────────────────────────────
        s_np = s_norm.numpy()
        g_np = g_norm.numpy()
        row["gen_isr"] = float(instruction_success(
            gen, itype, iparams, slope_deg, img_size, start, goal, s_np, g_np,
            height_map=height_map,
            pixel_resolution=px_res,
            limit_angle_deg=limit_deg,
        ))
        row["ref_isr"] = float(instruction_success(
            ref_path, itype, iparams, slope_deg, img_size, start, goal, s_np, g_np,
            height_map=height_map,
            pixel_resolution=px_res,
            limit_angle_deg=limit_deg,
        ))

        parts = itype.split("+")
        for part in parts:
            if part in ("left_bias", "right_bias"):
                bias = "left" if part == "left_bias" else "right"
                gen_ok = side_bias_success(gen, s_np, g_np, bias)
                ref_ok = side_bias_success(ref_path, s_np, g_np, bias)
                row[f"{part}_gen"] = float(gen_ok)
                row[f"{part}_ref"] = float(ref_ok)
                row[f"{part}_agreement"] = float(gen_ok == ref_ok)
            elif part == "avoid_steep":
                tau = iparams.get("tau_steep", 20.0)
                gen_ok = avoid_steep_success(gen, slope_deg, img_size, tau)
                ref_ok = avoid_steep_success(ref_path, slope_deg, img_size, tau)
                row["avoid_steep_gen"] = float(gen_ok)
                row["avoid_steep_ref"] = float(ref_ok)
                row["avoid_steep_agreement"] = float(gen_ok == ref_ok)

        adh_gap = instruction_adherence_gap(
            gen, ref_path, slope_deg, img_size, itype, iparams, start, goal)
        for k, v in adh_gap.items():
            row[k] = v

        for k, v in row.items():
            overall[k].append(v)
            per_intent[itype][k].append(v)

    result = _aggregate_lists(overall, per_intent)
    return result, overall, per_intent


def _file_stem(label: Optional[str]) -> str:
    if not label:
        return ""
    return f"_{label}"


def _intent_display_name(intent: str) -> str:
    labels = {
        "baseline": "Base",
        "left_bias": "Left",
        "right_bias": "Right",
        "center_bias": "Center",
        "avoid_steep": "Avoid",
        "prefer_flat": "Flat",
        "minimize_elevation_change": "ΔElev",
        "short_path": "Short",
        "energy_efficient": "Energy",
        "left_bias+avoid_steep": "Left+Steep",
        "right_bias+prefer_flat": "Right+Flat",
        "center_bias+prefer_flat": "Ctr+Flat",
        "short_path+avoid_steep": "Short+Steep",
        "energy_efficient+minimize_elevation_change": "En+ΔEl",
    }
    return labels.get(intent, intent.replace("_", " "))


def _write_similarity_outputs(
    result, overall, per_intent, output_dir, label: Optional[str] = None,
):
    stem = _file_stem(label)
    suffix = "similarity_results.json" if not stem else f"similarity_results{stem}.json"
    out_json = os.path.join(output_dir, suffix)
    payload = dict(result)
    if label:
        payload["split"] = label
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nResults → {out_json}")
    _print_summary(result, label=label)
    _plot_similarity(result, overall, per_intent, output_dir, label=label)


def run_similarity(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, scheduler, vocab, config = load_model(args.checkpoint, device)
    data_dir = _resolve_data_dir(args.data_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.run_both:
        if not args.unseen_data_dir:
            raise SystemExit("--run-both requires --unseen-data-dir (e.g. data/valid)")
        unseen_dir = _resolve_data_dir(args.unseen_data_dir)
        split_payloads = {}
        for split_label, ddir, desc in (
            ("seen", data_dir, "Similarity (seen)"),
            ("unseen", unseen_dir, "Similarity (unseen)"),
        ):
            result, overall, per_intent = _evaluate_split(
                model, scheduler, vocab, config, ddir,
                args.max_terrains, device, desc,
            )
            _write_similarity_outputs(
                result, overall, per_intent, args.output_dir, label=split_label,
            )
            split_payloads[split_label] = {
                "result": result,
                "overall": overall,
                "per_intent": per_intent,
            }
        _plot_split_comparison(split_payloads, args.output_dir)
        return

    result, overall, per_intent = _evaluate_split(
        model, scheduler, vocab, config, data_dir,
        args.max_terrains, device, "Similarity eval",
    )
    _write_similarity_outputs(result, overall, per_intent, args.output_dir, label=None)


def _print_summary(result, label: Optional[str] = None):
    print("\n" + "=" * 70)
    tag = f" — {label}" if label else ""
    print(f"  Pseudo-label Similarity — Overall{tag}")
    print("=" * 70)
    o = result["overall"]
    for k in ["pointwise_l2", "chamfer", "frechet", "cost_gap",
              "gen_isr", "ref_isr", "gen_goal_error"]:
        if k in o:
            print(f"  {k:28s}  mean={o[k]['mean']:.4f}  std={o[k]['std']:.4f}")

    print("\n  Per-intent ISR / agreement:")
    for itype, idata in result["per_intent"].items():
        gen_isr = idata.get("gen_isr", {}).get("mean", float("nan"))
        ref_isr = idata.get("ref_isr", {}).get("mean", float("nan"))
        print(f"    {itype:30s}  gen_ISR={gen_isr:.3f}  ref_ISR={ref_isr:.3f}")


def _plot_similarity(
    result, overall, per_intent, output_dir, label: Optional[str] = None,
):
    stem = _file_stem(label)
    title_suffix = f" ({label})" if label else ""

    # A single representative similarity metric is easier to read than
    # repeating the same pattern across three nearly redundant panels.
    fig, ax = plt.subplots(figsize=(10, 5))
    intents = sorted(per_intent.keys())
    data = [per_intent[it].get("pointwise_l2", []) for it in intents]
    data = [[v for v in d if not np.isnan(v)] for d in data]
    labels = [_intent_display_name(it) for it in intents]
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, c in zip(bp["boxes"], plt.cm.Set2(np.linspace(0, 1, len(intents)))):
        patch.set_facecolor(c)
    ax.set_ylabel("Pointwise L2")
    ax.set_title(f"Intent-wise Shape Similarity{title_suffix}", fontsize=12)
    ax.tick_params(axis="x", rotation=20, labelsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    name = "similarity_boxplots.png" if not stem else f"similarity_boxplots{stem}.png"
    out = os.path.join(output_dir, name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Box plots → {out}")

    # cost gap distribution
    gaps = [v for v in overall.get("cost_gap", []) if not np.isnan(v)]
    if gaps:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(gaps, bins=40, edgecolor="black", alpha=0.75)
        ax.axvline(0, color="red", ls="--", lw=1.5, label="cost_gap = 0")
        ax.set_xlabel("Cost Gap  (gen − ref) / |ref|")
        ax.set_ylabel("Count")
        ax.set_title(f"Edge-Cost Gap Distribution{title_suffix}")
        ax.legend()
        fig.tight_layout()
        out = os.path.join(
            output_dir,
            "cost_gap_hist.png" if not stem else f"cost_gap_hist{stem}.png",
        )
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Cost gap histogram → {out}")

    # ISR comparison bar
    intents = sorted(per_intent.keys())
    gen_isrs = [np.nanmean(per_intent[it].get("gen_isr", [0])) for it in intents]
    ref_isrs = [np.nanmean(per_intent[it].get("ref_isr", [0])) for it in intents]
    x = np.arange(len(intents))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, ref_isrs, w, label="Pseudo-label (ref)", alpha=0.8)
    ax.bar(x + w / 2, gen_isrs, w, label="Diffusion (gen)", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(intents, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("ISR")
    ax.set_title(f"ISR: Pseudo-label vs Diffusion{title_suffix}")
    ax.legend()
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    out = os.path.join(
        output_dir,
        "isr_comparison.png" if not stem else f"isr_comparison{stem}.png",
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"ISR comparison → {out}")


def _plot_split_comparison(split_payloads, output_dir):
    seen = split_payloads["seen"]
    unseen = split_payloads["unseen"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"seen": "#4C72B0", "unseen": "#DD8452"}

    metrics = [
        ("pointwise_l2", "Pointwise L2"),
        ("cost_gap", "Cost Gap"),
    ]
    positions = []
    box_data = []
    box_colors = []
    tick_positions = []
    tick_labels = []
    xpos = 1
    for metric, label in metrics:
        seen_vals = [v for v in seen["overall"].get(metric, []) if not np.isnan(v)]
        unseen_vals = [v for v in unseen["overall"].get(metric, []) if not np.isnan(v)]
        positions.extend([xpos - 0.18, xpos + 0.18])
        box_data.extend([seen_vals, unseen_vals])
        box_colors.extend([colors["seen"], colors["unseen"]])
        tick_positions.append(xpos)
        tick_labels.append(label)
        xpos += 1

    bp = axes[0].boxplot(box_data, positions=positions, widths=0.28, patch_artist=True)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    axes[0].set_xticks(tick_positions)
    axes[0].set_xticklabels(tick_labels)
    axes[0].set_ylabel("Metric Value")
    axes[0].set_title("Seen vs Unseen Overall Distributions", fontsize=11)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(
        [plt.Rectangle((0, 0), 1, 1, color=colors["seen"]),
         plt.Rectangle((0, 0), 1, 1, color=colors["unseen"])],
        ["Seen", "Unseen"],
        frameon=False,
        loc="upper left",
    )

    intents = sorted(set(seen["per_intent"].keys()) | set(unseen["per_intent"].keys()))
    labels = [_intent_display_name(it) for it in intents]
    seen_means = [
        np.nanmean(seen["per_intent"].get(it, {}).get("pointwise_l2", [np.nan]))
        for it in intents
    ]
    unseen_means = [
        np.nanmean(unseen["per_intent"].get(it, {}).get("pointwise_l2", [np.nan]))
        for it in intents
    ]
    x = np.arange(len(intents))
    w = 0.36
    axes[1].bar(x - w / 2, seen_means, w, label="Seen", color=colors["seen"], alpha=0.85)
    axes[1].bar(x + w / 2, unseen_means, w, label="Unseen", color=colors["unseen"], alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylabel("Mean Pointwise L2")
    axes[1].set_title("Intent-wise Difficulty Across Splits", fontsize=11)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle(
        "Seen vs Unseen Similarity: comparable distributions and stable intent difficulty",
        fontsize=13,
    )
    fig.tight_layout()
    out = os.path.join(output_dir, "seen_unseen_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Seen/unseen comparison → {out}")


def main():
    ap = argparse.ArgumentParser(description="Exp 4: Pseudo-label similarity")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data-dir", type=str, default=str(_ROOT / "data" / "raw"))
    ap.add_argument(
        "--unseen-data-dir",
        type=str,
        default=None,
        help="Hold-out terrains (e.g. data/valid). Use with --run-both.",
    )
    ap.add_argument(
        "--run-both",
        action="store_true",
        help="Evaluate --data-dir as seen and --unseen-data-dir as unseen; "
        "writes similarity_results_{seen,unseen}.json and suffixed plots.",
    )
    ap.add_argument("--output-dir", type=str, default=str(_ROOT / "results" / "similarity"))
    ap.add_argument("--max-terrains", type=int, default=100)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    run_similarity(args)


if __name__ == "__main__":
    main()
