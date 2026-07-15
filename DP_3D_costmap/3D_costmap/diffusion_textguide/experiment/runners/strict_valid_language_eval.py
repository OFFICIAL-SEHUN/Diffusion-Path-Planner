"""Strict valid-language evaluation over all validation instruction templates."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

_EXP_DIR = Path(__file__).resolve().parent
_ROOT = _EXP_DIR.parents[1]
sys.path.insert(0, str(_ROOT))

from experiment.core.metrics import chamfer_distance
from experiment.core.utils import load_terrain
from experiment.evaluators.text_encoder_ablation import _load_model
from experiment.runners.independent_learned_baselines import (
    load_bc_checkpoint,
    load_cvae_checkpoint,
    text_bank_lookup,
)
from experiment.runners.learned_baselines import (
    aggregate_records,
    evaluate_prediction,
    fmt,
    norm_start_goal,
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


def load_cache(cache_dir: Path, split: str) -> dict[str, Any]:
    return torch.load(cache_dir / f"{split}_independent_samples.pt", map_location="cpu", weights_only=False)


def build_strict_refs(valid_cache: dict[str, Any], valid_dir: Path, seed: int, num_terrains: int) -> list[dict[str, Any]]:
    templates = load_instruction_templates("valid")
    base_indices = selected_eval_indices(valid_cache, valid_dir, seed, num_terrains)
    refs: list[dict[str, Any]] = []
    for base_idx in base_indices:
        meta = dict(valid_cache["metadata"][base_idx])
        prompts = templates.get(meta["intent_type"], [meta.get("instruction", "")])
        for prompt_idx, instruction in enumerate(prompts):
            ref = dict(meta)
            ref["base_idx"] = int(base_idx)
            ref["prompt_idx"] = int(prompt_idx)
            ref["instruction"] = str(instruction)
            refs.append(ref)
    return refs


def batch_inputs(valid_cache: dict[str, Any], refs: list[dict[str, Any]], text_by_instr: dict[str, torch.Tensor], start: int, end: int):
    subset = refs[start:end]
    base_indices = torch.tensor([r["base_idx"] for r in subset], dtype=torch.long)
    terrain_indices = valid_cache["terrain_indices"][base_indices].long()
    costmaps = valid_cache["costmaps"][terrain_indices].float()
    text = torch.stack([text_by_instr[r["instruction"]].float() for r in subset]).float()
    start_goal = valid_cache["start_goal"][base_indices].float()
    return costmaps, text, start_goal, subset


@torch.inference_mode()
def eval_independent(args: argparse.Namespace, refs: list[dict[str, Any]], valid_cache: dict[str, Any], device: torch.device) -> dict[str, Any]:
    text_by_instr = text_bank_lookup(valid_cache)
    terrain_cache: dict[str, dict] = {}
    results: dict[str, Any] = {}

    if args.bc_checkpoint:
        bc = load_bc_checkpoint(resolve(args.bc_checkpoint), device)
        records = []
        t0 = time.perf_counter()
        for start in tqdm(range(0, len(refs), args.eval_batch_size), desc="Strict BC"):
            costmaps, text, start_goal, subset = batch_inputs(valid_cache, refs, text_by_instr, start, start + args.eval_batch_size)
            pred = bc(costmaps.to(device), text.to(device), start_goal.to(device)).cpu().numpy()
            latency = (time.perf_counter() - t0) / max(start + len(pred), 1)
            for j, path in enumerate(pred):
                rec = evaluate_prediction(path, subset[j], terrain_cache)
                rec.update({
                    "method": "bc_independent_strict_valid_language",
                    "method_label": "BC Planner (independent)",
                    "sample_index": 0,
                    "inference_time": latency,
                })
                records.append(rec)
        summary = aggregate_records(records)
        summary["best_of_k_isr"] = summary["by_metric"]["isr"]["mean"]
        summary["diversity"] = 0.0
        summary["checkpoint"] = str(resolve(args.bc_checkpoint))
        results["bc"] = {"summary": summary, "records": records}

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
            for start in tqdm(range(0, len(refs), args.eval_batch_size), desc=f"Strict CVAE k={k}"):
                costmaps, text, start_goal, subset = batch_inputs(valid_cache, refs, text_by_instr, start, start + args.eval_batch_size)
                pred = cvae.sample(costmaps.to(device), text.to(device), start_goal.to(device)).cpu().numpy()
                for j, path in enumerate(pred):
                    ref_idx = start + j
                    total += 1
                    rec = evaluate_prediction(path, subset[j], terrain_cache)
                    rec.update({
                        "method": "cvae_independent_strict_valid_language",
                        "method_label": "CVAE Planner (independent)",
                        "sample_index": k,
                        "inference_time": (time.perf_counter() - t0) / max(total, 1),
                    })
                    records.append(rec)
                    paths_by_ref[ref_idx].append(path)
                    isr_by_ref[ref_idx].append(float(rec["isr"]))
        summary = aggregate_records(records)
        summary["best_of_k_isr"] = float(np.mean([np.max(v) for v in isr_by_ref.values()]))
        div = []
        for plist in paths_by_ref.values():
            if len(plist) >= 2:
                div.append(float(np.mean([chamfer_distance(a, b) for a, b in itertools.combinations(plist, 2)])))
        summary["diversity"] = float(np.mean(div)) if div else float("nan")
        summary["checkpoint"] = str(resolve(args.cvae_checkpoint))
        results["cvae"] = {"summary": summary, "records": records}

    return results


@torch.inference_mode()
def eval_ours(args: argparse.Namespace, refs: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    ckpt_path = resolve(args.ours_checkpoint)
    model, scheduler, _, config, text_encoder_type, _ = _load_model(ckpt_path, device)
    model.eval()
    text_cfg = config.get("model", {}).get("text_encoder", {})
    feature_encoder = FrozenTextFeatureEncoder(
        text_encoder_type,
        model_name=text_cfg.get("model_name"),
        device=device,
        batch_size=args.text_batch_size,
    )
    unique_prompts = list(dict.fromkeys([r["instruction"] for r in refs]))
    raw_by_instr: dict[str, torch.Tensor] = {}
    for start in tqdm(range(0, len(unique_prompts), args.text_batch_size), desc="Encode Ours prompts"):
        batch = unique_prompts[start:start + args.text_batch_size]
        raw = feature_encoder.encode(batch).cpu().float()
        for instruction, vec in zip(batch, raw):
            raw_by_instr[instruction] = vec

    refs_by_terrain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ref in refs:
        refs_by_terrain[ref["terrain_path"]].append(ref)

    records = []
    paths_by_ref: dict[int, list[np.ndarray]] = defaultdict(list)
    isr_by_ref: dict[int, list[float]] = defaultdict(list)
    terrain_cache: dict[str, dict] = {}
    total = 0
    t0 = time.perf_counter()
    global_ref = 0
    terrain_items = sorted(refs_by_terrain.items())

    for terrain_i, (terrain_path, terrain_refs) in enumerate(tqdm(terrain_items, desc="Strict Ours terrain"), start=1):
        terrain = load_terrain(terrain_path)
        terrain_cache[terrain_path] = terrain
        img_size = int(terrain["img_size"])
        horizon = int(terrain.get("horizon", terrain["paths"].shape[1]))
        s_norm, g_norm = norm_start_goal(terrain)
        costmap_base = torch.from_numpy(terrain["costmap"]).float()
        terrain_refs = sorted(terrain_refs, key=lambda r: (int(r["prompt_idx"]), int(r["path_idx"])))
        ref_start = global_ref
        global_ref += len(terrain_refs)

        for k in range(args.eval_k):
            torch.manual_seed(args.eval_seed + k * 1_000_003 + terrain_i)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.eval_seed + k * 1_000_003 + terrain_i)
            for chunk_start in range(0, len(terrain_refs), args.ours_batch_size):
                chunk = terrain_refs[chunk_start:chunk_start + args.ours_batch_size]
                bsz = len(chunk)
                costmap = costmap_base.unsqueeze(0).repeat(bsz, 1, 1, 1).to(device)
                start_pos = torch.from_numpy(s_norm).float().repeat(bsz, 1).to(device)
                end_pos = torch.from_numpy(g_norm).float().repeat(bsz, 1).to(device)
                text_features = torch.stack([raw_by_instr[r["instruction"]] for r in chunk]).to(device)
                pred = scheduler.sample(
                    model,
                    costmap,
                    shape=(bsz, horizon, 2),
                    start_pos=start_pos,
                    end_pos=end_pos,
                    text_features=text_features,
                    show_progress=False,
                ).cpu().numpy()
                for j, path in enumerate(pred):
                    ref_idx = ref_start + chunk_start + j
                    total += 1
                    rec = evaluate_prediction(path, chunk[j], terrain_cache)
                    rec.update({
                        "method": "ours_strict_valid_language",
                        "method_label": "Ours (ResNet18+T5)",
                        "sample_index": k,
                        "inference_time": (time.perf_counter() - t0) / max(total, 1),
                    })
                    records.append(rec)
                    paths_by_ref[ref_idx].append(path)
                    isr_by_ref[ref_idx].append(float(rec["isr"]))
        if terrain_i == 1 or terrain_i % 10 == 0:
            print(f"[{now_kst()}] Ours terrain {terrain_i}/{len(terrain_items)}")

    summary = aggregate_records(records)
    summary["best_of_k_isr"] = float(np.mean([np.max(v) for v in isr_by_ref.values()]))
    div = []
    for plist in paths_by_ref.values():
        if len(plist) >= 2:
            div.append(float(np.mean([chamfer_distance(a, b) for a, b in itertools.combinations(plist, 2)])))
    summary["diversity"] = float(np.mean(div)) if div else float("nan")
    summary["checkpoint"] = str(ckpt_path)
    return {"summary": summary, "records": records}


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def row_from_astar(astar: dict[str, Any], key: str) -> str:
    d = astar["summary"][key]
    s = d["by_metric"]
    return f"| {d['label']} | {fmt(s['success']['mean'])} | {fmt(s['feasible']['mean'])} | {fmt(s['cumulative_cot']['mean'], 2)} | {fmt(s['risk_integral']['mean'], 2)} | {fmt(s['isr']['mean'])} | {fmt(s['inference_time']['mean'], 4)} |"


def overall_row(summary: dict[str, Any], label: str) -> str:
    s = summary["by_metric"]
    return f"| {label} | {fmt(s['success']['mean'])} | {fmt(s['feasible']['mean'])} | {fmt(s['cumulative_cot']['mean'], 2)} | {fmt(s['risk_integral']['mean'], 2)} | {fmt(s['isr']['mean'])} | {fmt(s['inference_time']['mean'], 4)} |"


def intent_row(summary: dict[str, Any], label: str) -> str:
    iw = summary["intentwise"]
    return f"| {label} | {fmt(iw['left_bias_score'])} | {fmt(iw['right_bias_score'])} | {fmt(iw['avoid_steep_isr'])} | {fmt(iw['prefer_flat_isr'])} | {fmt(iw['composite_isr'])} |"


def fidelity_row(summary: dict[str, Any], label: str) -> str:
    s = summary["by_metric"]
    return f"| {label} | {fmt(s['cost_gap']['mean'])} | {fmt(s['chamfer']['mean'])} | {fmt(s['frechet']['mean'])} | {fmt(summary['best_of_k_isr'])} | {fmt(summary['diversity'])} |"


def write_report(args: argparse.Namespace, payload: dict[str, Any], summary_path: Path, md_path: Path) -> None:
    astar = load_json(resolve(args.astar_summary)) if args.astar_summary else load_json(_ROOT / "results/baselines/valid_astar_seed42_20260519_003012/astar_baseline_summary.json")
    bc = payload["summaries"].get("bc")
    cvae = payload["summaries"].get("cvae")
    ours = payload["summaries"].get("ours")
    lines = [
        f"# Strict Valid-Language Evaluation Log ({payload['stamp']})",
        "",
        "## Run log",
        "",
        f"- Created: {payload['created_kst']}",
        f"- Summary JSON: `{summary_path}`",
        "- Instruction split: `data/instruction/valid/inst_valid.json`",
        f"- Protocol: all valid prompts per intent, {payload['eval_terrains']} terrains x 10 intents x 10 prompts = {payload['refs']} refs",
        f"- K: {payload['eval_k']} stochastic samples for CVAE/Ours",
        "- BC/CVAE protocol: independent raw costmap encoder + trainable text projection; no diffusion checkpoint encoder",
        f"- BC checkpoint: `{args.bc_checkpoint}`",
        f"- CVAE checkpoint: `{args.cvae_checkpoint}`",
        f"- Ours checkpoint: `{args.ours_checkpoint}`",
        f"- Elapsed: {payload['elapsed_seconds'] / 60:.2f} min",
        "",
        "## Table. Overall performance comparison",
        "",
        "| Method | Success ↑ | Feasible ↑ | CoT ↓ | Risk ↓ | ISR ↑ | Time ↓ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if astar:
        for key in ["vanilla_astar", "slope_aware_astar", "cot_aware_astar", "intent_aware_astar"]:
            lines.append(row_from_astar(astar, key))
    if bc:
        lines.append(overall_row(bc, "BC Planner (independent)"))
    if cvae:
        lines.append(overall_row(cvae, "CVAE Planner (independent)"))
    if ours:
        lines.append(overall_row(ours, "Ours (ResNet18+T5)"))
    lines.extend([
        "",
        "## Table. Intent-wise comparison",
        "",
        "| Method | Left-bias score ↑ | Right-bias score ↑ | Avoid-steep ISR ↑ | Prefer-flat ISR ↑ | Composite ISR ↑ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    if astar:
        iw = astar["intentwise"]["intent_aware_astar"]
        lines.append(f"| Intent-aware A* | {fmt(iw['left_bias_score'])} | {fmt(iw['right_bias_score'])} | {fmt(iw['avoid_steep_isr'])} | {fmt(iw['prefer_flat_isr'])} | {fmt(iw['composite_isr'])} |")
    if bc:
        lines.append(intent_row(bc, "BC Planner (independent)"))
    if cvae:
        lines.append(intent_row(cvae, "CVAE Planner (independent)"))
    if ours:
        lines.append(intent_row(ours, "Ours (ResNet18+T5)"))
    lines.extend([
        "",
        "## Table. Student-to-teacher fidelity and generation properties",
        "",
        "| Method | Cost Gap ↓ | Chamfer ↓ | Frechet ↓ | Best-of-K ISR ↑ | Diversity ↑ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    if bc:
        lines.append(fidelity_row(bc, "BC Planner (independent)"))
    if cvae:
        lines.append(fidelity_row(cvae, "CVAE Planner (independent)"))
    if ours:
        lines.append(fidelity_row(ours, "Ours (ResNet18+T5)"))
    lines.extend([
        "",
        "## Notes",
        "",
        "- This is the strict language-unseen check: each validation terrain-intent pair is evaluated with every template in `inst_valid.json`.",
        "- Classical A* rows are unchanged because A* uses symbolic intent, not language text.",
        "- BC is deterministic, so Best-of-K ISR equals single-sample ISR and Diversity is 0.",
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--valid-dir", default="data/valid")
    p.add_argument("--cache-dir", default="results/baselines/independent_feature_cache_t5")
    p.add_argument("--output-dir", default="results/baselines")
    p.add_argument("--bc-checkpoint", default="results/baselines/independent_learned_baselines_20260519_154801/bc_independent_best.pt")
    p.add_argument("--cvae-checkpoint", default="results/baselines/independent_learned_baselines_20260519_154801/cvae_independent_best.pt")
    p.add_argument("--ours-checkpoint", default="checkpoints/backbone_ablation_10intent/final_model.pt")
    p.add_argument("--astar-summary", default="results/baselines/valid_astar_seed42_20260519_003012/astar_baseline_summary.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-seed", type=int, default=42)
    p.add_argument("--eval-terrains", type=int, default=100)
    p.add_argument("--eval-k", type=int, default=3)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--ours-batch-size", type=int, default=50)
    p.add_argument("--text-batch-size", type=int, default=128)
    p.add_argument("--skip-ours", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    out_dir = resolve(args.output_dir) / f"strict_valid_language_eval_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "strict_valid_language_summary.json"
    md_path = _ROOT / "experiment" / f"external_comparison_strict_valid_language_log_{stamp}.md"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()

    print(f"[{now_kst()}] Load cache and build strict refs")
    valid_cache = load_cache(resolve(args.cache_dir), "valid")
    refs = build_strict_refs(valid_cache, resolve(args.valid_dir), args.eval_seed, args.eval_terrains)
    print(f"[{now_kst()}] refs={len(refs)} unique_prompts={len({r['instruction'] for r in refs})}")

    result_blocks = eval_independent(args, refs, valid_cache, device)
    if not args.skip_ours:
        result_blocks["ours"] = eval_ours(args, refs, device)

    for key, block in result_blocks.items():
        write_jsonl(out_dir / f"{key}_records.jsonl", block["records"])

    payload = {
        "stamp": stamp,
        "created_kst": now_kst(),
        "eval_seed": args.eval_seed,
        "eval_terrains": args.eval_terrains,
        "eval_k": args.eval_k,
        "refs": len(refs),
        "unique_prompts": len({r["instruction"] for r in refs}),
        "valid_dir": str(resolve(args.valid_dir)),
        "instruction_file": str(_ROOT / "data/instruction/valid/inst_valid.json"),
        "elapsed_seconds": time.perf_counter() - t0,
        "summaries": {k: v["summary"] for k, v in result_blocks.items()},
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    write_report(args, payload, summary_path, md_path)
    print(f"[{now_kst()}] Done")
    print(f"JSON {summary_path}")
    print(f"MD {md_path}")


if __name__ == "__main__":
    main()
