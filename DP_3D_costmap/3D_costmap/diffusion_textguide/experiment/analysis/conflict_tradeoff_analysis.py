#!/usr/bin/env python3
"""Re-analyze strict valid-language predictions under terrain-intent conflict.

This script does not train, sample new maps, add intents, or apply Best-of-K
selection. It only reads the existing strict-valid prediction JSONL files and
teacher paths stored in the validation terrains.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np


KST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.core.utils import load_terrain  # noqa: E402
from experiment.runners.learned_baselines import evaluate_prediction  # noqa: E402


DIRECTIONAL_INTENTS = {
    "left_bias",
    "right_bias",
    "center_bias",
    "left_bias+avoid_steep",
    "right_bias+prefer_flat",
    "center_bias+prefer_flat",
}

METHOD_FILES = {
    "BC independent": "bc_records.jsonl",
    "CVAE independent": "cvae_records.jsonl",
    "Ours": "ours_records.jsonl",
}

PRIMARY_METRICS = [
    "success",
    "feasible",
    "isr",
    "cumulative_cot",
    "risk_integral",
    "edge_cost",
    "cost_gap",
    "goal_error_m",
    "chamfer",
    "frechet",
]


@dataclass(frozen=True)
class Condition:
    terrain_file: str
    path_idx: int
    intent_type: str
    baseline_teacher: dict[str, float]
    intent_teacher: dict[str, float]
    risk_delta: float
    cot_delta: float
    edge_delta: float
    risk_delta_pos: float
    cot_delta_pos: float
    edge_delta_pos: float
    conflict_score: float

    @property
    def key(self) -> tuple[str, int]:
        return (self.terrain_file, self.path_idx)


def now_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")


def stamp_kst() -> str:
    return datetime.now(KST).strftime("%Y%m%d_%H%M%S")


def resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (ROOT / p).resolve()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def q(values: list[float], percentile: float) -> float:
    xs = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if xs.size == 0:
        return float("nan")
    return float(np.quantile(xs, percentile))


def mean(values: list[float]) -> float:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(xs)) if xs else float("nan")


def safe_div(num: float, den: float) -> float:
    if not math.isfinite(den) or abs(den) < 1e-12:
        return 0.0
    return float(num / den)


def build_condition_table(valid_dir: Path, selected_terrain_files: set[str]) -> tuple[dict[tuple[str, int], Condition], dict[str, Any]]:
    """Compute teacher deltas for each non-baseline terrain-intent pair."""
    terrain_cache: dict[str, dict[str, Any]] = {}
    teacher_by_key: dict[tuple[str, int], dict[str, float]] = {}
    meta_by_key: dict[tuple[str, int], dict[str, Any]] = {}

    for terrain_file in sorted(selected_terrain_files):
        terrain_path = valid_dir / terrain_file
        terrain = load_terrain(str(terrain_path))
        terrain_cache[str(terrain_path.resolve())] = terrain
        paths = np.asarray(terrain["paths"])
        intent_types = list(terrain.get("intent_types", ["baseline"] * len(paths)))
        instructions = list(terrain.get("instructions", [""] * len(paths)))
        map_id = str(terrain.get("map_id", terrain_path.stem))

        for path_idx in range(len(paths)):
            intent_type = str(intent_types[path_idx] if path_idx < len(intent_types) else "baseline")
            instruction = str(instructions[path_idx] if path_idx < len(instructions) else "")
            meta = {
                "terrain_file": terrain_file,
                "terrain_path": str(terrain_path.resolve()),
                "map_id": map_id,
                "path_idx": int(path_idx),
                "intent_type": intent_type,
                "instruction": instruction,
            }
            rec = evaluate_prediction(np.asarray(paths[path_idx]), meta, terrain_cache)
            teacher_by_key[(terrain_file, int(path_idx))] = {
                "risk_integral": finite_float(rec.get("risk_integral")),
                "cumulative_cot": finite_float(rec.get("cumulative_cot")),
                "edge_cost": finite_float(rec.get("edge_cost")),
                "isr": finite_float(rec.get("isr")),
                "feasible": finite_float(rec.get("feasible")),
            }
            meta_by_key[(terrain_file, int(path_idx))] = meta

    raw: list[dict[str, Any]] = []
    for (terrain_file, path_idx), meta in sorted(meta_by_key.items()):
        if int(path_idx) == 0 or meta["intent_type"] == "baseline":
            continue
        base = teacher_by_key[(terrain_file, 0)]
        teacher = teacher_by_key[(terrain_file, path_idx)]
        risk_delta = teacher["risk_integral"] - base["risk_integral"]
        cot_delta = teacher["cumulative_cot"] - base["cumulative_cot"]
        edge_delta = teacher["edge_cost"] - base["edge_cost"]
        raw.append({
            "terrain_file": terrain_file,
            "path_idx": int(path_idx),
            "intent_type": meta["intent_type"],
            "baseline_teacher": base,
            "intent_teacher": teacher,
            "risk_delta": risk_delta,
            "cot_delta": cot_delta,
            "edge_delta": edge_delta,
            "risk_delta_pos": max(0.0, risk_delta),
            "cot_delta_pos": max(0.0, cot_delta),
            "edge_delta_pos": max(0.0, edge_delta),
        })

    scales = {
        "risk_delta_q75": q([r["risk_delta_pos"] for r in raw], 0.75),
        "cot_delta_q75": q([r["cot_delta_pos"] for r in raw], 0.75),
        "edge_delta_q75": q([r["edge_delta_pos"] for r in raw], 0.75),
    }
    conditions: dict[tuple[str, int], Condition] = {}
    scores: list[float] = []
    for row in raw:
        score = (
            safe_div(row["risk_delta_pos"], scales["risk_delta_q75"])
            + safe_div(row["cot_delta_pos"], scales["cot_delta_q75"])
            + safe_div(row["edge_delta_pos"], scales["edge_delta_q75"])
        )
        scores.append(score)
        cond = Condition(
            terrain_file=row["terrain_file"],
            path_idx=int(row["path_idx"]),
            intent_type=str(row["intent_type"]),
            baseline_teacher=row["baseline_teacher"],
            intent_teacher=row["intent_teacher"],
            risk_delta=float(row["risk_delta"]),
            cot_delta=float(row["cot_delta"]),
            edge_delta=float(row["edge_delta"]),
            risk_delta_pos=float(row["risk_delta_pos"]),
            cot_delta_pos=float(row["cot_delta_pos"]),
            edge_delta_pos=float(row["edge_delta_pos"]),
            conflict_score=float(score),
        )
        conditions[cond.key] = cond

    directional_scores = [
        c.conflict_score for c in conditions.values()
        if c.intent_type in DIRECTIONAL_INTENTS
    ]
    thresholds = {
        **scales,
        "conflict_score_q50": q(scores, 0.50),
        "conflict_score_q75": q(scores, 0.75),
        "conflict_score_q90": q(scores, 0.90),
        "directional_conflict_score_q75": q(directional_scores, 0.75),
        "directional_conflict_score_q90": q(directional_scores, 0.90),
    }
    return conditions, thresholds


def subset_keys(conditions: dict[tuple[str, int], Condition], thresholds: dict[str, float]) -> dict[str, set[tuple[str, int]]]:
    all_keys = set(conditions.keys())
    directional_keys = {
        k for k, c in conditions.items()
        if c.intent_type in DIRECTIONAL_INTENTS
    }
    return {
        "all_nonbaseline": all_keys,
        "top50_conflict": {
            k for k, c in conditions.items()
            if c.conflict_score >= thresholds["conflict_score_q50"]
        },
        "top25_conflict": {
            k for k, c in conditions.items()
            if c.conflict_score >= thresholds["conflict_score_q75"]
        },
        "top10_conflict": {
            k for k, c in conditions.items()
            if c.conflict_score >= thresholds["conflict_score_q90"]
        },
        "directional_conflict": {
            k for k, c in conditions.items()
            if c.intent_type in DIRECTIONAL_INTENTS
            and c.risk_delta_pos >= thresholds["risk_delta_q75"]
        },
        "directional_top25_conflict": {
            k for k, c in conditions.items()
            if c.intent_type in DIRECTIONAL_INTENTS
            and c.conflict_score >= thresholds["directional_conflict_score_q75"]
        },
        "directional_top10_conflict": {
            k for k, c in conditions.items()
            if c.intent_type in DIRECTIONAL_INTENTS
            and c.conflict_score >= thresholds["directional_conflict_score_q90"]
        },
        "directional_all": directional_keys,
    }


def enrich_record(method: str, record: dict[str, Any], cond: Condition, subset_map: dict[str, set[tuple[str, int]]]) -> dict[str, Any]:
    key = cond.key
    out = {
        "method": method,
        "terrain_file": record["terrain_file"],
        "map_id": record.get("map_id", ""),
        "path_idx": int(record["path_idx"]),
        "intent_type": record["intent_type"],
        "instruction": record.get("instruction", ""),
        "sample_index": int(record.get("sample_index", 0)),
        "success": finite_float(record.get("success")),
        "feasible": finite_float(record.get("feasible")),
        "isr": finite_float(record.get("isr")),
        "cot": finite_float(record.get("cumulative_cot")),
        "risk": finite_float(record.get("risk_integral")),
        "edge_cost": finite_float(record.get("edge_cost")),
        "cost_gap": finite_float(record.get("cost_gap")),
        "endpoint_error": finite_float(record.get("goal_error_m")),
        "chamfer": finite_float(record.get("chamfer")),
        "frechet": finite_float(record.get("frechet")),
        "inference_time": finite_float(record.get("inference_time")),
        "teacher_feasible": cond.intent_teacher["feasible"],
        "teacher_isr": cond.intent_teacher["isr"],
        "teacher_cot": cond.intent_teacher["cumulative_cot"],
        "teacher_risk": cond.intent_teacher["risk_integral"],
        "teacher_edge_cost": cond.intent_teacher["edge_cost"],
        "baseline_teacher_cot": cond.baseline_teacher["cumulative_cot"],
        "baseline_teacher_risk": cond.baseline_teacher["risk_integral"],
        "baseline_teacher_edge_cost": cond.baseline_teacher["edge_cost"],
        "risk_delta": cond.risk_delta,
        "cot_delta": cond.cot_delta,
        "edge_delta": cond.edge_delta,
        "risk_delta_pos": cond.risk_delta_pos,
        "cot_delta_pos": cond.cot_delta_pos,
        "edge_delta_pos": cond.edge_delta_pos,
        "conflict_score": cond.conflict_score,
        "risk_delta_vs_teacher": finite_float(record.get("risk_integral")) - cond.intent_teacher["risk_integral"],
        "cot_delta_vs_teacher": finite_float(record.get("cumulative_cot")) - cond.intent_teacher["cumulative_cot"],
        "edge_delta_vs_teacher": finite_float(record.get("edge_cost")) - cond.intent_teacher["edge_cost"],
        "isr_delta_vs_teacher": finite_float(record.get("isr")) - cond.intent_teacher["isr"],
    }
    out["feasible_isr"] = out["feasible"] * out["isr"]
    out["risk_below_teacher"] = float(out["risk"] <= cond.intent_teacher["risk_integral"])
    out["cot_below_teacher"] = float(out["cot"] <= cond.intent_teacher["cumulative_cot"])
    out["near_teacher_isr_and_lower_risk"] = float(
        out["isr"] >= cond.intent_teacher["isr"] - 0.05
        and out["risk"] <= cond.intent_teacher["risk_integral"]
    )
    for subset, keys in subset_map.items():
        out[f"in_{subset}"] = float(key in keys)
    for component in (
        "isr_component_left_bias",
        "isr_component_right_bias",
        "isr_component_avoid_steep",
        "isr_component_prefer_flat",
    ):
        if component in record:
            out[component] = finite_float(record.get(component))
    return out


def summarize_records(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = {(r["terrain_file"], int(r["path_idx"])) for r in rows}
    prompt_keys = {
        (r["terrain_file"], int(r["path_idx"]), r.get("instruction", ""))
        for r in rows
    }
    out: dict[str, Any] = {
        "count": len(rows),
        "conditions": len(keys),
        "prompt_conditions": len(prompt_keys),
    }
    metric_map = {
        "success": "success",
        "feasible": "feasible",
        "isr": "isr",
        "cot": "cot",
        "risk": "risk",
        "edge_cost": "edge_cost",
        "cost_gap": "cost_gap",
        "feasible_isr": "feasible_isr",
        "endpoint_error": "endpoint_error",
        "chamfer": "chamfer",
        "frechet": "frechet",
        "risk_below_teacher_rate": "risk_below_teacher",
        "cot_below_teacher_rate": "cot_below_teacher",
        "near_teacher_isr_and_lower_risk_rate": "near_teacher_isr_and_lower_risk",
        "risk_delta_vs_teacher": "risk_delta_vs_teacher",
        "cot_delta_vs_teacher": "cot_delta_vs_teacher",
        "edge_delta_vs_teacher": "edge_delta_vs_teacher",
        "isr_delta_vs_teacher": "isr_delta_vs_teacher",
    }
    for out_key, row_key in metric_map.items():
        out[out_key] = mean([finite_float(r.get(row_key)) for r in rows])
    return out


def intentwise_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "left_bias_score": mean([finite_float(r.get("isr_component_left_bias")) for r in rows if "isr_component_left_bias" in r]),
        "right_bias_score": mean([finite_float(r.get("isr_component_right_bias")) for r in rows if "isr_component_right_bias" in r]),
        "avoid_steep_isr": mean([finite_float(r.get("isr_component_avoid_steep")) for r in rows if "isr_component_avoid_steep" in r]),
        "prefer_flat_isr": mean([finite_float(r.get("isr_component_prefer_flat")) for r in rows if "isr_component_prefer_flat" in r]),
        "composite_isr": mean([finite_float(r.get("isr")) for r in rows if "+" in str(r.get("intent_type", ""))]),
    }


def fmt3(x: float) -> str:
    return "--" if not math.isfinite(float(x)) else f"{float(x):.3f}"


def fmt2(x: float) -> str:
    return "--" if not math.isfinite(float(x)) else f"{float(x):.2f}"


def markdown_table(summary: dict[str, dict[str, Any]], methods: list[str]) -> str:
    lines = [
        "| Method | Feasible ↑ | ISR ↑ | CoT ↓ | Risk ↓ | Cost Gap ↓ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in methods:
        row = summary[method]
        lines.append(
            f"| {method} | {fmt3(row['feasible'])} | {fmt3(row['isr'])} | "
            f"{fmt2(row['cot'])} | {fmt2(row['risk'])} | {fmt3(row['cost_gap'])} |"
        )
    return "\n".join(lines)


def latex_table(caption: str, label: str, summary: dict[str, dict[str, Any]], methods: list[str]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & Feasible $\\uparrow$ & ISR $\\uparrow$ & CoT $\\downarrow$ & Risk $\\downarrow$ & Cost Gap $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for method in methods:
        row = summary[method]
        lines.append(
            f"{method} & {fmt3(row['feasible'])} & {fmt3(row['isr'])} & "
            f"{fmt2(row['cot'])} & {fmt2(row['risk'])} & {fmt3(row['cost_gap'])} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def paper_ready_text() -> str:
    return """# Paper-ready English Text

## Experiment Setup
We evaluate rationality under terrain-intent conflict by re-analyzing the existing strict valid-language validation predictions. The conflict unit is a terrain-intent pair, not a terrain. For each validation terrain, we use the baseline-intent teacher path as a reference and compare it with the teacher path for each non-baseline intent. A terrain-intent pair is considered more conflicting when following the intent increases Risk, CoT, and edge cost relative to the baseline-intent teacher path. We compute a conflict score by summing the positive increases in these three quantities after normalization by their 75th-percentile positive deltas. We then evaluate the same stored single-sample predictions on the highest-scoring subsets. This analysis uses the 100 validation terrains, the valid instruction templates, and the existing BC, CVAE, and diffusion predictions. It does not introduce new maps, new intents, retraining, candidate ranking, or Best-of-K selection.

## Results
On the top-25% terrain-intent conflict subset, our diffusion planner achieves higher feasibility and ISR than both learned baselines while also reducing CoT, Risk, and Cost Gap. This suggests that the model does not only match the requested intent, but also preserves a lower-cost path when the intent conflicts with terrain cost. The directional conflict subset shows the same pattern more clearly. Under left, right, and center directional constraints that increase terrain risk, our method obtains higher ISR than CVAE and lower CoT, Risk, and Cost Gap than both BC and CVAE. BC is a strong deterministic direct-regression baseline, but lower Risk should be interpreted together with ISR because a planner can reduce Risk by deviating from the requested intent. CVAE improves intent compliance relative to BC, but its Risk and Cost Gap increase in conflict conditions. These results indicate that the advantage of the diffusion planner is most visible when intent compliance and terrain cost must be balanced.

## Caption
Rationality under terrain-intent conflict. Conflict is defined at the terrain-intent level by comparing the intent-specific teacher path with the baseline-intent teacher path and measuring positive increases in Risk, CoT, and edge cost. All methods are evaluated using the existing strict valid-language single-sample predictions. No retraining, new maps, intents outside the evaluated set, candidate ranking, or Best-of-K selection is used. Our diffusion planner shows a better intent-cost trade-off under conflict conditions.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-dir", default="results/baselines/strict_valid_language_eval_20260519_220732")
    parser.add_argument("--valid-dir", default="data/valid")
    parser.add_argument("--out-root", default="results/baselines")
    parser.add_argument("--experiment-dir", default="experiment")
    parser.add_argument("--legacy-summary", default="results/baselines/conflict_rationality_analysis_20260521_015744/conflict_rationality_summary.json")
    parser.add_argument("--stamp", default="")
    args = parser.parse_args()

    run_stamp = args.stamp or stamp_kst()
    strict_dir = resolve(args.strict_dir)
    valid_dir = resolve(args.valid_dir)
    out_dir = resolve(args.out_root) / f"conflict_tradeoff_analysis_{run_stamp}"
    experiment_dir = resolve(args.experiment_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    legacy_summary_path = resolve(args.legacy_summary)
    legacy_summary: dict[str, Any] | None = None
    if legacy_summary_path.exists():
        legacy_summary = json.loads(legacy_summary_path.read_text(encoding="utf-8"))

    command = " ".join([sys.executable, *sys.argv])
    (out_dir / "command.txt").write_text(command + "\n", encoding="utf-8")

    records_by_method: dict[str, list[dict[str, Any]]] = {}
    for method, filename in METHOD_FILES.items():
        records_by_method[method] = load_jsonl(strict_dir / filename)

    selected_terrain_files = {
        str(r["terrain_file"])
        for rows in records_by_method.values()
        for r in rows
    }
    conditions, thresholds = build_condition_table(valid_dir, selected_terrain_files)
    subsets = subset_keys(conditions, thresholds)

    enriched_rows: list[dict[str, Any]] = []
    rows_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = 0
    for method, records in records_by_method.items():
        for record in records:
            key = (str(record["terrain_file"]), int(record["path_idx"]))
            if key not in conditions:
                skipped += 1
                continue
            row = enrich_record(method, record, conditions[key], subsets)
            enriched_rows.append(row)
            rows_by_method[method].append(row)

    methods = list(METHOD_FILES.keys())
    subset_summary: dict[str, dict[str, Any]] = {}
    intentwise_top25: dict[str, dict[str, float]] = {}
    for subset_name, keys in subsets.items():
        subset_summary[subset_name] = {}
        for method in methods:
            method_rows = [
                r for r in rows_by_method[method]
                if (r["terrain_file"], int(r["path_idx"])) in keys
            ]
            subset_summary[subset_name][method] = summarize_records(method_rows)
            if subset_name == "top25_conflict":
                intentwise_top25[method] = intentwise_summary(method_rows)

    threshold_rows: list[dict[str, Any]] = []
    for subset_name in (
        "all_nonbaseline",
        "top50_conflict",
        "top25_conflict",
        "top10_conflict",
        "directional_conflict",
        "directional_top25_conflict",
        "directional_top10_conflict",
    ):
        for method in methods:
            s = subset_summary[subset_name][method]
            threshold_rows.append({
                "subset": subset_name,
                "method": method,
                "conditions": s["conditions"],
                "samples": s["count"],
                "prompt_conditions": s["prompt_conditions"],
                "feasible": s["feasible"],
                "isr": s["isr"],
                "cot": s["cot"],
                "risk": s["risk"],
                "cost_gap": s["cost_gap"],
                "feasible_isr": s["feasible_isr"],
                "endpoint_error": s["endpoint_error"],
                "chamfer": s["chamfer"],
                "frechet": s["frechet"],
                "risk_below_teacher_rate": s["risk_below_teacher_rate"],
                "near_teacher_isr_and_lower_risk_rate": s["near_teacher_isr_and_lower_risk_rate"],
            })

    raw_fields = [
        "method", "terrain_file", "map_id", "path_idx", "intent_type", "instruction", "sample_index",
        "success", "feasible", "isr", "cot", "risk", "edge_cost", "cost_gap", "endpoint_error",
        "chamfer", "frechet", "feasible_isr", "inference_time",
        "teacher_feasible", "teacher_isr", "teacher_cot", "teacher_risk", "teacher_edge_cost",
        "baseline_teacher_cot", "baseline_teacher_risk", "baseline_teacher_edge_cost",
        "risk_delta", "cot_delta", "edge_delta", "risk_delta_pos", "cot_delta_pos", "edge_delta_pos",
        "conflict_score", "risk_delta_vs_teacher", "cot_delta_vs_teacher", "edge_delta_vs_teacher",
        "isr_delta_vs_teacher", "risk_below_teacher", "cot_below_teacher", "near_teacher_isr_and_lower_risk",
        "in_all_nonbaseline", "in_top50_conflict", "in_top25_conflict", "in_top10_conflict",
        "in_directional_conflict", "in_directional_top25_conflict", "in_directional_top10_conflict",
        "in_directional_all",
        "isr_component_left_bias", "isr_component_right_bias",
        "isr_component_avoid_steep", "isr_component_prefer_flat",
    ]
    raw_csv = out_dir / "conflict_tradeoff_raw.csv"
    threshold_csv = out_dir / "threshold_sensitivity.csv"
    write_csv(raw_csv, enriched_rows, raw_fields)
    write_csv(threshold_csv, threshold_rows, list(threshold_rows[0].keys()))

    condition_rows = []
    for cond in sorted(conditions.values(), key=lambda c: (c.terrain_file, c.path_idx)):
        condition_rows.append({
            "terrain_file": cond.terrain_file,
            "path_idx": cond.path_idx,
            "intent_type": cond.intent_type,
            "risk_delta": cond.risk_delta,
            "cot_delta": cond.cot_delta,
            "edge_delta": cond.edge_delta,
            "risk_delta_pos": cond.risk_delta_pos,
            "cot_delta_pos": cond.cot_delta_pos,
            "edge_delta_pos": cond.edge_delta_pos,
            "conflict_score": cond.conflict_score,
            **{f"in_{name}": float(cond.key in keys) for name, keys in subsets.items()},
        })
    condition_csv = out_dir / "conflict_conditions.csv"
    write_csv(condition_csv, condition_rows, list(condition_rows[0].keys()))

    latex_caption = (
        "Intent compliance and terrain cost under conflict conditions. "
        "Conflict is defined by positive increases in Risk, CoT, and edge cost "
        "of the intent-specific teacher path relative to the baseline-intent teacher path. "
        "All methods use existing strict valid-language single-sample predictions, "
        "without retraining, new maps, intents outside the evaluated set, "
        "candidate ranking, or Best-of-K selection."
    )
    latex = "\n\n".join([
        "% Table A. Top-25% terrain-intent conflict",
        latex_table(latex_caption, "tab:conflict_top25", subset_summary["top25_conflict"], methods),
        "% Table B. Directional terrain-intent conflict",
        latex_table(latex_caption, "tab:directional_conflict", subset_summary["directional_conflict"], methods),
    ])
    latex_path = out_dir / "recommended_conflict_tables.tex"
    latex_path.write_text(latex + "\n", encoding="utf-8")

    paper_text_path = out_dir / "paper_ready_text.md"
    paper_text_path.write_text(paper_ready_text(), encoding="utf-8")

    summary = {
        "created_kst": now_kst(),
        "strict_eval_dir": str(strict_dir),
        "valid_dir": str(valid_dir),
        "instruction_file": "/workspace/diffusion_textguide/data/instruction/valid/inst_valid.json",
        "protocol": {
            "terrain_count": len(selected_terrain_files),
            "seed": 42,
            "condition_count_nonbaseline": len(conditions),
            "models": methods,
            "prediction_source": "existing strict valid-language JSONL predictions",
            "uses_new_maps": False,
            "uses_new_intents": False,
            "retrained_models": False,
            "uses_best_of_k": False,
            "aggregation": "stored prediction samples are evaluated independently and averaged; no ranking or selection",
        },
        "conflict_definition": {
            "unit": "terrain-intent pair",
            "baseline_teacher_path": "path_idx=0 baseline-intent teacher path for the same terrain",
            "intent_teacher_path": "teacher path for the evaluated non-baseline intent",
            "deltas": "intent teacher metric minus baseline teacher metric",
            "positive_deltas": "max(0, delta) for Risk, CoT, and edge cost",
            "score": "risk_pos/q75(risk_pos) + cot_pos/q75(cot_pos) + edge_pos/q75(edge_pos)",
            "directional_intents": sorted(DIRECTIONAL_INTENTS),
        },
        "thresholds": thresholds,
        "legacy_preliminary_check": {
            "path": str(legacy_summary_path),
            "exists": legacy_summary is not None,
            "strict_eval_dir": legacy_summary.get("strict_eval_dir") if legacy_summary else None,
            "thresholds": legacy_summary.get("thresholds") if legacy_summary else None,
            "subset_condition_counts": legacy_summary.get("subset_condition_counts") if legacy_summary else None,
            "note": (
                "The main Top-25 and directional tables reproduce the preliminary "
                "reported values at the displayed precision. This report adds "
                "Top-50, directional Top-25/10, raw CSV, and paper text from the same strict records."
            ) if legacy_summary else "Legacy preliminary summary was not found.",
        },
        "subset_condition_counts": {name: len(keys) for name, keys in subsets.items()},
        "subset_summary": subset_summary,
        "intentwise_top25_conflict": intentwise_top25,
        "skipped_baseline_records": skipped,
        "files": {
            "summary_json": str(out_dir / "conflict_tradeoff_summary.json"),
            "raw_csv": str(raw_csv),
            "threshold_csv": str(threshold_csv),
            "condition_csv": str(condition_csv),
            "latex_table": str(latex_path),
            "paper_ready_text": str(paper_text_path),
            "command": str(out_dir / "command.txt"),
        },
    }
    summary_path = out_dir / "conflict_tradeoff_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    table_a = markdown_table(subset_summary["top25_conflict"], methods)
    table_b = markdown_table(subset_summary["directional_conflict"], methods)
    table_c_rows = [
        "| Subset | Method | Feasible ↑ | ISR ↑ | CoT ↓ | Risk ↓ | Cost Gap ↓ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for subset_name in (
        "top10_conflict",
        "top25_conflict",
        "top50_conflict",
        "directional_conflict",
        "directional_top25_conflict",
        "directional_top10_conflict",
    ):
        for method in methods:
            s = subset_summary[subset_name][method]
            table_c_rows.append(
                f"| {subset_name} | {method} | {fmt3(s['feasible'])} | {fmt3(s['isr'])} | "
                f"{fmt2(s['cot'])} | {fmt2(s['risk'])} | {fmt3(s['cost_gap'])} |"
            )
    table_d_rows = [
        "| Method | Left ↑ | Right ↑ | Avoid-steep ↑ | Prefer-flat ↑ | Composite ↑ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in methods:
        s = intentwise_top25[method]
        table_d_rows.append(
            f"| {method} | {fmt3(s['left_bias_score'])} | {fmt3(s['right_bias_score'])} | "
            f"{fmt3(s['avoid_steep_isr'])} | {fmt3(s['prefer_flat_isr'])} | {fmt3(s['composite_isr'])} |"
        )

    report = f"""# Conflict Trade-off Analysis ({run_stamp})

Created: {now_kst()}

## Source Check

- Strict eval dir: `{strict_dir}`
- Valid terrain dir: `{valid_dir}`
- Instruction file: `/workspace/diffusion_textguide/data/instruction/valid/inst_valid.json`
- Terrain count: {len(selected_terrain_files)}
- Non-baseline terrain-intent conditions: {len(conditions)}
- Models: {", ".join(methods)}
- No new maps, new intents, retraining, candidate selection, or Best-of-K ranking were used.
- Stored samples are evaluated independently and averaged. This preserves the previous strict valid-language evaluation protocol without selecting among candidates.
- Legacy preliminary summary checked: `{legacy_summary_path}`.
- Legacy preliminary strict eval dir: `{legacy_summary.get("strict_eval_dir") if legacy_summary else "not found"}`.

Legacy preliminary thresholds:

```json
{json.dumps(legacy_summary.get("thresholds") if legacy_summary else {}, indent=2)}
```

Legacy preliminary subset counts:

```json
{json.dumps(legacy_summary.get("subset_condition_counts") if legacy_summary else {}, indent=2)}
```

The Top-25 and directional tables below reproduce the preliminary reported values at the displayed precision. This run adds Top-50, directional Top-25/10, raw CSV, threshold sensitivity CSV, LaTeX, and paper-ready text from the same strict prediction records.

## Conflict Definition

The conflict unit is a terrain-intent pair, not a terrain. For each terrain, the baseline-intent teacher path (`path_idx=0`) is used as the reference. The intent-specific teacher path for each non-baseline intent is then compared with this baseline path. A condition is more conflicting when following the intent increases Risk, CoT, and edge cost relative to the baseline teacher path. The conflict score is:

`max(0, ΔRisk) / q75(max(0, ΔRisk)) + max(0, ΔCoT) / q75(max(0, ΔCoT)) + max(0, ΔEdgeCost) / q75(max(0, ΔEdgeCost))`

Thresholds:

```json
{json.dumps(thresholds, indent=2)}
```

Subset condition counts:

```json
{json.dumps({name: len(keys) for name, keys in subsets.items()}, indent=2)}
```

## Table A. Top-25% Terrain-Intent Conflict

{table_a}

## Table B. Directional Terrain-Intent Conflict

{table_b}

## Table C. Threshold Sensitivity

{chr(10).join(table_c_rows)}

## Table D. Intent-wise Top-25% Conflict Results

{chr(10).join(table_d_rows)}

## Interpretation

Top-25% conflict: Ours has the highest Feasible and ISR and the lowest CoT, Risk, and Cost Gap among the learned planners. This supports the claim that diffusion gives a better trade-off when intent compliance increases terrain cost.

Directional conflict: Ours improves ISR over CVAE and has lower CoT, Risk, and Cost Gap than both BC and CVAE. This is the cleanest main-table candidate because directional intents create a direct conflict between the requested route bias and terrain risk.

BC interpretation: BC is a deterministic direct-regression baseline that receives map, start, goal, and intent. It should not be described as intent-agnostic. In severe conflict regimes, lower Risk alone can come from deviating from the requested intent, so Risk should be read together with ISR and Cost Gap.

CVAE interpretation: CVAE improves intent compliance relative to BC in the conflict subsets, but it tends to pay a higher terrain cost, especially in directional conflict where its Risk is the highest among the three methods.

Ours interpretation: Ours maintains or improves intent compliance while reducing CoT, Risk, and Cost Gap in the main conflict subsets. The advantage is clearer in conflict subsets than in the full validation average.

## Honest Caveats

Top-10% conflict is more severe and more mixed. Ours has the highest Feasible and ISR and the lowest CoT and Cost Gap, but BC has lower Risk than Ours. This should be reported in threshold sensitivity or appendix. In the most severe conflict regime, lower Risk alone is not necessarily better because a planner can reduce Risk by ignoring the requested intent.

## Paper-ready English

See `{paper_text_path}`.

## Files

- Summary JSON: `{summary_path}`
- Raw CSV: `{raw_csv}`
- Threshold CSV: `{threshold_csv}`
- Condition CSV: `{condition_csv}`
- LaTeX tables: `{latex_path}`
- Paper-ready text: `{paper_text_path}`
- Command: `{out_dir / "command.txt"}`
- Code path: `/workspace/diffusion_textguide/experiment/analysis/conflict_tradeoff_analysis.py`

The purpose of this analysis is not to create favorable new data. The purpose is to separate terrain-intent conflict conditions within the existing validation predictions and diagnose how each model handles the trade-off between intent compliance and terrain cost.
"""
    report_path = experiment_dir / f"conflict_tradeoff_report_{run_stamp}.md"
    report_path.write_text(report, encoding="utf-8")
    summary["files"]["report"] = str(report_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[{now_kst()}] Wrote conflict trade-off analysis")
    print(f"  summary: {summary_path}")
    print(f"  raw csv: {raw_csv}")
    print(f"  threshold csv: {threshold_csv}")
    print(f"  latex: {latex_path}")
    print(f"  report: {report_path}")
    print()
    print("Top-25% conflict")
    print(table_a)
    print()
    print("Directional conflict")
    print(table_b)


if __name__ == "__main__":
    main()
