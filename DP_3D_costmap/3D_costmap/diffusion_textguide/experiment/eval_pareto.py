"""
Experiment 1 — Intent Path Weight Justification via Pareto Frontier

Sweeps (α, β, γ, δ) cost weights and generates A* paths for each
weight configuration × intent.  Two complementary analyses:

  • **Pareto front** — non-dominated points in (cot_mean, risk_mean, isr_mean)
    per intent (multi-objective trade-off surface).

  • **Constrained min-CoT pick** (optional, YAML `constrained_selection`) —
    Step 1: keep weight tuples whose aggregated feasibility ≥ floor and
    aggregated ISR ≥ floor (over intents in the sweep).
    Step 2: among survivors, choose the tuple with **lowest mean CoT**.
    This is *not* a Pareto computation; it is a standard “reliability +
    instruction floor, then minimize energy” rule for picking one default.

Usage:
  python -m experiment.eval_pareto \
      --config experiment/configs/pareto_sweep.yaml \
      --output-dir results/pareto
"""

import argparse
import itertools
import json
import os
import sys
from collections import defaultdict
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
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
from experiment.metrics import (
    cumulative_cot,
    risk_integral,
    mean_slope_along_path,
    max_slope_along_path,
    path_length_metres,
    instruction_success,
)

DEFAULT_CONFIG = {
    "terrain": {
        "num_terrains": 20,
        "img_size": 100,
        "height_range": [0, 5],
        "terrain_scales": [[20, 10], [10, 5], [5, 2]],
        "limit_angle_deg": 25,
        "pixel_resolution": 0.5,
        "max_iterations": 20000,
    },
    "sweep": {
        "alpha": [0.5, 1.0, 2.0],
        "beta":  [0.0, 0.4, 0.8, 1.5],
        "gamma": [0.0, 0.1, 0.5, 1.0],
        "delta": [0.0, 0.5, 1.0, 2.0],
    },
    "intents": ["baseline", "left_bias", "right_bias", "avoid_steep",
                "prefer_flat", "via_flat_region"],
    "horizon": 120,
    "risk_threshold_deg": 15.0,
    "seed": 42,
    # Optional: see constrained_min_cot_pick()
    "constrained_selection": None,
}


def _intent_params_for(itype: str) -> dict:
    for entry in INTENT_CATALOG:
        if entry["type"] == itype:
            return dict(entry["params"])
    return {}


def _generate_terrains(cfg: dict, seed: int):
    """Pre-generate terrains with valid start/goal pairs."""
    np.random.seed(seed)
    tc = cfg["terrain"]
    gen = SlopeCotGenerator(
        img_size=tc["img_size"],
        height_range=tc["height_range"],
        mass=10.0, gravity=9.8,
        limit_angle_deg=tc["limit_angle_deg"],
        max_iterations=tc["max_iterations"],
        pixel_resolution=tc["pixel_resolution"],
    )
    margin = tc["img_size"] // 10
    terrains = []
    attempts = 0
    while len(terrains) < tc["num_terrains"] and attempts < tc["num_terrains"] * 30:
        attempts += 1
        h_map, s_map = gen.generate(terrain_scales=tc["terrain_scales"])
        slope_deg = np.degrees(s_map)
        if np.mean(slope_deg) < 8 or np.mean(slope_deg) > 32 or np.max(slope_deg) > 55:
            continue
        edge_lo, edge_hi = margin, margin * 3
        far_lo, far_hi = tc["img_size"] - margin * 3, tc["img_size"] - margin
        start = goal = None
        for _ in range(200):
            s = (np.random.randint(edge_lo, edge_hi), np.random.randint(edge_lo, edge_hi))
            g = (np.random.randint(far_lo, far_hi), np.random.randint(far_lo, far_hi))
            d = np.hypot(g[0] - s[0], g[1] - s[1])
            if d >= tc["img_size"] * 0.6 and s_map[s] < gen.limit_angle and s_map[g] < gen.limit_angle:
                start, goal = s, g
                break
        if start is None:
            continue
        terrains.append({
            "height_map": h_map.copy(),
            "slope_map_rad": s_map.copy(),
            "slope_map_deg": slope_deg.copy(),
            "start": start,
            "goal": goal,
            "gen": gen,
        })
    return terrains


def run_sweep(cfg: dict):
    """Run the full weight sweep and return list of result dicts."""
    tc = cfg["terrain"]
    sw = cfg["sweep"]
    horizon = cfg["horizon"]
    risk_th = cfg["risk_threshold_deg"]
    img_size = tc["img_size"]
    px_res = tc["pixel_resolution"]
    limit_deg = tc["limit_angle_deg"]

    terrains = _generate_terrains(cfg, cfg.get("seed", 42))
    print(f"Generated {len(terrains)} valid terrains")

    weight_combos = list(itertools.product(
        sw["alpha"], sw["beta"], sw["gamma"], sw["delta"]
    ))
    print(f"Weight combinations: {len(weight_combos)}")

    results = []
    for a, b, g, d in tqdm(weight_combos, desc="Weight sweep"):
        for intent_name in cfg["intents"]:
            iparams = _intent_params_for(intent_name)
            row = {
                "alpha": a, "beta": b, "gamma": g, "delta": d,
                "intent": intent_name,
                "n_success": 0, "n_total": 0,
                "cot_list": [], "risk_list": [], "mean_slope_list": [],
                "max_slope_list": [], "length_m_list": [], "isr_list": [],
            }
            for ter in terrains:
                gen = SlopeCotGenerator(
                    img_size=img_size, height_range=tc["height_range"],
                    mass=10.0, gravity=9.8,
                    limit_angle_deg=limit_deg,
                    max_iterations=tc["max_iterations"],
                    pixel_resolution=px_res,
                )
                gen.height_map = ter["height_map"]
                gen.slope_map = ter["slope_map_rad"]

                path_px = gen.find_path_with_intent(
                    ter["start"], ter["goal"],
                    alpha=a, beta=b, gamma=g, delta=d,
                    risk_threshold_deg=risk_th,
                    intent_type=intent_name,
                    intent_params=iparams,
                )
                row["n_total"] += 1
                if path_px is None or len(path_px) < 5:
                    continue
                row["n_success"] += 1

                norm = _path_pixels_to_normalized(path_px, img_size)
                fixed = _resample_path(norm, horizon)

                row["cot_list"].append(
                    cumulative_cot(fixed, ter["height_map"], img_size, px_res, limit_deg))
                row["risk_list"].append(
                    risk_integral(fixed, ter["slope_map_deg"], img_size, risk_th))
                row["mean_slope_list"].append(
                    mean_slope_along_path(fixed, ter["slope_map_deg"], img_size))
                row["max_slope_list"].append(
                    max_slope_along_path(fixed, ter["slope_map_deg"], img_size))
                row["length_m_list"].append(
                    path_length_metres(fixed, img_size, px_res))
                row["isr_list"].append(float(instruction_success(
                    fixed, intent_name, iparams,
                    ter["slope_map_deg"], img_size,
                    ter["start"], ter["goal"],
                )))

            for key in ("cot_list", "risk_list", "mean_slope_list",
                        "max_slope_list", "length_m_list", "isr_list"):
                vals = row[key]
                prefix = key.replace("_list", "")
                row[f"{prefix}_mean"] = float(np.mean(vals)) if vals else float("nan")
                row[f"{prefix}_std"] = float(np.std(vals)) if vals else float("nan")
                del row[key]

            row["feasibility"] = row["n_success"] / max(row["n_total"], 1)
            results.append(row)
    return results


def _is_dominated(a: dict, b: dict, objectives: list[str], directions: list[str]) -> bool:
    """Return True if *a* is dominated by *b* (b is at least as good on all,
    strictly better on at least one)."""
    at_least_one_better = False
    for obj, direction in zip(objectives, directions):
        va, vb = a[obj], b[obj]
        if direction == "min":
            if vb > va:
                return False
            if vb < va:
                at_least_one_better = True
        else:
            if vb < va:
                return False
            if vb > va:
                at_least_one_better = True
    return at_least_one_better


def pareto_front(results: list[dict], objectives: list[str],
                 directions: list[str]) -> list[dict]:
    """Extract non-dominated points."""
    valid = [r for r in results if all(not np.isnan(r.get(o, float("nan"))) for o in objectives)]
    front = []
    for r in valid:
        if not any(_is_dominated(r, other, objectives, directions) for other in valid if other is not r):
            front.append(r)
    return front


def constrained_min_cot_pick(
    results: list[dict],
    intents: list[str],
    feasibility_min: float,
    isr_min: float,
    aggregate: str = "mean",
):
    """Pick (α,β,γ,δ) with feasibility / ISR floors, then minimize mean CoT.

    Groups sweep rows by weight tuple. Only groups that have exactly one row
    per intent in *intents* are kept.

    *aggregate*:
      - ``mean``: require mean(feasibility) ≥ *feasibility_min* and
        mean(isr_mean) ≥ *isr_min* across intents; objective = mean(cot_mean).
      - ``min``: require min(feasibility) and min(isr) (worst intent) to pass;
        objective = mean(cot_mean).

    Returns (best_dict | None, all_feasible_candidates_list).
    """
    intent_set = set(intents)
    groups = defaultdict(list)
    for r in results:
        if r["intent"] not in intent_set:
            continue
        key = (r["alpha"], r["beta"], r["gamma"], r["delta"])
        groups[key].append(r)

    candidates = []
    for key, rows in groups.items():
        if {x["intent"] for x in rows} != intent_set:
            continue

        feas = np.array([x["feasibility"] for x in rows], dtype=np.float64)
        isrs = np.array([x["isr_mean"] for x in rows], dtype=np.float64)
        cots = np.array([x["cot_mean"] for x in rows], dtype=np.float64)

        if aggregate == "min":
            f_gate = float(np.min(feas))
            i_gate = float(np.min(isrs))
        else:
            f_gate = float(np.mean(feas))
            i_gate = float(np.mean(isrs))

        cot_obj = float(np.nanmean(cots))
        if np.isnan(cot_obj):
            continue

        if f_gate < feasibility_min or i_gate < isr_min:
            continue

        per_intent = {
            x["intent"]: {
                "feasibility": x["feasibility"],
                "isr_mean": x["isr_mean"],
                "cot_mean": x["cot_mean"],
                "risk_mean": x["risk_mean"],
            }
            for x in rows
        }
        candidates.append({
            "alpha": key[0],
            "beta": key[1],
            "gamma": key[2],
            "delta": key[3],
            "aggregate": aggregate,
            "gate_feasibility": f_gate,
            "gate_isr": i_gate,
            "objective_cot_mean_across_intents": cot_obj,
            "per_intent": per_intent,
        })

    if not candidates:
        return None, []
    best = min(candidates, key=lambda c: c["objective_cot_mean_across_intents"])
    return best, candidates


def balanced_pick(
    results: list[dict],
    intents: list[str],
    feasibility_min: float = 0.90,
    isr_min: float = 0.70,
):
    """Pick the "most balanced" (α,β,γ,δ) — per-intent floors then utopia distance.

    1. Per-intent (not average) feasibility ≥ floor, ISR ≥ floor.
    2. Among survivors, normalise CoT / Risk / ISR to [0,1] across the pool.
    3. Pick the point closest to the utopia corner (min CoT, min Risk, max ISR).
       → Euclidean distance in normalised space = balance.

    Returns (best_dict | None, all_feasible_candidates_list).
    """
    intent_set = set(intents)
    groups = defaultdict(list)
    for r in results:
        if r["intent"] not in intent_set:
            continue
        key = (r["alpha"], r["beta"], r["gamma"], r["delta"])
        groups[key].append(r)

    candidates = []
    for key, rows in groups.items():
        if {x["intent"] for x in rows} != intent_set:
            continue

        per_intent = {}
        pass_gate = True
        for x in rows:
            if x["feasibility"] < feasibility_min:
                pass_gate = False
                break
            if np.isnan(x["isr_mean"]) or x["isr_mean"] < isr_min:
                pass_gate = False
                break
            per_intent[x["intent"]] = {
                "feasibility": x["feasibility"],
                "isr_mean": x["isr_mean"],
                "cot_mean": x["cot_mean"],
                "risk_mean": x["risk_mean"],
                "mean_slope_mean": x.get("mean_slope_mean", float("nan")),
            }
        if not pass_gate:
            continue

        feas_arr = np.array([pi["feasibility"] for pi in per_intent.values()])
        isr_arr = np.array([pi["isr_mean"] for pi in per_intent.values()])
        cot_arr = np.array([pi["cot_mean"] for pi in per_intent.values()])
        risk_arr = np.array([pi["risk_mean"] for pi in per_intent.values()])

        candidates.append({
            "alpha": key[0], "beta": key[1], "gamma": key[2], "delta": key[3],
            "min_feasibility": float(np.min(feas_arr)),
            "min_isr": float(np.min(isr_arr)),
            "mean_isr": float(np.mean(isr_arr)),
            "mean_cot": float(np.nanmean(cot_arr)),
            "mean_risk": float(np.nanmean(risk_arr)),
            "per_intent": per_intent,
        })

    if not candidates:
        return None, []

    cots = np.array([c["mean_cot"] for c in candidates])
    risks = np.array([c["mean_risk"] for c in candidates])
    isrs = np.array([c["mean_isr"] for c in candidates])

    def _norm(arr, direction):
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-12:
            return np.zeros_like(arr)
        n = (arr - lo) / (hi - lo)
        return n if direction == "max" else 1.0 - n

    cot_n = _norm(cots, "min")
    risk_n = _norm(risks, "min")
    isr_n = _norm(isrs, "max")

    dist_to_utopia = np.sqrt((1.0 - cot_n) ** 2 + (1.0 - risk_n) ** 2 + (1.0 - isr_n) ** 2)

    for i, c in enumerate(candidates):
        c["utopia_distance"] = float(dist_to_utopia[i])
        c["norm_cot"] = float(cot_n[i])
        c["norm_risk"] = float(risk_n[i])
        c["norm_isr"] = float(isr_n[i])

    best_idx = int(np.argmin(dist_to_utopia))
    return candidates[best_idx], candidates


def plot_pareto_2d(results, front, x_obj, y_obj, x_dir, y_dir,
                   out_path, title=""):
    """2-D Pareto scatter: all points + highlighted front."""
    fig, ax = plt.subplots(figsize=(8, 6))
    xs = [r[x_obj] for r in results if not np.isnan(r.get(x_obj, float("nan")))]
    ys = [r[y_obj] for r in results if not np.isnan(r.get(y_obj, float("nan")))]
    ax.scatter(xs, ys, alpha=0.3, s=20, label="all configs")

    if front:
        fx = [r[x_obj] for r in front]
        fy = [r[y_obj] for r in front]
        order = np.argsort(fx)
        ax.plot(np.array(fx)[order], np.array(fy)[order], "r-o", lw=2,
                markersize=6, label="Pareto front")
    ax.set_xlabel(x_obj)
    ax.set_ylabel(y_obj)
    ax.set_title(title or f"Pareto: {x_obj} vs {y_obj}")
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Exp 1: Pareto weight sweep")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default=str(_ROOT / "results" / "pareto"))
    args = ap.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = DEFAULT_CONFIG

    os.makedirs(args.output_dir, exist_ok=True)

    results = run_sweep(cfg)
    out_json = os.path.join(args.output_dir, "sweep_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved {len(results)} rows → {out_json}")

    # Pareto analysis: energy vs safety (per intent)
    for intent in cfg["intents"]:
        subset = [r for r in results if r["intent"] == intent]
        if not subset:
            continue
        objectives = ["cot_mean", "risk_mean", "isr_mean"]
        directions = ["min", "min", "max"]
        front = pareto_front(subset, objectives, directions)

        plot_pareto_2d(
            subset, front, "cot_mean", "risk_mean", "min", "min",
            os.path.join(args.output_dir, f"pareto_cot_risk_{intent}.png"),
            title=f"Pareto: Energy vs Safety  [{intent}]",
        )
        plot_pareto_2d(
            subset, front, "cot_mean", "isr_mean", "min", "max",
            os.path.join(args.output_dir, f"pareto_cot_isr_{intent}.png"),
            title=f"Pareto: Energy vs ISR  [{intent}]",
        )

        print(f"\n[{intent}] Pareto front ({len(front)} points):")
        for pt in front[:5]:
            print(f"  α={pt['alpha']}, β={pt['beta']}, γ={pt['gamma']}, δ={pt['delta']}  "
                  f"CoT={pt['cot_mean']:.2f}  Risk={pt['risk_mean']:.2f}  "
                  f"ISR={pt['isr_mean']:.2f}  feas={pt['feasibility']:.2f}")

    # Summary table
    summary_path = os.path.join(args.output_dir, "pareto_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Intent | #Pareto | Best CoT | Best Risk | Best ISR\n")
        f.write("-" * 60 + "\n")
        for intent in cfg["intents"]:
            subset = [r for r in results if r["intent"] == intent]
            front = pareto_front(
                subset, ["cot_mean", "risk_mean", "isr_mean"],
                ["min", "min", "max"],
            )
            best_cot = min((r["cot_mean"] for r in subset if not np.isnan(r["cot_mean"])), default=float("nan"))
            best_risk = min((r["risk_mean"] for r in subset if not np.isnan(r["risk_mean"])), default=float("nan"))
            best_isr = max((r["isr_mean"] for r in subset if not np.isnan(r["isr_mean"])), default=float("nan"))
            f.write(f"{intent:26s} | {len(front):7d} | {best_cot:8.2f} | {best_risk:9.2f} | {best_isr:8.2f}\n")

    # ── Balanced pick: per-intent floors + utopia distance ────────────
    bal = cfg.get("balanced_selection", {})
    if isinstance(bal, dict) and bal.get("enabled", False):
        b_feas = float(bal.get("feasibility_min", 0.90))
        b_isr = float(bal.get("isr_min", 0.70))
        best_bal, pool_bal = balanced_pick(
            results, cfg["intents"], feasibility_min=b_feas, isr_min=b_isr,
        )
        bal_path = os.path.join(args.output_dir, "balanced_pick.json")
        bal_payload = {
            "method": "per-intent floors + utopia distance (CoT↓ Risk↓ ISR↑)",
            "constraints": {"feasibility_min_per_intent": b_feas,
                            "isr_min_per_intent": b_isr},
            "num_candidates_passing": len(pool_bal),
            "best": best_bal,
            "top5": sorted(pool_bal, key=lambda c: c["utopia_distance"])[:5],
        }
        with open(bal_path, "w") as f:
            json.dump(bal_payload, f, indent=2, default=str)

        with open(summary_path, "a") as f:
            f.write("\n--- Balanced pick (per-intent floors + utopia distance) ---\n")
            f.write(f"feasibility_min={b_feas} (per intent), isr_min={b_isr} (per intent)\n")
            if best_bal:
                f.write(
                    f"BEST: α={best_bal['alpha']}, β={best_bal['beta']}, "
                    f"γ={best_bal['gamma']}, δ={best_bal['delta']}  "
                    f"min_feas={best_bal['min_feasibility']:.4f}  "
                    f"min_ISR={best_bal['min_isr']:.4f}  "
                    f"mean_ISR={best_bal['mean_isr']:.4f}  "
                    f"mean_CoT={best_bal['mean_cot']:.4f}  "
                    f"mean_Risk={best_bal['mean_risk']:.4f}  "
                    f"utopia_dist={best_bal['utopia_distance']:.4f}\n"
                )
                f.write("  Per-intent breakdown:\n")
                for it, iv in best_bal["per_intent"].items():
                    f.write(f"    {it:30s}  ISR={iv['isr_mean']:.3f}  "
                            f"CoT={iv['cot_mean']:.2f}  Risk={iv['risk_mean']:.2f}\n")
            else:
                f.write("No (α,β,γ,δ) satisfied per-intent constraints.\n")

        print(f"\nBalanced pick → {bal_path}")
        if best_bal:
            print(
                f"  BALANCED: α={best_bal['alpha']}, β={best_bal['beta']}, "
                f"γ={best_bal['gamma']}, δ={best_bal['delta']}\n"
                f"  min_ISR={best_bal['min_isr']:.4f}  mean_ISR={best_bal['mean_isr']:.4f}  "
                f"mean_CoT={best_bal['mean_cot']:.4f}  mean_Risk={best_bal['mean_risk']:.4f}\n"
                f"  utopia_dist={best_bal['utopia_distance']:.4f}  "
                f"({len(pool_bal)} passing tuples)"
            )
        else:
            print("  No weight tuple passed per-intent constraints.")

    # ── Constrained min-CoT (legacy) ───────────────────────────────
    sel = cfg.get("constrained_selection")
    if isinstance(sel, dict) and sel.get("enabled", False):
        feas_floor = float(sel.get("feasibility_min", 0.95))
        isr_floor = float(sel.get("isr_min", 0.80))
        agg = str(sel.get("aggregate", "mean"))
        best, pool = constrained_min_cot_pick(
            results, cfg["intents"], feas_floor, isr_floor, aggregate=agg,
        )
        pick_path = os.path.join(args.output_dir, "constrained_min_cot_pick.json")
        pick_payload = {
            "constraints": {
                "feasibility_min": feas_floor,
                "isr_min": isr_floor,
                "aggregate": agg,
            },
            "num_candidates_passing": len(pool),
            "best": best,
            "all_passing": sorted(
                pool, key=lambda c: c["objective_cot_mean_across_intents"]
            ),
        }
        with open(pick_path, "w") as f:
            json.dump(pick_payload, f, indent=2, default=str)

        with open(summary_path, "a") as f:
            f.write("\n--- Constrained min-CoT (feas & ISR floors, then min mean CoT) ---\n")
            f.write(f"feasibility_min={feas_floor}, isr_min={isr_floor}, aggregate={agg}\n")
            if best:
                f.write(
                    f"BEST: α={best['alpha']}, β={best['beta']}, γ={best['gamma']}, δ={best['delta']}  "
                    f"g_feas={best['gate_feasibility']:.4f}  g_isr={best['gate_isr']:.4f}  "
                    f"mean_CoT={best['objective_cot_mean_across_intents']:.4f}\n"
                )
            else:
                f.write("No (α,β,γ,δ) satisfied constraints for all intents.\n")
        print(f"\nConstrained pick → {pick_path}")
        if best:
            print(
                f"  Chosen weights: α={best['alpha']}, β={best['beta']}, γ={best['gamma']}, δ={best['delta']}\n"
                f"  Gate feas={best['gate_feasibility']:.4f}, ISR={best['gate_isr']:.4f}, "
                f"mean CoT={best['objective_cot_mean_across_intents']:.4f}  "
                f"({len(pool)} passing tuples)"
            )
        else:
            print("  No weight tuple passed constraints (see JSON for empty best).")

    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
