"""
Unified metric library for all four experiment tracks.

All path inputs are in **normalised** [-1,1] (x=col, y=row) unless noted.
Terrain-aware metrics require slope_map_deg, height_map, and img_size.

Metric catalogue
─────────────────
 Category           Metric                  Function
 ──────────        ──────                  ────────
 Goal quality       goal_error              goal_error()
                    feasibility             infeasible_rate()
 Instruction        ISR score [0,1]         instruction_success() (atomic scores +
                                                          geometric mean for +)
                    ISR mean                instruction_success_rate()
                    side_bias_success       side_bias_success() (legacy bool)
                    avoid_steep_success     avoid_steep_success() (legacy bool)
 Energy / safety    cumulative_cot          cumulative_cot()
                    mean_slope              mean_slope_along_path()
                    max_slope               max_slope_along_path()
                    risk_integral           risk_integral()
 Path geometry      path_length_norm        path_length()
                    path_length_metres      path_length_metres()
 Similarity         pointwise_l2            pointwise_l2()
                    chamfer_distance        chamfer_distance()
                    frechet_distance        frechet_distance()
 Cost-based         edge_cost_total         edge_cost_total()
                    cost_gap                cost_gap()
                    instr_adherence_gap     instruction_adherence_gap()
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_data import (
    _calculate_directional_cot,
    _calculate_risk,
    _calculate_intent_penalty,
    _precompute_side_bias,
)


# ═══════════════════════════════════════════════════════════════════════════
# Coordinate helpers (duplicated lite version to keep metrics self-contained)
# ═══════════════════════════════════════════════════════════════════════════

def _norm_to_px(path_norm: np.ndarray, img_size: int) -> np.ndarray:
    """[-1,1] (x,y) → pixel (row, col), float."""
    px = (path_norm + 1.0) / 2.0 * img_size
    px = np.clip(px, 0, img_size - 1)
    return np.stack([px[:, 1], px[:, 0]], axis=1)


def _px_int(path_px: np.ndarray, img_size: int) -> np.ndarray:
    """Clamp to valid integer indices."""
    return np.clip(path_px.astype(int), 0, img_size - 1)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Goal-quality metrics
# ═══════════════════════════════════════════════════════════════════════════

def goal_error(path_norm: np.ndarray, goal_norm: np.ndarray) -> float:
    """L2 between last waypoint and goal in normalised space."""
    return float(np.linalg.norm(path_norm[-1] - goal_norm))


def goal_error_metres(
    path_norm: np.ndarray,
    goal_norm: np.ndarray,
    img_size: int,
    pixel_resolution: float = 0.5,
) -> float:
    """Goal error in metres (pixel-space L2 × resolution)."""
    p = _norm_to_px(path_norm[-1:], img_size)[0]
    g = _norm_to_px(goal_norm.reshape(1, 2), img_size)[0]
    return float(np.linalg.norm(p - g) * pixel_resolution)


def infeasible_rate(
    paths_norm: np.ndarray,
    slope_map_deg: np.ndarray,
    img_size: int,
    limit_angle_deg: float = 35.0,
) -> float:
    """Fraction of paths that cross at least one cell ≥ limit_angle_deg."""
    n_infeasible = 0
    for p in paths_norm:
        rc = _px_int(_norm_to_px(p, img_size), img_size)
        slopes = slope_map_deg[rc[:, 0], rc[:, 1]]
        if np.any(slopes >= limit_angle_deg):
            n_infeasible += 1
    return n_infeasible / max(len(paths_norm), 1)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Instruction Success Rate (ISR) — per-intent checkers
# ═══════════════════════════════════════════════════════════════════════════

def _path_centroid_offset(
    path_norm: np.ndarray,
    start_norm: np.ndarray,
    goal_norm: np.ndarray,
) -> float:
    """Signed offset of path centroid w.r.t. start→goal line.

    Positive = left of travel direction (screen coords with origin='lower').
    """
    fwd = goal_norm - start_norm
    norm_perp = np.array([-fwd[1], fwd[0]])
    length = np.linalg.norm(norm_perp)
    if length < 1e-8:
        return 0.0
    norm_perp /= length
    centroid = path_norm.mean(axis=0)
    return float(np.dot(centroid - start_norm, norm_perp))


def _ref_polyline_lateral(
    baseline_path_norm: Optional[np.ndarray],
    start_norm: np.ndarray,
    goal_norm: np.ndarray,
) -> np.ndarray:
    """Baseline route τ^base in normalised space; chord start→goal if unknown."""
    if baseline_path_norm is not None and len(baseline_path_norm) >= 2:
        return np.asarray(baseline_path_norm, dtype=np.float32)
    return np.stack(
        [np.asarray(start_norm, dtype=np.float32), np.asarray(goal_norm, dtype=np.float32)],
        axis=0,
    )


def lateral_deltas_vs_reference(
    path_norm: np.ndarray,
    ref_path_norm: np.ndarray,
) -> np.ndarray:
    """Per-step Δ_t: signed lateral offset of each waypoint w.r.t. reference polyline.

    Positive = left of local reference tangent in normalised (x, y) space
    (matches centroid / A* side_bias convention in this codebase).
    """
    path_norm = np.asarray(path_norm, dtype=np.float32)
    ref = np.asarray(ref_path_norm, dtype=np.float32)
    t_n = len(path_norm)
    deltas = np.zeros(t_n, dtype=np.float32)
    if len(ref) < 2:
        return deltas
    for t in range(t_n):
        p = path_norm[t]
        d2 = np.sum((ref - p) ** 2, axis=1)
        k = int(np.argmin(d2))
        i0 = max(0, k - 1)
        i1 = min(len(ref) - 1, k + 1)
        tang = ref[i1] - ref[i0]
        n = float(np.linalg.norm(tang))
        if n < 1e-8:
            continue
        left = np.array([-tang[1], tang[0]], dtype=np.float32) / n
        deltas[t] = float(np.dot(p - ref[k], left))
    return deltas


def _tilde_linear(x: float, scale: float) -> float:
    """Normalise raw x ≥ 0 into [0, 1] via x / scale (clipped)."""
    return float(min(max(x, 0.0) / max(scale, 1e-8), 1.0))


def _path_centroid_offset_from_reference(
    path_norm: np.ndarray,
    ref_path_norm: np.ndarray,
) -> float:
    """Signed lateral offset of path centroid w.r.t. local tangent of ref path.

    Positive means left of reference tangent in the same normalized (x,y) space.
    """
    if ref_path_norm is None or len(ref_path_norm) < 2:
        return 0.0
    centroid = np.asarray(path_norm, dtype=np.float32).mean(axis=0)
    ref = np.asarray(ref_path_norm, dtype=np.float32)
    d2 = np.sum((ref - centroid) ** 2, axis=1)
    k = int(np.argmin(d2))
    i0 = max(0, k - 1)
    i1 = min(len(ref) - 1, k + 1)
    t = ref[i1] - ref[i0]
    n = float(np.linalg.norm(t))
    if n < 1e-8:
        return 0.0
    left = np.array([-t[1], t[0]], dtype=np.float32) / n
    return float(np.dot(centroid - ref[k], left))


def side_bias_success(
    path_norm: np.ndarray,
    start_norm: np.ndarray,
    goal_norm: np.ndarray,
    bias: str = "left",
    threshold: float = 0.0,
) -> bool:
    """Check if path has the requested directional bias.

    bias ∈ {"left", "right"}.
    """
    offset = _path_centroid_offset(path_norm, start_norm, goal_norm)
    if bias == "left":
        return offset > threshold
    return offset < -threshold


def avoid_steep_success(
    path_norm: np.ndarray,
    slope_map_deg: np.ndarray,
    img_size: int,
    tau_steep: float = 20.0,
    tolerance_frac: float = 0.05,
) -> bool:
    """Check if path mostly avoids cells steeper than tau_steep.

    Allows up to tolerance_frac of waypoints to exceed threshold.
    """
    rc = _px_int(_norm_to_px(path_norm, img_size), img_size)
    slopes = slope_map_deg[rc[:, 0], rc[:, 1]]
    frac_steep = np.mean(slopes >= tau_steep)
    return frac_steep <= tolerance_frac


def _atomic_instruction_score(
    part: str,
    path_norm: np.ndarray,
    intent_params: dict,
    slope_map_deg: np.ndarray,
    img_size: int,
    start_pos: tuple,
    goal_pos: tuple,
    start_norm: np.ndarray,
    goal_norm: np.ndarray,
    baseline_path_norm: Optional[np.ndarray],
    height_map: Optional[np.ndarray],
    pixel_resolution: float,
    limit_angle_deg: float,
) -> float:
    """Single intent component score in [0, 1] (paper ISR definitions)."""
    ref = _ref_polyline_lateral(baseline_path_norm, start_norm, goal_norm)

    if part == "left_bias":
        m = float(intent_params.get("lateral_margin", 0.02))
        d = lateral_deltas_vs_reference(path_norm, ref)
        return float(np.clip(np.mean(d > m), 0.0, 1.0))

    if part == "right_bias":
        m = float(intent_params.get("lateral_margin", 0.02))
        d = lateral_deltas_vs_reference(path_norm, ref)
        return float(np.clip(np.mean(d < -m), 0.0, 1.0))

    if part == "center_bias":
        scale = float(intent_params.get("isr_center_delta_scale", 0.25))
        d = lateral_deltas_vs_reference(path_norm, ref)
        mean_abs = float(np.mean(np.abs(d)))
        return float(np.clip(1.0 - _tilde_linear(mean_abs, scale), 0.0, 1.0))

    if part == "avoid_steep":
        tau = float(intent_params.get("tau_steep", 20.0))
        rc = _px_int(_norm_to_px(path_norm, img_size), img_size)
        slopes = slope_map_deg[rc[:, 0], rc[:, 1]]
        frac = float(np.mean(slopes > tau))
        return float(np.clip(1.0 - frac, 0.0, 1.0))

    if part == "prefer_flat":
        scale_deg = float(intent_params.get("isr_flat_slope_scale_deg", 45.0))
        rc = _px_int(_norm_to_px(path_norm, img_size), img_size)
        mean_s = float(np.mean(slope_map_deg[rc[:, 0], rc[:, 1]]))
        return float(np.clip(1.0 - _tilde_linear(mean_s, scale_deg), 0.0, 1.0))

    if part == "minimize_elevation_change":
        if height_map is None:
            return 0.0
        scale_v = float(intent_params.get("isr_elev_change_scale", 80.0))
        rc = _px_int(_norm_to_px(path_norm, img_size), img_size)
        h = height_map[rc[:, 0], rc[:, 1]].astype(np.float64)
        v = float(np.sum(np.abs(np.diff(h))))
        return float(np.clip(1.0 - _tilde_linear(v, scale_v), 0.0, 1.0))

    if part == "short_path":
        scale_f = float(intent_params.get("isr_short_len_scale_factor", 2.5))
        L = path_length(path_norm)
        chord = float(np.linalg.norm(goal_norm - start_norm))
        l_max = max(scale_f * max(chord, 1e-6), 1e-6)
        return float(np.clip(1.0 - _tilde_linear(L, l_max), 0.0, 1.0))

    if part == "energy_efficient":
        if height_map is None:
            return 0.0
        scale_cot = float(intent_params.get("isr_cot_scale", 800.0))
        cot = cumulative_cot(path_norm, height_map, img_size, pixel_resolution, limit_angle_deg)
        return float(np.clip(1.0 - _tilde_linear(cot, scale_cot), 0.0, 1.0))

    return 1.0


def instruction_success(
    path_norm: np.ndarray,
    intent_type: str,
    intent_params: dict,
    slope_map_deg: np.ndarray,
    img_size: int,
    start_pos: tuple,
    goal_pos: tuple,
    start_norm: Optional[np.ndarray] = None,
    goal_norm: Optional[np.ndarray] = None,
    baseline_path_norm: Optional[np.ndarray] = None,
    height_map: Optional[np.ndarray] = None,
    pixel_resolution: float = 0.5,
    limit_angle_deg: float = 35.0,
) -> float:
    """ISR adherence score in [0, 1] for intent_type (composite = geometric mean)."""
    if start_norm is None:
        start_norm = np.array([(start_pos[1] / img_size) * 2 - 1,
                               (start_pos[0] / img_size) * 2 - 1], dtype=np.float32)
    if goal_norm is None:
        goal_norm = np.array([(goal_pos[1] / img_size) * 2 - 1,
                              (goal_pos[0] / img_size) * 2 - 1], dtype=np.float32)

    parts = [p for p in intent_type.split("+") if p and p != "baseline"]
    if not parts:
        return 1.0

    scores = []
    for part in parts:
        scores.append(
            _atomic_instruction_score(
                part, path_norm, intent_params, slope_map_deg, img_size,
                start_pos, goal_pos, start_norm, goal_norm, baseline_path_norm,
                height_map, pixel_resolution, limit_angle_deg,
            )
        )
    scores_arr = np.clip(np.array(scores, dtype=np.float64), 0.0, 1.0)
    log_s = np.log(np.maximum(scores_arr, 1e-15))
    return float(np.exp(np.mean(log_s)))


def instruction_success_rate(results: list) -> float:
    """Mean ISR score (each entry in [0, 1])."""
    return float(np.mean(results)) if results else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 3. Energy / Safety metrics
# ═══════════════════════════════════════════════════════════════════════════

def cumulative_cot(
    path_norm: np.ndarray,
    height_map: np.ndarray,
    img_size: int,
    pixel_resolution: float = 0.5,
    limit_angle_deg: float = 35.0,
) -> float:
    """Sum of directional CoT × step_distance along the path."""
    path_px = _norm_to_px(path_norm, img_size)
    total = 0.0
    for i in range(len(path_px) - 1):
        r0, c0 = int(np.clip(path_px[i, 0], 0, img_size - 1)), int(np.clip(path_px[i, 1], 0, img_size - 1))
        r1, c1 = int(np.clip(path_px[i + 1, 0], 0, img_size - 1)), int(np.clip(path_px[i + 1, 1], 0, img_size - 1))
        d_px = np.hypot(r1 - r0, c1 - c0)
        d_m = d_px * pixel_resolution
        if d_m < 1e-8:
            continue
        cot = _calculate_directional_cot(
            height_map[r0, c0], height_map[r1, c1], d_m, limit_angle_deg
        )
        if np.isinf(cot):
            cot = 10.0
        total += cot * d_m
    return total


def mean_slope_along_path(
    path_norm: np.ndarray,
    slope_map_deg: np.ndarray,
    img_size: int,
) -> float:
    rc = _px_int(_norm_to_px(path_norm, img_size), img_size)
    return float(np.mean(slope_map_deg[rc[:, 0], rc[:, 1]]))


def max_slope_along_path(
    path_norm: np.ndarray,
    slope_map_deg: np.ndarray,
    img_size: int,
) -> float:
    rc = _px_int(_norm_to_px(path_norm, img_size), img_size)
    return float(np.max(slope_map_deg[rc[:, 0], rc[:, 1]]))


def risk_integral(
    path_norm: np.ndarray,
    slope_map_deg: np.ndarray,
    img_size: int,
    risk_threshold_deg: float = 15.0,
) -> float:
    """Σ max(0, slope_i − τ_risk) along path."""
    rc = _px_int(_norm_to_px(path_norm, img_size), img_size)
    slopes = slope_map_deg[rc[:, 0], rc[:, 1]]
    return float(np.sum(np.maximum(0, slopes - risk_threshold_deg)))


# ═══════════════════════════════════════════════════════════════════════════
# 4. Path geometry
# ═══════════════════════════════════════════════════════════════════════════

def path_length(path_norm: np.ndarray) -> float:
    """Cumulative L2 length in normalised space."""
    diffs = np.diff(path_norm, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def path_length_metres(
    path_norm: np.ndarray,
    img_size: int,
    pixel_resolution: float = 0.5,
) -> float:
    """Cumulative L2 in metres."""
    path_px = _norm_to_px(path_norm, img_size)
    diffs = np.diff(path_px, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)) * pixel_resolution)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Path similarity (pseudo-label vs. generated)
# ═══════════════════════════════════════════════════════════════════════════

def pointwise_l2(path_a: np.ndarray, path_b: np.ndarray) -> float:
    """Mean pointwise L2 (MSE-style). Paths must be same length."""
    assert len(path_a) == len(path_b), "Paths must be same horizon"
    return float(np.mean(np.linalg.norm(path_a - path_b, axis=1)))


def chamfer_distance(path_a: np.ndarray, path_b: np.ndarray) -> float:
    """Symmetric Chamfer distance: mean of nearest-neighbour distances."""
    from scipy.spatial.distance import cdist
    D = cdist(path_a, path_b)
    d_a2b = D.min(axis=1).mean()
    d_b2a = D.min(axis=0).mean()
    return float((d_a2b + d_b2a) / 2.0)


def frechet_distance(path_a: np.ndarray, path_b: np.ndarray) -> float:
    """Discrete Fréchet distance (DP approach)."""
    n, m = len(path_a), len(path_b)
    ca = np.full((n, m), -1.0)

    def _dist(i, j):
        return np.linalg.norm(path_a[i] - path_b[j])

    def _c(i, j):
        if ca[i, j] > -0.5:
            return ca[i, j]
        d = _dist(i, j)
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i == 0:
            ca[i, j] = max(_c(0, j - 1), d)
        elif j == 0:
            ca[i, j] = max(_c(i - 1, 0), d)
        else:
            ca[i, j] = max(min(_c(i - 1, j), _c(i - 1, j - 1), _c(i, j - 1)), d)
        return ca[i, j]

    return float(_c(n - 1, m - 1))


# ═══════════════════════════════════════════════════════════════════════════
# 6. Cost-based metrics (edge cost along path)
# ═══════════════════════════════════════════════════════════════════════════

def edge_cost_total(
    path_norm: np.ndarray,
    slope_map_rad: np.ndarray,
    height_map: np.ndarray,
    img_size: int,
    alpha: float = 1.0,
    beta: float = 0.8,
    gamma: float = 0.1,
    delta: float = 1.0,
    risk_threshold_deg: float = 15.0,
    intent_type: str = "baseline",
    intent_params: Optional[dict] = None,
    pixel_resolution: float = 0.5,
    limit_angle_deg: float = 35.0,
    start_pos: Optional[tuple] = None,
    goal_pos: Optional[tuple] = None,
) -> float:
    """Compute total 4-term edge cost along a normalised path, using the same
    formula as the A* planner in generate_data.py."""
    if intent_params is None:
        intent_params = {}

    path_px = _norm_to_px(path_norm, img_size)
    n = len(path_px)
    if n < 2:
        return 0.0

    side_info = None
    if start_pos is not None and goal_pos is not None:
        side_info = _precompute_side_bias(start_pos, goal_pos, img_size)

    total = 0.0
    for i in range(n - 1):
        r0 = int(np.clip(path_px[i, 0], 0, img_size - 1))
        c0 = int(np.clip(path_px[i, 1], 0, img_size - 1))
        r1 = int(np.clip(path_px[i + 1, 0], 0, img_size - 1))
        c1 = int(np.clip(path_px[i + 1, 1], 0, img_size - 1))

        d_px = np.hypot(r1 - r0, c1 - c0)
        real_d = d_px * pixel_resolution
        if real_d < 1e-8:
            continue

        cot = _calculate_directional_cot(
            height_map[r0, c0], height_map[r1, c1], real_d, limit_angle_deg
        )
        if np.isinf(cot):
            cot = 10.0

        risk = _calculate_risk(slope_map_rad[r1, c1], risk_threshold_deg)
        intent_pen = _calculate_intent_penalty(
            intent_type, intent_params, (r1, c1), img_size, slope_map_rad, side_info
        )

        step = alpha * real_d + beta * cot * real_d + gamma * risk + delta * intent_pen
        total += step
    return total


def cost_gap(
    ref_cost: float,
    gen_cost: float,
) -> float:
    """Relative cost gap: (gen - ref) / max(ref, 1e-8)."""
    return (gen_cost - ref_cost) / max(abs(ref_cost), 1e-8)


def instruction_adherence_gap(
    path_norm: np.ndarray,
    ref_path_norm: np.ndarray,
    slope_map_deg: np.ndarray,
    img_size: int,
    intent_type: str,
    intent_params: dict,
    start_pos: tuple,
    goal_pos: tuple,
) -> Dict[str, float]:
    """Per-intent adherence gap: difference in sub-metric values between
    generated and reference path."""
    gap = {}
    parts = intent_type.split("+")
    s_norm = np.array([(start_pos[1] / img_size) * 2 - 1,
                       (start_pos[0] / img_size) * 2 - 1])
    g_norm = np.array([(goal_pos[1] / img_size) * 2 - 1,
                       (goal_pos[0] / img_size) * 2 - 1])
    for part in parts:
        if part in ("left_bias", "right_bias"):
            off_gen = _path_centroid_offset(path_norm, s_norm, g_norm)
            off_ref = _path_centroid_offset(ref_path_norm, s_norm, g_norm)
            gap[f"{part}_offset_gap"] = off_gen - off_ref
        elif part in ("avoid_steep", "prefer_flat", "minimize_elevation_change", "energy_efficient"):
            ms_gen = mean_slope_along_path(path_norm, slope_map_deg, img_size)
            ms_ref = mean_slope_along_path(ref_path_norm, slope_map_deg, img_size)
            gap[f"{part}_mean_slope_gap"] = ms_gen - ms_ref
        elif part == "center_bias":
            off_gen = abs(_path_centroid_offset(path_norm, s_norm, g_norm))
            off_ref = abs(_path_centroid_offset(ref_path_norm, s_norm, g_norm))
            gap["center_bias_abs_offset_gap"] = off_gen - off_ref
        elif part == "short_path":
            gap["short_path_len_gap"] = path_length(path_norm) - path_length(ref_path_norm)
    return gap


# ═══════════════════════════════════════════════════════════════════════════
# 7. Aggregate helper
# ═══════════════════════════════════════════════════════════════════════════

def compute_all_metrics(
    path_norm: np.ndarray,
    goal_norm: np.ndarray,
    slope_map_deg: np.ndarray,
    height_map: np.ndarray,
    img_size: int,
    intent_type: str = "baseline",
    intent_params: Optional[dict] = None,
    start_pos: Optional[tuple] = None,
    goal_pos: Optional[tuple] = None,
    ref_path_norm: Optional[np.ndarray] = None,
    pixel_resolution: float = 0.5,
    limit_angle_deg: float = 35.0,
    risk_threshold_deg: float = 15.0,
    alpha: float = 1.0,
    beta: float = 0.8,
    gamma: float = 0.1,
    delta: float = 1.0,
) -> Dict[str, float]:
    """Compute the full metric dict for one (path, terrain, intent) sample."""
    if intent_params is None:
        intent_params = {}

    slope_map_rad = np.radians(slope_map_deg)

    m: Dict[str, float] = {}

    # goal
    m["goal_error"] = goal_error(path_norm, goal_norm)
    m["goal_error_m"] = goal_error_metres(path_norm, goal_norm, img_size, pixel_resolution)

    # instruction
    start_norm = np.array([(start_pos[1] / img_size) * 2 - 1,
                           (start_pos[0] / img_size) * 2 - 1], dtype=np.float32) if start_pos else path_norm[0]
    goal_norm_v = np.array([(goal_pos[1] / img_size) * 2 - 1,
                            (goal_pos[0] / img_size) * 2 - 1], dtype=np.float32) if goal_pos else path_norm[-1]
    m["isr"] = float(instruction_success(
        path_norm, intent_type, intent_params,
        slope_map_deg, img_size, start_pos or (0, 0), goal_pos or (0, 0),
        start_norm, goal_norm_v,
        height_map=height_map,
        pixel_resolution=pixel_resolution,
        limit_angle_deg=limit_angle_deg,
    ))

    # energy / safety
    m["cumulative_cot"] = cumulative_cot(path_norm, height_map, img_size, pixel_resolution, limit_angle_deg)
    m["mean_slope"] = mean_slope_along_path(path_norm, slope_map_deg, img_size)
    m["max_slope"] = max_slope_along_path(path_norm, slope_map_deg, img_size)
    m["risk_integral"] = risk_integral(path_norm, slope_map_deg, img_size, risk_threshold_deg)

    # geometry
    m["path_length"] = path_length(path_norm)
    m["path_length_m"] = path_length_metres(path_norm, img_size, pixel_resolution)

    # cost
    m["edge_cost"] = edge_cost_total(
        path_norm, slope_map_rad, height_map, img_size,
        alpha, beta, gamma, delta, risk_threshold_deg,
        intent_type, intent_params, pixel_resolution, limit_angle_deg,
        start_pos, goal_pos,
    )

    # similarity (if reference available)
    if ref_path_norm is not None:
        m["pointwise_l2"] = pointwise_l2(path_norm, ref_path_norm)
        m["chamfer"] = chamfer_distance(path_norm, ref_path_norm)
        m["frechet"] = frechet_distance(path_norm, ref_path_norm)
        ref_cost = edge_cost_total(
            ref_path_norm, slope_map_rad, height_map, img_size,
            alpha, beta, gamma, delta, risk_threshold_deg,
            intent_type, intent_params, pixel_resolution, limit_angle_deg,
            start_pos, goal_pos,
        )
        m["ref_edge_cost"] = ref_cost
        m["cost_gap"] = cost_gap(ref_cost, m["edge_cost"])

    return m
