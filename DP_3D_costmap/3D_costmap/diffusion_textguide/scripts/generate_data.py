"""
Intent-conditioned pseudo-label path generation (CoRL submission, v2).

Pipeline (per terrain):
  1. height map  : multi-scale Gaussian-smoothed noise
  2. slope map   : finite-difference gradient → arctan magnitude
  3. start/goal  : sampled near opposing corners with feasibility filter
  4. intents     : INTENT_CATALOG (penalty intents + weight modulators)
  5. A* search   : 4-term **dimensionless** step cost (every term in [0, 1])
  6. instruction : sampled from inst_train.json templates per intent

Step cost (v2, all terms ∈ [0, 1]):
  ĉ_ij = α · d̂_ij  +  β · ê(s_j) · d̂_ij  +  γ · R̂(s_j)  +  δ · Î_j(intent)

  • d̂ = real_d / d_max,  d_max = √2 · pixel_resolution
  • ê = (g(s) − g_min) / (g_max − g_min);  g(s) = Go2 4th-degree polynomial in s (deg),
        from ``CoT-Regression/`` (``np.polyfit`` on Table 2). Export JSON via
        ``python3 .../CoT-Regression/Planetary_CoT.py`` → ``data/cot_model.json``.
  • R̂ = clip((|s| − τ_R) / (s_lim − τ_R), 0, 1)
  • Î = mean of atomic intent scores (composites use soft-AND mean)

Two intent classes are handled jointly:
  • Penalty intents  → contribute to δ·Î (left/right/center, avoid_steep,
                       prefer_flat).
  • Weight modulators → scale the base (α, β) before A* runs and contribute
                       0 to δ·Î (short_path, energy_efficient).

Lateral intents use the deterministic straight start→goal line as their
reference (no chicken-and-egg dependency on a baseline A* call).
"""

import os
import heapq
import argparse
import random
import sys
import numpy as np
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

np.random.seed(42)
random.seed(42)

_SCRIPT_DIR = Path(__file__).resolve().parents[0]
_ROOT = _SCRIPT_DIR.parent
_DATA_RAW = _ROOT / "data" / "raw"
sys.path.insert(0, str(_ROOT))

from instruction_utils import load_instruction_templates


# ============================================================================
# Instruction Templates: pseudo label → 자연어 매핑
# ============================================================================

INSTRUCTION_TEMPLATES = load_instruction_templates("train")


def _sample_instruction(intent_type):
    """pseudo label에 대응하는 자연어 instruction을 랜덤 샘플링."""
    templates = INSTRUCTION_TEMPLATES.get(intent_type)
    if templates is not None:
        return random.choice(templates)
    parts = intent_type.split("+")
    combined = []
    for p in parts:
        ts = INSTRUCTION_TEMPLATES.get(p, [])
        if ts:
            combined.append(random.choice(ts))
    return " 그리고 ".join(combined) if combined else intent_type


# ============================================================================
# Intent Catalog (v2)
# ============================================================================
# Two intent classes:
#   (1) Penalty intents  → contribute to the δ·Î_j term in the step cost.
#                          Î ∈ [0, 1] regardless of intent (uniform scale).
#   (2) Weight modulators → adjust the base (α, β) before A* runs;
#                          contribute 0 to δ·Î.
# Composite intents may mix both classes.

# Multipliers applied to (α, β) when the intent string contains the key.
# 1.0 means unchanged.
WEIGHT_MODULATORS = {
    "short_path":       {"alpha_mult": 2.0, "beta_mult": 0.5},
    "energy_efficient": {"alpha_mult": 0.5, "beta_mult": 2.0},
}

# Atomic penalty intents (used by _I_atomic). All others contribute 0 penalty.
PENALTY_INTENTS = {
    "left_bias", "right_bias", "center_bias",
    "avoid_steep", "prefer_flat",
}

INTENT_CATALOG = [
    # --- baseline ---
    {"type": "baseline",                                    "params": {}},

    # --- lateral (deterministic straight-line reference) ---
    {"type": "left_bias",                                   "params": {}},
    {"type": "right_bias",                                  "params": {}},
    {"type": "center_bias",                                 "params": {}},

    # --- terrain-based penalty intents ---
    {"type": "avoid_steep",                                 "params": {"tau_steep_deg": 20.0}},
    {"type": "prefer_flat",                                 "params": {}},

    # --- pure weight modulators (no δ-penalty) ---
    {"type": "short_path",                                  "params": {}},
    {"type": "energy_efficient",                            "params": {}},

    # --- composite (penalty + penalty) ---
    {"type": "left_bias+avoid_steep",                       "params": {"tau_steep_deg": 20.0}},
    {"type": "right_bias+prefer_flat",                      "params": {}},
    {"type": "center_bias+prefer_flat",                     "params": {}},

    # --- composite (modulator + penalty) ---
    {"type": "short_path+avoid_steep",                      "params": {"tau_steep_deg": 20.0}},
]


# ============================================================================
# Cost of Transport — 4th-degree polynomial (Isaac Lab Go2, Table 2)
# ============================================================================
# g(s) = a·s⁴ + b·s³ + c·s² + d·s + e   (s in degrees)
#
# Source of truth: ``CoT-Regression/`` (``coefficient.py`` + ``Planetary_CoT.py``).
# Export JSON for planners:
#     python3 DP_3D_costmap/3D_costmap/CoT-Regression/Planetary_CoT.py
#   → writes ``diffusion_textguide/data/cot_model.json``
#
# Default coefficients match the regression used in ``Planetary_CoT.py``
# before export; after running the script, JSON is overwritten by np.polyfit.

SLOPE_LIMIT_DEG = 25.0
RISK_THRESHOLD_DEG_DEFAULT = 15.0

# Fallback if ``data/cot_model.json`` is missing (same as CoT-Regression/Planetary_CoT.py)
COT_POLY_COEFFS = {
    "a": 6.19e-07,
    "b": 3.72e-05,
    "c": 1.14e-03,
    "d": 2.37e-03,
    "e": 0.44,
}

_G_COT_NORM_CACHE = {"g_min": None, "g_max": None}
_DEFAULT_COT_JSON = _ROOT / "data" / "cot_model.json"


def _g_cot_poly(slope_deg, coeffs=None):
    """Go2 CoT(slope) — degree-4 polynomial in degrees (vectorised)."""
    c = coeffs if coeffs is not None else COT_POLY_COEFFS
    s = np.asarray(slope_deg, dtype=np.float64)
    a, b, cc, d, e = c["a"], c["b"], c["c"], c["d"], c["e"]
    return (a * s**4 + b * s**3 + cc * s**2 + d * s + e).astype(np.float32)


def _refresh_cot_norm_cache():
    """Recompute g_min / g_max over [-SLOPE_LIMIT_DEG, +SLOPE_LIMIT_DEG]."""
    grid = np.linspace(-SLOPE_LIMIT_DEG, SLOPE_LIMIT_DEG, 257)
    vals = _g_cot_poly(grid)
    _G_COT_NORM_CACHE["g_min"] = float(np.min(vals))
    _G_COT_NORM_CACHE["g_max"] = float(np.max(vals))


def set_cot_model(coefficients=None, slope_limit_deg=None):
    """Override the global CoT polynomial at runtime.

    ``coefficients`` dict may contain any subset of keys ``a, b, c, d, e``.
    """
    global SLOPE_LIMIT_DEG
    if coefficients:
        for k in ("a", "b", "c", "d", "e"):
            if k in coefficients:
                COT_POLY_COEFFS[k] = float(coefficients[k])
    if slope_limit_deg is not None:
        SLOPE_LIMIT_DEG = float(slope_limit_deg)
    _refresh_cot_norm_cache()


def _try_load_cot_json():
    if not _DEFAULT_COT_JSON.exists():
        _refresh_cot_norm_cache()
        return
    try:
        import json as _json
        data = _json.loads(_DEFAULT_COT_JSON.read_text())
        model = data.get("model", "poly4_deg")
        if model != "poly4_deg":
            print(f"[generate_data] unsupported cot_model.json model={model!r}; "
                  f"expected 'poly4_deg'. Using built-in coefficients.")
            _refresh_cot_norm_cache()
            return
        coef = data.get("coefficients", {})
        norm = data.get("normalization", {})
        set_cot_model(coefficients=coef, slope_limit_deg=norm.get("slope_limit_deg"))
    except Exception as exc:  # noqa: BLE001
        print(f"[generate_data] failed to load {_DEFAULT_COT_JSON}: {exc}")
        _refresh_cot_norm_cache()


_try_load_cot_json()


def _e_hat(slope_deg):
    """Normalized CoT score ê ∈ [0, 1]."""
    g = float(_g_cot_poly(slope_deg))
    g_min = _G_COT_NORM_CACHE["g_min"]
    g_max = _G_COT_NORM_CACHE["g_max"]
    span = max((g_max - g_min) if (g_max is not None and g_min is not None) else 1.0,
               1e-8)
    return float(np.clip((g - (g_min if g_min is not None else 0.0)) / span,
                         0.0, 1.0))


def _calculate_paper_cot(slope_deg):
    """[Backward compat] alias for the Go2 polynomial CoT."""
    return _g_cot_poly(slope_deg)


def _cot_floor():
    """Minimum CoT over the configured slope band (for directional clamp)."""
    gmn = _G_COT_NORM_CACHE.get("g_min")
    if gmn is not None:
        return float(gmn)
    return float(COT_POLY_COEFFS["e"])


def _calculate_directional_cot(height_curr, height_next, distance, limit_angle_deg=None):
    """Edge-level CoT (i→j). Returns ∞ if implied slope exceeds the limit."""
    if limit_angle_deg is None:
        limit_angle_deg = SLOPE_LIMIT_DEG
    height_diff = height_next - height_curr
    slope_deg = np.degrees(np.arctan2(height_diff, distance))
    if abs(slope_deg) >= limit_angle_deg:
        return np.inf
    cot = float(_g_cot_poly(slope_deg))
    return max(cot, _cot_floor())


# ============================================================================
# Risk (slope safety) — raw and normalized
# ============================================================================

def _calculate_risk(slope_rad_j, risk_threshold_deg=None):
    """[Backward compat] Raw risk = max(0, slope_deg - τ_R) in degrees.

    For the new normalized step cost, prefer ``_R_hat()``.
    """
    if risk_threshold_deg is None:
        risk_threshold_deg = RISK_THRESHOLD_DEG_DEFAULT
    slope_deg_j = np.degrees(slope_rad_j)
    return float(max(0.0, slope_deg_j - risk_threshold_deg))


def _R_hat(slope_rad_j, risk_threshold_deg=None, slope_limit_deg=None):
    """Normalized risk R̂ ∈ [0, 1]."""
    if risk_threshold_deg is None:
        risk_threshold_deg = RISK_THRESHOLD_DEG_DEFAULT
    if slope_limit_deg is None:
        slope_limit_deg = SLOPE_LIMIT_DEG
    s_deg = abs(np.degrees(slope_rad_j))
    span = max(slope_limit_deg - risk_threshold_deg, 1e-6)
    return float(np.clip((s_deg - risk_threshold_deg) / span, 0.0, 1.0))


# ============================================================================
# Lateral reference (deterministic straight start→goal line)
# ============================================================================
# Replaces the v1 baseline-A* dependency for left/right/center intents,
# removing the chicken-and-egg problem (lateral reference depended on the
# very baseline path that was supposed to differ across intents).

# Half-corridor (in pixels) used to normalize signed perpendicular offsets
# for lateral intents. Override at runtime via set_lateral_corridor().
LATERAL_HALF_CORRIDOR_PIXELS = 30.0


def set_lateral_corridor(half_corridor_pixels):
    global LATERAL_HALF_CORRIDOR_PIXELS
    LATERAL_HALF_CORRIDOR_PIXELS = float(half_corridor_pixels)


def _straight_line_reference(start, goal):
    """Pre-compute the straight start→goal reference for lateral intents.

    Returns dict with unit forward / left vectors in (row, col) frame, or
    None if start == goal. The "left" perpendicular matches the visual
    coordinate system used elsewhere (origin='lower'): visual forward
    = (dc, dr), CCW left = (-dr, dc) → (row,col) form left_r = fwd_c,
    left_c = -fwd_r.
    """
    sr, sc = start
    gr, gc = goal
    dr = gr - sr
    dc = gc - sc
    norm = np.hypot(dr, dc)
    if norm < 1e-6:
        return None
    fwd_r = dr / norm
    fwd_c = dc / norm
    return {
        "start": (sr, sc),
        "fwd_r": float(fwd_r), "fwd_c": float(fwd_c),
        "left_r": float(fwd_c), "left_c": float(-fwd_r),
        "length": float(norm),
    }


def _signed_perpendicular_offset(node, ref):
    """Signed perpendicular offset (in pixels, positive = left of forward dir)."""
    if ref is None:
        return 0.0
    rel_r = node[0] - ref["start"][0]
    rel_c = node[1] - ref["start"][1]
    return float(rel_r * ref["left_r"] + rel_c * ref["left_c"])


# ----- Backward-compatible wrappers (used by experiment/core/metrics.py) -----

def _precompute_side_bias(start, goal, img_size):
    """[Backward compat] returns the legacy side-bias dict shape, but the
    underlying reference is the deterministic straight start→goal line."""
    ref = _straight_line_reference(start, goal)
    if ref is None:
        return None
    return {
        "left_r": ref["left_r"],
        "left_c": ref["left_c"],
        "half_range": float(img_size) / 2.0,
        "ref_start": ref["start"],
    }


def _signed_offset_from_baseline(node_j, baseline_points):
    """[Backward compat] signed lateral offset from a discrete baseline.

    Kept only for legacy ablations that pass an explicit baseline curve;
    new code should rely on _straight_line_reference instead.
    """
    if baseline_points is None or len(baseline_points) < 2:
        return 0.0
    p = np.array(node_j, dtype=np.float32)
    pts = np.asarray(baseline_points, dtype=np.float32)
    d2 = np.sum((pts - p) ** 2, axis=1)
    k = int(np.argmin(d2))
    i0 = max(0, k - 1)
    i1 = min(len(pts) - 1, k + 1)
    t = pts[i1] - pts[i0]
    norm = float(np.hypot(t[0], t[1]))
    if norm < 1e-6:
        return 0.0
    left_r = t[1] / norm
    left_c = -t[0] / norm
    rel = p - pts[k]
    return float(rel[0] * left_r + rel[1] * left_c)


# ============================================================================
# Intent score Î ∈ [0, 1]  (penalty intents only; modulators contribute 0)
# ============================================================================

def _I_atomic(intent, node_j, prev_node, slope_map_rad, height_map,
              params, ref):
    """Atomic per-intent score Î ∈ [0, 1]. Modulator/unknown intents → 0."""
    if intent not in PENALTY_INTENTS:
        return 0.0

    # ---- Lateral intents (deterministic straight start→goal line) ----
    if intent in ("left_bias", "right_bias", "center_bias"):
        if ref is None:
            return 0.0
        proj = _signed_perpendicular_offset(node_j, ref)
        u = proj / max(LATERAL_HALF_CORRIDOR_PIXELS, 1e-6)  # +left, -right
        if intent == "left_bias":
            # higher score (worse) when on the right (u < 0)
            return float(np.clip(0.5 - 0.5 * u, 0.0, 1.0))
        if intent == "right_bias":
            return float(np.clip(0.5 + 0.5 * u, 0.0, 1.0))
        # center_bias: penalty grows with |offset|
        return float(np.clip(abs(u), 0.0, 1.0))

    # ---- Slope-based intents ----
    r, c = node_j
    s_deg = float(np.degrees(slope_map_rad[r, c]))

    if intent == "avoid_steep":
        tau = float(params.get("tau_steep_deg", 20.0))
        span = max(SLOPE_LIMIT_DEG - tau, 1e-6)
        return float(np.clip((s_deg - tau) / span, 0.0, 1.0))

    if intent == "prefer_flat":
        return float(np.clip(abs(s_deg) / max(SLOPE_LIMIT_DEG, 1e-6), 0.0, 1.0))

    return 0.0


def _calculate_intent_penalty(intent_type, intent_params, node_j, img_size,
                              slope_map_rad, side_info=None,
                              prev_node=None, height_map=None,
                              ref=None):
    """Î_total ∈ [0, 1] for the intent at node_j.

    Composites combine atomic scores via mean (soft-AND of partial
    satisfactions). Pure modulator intents (e.g. ``short_path``,
    ``energy_efficient``) return 0 — they only modulate (α, β) at planner
    setup time, not the per-step δ·Î term.
    """
    if intent_type == "baseline":
        return 0.0

    # Resolve reference: prefer new ref; otherwise reconstruct from legacy side_info.
    if ref is None and isinstance(side_info, dict):
        if "start" in side_info and "fwd_r" in side_info:
            ref = side_info
        elif "ref_start" in side_info and "left_r" in side_info:
            ref = {
                "start": side_info["ref_start"],
                "left_r": float(side_info["left_r"]),
                "left_c": float(side_info["left_c"]),
                "fwd_r": float(-side_info["left_c"]),
                "fwd_c": float(side_info["left_r"]),
                "length": float(side_info.get("half_range", 1.0)) * 2.0,
            }

    parts = intent_type.split("+")
    penalty_parts = [p for p in parts if p in PENALTY_INTENTS]
    if not penalty_parts:
        return 0.0

    scores = [
        _I_atomic(sub, node_j, prev_node, slope_map_rad,
                  height_map, intent_params or {}, ref)
        for sub in penalty_parts
    ]
    return float(np.mean(scores))


def _single_intent_penalty(intent_type, params, node_j, img_size,
                           slope_map_rad, side_info=None):
    """[Backward compat] thin wrapper over the new _calculate_intent_penalty."""
    return _calculate_intent_penalty(intent_type, params, node_j, img_size,
                                     slope_map_rad, side_info=side_info)


# ============================================================================
# A* with 4-term transition cost
# ============================================================================

def _a_star_intent_search(slope_map, height_map, start, goal,
                          limit_angle_rad, max_iterations,
                          pixel_resolution=0.5,
                          alpha=1.0, beta=0.8, gamma=0.1, delta=1.0,
                          risk_threshold_deg=None,
                          intent_type="baseline", intent_params=None,
                          side_info=None, ref=None):
    """A* over a normalized 4-term step cost (each term ∈ [0, 1]):

        ĉ_ij = α · d̂_ij  +  β · ê(s_j) · d̂_ij  +  γ · R̂(s_j)  +  δ · Î_j(intent)

    where d̂ = real_d / d_max, ê = (g(s) - g_min) / (g_max - g_min),
    R̂ = clip((|s| - τ_R) / (s_lim - τ_R), 0, 1), Î ∈ [0, 1] from
    _calculate_intent_penalty().
    """
    if intent_params is None:
        intent_params = {}
    if risk_threshold_deg is None:
        risk_threshold_deg = RISK_THRESHOLD_DEG_DEFAULT

    rows, cols = height_map.shape
    start, goal = tuple(start), tuple(goal)
    limit_angle_deg = float(np.degrees(limit_angle_rad))

    if slope_map[start] >= limit_angle_rad or slope_map[goal] >= limit_angle_rad:
        return None

    # Lateral reference: use the deterministic straight start→goal line.
    if ref is None:
        ref = _straight_line_reference(start, goal)

    map_size = rows * cols
    if max_iterations < map_size * 10:
        max_iterations = int(map_size * 10)

    d_max = float(np.sqrt(2.0) * pixel_resolution)
    g_min = _G_COT_NORM_CACHE["g_min"] or 0.0
    g_max = _G_COT_NORM_CACHE["g_max"] or 1.0
    e_span = max(g_max - g_min, 1e-8)

    # Admissible heuristic: α · D(n, goal) / d_max  (normalized distance only).
    def heuristic(a, b):
        D = np.hypot(a[0] - b[0], a[1] - b[1]) * pixel_resolution
        return alpha * (D / d_max)

    open_heap = [(heuristic(start, goal), 0, start)]
    came_from = {}
    g_score = np.full((rows, cols), np.inf, dtype=np.float32)
    g_score[start] = 0.0
    closed_set = set()
    counter = 0
    iterations = 0

    while open_heap and iterations < max_iterations:
        iterations += 1
        _, _, current = heapq.heappop(open_heap)
        if current in closed_set:
            continue
        closed_set.add(current)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        cr, cc = current
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or (nr, nc) in closed_set:
                continue

            pixel_d = np.sqrt(2.0) if (dr != 0 and dc != 0) else 1.0
            real_d = pixel_d * pixel_resolution
            d_hat = real_d / d_max

            cot_raw = _calculate_directional_cot(
                height_map[cr, cc], height_map[nr, nc], real_d, limit_angle_deg
            )
            if np.isinf(cot_raw) or slope_map[nr, nc] >= limit_angle_rad:
                continue
            if abs(dr) + abs(dc) == 2:
                if (slope_map[cr + dr, cc] >= limit_angle_rad or
                        slope_map[cr, cc + dc] >= limit_angle_rad):
                    continue

            e_hat = float(np.clip((cot_raw - g_min) / e_span, 0.0, 1.0))
            r_hat = _R_hat(slope_map[nr, nc], risk_threshold_deg, limit_angle_deg)
            i_hat = _calculate_intent_penalty(
                intent_type, intent_params, (nr, nc), rows, slope_map,
                ref=ref, prev_node=current, height_map=height_map,
            )

            step_cost = (alpha * d_hat
                         + beta * e_hat * d_hat
                         + gamma * r_hat
                         + delta * i_hat)

            g_new = g_score[cr, cc] + step_cost
            if g_new < g_score[nr, nc]:
                came_from[(nr, nc)] = current
                g_score[nr, nc] = g_new
                counter += 1
                heapq.heappush(
                    open_heap,
                    (g_new + heuristic((nr, nc), goal) * (1.0 + 1e-3), counter, (nr, nc))
                )

    return None


# ============================================================================
# Path utilities
# ============================================================================

def _resample_path(path, horizon):
    """[N,2] → [horizon,2] 고정 길이 리샘플링."""
    path = np.asarray(path, dtype=np.float32)
    n = path.shape[0]
    if n == 0:
        return np.zeros((horizon, 2), dtype=np.float32)
    t_cur = np.linspace(0, 1, n)
    t_tgt = np.linspace(0, 1, horizon)
    x = np.interp(t_tgt, t_cur, path[:, 0])
    y = np.interp(t_tgt, t_cur, path[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def _path_pixels_to_normalized(path_pixels, img_size):
    """[(row,col),...] → [N,2] (x,y) 정규화 [-1,1]."""
    if not path_pixels:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.array(path_pixels, dtype=np.float32)[:, [1, 0]]
    return ((arr / img_size) * 2 - 1).astype(np.float32)


# ============================================================================
# SlopeCotGenerator
# ============================================================================

class SlopeCotGenerator:
    """Slope + CoT 지형 생성 및 intent 기반 A* 경로 계획."""

    def __init__(self, img_size, height_range, mass, gravity, limit_angle_deg,
                 max_iterations, pixel_resolution=0.5):
        self.img_size = img_size
        self.height_range = tuple(height_range)
        self.mass = mass
        self.gravity = gravity
        self.limit_angle = np.radians(limit_angle_deg)
        self.pixel_resolution = pixel_resolution
        self.max_iterations = max_iterations
        self.height_map = None
        self.slope_map = None

    def generate(self, terrain_scales):
        """지형 생성. terrain_scales: [(scale, weight), ...]."""
        if terrain_scales is None:
            raise ValueError("terrain_scales must be provided")
        self.height_map = self._height_map(terrain_scales)
        self.slope_map = self._slope_map(self.height_map)
        return self.height_map, self.slope_map

    def _height_map(self, terrain_scales):
        h = np.zeros((self.img_size, self.img_size))
        for scale, weight in terrain_scales:
            noise = np.random.rand(self.img_size, self.img_size)
            h += gaussian_filter(noise, sigma=scale) * weight
        h = (h - h.min()) / (h.max() - h.min() + 1e-12)
        h = h * (self.height_range[1] - self.height_range[0]) + self.height_range[0]
        return h.astype(np.float32)

    def _slope_map(self, height_map):
        gy, gx = np.gradient(height_map, self.pixel_resolution)
        mag = np.sqrt(gx**2 + gy**2)
        return np.arctan(mag).astype(np.float32)

    def find_path_with_intent(self, start, goal,
                              alpha=1.0, beta=0.8, gamma=0.1, delta=1.0,
                              risk_threshold_deg=None,
                              intent_type="baseline", intent_params=None):
        """Plan an intent-conditioned path with the v2 normalized cost.

        Two intent classes are handled here:

        * **Penalty intents** (left_bias / right_bias / center_bias /
          avoid_steep / prefer_flat) contribute to
          the δ·Î term inside A*; they all use the deterministic straight
          start→goal line as their lateral reference (no chicken-and-egg
          dependency on a baseline A* call).

        * **Weight modulators** (short_path, energy_efficient) instead scale
          the base (α, β) before A* runs and contribute 0 to δ·Î. Their
          intuition (“take a shorter route”, “prefer low-energy ground”) is
          encoded directly in the cost weights rather than as an extra
          penalty term.
        """
        if self.height_map is None or self.slope_map is None:
            raise RuntimeError("generate()를 먼저 호출하세요.")
        if risk_threshold_deg is None:
            risk_threshold_deg = RISK_THRESHOLD_DEG_DEFAULT

        a_eff, b_eff = float(alpha), float(beta)
        for sub in intent_type.split("+"):
            mod = WEIGHT_MODULATORS.get(sub)
            if mod is not None:
                a_eff *= mod.get("alpha_mult", 1.0)
                b_eff *= mod.get("beta_mult", 1.0)

        ref = _straight_line_reference(start, goal)
        return _a_star_intent_search(
            self.slope_map, self.height_map, start, goal,
            self.limit_angle, self.max_iterations,
            pixel_resolution=self.pixel_resolution,
            alpha=a_eff, beta=b_eff, gamma=gamma, delta=delta,
            risk_threshold_deg=risk_threshold_deg,
            intent_type=intent_type,
            intent_params=intent_params or {},
            ref=ref,
        )


# ============================================================================
# 데이터 생성
# ============================================================================

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_terrain_data(
    num_terrains=100,
    intents=None,
    img_size=100,
    horizon=120,
    height_range=(0, 5),
    mass=10.0,
    gravity=9.8,
    limit_angle_deg=25,
    pixel_resolution=0.5,
    terrain_scales=None,
    min_distance_factor=1.5,
    max_iterations=20000,
    alpha=1.0, beta=0.8, gamma=0.1, delta=1.0,
    risk_threshold_deg=15.0,
    output_dir="data/raw",
    debug=False,
):
    """동일 terrain·start-goal에서 intent별 경로를 생성하고 .pt로 저장."""
    if intents is None:
        intents = INTENT_CATALOG

    os.makedirs(output_dir, exist_ok=True)
    margin = img_size // 10
    min_distance = int(img_size // min_distance_factor)

    print("=" * 60)
    print("Intent-based Terrain & Path Generation")
    print("=" * 60)
    print(f"  Grid       : {img_size}x{img_size}")
    print(f"  Horizon    : {horizon}")
    print(f"  Terrains   : {num_terrains}")
    print(f"  Intents    : {[i['type'] for i in intents]}")
    print(f"  Cost (α,β,γ,δ) : ({alpha}, {beta}, {gamma}, {delta})")
    print(f"  Risk thresh: {risk_threshold_deg}°")
    print(f"  Output     : {output_dir}")
    print("=" * 60)

    generated = []
    stats = {"paths_count": [], "mean_slope": []}
    pbar = tqdm(total=num_terrains, desc="Generating")
    n_done = 0
    attempts = 0
    max_attempts = num_terrains * 30

    while n_done < num_terrains and attempts < max_attempts:
        attempts += 1

        gen = SlopeCotGenerator(
            img_size=img_size,
            height_range=height_range,
            mass=mass, gravity=gravity,
            limit_angle_deg=limit_angle_deg,
            max_iterations=max_iterations,
            pixel_resolution=pixel_resolution,
        )
        h_map, s_map = gen.generate(terrain_scales=terrain_scales)

        slope_deg = np.degrees(s_map)
        mean_slope = float(np.mean(slope_deg))
        max_slope = float(np.max(slope_deg))
        steep_ratio = np.sum(slope_deg > 30.0) / slope_deg.size
        if mean_slope < 8.0 or mean_slope > 32.0 or max_slope > 55.0 or steep_ratio > 0.55:
            continue

        # --- start-goal 샘플링 (반대편 벽 근처) ---
        # start는 맵 한쪽 코너 근처, goal은 대각선 반대쪽 코너 근처
        edge_lo = margin
        edge_hi = margin * 3
        far_lo = img_size - margin * 3
        far_hi = img_size - margin
        start, goal = None, None
        for _ in range(200):
            side = np.random.randint(4)
            if side == 0:      # 좌하 → 우상
                s = (np.random.randint(edge_lo, edge_hi), np.random.randint(edge_lo, edge_hi))
                g = (np.random.randint(far_lo, far_hi),   np.random.randint(far_lo, far_hi))
            elif side == 1:    # 우하 → 좌상
                s = (np.random.randint(edge_lo, edge_hi), np.random.randint(far_lo, far_hi))
                g = (np.random.randint(far_lo, far_hi),   np.random.randint(edge_lo, edge_hi))
            elif side == 2:    # 좌상 → 우하
                s = (np.random.randint(far_lo, far_hi),   np.random.randint(edge_lo, edge_hi))
                g = (np.random.randint(edge_lo, edge_hi), np.random.randint(far_lo, far_hi))
            else:              # 우상 → 좌하
                s = (np.random.randint(far_lo, far_hi),   np.random.randint(far_lo, far_hi))
                g = (np.random.randint(edge_lo, edge_hi), np.random.randint(edge_lo, edge_hi))
            d = np.sqrt((g[0]-s[0])**2 + (g[1]-s[1])**2)
            if (d >= img_size * 0.6
                    and s_map[s] < gen.limit_angle
                    and s_map[g] < gen.limit_angle):
                start, goal = s, g
                break
        if start is None:
            continue

        # --- intent별 경로 생성 ---
        paths_data = []
        min_path_len = max(5, int(min_distance * 0.1))

        for intent_def in intents:
            itype = intent_def["type"]
            iparams = dict(intent_def["params"])

            # Resolve effective (α, β) after applying weight modulators —
            # we record the *applied* values for reproducibility.
            a_eff, b_eff = float(alpha), float(beta)
            modulators = []
            for sub in itype.split("+"):
                mod = WEIGHT_MODULATORS.get(sub)
                if mod is not None:
                    a_eff *= mod.get("alpha_mult", 1.0)
                    b_eff *= mod.get("beta_mult", 1.0)
                    modulators.append({"intent": sub, **mod})

            path_pixels = gen.find_path_with_intent(
                start, goal,
                alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                risk_threshold_deg=risk_threshold_deg,
                intent_type=itype,
                intent_params=iparams,
            )

            if path_pixels is None or len(path_pixels) <= min_path_len:
                if debug:
                    print(f"  [skip] terrain attempt {attempts}, intent={itype}: no valid path")
                continue

            norm = _path_pixels_to_normalized(path_pixels, img_size)
            fixed = _resample_path(norm, horizon)
            instruction = _sample_instruction(itype)

            pseudo_label = {
                "intent_type": itype,
                "intent_params": iparams,
                "cost_weights_base": {
                    "alpha": float(alpha), "beta": float(beta),
                    "gamma": float(gamma), "delta": float(delta),
                },
                "cost_weights_effective": {
                    "alpha": float(a_eff), "beta": float(b_eff),
                    "gamma": float(gamma), "delta": float(delta),
                },
                "modulators_applied": modulators,
                "risk_threshold_deg": float(risk_threshold_deg),
                "cot_model": {"model": "poly4_deg", **dict(COT_POLY_COEFFS)},
                "slope_limit_deg": float(SLOPE_LIMIT_DEG),
            }

            paths_data.append({
                "path_normalized": fixed,
                "intent_type": itype,
                "intent_params": iparams,
                "instruction": instruction,
                "pseudo_label": pseudo_label,
            })

        if len(paths_data) < 2:
            continue

        # --- 저장 ---
        n_done += 1
        slope_norm = slope_deg / 90.0
        height_norm = (h_map - h_map.min()) / (h_map.max() - h_map.min() + 1e-8)
        costmap = np.stack([slope_norm, height_norm], axis=0)

        map_id = f"terrain_{n_done:05d}"
        save_path = os.path.join(output_dir, f"{map_id}.pt")

        torch.save({
            "map_id": map_id,
            "costmap": torch.from_numpy(costmap).float(),
            "height_map": torch.from_numpy(h_map).float(),
            "slope_map": torch.from_numpy(slope_deg).float(),
            "paths": torch.from_numpy(
                np.array([p["path_normalized"] for p in paths_data])
            ).float(),
            "intent_types": [p["intent_type"] for p in paths_data],
            "intent_params": [p["intent_params"] for p in paths_data],
            "instructions": [p["instruction"] for p in paths_data],
            "pseudo_labels": [p["pseudo_label"] for p in paths_data],
            "start_position": start,
            "goal_position": goal,
            "cost_weights": {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta},
            "risk_threshold_deg": risk_threshold_deg,
            "img_size": img_size,
            "horizon": horizon,
            "pixel_resolution": pixel_resolution,
            "limit_angle_deg": limit_angle_deg,
        }, save_path)

        generated.append(save_path)
        stats["paths_count"].append(len(paths_data))
        stats["mean_slope"].append(mean_slope)
        pbar.update(1)
        pbar.set_postfix(paths=len(paths_data), slope=f"{mean_slope:.1f}°")

    pbar.close()
    print("\n" + "=" * 60)
    print("Summary")
    print(f"  Terrains generated : {n_done}/{num_terrains}")
    print(f"  Files saved        : {len(generated)}")
    if stats["paths_count"]:
        print(f"  Avg paths/terrain  : {np.mean(stats['paths_count']):.1f}")
        print(f"  Avg mean slope     : {np.mean(stats['mean_slope']):.1f}°")
    print("=" * 60)
    return generated


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Generate intent-based terrain + path data")
    ap.add_argument("--config", type=str, default=None, help="Config YAML path")
    ap.add_argument("--num-terrains", type=int, default=100)
    ap.add_argument("--output-dir", type=str, default=None,
                    help=f"Output directory (default: {_DATA_RAW})")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    output_dir = args.output_dir or str(_DATA_RAW)

    if args.config:
        config = load_config(args.config)
    else:
        cfg_path = _ROOT / "configs" / "default.yaml"
        if not cfg_path.exists():
            cfg_path = Path(__file__).resolve().parents[2] / "diffusion_textguide" / "configs" / "default.yaml"
        if cfg_path.exists():
            config = load_config(str(cfg_path))
            print(f"Using config: {cfg_path}")
        else:
            config = None

    if config:
        d = config.get("data", {})
        g = config.get("gradient", {})
        ic = config.get("intent", {})
        cw = ic.get("cost_weights", {})

        generate_terrain_data(
            num_terrains=args.num_terrains,
            img_size=d.get("img_size", 100),
            horizon=d.get("horizon", 120),
            height_range=tuple(g.get("height_range", [0, 5])),
            mass=g.get("mass", 10.0),
            gravity=g.get("gravity", 9.8),
            limit_angle_deg=g.get("limit_angle_deg", 25),
            pixel_resolution=g.get("pixel_resolution", 0.5),
            terrain_scales=g.get("terrain_scales"),
            min_distance_factor=d.get("min_distance_factor", 1.5),
            max_iterations=g.get("max_iterations", 20000),
            alpha=cw.get("alpha", 1.0),
            beta=cw.get("beta", 0.8),
            gamma=cw.get("gamma", 0.1),
            delta=cw.get("delta", 1.0),
            risk_threshold_deg=ic.get("risk_threshold_deg", 15.0),
            output_dir=output_dir,
            debug=args.debug,
        )
    else:
        generate_terrain_data(
            num_terrains=args.num_terrains,
            output_dir=output_dir,
            debug=args.debug,
        )


if __name__ == "__main__":
    main()
