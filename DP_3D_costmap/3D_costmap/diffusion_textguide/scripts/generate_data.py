"""
Intent 기반 지형 경로 생성 스크립트 (Section 6 구현)

파이프라인:
  1. height map 생성 (Gaussian-smoothed multi-scale)
  2. slope map 계산
  3. start-goal 샘플링
  4. pseudo label 기반 intent 정의
  5. 4-term cost A* (dist + CoT + Risk + IntentPenalty) 로 GT path 생성
  6. 자연어 instruction 자동 부착

4-term transition cost:
  c_{ij} = α·d_{ij} + β·CoT_{ij}·d_{ij} + γ·Risk_{ij} + δ·IntentPenalty_{ij}

동일 terrain + 동일 start-goal에서 intent만 변경하여 복수 경로 생성
"""

import os
import heapq
import argparse
import random
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


# ============================================================================
# Instruction Templates: pseudo label → 자연어 매핑
# ============================================================================

INSTRUCTION_TEMPLATES = {
    "baseline": [
        "Navigate along the default route",
        "Follow a balanced path",
        "Take the standard route",
        "Go without any special constraints",
        "Find an efficient path to the goal",
    ],
    "left_bias": [
        "Stay to the left side",
        "Follow the left corridor",
        "Keep to the left as much as possible",
        "Veer left on your way to the goal",
        "Take the leftward path",
    ],
    "right_bias": [
        "Stay to the right side",
        "Follow the right corridor",
        "Keep to the right as much as possible",
        "Veer right on your way to the goal",
        "Take the rightward path",
    ],
    "center_bias": [
        "Stay close to the baseline route",
        "Follow the center-aligned path",
        "Keep near the middle route",
        "Do not drift too far from the baseline path",
        "Take a path close to the standard route",
    ],
    "avoid_steep": [
        "Avoid steep slopes",
        "Stay away from steep terrain",
        "Bypass any high-gradient areas",
        "Do not cross steep inclines",
        "Route around the steepest sections",
    ],
    "prefer_flat": [
        "Follow flat terrain",
        "Prefer gentle slopes",
        "Stick to the flattest ground possible",
        "Choose the most level path available",
        "Stay on the smoothest terrain you can find",
    ],
    "minimize_elevation_change": [
        "Minimize elevation changes along the way",
        "Take a route with as little height change as possible",
        "Avoid unnecessary ups and downs",
        "Follow a path with minimal elevation variation",
        "Keep the altitude changes as small as possible",
    ],
    "short_path": [
        "Take the shortest route possible",
        "Use the most direct path to the goal",
        "Keep the route as short as you can",
        "Choose the most direct way forward",
        "Minimize the travel distance",
    ],
    "energy_efficient": [
        "Choose the most energy-efficient route",
        "Minimize traversal effort on the way to the goal",
        "Follow a path that uses the least energy",
        "Take the most efficient route in terms of effort",
        "Reduce the energy cost as much as possible",
    ],
    "left_bias+avoid_steep": [
        "Stay left and avoid steep areas",
        "Keep to the left while bypassing steep slopes",
        "Veer left but steer clear of high gradients",
        "Follow the left side and route around steep terrain",
        "Take a leftward path that avoids steep sections",
    ],
    "right_bias+prefer_flat": [
        "Stay right and follow flat terrain",
        "Keep to the right along gentle slopes",
        "Veer right while preferring level ground",
        "Take the rightward path on the flattest terrain",
        "Follow the right side while staying on smooth ground",
    ],
    "center_bias+prefer_flat": [
        "Stay near the baseline route and follow flat terrain",
        "Keep close to the standard path while preferring gentle slopes",
        "Take a center-aligned route over level ground",
        "Follow the baseline direction on the flattest terrain",
        "Stay near the middle route while keeping to smooth ground",
    ],
    "short_path+avoid_steep": [
        "Take a short route while avoiding steep areas",
        "Choose the most direct path that avoids steep slopes",
        "Keep the path short but stay away from steep terrain",
        "Find a compact route around steep sections",
        "Minimize distance without crossing steep areas",
    ],
    "energy_efficient+minimize_elevation_change": [
        "Choose an energy-efficient route with minimal elevation change",
        "Minimize effort while avoiding unnecessary ups and downs",
        "Take a low-energy path with as little height variation as possible",
        "Follow the most efficient route while keeping elevation changes small",
        "Reduce energy cost and keep the route as smooth in height as possible",
    ],
}


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
# Intent Catalog
# ============================================================================

INTENT_CATALOG = [
    {"type": "baseline",                             "params": {}},
    {"type": "left_bias",                            "params": {"lambda_side": 5.0}},
    {"type": "right_bias",                           "params": {"lambda_side": 5.0}},
    {"type": "center_bias",                          "params": {"lambda_center": 5.0}},
    {"type": "avoid_steep",                          "params": {"lambda_steep": 5.0, "tau_steep": 20.0}},
    {"type": "prefer_flat",                          "params": {"lambda_flat": 1.0}},
    {"type": "minimize_elevation_change",           "params": {"lambda_elev": 0.08}},
    {"type": "short_path",                           "params": {"lambda_detour": 4.0}},
    {"type": "energy_efficient",                    "params": {"lambda_energy": 0.12}},
    {"type": "left_bias+avoid_steep",               "params": {
        "lambda_side": 5.0, "lambda_steep": 5.0, "tau_steep": 20.0}},
    {"type": "right_bias+prefer_flat",               "params": {
        "lambda_side": 5.0, "lambda_flat": 1.0}},
    {"type": "center_bias+prefer_flat",              "params": {
        "lambda_center": 5.0, "lambda_flat": 1.0}},
    {"type": "short_path+avoid_steep",               "params": {
        "lambda_detour": 4.0, "lambda_steep": 5.0, "tau_steep": 20.0}},
    {"type": "energy_efficient+minimize_elevation_change", "params": {
        "lambda_energy": 0.12, "lambda_elev": 0.08}},
]


# ============================================================================
# CoT 계산
# ============================================================================

def _calculate_paper_cot(slope_deg):
    """4차 다항식 기반 CoT (Minetti et al.)."""
    a, b, c, d, e = -1.53e-06, 2.07e-05, 2.20e-03, -3.24e-02, 0.65
    return (a * slope_deg**4) + (b * slope_deg**3) + (c * slope_deg**2) + (d * slope_deg) + e


def _calculate_directional_cot(height_curr, height_next, distance, limit_angle_deg=35.0):
    """이동 방향 기반 CoT. 등반 불가 시 np.inf."""
    height_diff = height_next - height_curr
    slope_deg = np.degrees(np.arctan2(height_diff, distance))
    if abs(slope_deg) >= limit_angle_deg:
        return np.inf
    cot = _calculate_paper_cot(slope_deg)
    return max(cot, 0.1)


# ============================================================================
# Risk_{ij} 계산
# ============================================================================

def _calculate_risk(slope_rad_j, risk_threshold_deg=15.0):
    """Slope-based soft safety cost.

    Risk_{ij} = max(0, slope_deg(v_j) - τ_risk)
    """
    slope_deg_j = np.degrees(slope_rad_j)
    return max(0.0, slope_deg_j - risk_threshold_deg)


# ============================================================================
# IntentPenalty_{ij} 계산
# ============================================================================

def _precompute_side_bias(start, goal, img_size):
    """start→goal 방향 기준 좌/우 판별용 벡터 사전 계산.

    시각화에서 origin='lower' (row↑=화면↑)이므로,
    시각적 좌표계(right-handed)에서의 90° CCW 회전:
      visual forward = (dc, dr),  visual left = (-dr, dc)
      → (row,col) 형식: left_r = dc/norm, left_c = -dr/norm

    반환: dict(side reference) 또는 None (start==goal).
    """
    dr = goal[0] - start[0]
    dc = goal[1] - start[1]
    norm = np.hypot(dr, dc)
    if norm < 1e-6:
        return None
    left_r = dc / norm
    left_c = -dr / norm
    half_range = img_size / 2.0
    return {
        "left_r": left_r,
        "left_c": left_c,
        "half_range": half_range,
        "ref_start": start,
    }


def _signed_offset_from_baseline(node_j, baseline_points):
    """Return signed lateral offset (pixels) from nearest baseline segment.

    Positive means left of local baseline tangent in (row, col) coordinates.
    """
    if baseline_points is None or len(baseline_points) < 2:
        return 0.0
    p = np.array(node_j, dtype=np.float32)
    pts = np.asarray(baseline_points, dtype=np.float32)
    d2 = np.sum((pts - p) ** 2, axis=1)
    k = int(np.argmin(d2))
    i0 = max(0, k - 1)
    i1 = min(len(pts) - 1, k + 1)
    t = pts[i1] - pts[i0]  # local tangent in (row, col)
    norm = float(np.hypot(t[0], t[1]))
    if norm < 1e-6:
        return 0.0
    left_r = t[1] / norm
    left_c = -t[0] / norm
    rel = p - pts[k]
    return float(rel[0] * left_r + rel[1] * left_c)


def _calculate_intent_penalty(intent_type, intent_params, node_j, img_size,
                               slope_map_rad, side_info=None):
    """IntentPenalty_{ij} 계산. compositional intent는 '+' 구분자로 합산."""
    if intent_type == "baseline":
        return 0.0
    if "+" in intent_type:
        total = 0.0
        for sub in intent_type.split("+"):
            total += _single_intent_penalty(sub, intent_params, node_j,
                                            img_size, slope_map_rad, side_info)
        return total
    return _single_intent_penalty(intent_type, intent_params, node_j,
                                  img_size, slope_map_rad, side_info)


def _single_intent_penalty(intent_type, params, node_j, img_size,
                            slope_map_rad, side_info=None):
    """단일 intent에 대한 penalty.

    - left_bias:    start→goal 이동 방향 기준 오른쪽일수록 penalty ↑
    - right_bias:   start→goal 이동 방향 기준 왼쪽일수록 penalty ↑
    - avoid_steep:  λ_steep · max(0, S - τ)   (급경사 회피)
    - prefer_flat:  λ_flat · S(v_j)           (slope 비례 penalty)
    """
    r, c = node_j

    if intent_type in ("left_bias", "right_bias"):
        if side_info is None:
            return 0.0
        if isinstance(side_info, dict) and side_info.get("baseline_points") is not None:
            proj = _signed_offset_from_baseline((r, c), side_info.get("baseline_points"))
            half_range = float(side_info.get("half_range", img_size / 2.0))
        else:
            left_r = side_info["left_r"] if isinstance(side_info, dict) else side_info[0]
            left_c = side_info["left_c"] if isinstance(side_info, dict) else side_info[1]
            half_range = side_info["half_range"] if isinstance(side_info, dict) else side_info[2]
            ref_start = side_info["ref_start"] if isinstance(side_info, dict) else side_info[3]
            proj = (r - ref_start[0]) * left_r + (c - ref_start[1]) * left_c
        u_perp = np.clip(0.5 - proj / (2.0 * half_range), 0.0, 1.0)
        lam = params.get("lambda_side", 3.0)
        if intent_type == "left_bias":
            return lam * u_perp
        return lam * (1.0 - u_perp)

    if intent_type == "avoid_steep":
        slope_deg_j = np.degrees(slope_map_rad[r, c])
        tau = params.get("tau_steep", 20.0)
        lam = params.get("lambda_steep", 5.0)
        return lam * max(0.0, slope_deg_j - tau)

    if intent_type == "prefer_flat":
        slope_deg_j = np.degrees(slope_map_rad[r, c])
        return params.get("lambda_flat", 1.0) * slope_deg_j

    if intent_type == "center_bias":
        if side_info is None:
            return 0.0
        if isinstance(side_info, dict) and side_info.get("baseline_points") is not None:
            proj = _signed_offset_from_baseline((r, c), side_info.get("baseline_points"))
            half_range = float(side_info.get("half_range", img_size / 2.0))
        else:
            left_r = side_info["left_r"] if isinstance(side_info, dict) else side_info[0]
            left_c = side_info["left_c"] if isinstance(side_info, dict) else side_info[1]
            half_range = side_info["half_range"] if isinstance(side_info, dict) else side_info[2]
            ref_start = side_info["ref_start"] if isinstance(side_info, dict) else side_info[3]
            proj = (r - ref_start[0]) * left_r + (c - ref_start[1]) * left_c
        u = proj / max(2.0 * half_range, 1e-6)
        lam = params.get("lambda_center", 5.0)
        return lam * abs(u)

    if intent_type == "minimize_elevation_change":
        slope_deg_j = np.degrees(slope_map_rad[r, c])
        lam = params.get("lambda_elev", 0.08)
        return lam * (slope_deg_j ** 2) / 100.0

    if intent_type == "short_path":
        if side_info is None:
            return 0.0
        if isinstance(side_info, dict) and side_info.get("baseline_points") is not None:
            proj = _signed_offset_from_baseline((r, c), side_info.get("baseline_points"))
            half_range = float(side_info.get("half_range", img_size / 2.0))
        else:
            left_r = side_info["left_r"] if isinstance(side_info, dict) else side_info[0]
            left_c = side_info["left_c"] if isinstance(side_info, dict) else side_info[1]
            half_range = side_info["half_range"] if isinstance(side_info, dict) else side_info[2]
            ref_start = side_info["ref_start"] if isinstance(side_info, dict) else side_info[3]
            proj = (r - ref_start[0]) * left_r + (c - ref_start[1]) * left_c
        u = abs(proj) / max(2.0 * half_range, 1e-6)
        lam = params.get("lambda_detour", 4.0)
        return lam * (u ** 2)

    if intent_type == "energy_efficient":
        slope_deg_j = np.degrees(slope_map_rad[r, c])
        cot = _calculate_paper_cot(slope_deg_j)
        lam = params.get("lambda_energy", 0.12)
        return lam * max(cot, 0.0)

    return 0.0


# ============================================================================
# A* with 4-term transition cost
# ============================================================================

def _a_star_intent_search(slope_map, height_map, start, goal,
                          limit_angle_rad, max_iterations,
                          pixel_resolution=0.5,
                          alpha=1.0, beta=0.8, gamma=0.1, delta=1.0,
                          risk_threshold_deg=15.0,
                          intent_type="baseline", intent_params=None,
                          side_info=None):
    """4-term cost A*.

    c_{ij} = α·d_{ij} + β·CoT_{ij}·d_{ij} + γ·Risk_{ij} + δ·IntentPenalty_{ij}
    """
    if intent_params is None:
        intent_params = {}

    rows, cols = height_map.shape
    start, goal = tuple(start), tuple(goal)
    limit_angle_deg = np.degrees(limit_angle_rad)

    if slope_map[start] >= limit_angle_rad or slope_map[goal] >= limit_angle_rad:
        return None

    if side_info is None:
        side_info = _precompute_side_bias(start, goal, rows)

    map_size = rows * cols
    if max_iterations < map_size * 10:
        max_iterations = int(map_size * 10)

    h_scale = alpha * 0.1

    def heuristic(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1]) * pixel_resolution * h_scale

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

            cot = _calculate_directional_cot(
                height_map[cr, cc], height_map[nr, nc], real_d, limit_angle_deg
            )
            if np.isinf(cot) or slope_map[nr, nc] >= limit_angle_rad:
                continue
            if abs(dr) + abs(dc) == 2:
                if (slope_map[cr + dr, cc] >= limit_angle_rad or
                        slope_map[cr, cc + dc] >= limit_angle_rad):
                    continue

            risk = _calculate_risk(slope_map[nr, nc], risk_threshold_deg)
            intent_pen = _calculate_intent_penalty(
                intent_type, intent_params, (nr, nc), rows, slope_map, side_info
            )

            step_cost = (alpha * real_d
                         + beta * cot * real_d
                         + gamma * risk
                         + delta * intent_pen)

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
                              risk_threshold_deg=15.0,
                              intent_type="baseline", intent_params=None):
        """intent 기반 경로 탐색."""
        if self.height_map is None or self.slope_map is None:
            raise RuntimeError("generate()를 먼저 호출하세요.")

        side_info = _precompute_side_bias(start, goal, self.img_size)
        intent_parts = set(intent_type.split("+"))
        uses_lateral = bool(intent_parts & {"left_bias", "right_bias", "center_bias", "short_path"})
        # For lateral intents, anchor left/right/center to the baseline route
        # rather than a single straight start->goal line.
        if uses_lateral and intent_type != "baseline":
            baseline_path = _a_star_intent_search(
                self.slope_map, self.height_map, start, goal,
                self.limit_angle, self.max_iterations,
                pixel_resolution=self.pixel_resolution,
                alpha=alpha, beta=beta, gamma=gamma, delta=0.0,
                risk_threshold_deg=risk_threshold_deg,
                intent_type="baseline",
                intent_params={},
                side_info=side_info,
            )
            if baseline_path is not None and len(baseline_path) >= 2 and side_info is not None:
                side_info = dict(side_info)
                side_info["baseline_points"] = np.array(baseline_path, dtype=np.float32)

        return _a_star_intent_search(
            self.slope_map, self.height_map, start, goal,
            self.limit_angle, self.max_iterations,
            pixel_resolution=self.pixel_resolution,
            alpha=alpha, beta=beta, gamma=gamma, delta=delta,
            risk_threshold_deg=risk_threshold_deg,
            intent_type=intent_type,
            intent_params=intent_params or {},
            side_info=side_info,
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
                "cost_weights": {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta},
                "risk_threshold_deg": risk_threshold_deg,
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
        cfg_path = _ROOT / "configs" / "default_config.yaml"
        if not cfg_path.exists():
            cfg_path = Path(__file__).resolve().parents[2] / "diffusion_patch" / "configs" / "default_config.yaml"
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
