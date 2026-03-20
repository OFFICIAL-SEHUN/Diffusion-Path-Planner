import os
import torch
import numpy as np
import yaml
import argparse
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import heapq
from path_utils import check_path_linearity
import matplotlib.pyplot as plt


def calculate_paper_cot(slope_deg):
    """
    논문 기반 CoT 계산 (4차 다항식)
    
    경사각 정의:
    - 음수: 내리막 (downhill, -20° ~ 0°)
    - 0: 평지
    - 양수: 오르막 (uphill, 0° ~ 20°)
    
    특징:
    - 내리막은 브레이킹으로 인해 CoT 높음
    - 약간의 오르막(+5~10°)이 가장 효율적
    - 가파른 오르막은 CoT 증가
    
    Args:
        slope_deg (float or ndarray): 경사각(도) - 방향성 포함
    
    Returns:
        float or ndarray: CoT 값
    """
    # 논문 Fig 4의 회귀 계수
    a = -1.53e-06
    b = 2.07e-05
    c = 2.20e-03
    d = -3.24e-02
    e = 0.65
    
    # 4차 다항식 계산
    cot = (a * slope_deg**4) + (b * slope_deg**3) + (c * slope_deg**2) + (d * slope_deg) + e
    return cot


def calculate_directional_cot(height_curr, height_next, distance, limit_angle_deg=35.0):
    """
    이동 방향을 고려한 CoT 계산
    
    Args:
        height_curr (float): 현재 셀의 높이
        height_next (float): 다음 셀의 높이
        distance (float): 이동 거리 (1.0 or √2)
        limit_angle_deg (float): 등반 불가능한 최대 경사각
    
    Returns:
        float: 방향성 CoT (등반 불가능 시 np.inf)
    """
    # 높이 차이 → 경사각 계산
    height_diff = height_next - height_curr
    slope_rad = np.arctan2(height_diff, distance)
    slope_deg = np.degrees(slope_rad)
    
    # 등반 불가능 체크 (절대값) - limit_angle_deg 이상이면 등반 불가능 (회피)
    if abs(slope_deg) >= limit_angle_deg:
        return np.inf
    
    # 논문 수식으로 CoT 계산 (방향성 포함)
    cot = calculate_paper_cot(slope_deg)
    
    # 안전 장치: 최소값 보장
    cot = max(cot, 0.1)
    
    return cot


"""
Slope + Height 기반 학습 데이터 생성
경사각과 높이 정보를 2채널로 제공하고, CoT 효율적 경로를 생성합니다.
"""

# --- Config 로드 ---
def load_config(config_path="configs/default_config.yaml"):
    """설정 파일 로드"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# 기본 설정 (config 파일이 없을 경우 fallback)
DEFAULT_CONFIG = {
    'data': {
        'num_samples': 5000,
        'paths_per_terrain': 5,
        'img_size': 100,
        'horizon': 120
    },
    'gradient': {
        'height_range': [0, 5],
        'terrain_scales': [[20, 30], [10, 20], [5, 10]],
        'mass': 10.0,
        'gravity': 9.8,
        'limit_angle_deg': 30,
        'pixel_resolution': 0.5  # m/pixel (100x100 픽셀 = 50m x 50m)
    }
}

# --- 저장 설정 ---
SAVE_DIR = "data"
SAVE_NAME = "test_dataset.pt"


class SlopeCotGenerator:
    """Slope + CoT 기반 지형 생성 및 경로 계획"""
    
    def __init__(self, img_size, height_range, mass, gravity, limit_angle_deg, max_iterations, pixel_resolution=0.5):
        """
        Args:
            img_size (int): 맵 크기 (img_size x img_size)
            height_range (tuple): 높이 범위 (min_m, max_m)
            mass (float): 로봇 질량 (kg)
            gravity (float): 중력 가속도 (m/s^2)
            limit_angle_deg (float): 최대 등반 가능 각도 (도)
            max_iterations (int): 최대 반복 횟수 (타임아웃 방지)
            pixel_resolution (float): 픽셀당 실제 거리 (m/pixel)
                                      예: 0.5 → 100x100 픽셀 = 50m x 50m
        """
        self.img_size = img_size
        self.height_range = height_range
        self.mass = mass
        self.gravity = gravity
        self.limit_angle = np.radians(limit_angle_deg)
        self.pixel_resolution = pixel_resolution
        self.max_iterations = max_iterations
        
        # 생성된 데이터 저장
        self.height_map = None
        self.slope_map = None
    
    def generate(self, terrain_scales):
        """
        Slope + Height 기반 지형 생성
        
        Args:
            terrain_scales (list): [(scale, weight), ...] 지형 노이즈 파라미터
            
        Returns:
            tuple: (height_map, slope_map)
        """
        # 1) 높이 맵 생성
        self.height_map = self._generate_height_map(terrain_scales)
        
        # 2) 경사 맵 계산
        self.slope_map = self._calculate_slope_map(self.height_map)
        
        # CoT는 A* 탐색 시 동적으로 계산 (방향성 고려)
        # Static CoT map은 방향성이 없어 부정확하므로 제거
        
        return self.height_map, self.slope_map
    
    def _generate_height_map(self, terrain_scales):
        """
        다중 스케일 노이즈로 높이 맵 생성
        
        Args:
            terrain_scales: list of (scale, weight) 또는 None
                           None이면 랜덤하게 다양한 terrain 생성 (Augmentation)
        """
        height_map = np.zeros((self.img_size, self.img_size))
        
        # Data Augmentation: terrain_scales가 None이면 랜덤 생성
        if terrain_scales is None:
            print("No terrain scales provided")
            raise ValueError("No terrain scales provided")
        
        # 다중 스케일 노이즈 합성
        for scale, weight in terrain_scales:
            noise = np.random.rand(self.img_size, self.img_size)
            height_map += gaussian_filter(noise, sigma=scale) * weight 
            # sigma = scale : how much noise to add, The larger the sigma, the more it blurs.
        
        # 정규화
        height_map = (height_map - height_map.min()) / (height_map.max() - height_map.min())
        height_map = height_map * (self.height_range[1] - self.height_range[0]) + self.height_range[0]
        
        return height_map.astype(np.float32)
    
    def _calculate_slope_map(self, height_map):
        """
        높이 맵으로부터 경사각 맵 계산 (물리적으로 정확한 계산)
        
        gradient를 실제 거리(m)로 계산하여 정확한 경사각 도출
        """
        
        # [현재] 정확한 계산 (pixel_resolution 사용)
        gy, gx = np.gradient(height_map, self.pixel_resolution)
        
        
        # gradient magnitude (m/m, 즉 rise/run)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        
        # 경사각 계산 (라디안)
        slope_map = np.arctan(gradient_magnitude)
        
        return slope_map.astype(np.float32)
    
    
    def find_path(self, start, goal, weight=0.5):
        """
        A* 알고리즘으로 CoT 효율적 경로 탐색
        
        Args:
            start (tuple): 시작 위치 (row, col)
            goal (tuple): 목표 위치 (row, col)
            weight (float): 거리 vs CoT 균형 가중치 [0, 1]
            
        Returns:
            list: 경로 [(row, col), ...] 또는 None
        """
        if self.height_map is None or self.slope_map is None:
            raise RuntimeError("먼저 generate()를 호출하여 맵을 생성해야 합니다.")
        
        return a_star_cot_search(
            self.slope_map,
            self.height_map,  # 방향성 CoT 계산을 위한 height_map
            start, 
            goal,
            self.limit_angle,
            self.pixel_resolution,  # 실제 거리 계산을 위한 pixel_resolution 추가
            self.max_iterations, # 최대 반복 횟수
            weight=weight  # Weight parameter 추가
        )


def a_star_cot_search(slope_map, height_map, start, goal, limit_angle_rad, max_iterations, pixel_resolution=0.5,  weight=0.5):
    """
    CoT 효율 기반 A* 경로 탐색 (방향성 CoT 적용) + Weight balancing
    
    Args:
        slope_map (ndarray): 경사각 맵 (라디안, 등반 불가 체크용)
        height_map (ndarray): 높이 맵 (방향성 CoT 계산용)
        start (tuple): 시작 위치 (row, col)
        goal (tuple): 목표 위치 (row, col)
        limit_angle_rad (float): 등반 불가능한 최대 경사각 (라디안)
        max_iterations (int): 최대 반복 횟수 (타임아웃 방지)
        pixel_resolution (float): 픽셀당 실제 거리 (m/pixel) - CoT 계산에 필수!
        weight (float): 거리 vs CoT 균형 가중치 [0, 1]
                       - w=0.1: 거리 우선 (빠른 경로, "Quickly", "Shortest path")
                       - w=0.9: CoT 우선 (안전한 경로, "Safe route", "Energy efficient")
                       - w=0.5: 균형 (기본값)
        
    Returns:
        list: 경로 또는 None
    """
    
    rows, cols = height_map.shape
    start = tuple(start)
    goal = tuple(goal)
    
    # limit_angle을 도(degree)로 변환 (calculate_directional_cot에서 사용)
    limit_angle_deg = np.degrees(limit_angle_rad)
    
    # 시작/끝 지점이 등반 불가능하면 실패 (limit_angle_rad 이상이면 등반 불가능 - 회피)
    if slope_map[start] >= limit_angle_rad or slope_map[goal] >= limit_angle_rad:
        return None
    
    # 맵 크기에 비례하여 max_iterations 조정 (너무 큰 값 방지)
    map_size = rows * cols

    if max_iterations < map_size * 10:
        max_iterations = map_size * 10
    
    def heuristic(a, b):
        """
        유클리드 거리 휴리스틱 (실제 거리 * 최소 CoT)
        CoT는 항상 0.1 이상이므로, 실제 거리 * 0.1을 최소 비용으로 사용
        """
        pixel_distance = np.hypot(a[0] - b[0], a[1] - b[1])
        real_distance = pixel_distance * pixel_resolution
        min_cot = 0.1  # calculate_directional_cot의 최소값
        return real_distance * min_cot
    
    # 우선순위 큐
    counter = 0
    open_heap = [(heuristic(start, goal), counter, start)]
    came_from = {}
    
    # g_score: numpy 배열로 메모리 효율 향상
    g_score = np.full((rows, cols), np.inf, dtype=np.float32)
    g_score[start] = 0.0
    
    # closed set으로 중복 방문 방지
    closed_set = set()
    
    # 타임아웃 방지: 최대 반복 횟수 제한
    iterations = 0
    
    while open_heap and iterations < max_iterations:
        iterations += 1
        _, _, current = heapq.heappop(open_heap)
        
        # 이미 방문한 노드는 스킵
        if current in closed_set:
            continue
        closed_set.add(current)
        
        # 목표 도달
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        
        cr, cc = current
        
        # 8방향 탐색
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0), (-1,-1), (-1,1), (1,-1), (1,1)]:
            nr, nc = cr + dr, cc + dc
            
            # 범위 체크
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            
            # 이미 방문했으면 스킵
            if (nr, nc) in closed_set:
                continue
            
            # 이동 거리 (픽셀 단위)
            pixel_distance = np.sqrt(2.0) if (dr != 0 and dc != 0) else 1.0
            # 실제 거리 (미터 단위) - CoT 계산에 필수!
            real_distance = pixel_distance * pixel_resolution
            
            # 방향성 CoT 비용 계산 (논문 수식 기반) - 먼저 체크!
            # calculate_directional_cot는 실제 이동 방향의 경사각을 정확히 계산
            height_curr = height_map[cr, cc]
            height_next = height_map[nr, nc]
            directional_cot = calculate_directional_cot(
                height_curr, height_next, real_distance, limit_angle_deg
            )
            
            # 등반 불가능한 경로는 스킵 (20도 이상이면 회피)
            if np.isinf(directional_cot):
                continue
            
            # 등반 불가능 지역 체크 (slope_map 사용) - 추가 안전장치
            # limit_angle_rad 이상이면 등반 불가능 (회피)
            if slope_map[nr, nc] >= limit_angle_rad:
                continue
            
            # 대각선 이동 시 코너컷 방지
            if abs(dr) + abs(dc) == 2:
                if (slope_map[cr + dr, cc] >= limit_angle_rad or 
                    slope_map[cr, cc + dc] >= limit_angle_rad):
                    continue
            
            # 총 비용 = (1-w) * 거리 + w * CoT * 실제 거리
            # w=0.1: 거리 우선 → 빠른 경로
            # w=0.9: CoT 우선 → 에너지 효율적 경로
            distance_cost = real_distance
            cot_cost = directional_cot * real_distance
            
            # Weighted combination
            step_cost = (1 - weight) * distance_cost + weight * cot_cost
            tentative_g = g_score[cr, cc] + step_cost
             
            # 더 나은 경로 발견
            if tentative_g < g_score[nr, nc]:
                came_from[(nr, nc)] = current
                g_score[nr, nc] = tentative_g
                
                # f = g + h (약간의 tie-breaking 추가)
                h = heuristic((nr, nc), goal)
                f = tentative_g + h * (1.0 + 1e-3)
                
                counter += 1
                heapq.heappush(open_heap, (f, counter, (nr, nc)))
    
    return None

def resample_path(path, horizon):
    """Path 리샘플링 (Variable length -> Fixed horizon)"""
    original_len = path.shape[0]
    if original_len == 0:
        return np.zeros((horizon, 2), dtype=np.float32)
    
    t_current = np.linspace(0, 1, original_len)
    t_target = np.linspace(0, 1, horizon)
    
    x_interp = np.interp(t_target, t_current, path[:, 0])
    y_interp = np.interp(t_target, t_current, path[:, 1])
    path_fixed = np.stack([x_interp, y_interp], axis=1)
    
    return path_fixed

def path_pixels_to_normalized(path_pixels, img_size):
    """픽셀 좌표 [(r,c), ...] -> 정규화 좌표 [-1, 1] (x, y) 형식"""
    if path_pixels is None or len(path_pixels) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    
    path_array = np.array(path_pixels, dtype=np.float32)
    
    # ✅ (row, col) -> (x, y) = (col, row) 형식으로 변환 (대각선 반전)
    # A* 반환값: [(row, col), ...] → 저장: [(x, y), ...]
    path_array = path_array[:, [1, 0]]  # [col, row] 순서로 변경
    
    # 정규화: [0, img_size] -> [-1, 1]
    path_normalized = (path_array / img_size) * 2 - 1
    
    return path_normalized


# Text label mapping: weight -> text labels
TEXT_LABELS = {
    0.1: ["Quickly", "Shortest path", "Ignore the hills"],
    0.3: ["Fast route", "Minimize distance", "Take direct path"],
    0.5: ["Balanced path", "Moderate route", "Consider terrain"],
    0.7: ["Safe route", "Avoid steep slopes", "Prefer flat terrain"],
    0.9: ["Energy efficient", "Safe route", "Avoid steep slope", "Minimize elevation gain"]
}


def weight_to_text_label(weight):
    """
    Weight 값을 텍스트 라벨로 변환
    
    
    
    Args:
        weight (float): Weight 값 [0, 1]
        
    Returns:
        str: 텍스트 라벨
    """
    # 가장 가까운 weight 값 찾기
    closest_weight = min(TEXT_LABELS.keys(), key=lambda x: abs(x - weight))
    labels = TEXT_LABELS[closest_weight]
    # 랜덤으로 하나 선택
    return np.random.choice(labels)

def generate_slope_cot_data(num_samples=5000, paths_per_terrain=5, img_size=100, horizon=120,
                            height_range=(0, 10), mass=10.0, gravity=9.8, 
                            limit_angle_deg=30, pixel_resolution=0.5, terrain_scales=None,
                            min_distance_factor=1.0, pca_linearity_threshold=0.95, max_iterations=10000):
    """
    Slope + Height 기반 데이터 생성 (Multi-path augmentation)
    
    하나의 terrain에서 여러 개의 path를 생성하여 데이터 효율성 향상!
    
    Args:
        num_samples (int): 생성할 terrain 개수
        paths_per_terrain (int): 각 terrain당 생성할 path 개수 (Data Augmentation)
                                 총 path 개수 = num_samples × paths_per_terrain
        img_size (int): 맵 크기
        horizon (int): 경로 길이
        height_range (tuple): 높이 범위
        mass (float): 로봇 질량 (A* CoT 계산에 사용)
        gravity (float): 중력 (A* CoT 계산에 사용)
        limit_angle_deg (float): 최대 등반 각도
        pixel_resolution (float): 픽셀당 실제 거리 (m/pixel)
                                  예: 0.5 → 100x100픽셀 = 50m x 50m
        terrain_scales (list): 지형 스케일 파라미터 (None이면 랜덤 생성)
        min_distance_factor (float): 최소 거리 팩터 (min_distance = img_size / factor)
                                     예: 1.0 → min_distance = img_size
                                         0.7 → min_distance = img_size / 0.7
        pca_linearity_threshold (float): PCA 선형성 검사 임계값 (0.98 = 98%)
                                         첫 번째 주성분 분산 비율이 이 값 이상이면 경로 거부
                                         높을수록 더 엄격 (직선 경로만 거부), 낮을수록 더 느슨
    
    Returns:
        - Costmaps: [Slope map, Height map] 2채널
          - Slope: 경사 크기 (물리적으로 정확하게 계산됨)
          - Height: 고도 정보 (AI가 gradient 방향을 추론하여 오르막/내리막 판단)
        - Paths: CoT 기반 A* 경로 (GT, 방향성 CoT로 동적 계산)
    """
    num_terrains = num_samples  # num_samples = terrain 개수
    total_paths = num_samples * paths_per_terrain
    print(f"[Slope + Height Generator] Generating dataset...")
    print(f"  → Terrains: {num_terrains}")
    print(f"  → Paths per terrain: {paths_per_terrain}")
    print(f"  → Total paths: {total_paths}")
    print(f"  → Diffusion Input: [Slope map, Height map] (2-channel)")
    print(f"      • Slope: Physical terrain steepness")
    print(f"      • Height: Elevation info (AI infers uphill/downhill direction)")
    print(f"  → GT Generation: CoT-based A* search (directional CoT)")
    print(f"  → PCA Linearity Check: Threshold = {pca_linearity_threshold} (exclude too linear paths)")
    print(f"  → Data Augmentation: Same terrain, different start/goal positions\n")
    
    costmaps_list = []
    paths_list = []
    height_maps_list = []
    slope_maps_list = []
    text_labels_list = []  # 텍스트 라벨 저장
    
    # 통계 수집
    slope_stats = {'mean': [], 'max': [], 'std': []}
    terrain_path_counts = []  # 각 terrain에서 생성된 path 개수
    
    # PCA 검사 통계
    total_pca_rejections = 0
    
    margin = img_size // 10
    min_distance = int(img_size // min_distance_factor)
    
    pbar = tqdm(total=total_paths, desc="Generating paths")
    samples_generated = 0  # 총 path 개수
    terrains_generated = 0  # terrain 개수
    
    # 무한 루프 방지: 최대 terrain 시도 횟수
    max_terrain_attempts = num_terrains * 10  # 각 terrain당 최대 10번 시도
    terrain_attempts = 0
    
    while terrains_generated < num_terrains and terrain_attempts < max_terrain_attempts:
        terrain_attempts += 1
        # 1. 지형 생성 (Slope + Height)
        generator = SlopeCotGenerator(
            img_size=img_size,
            height_range=height_range,
            mass=mass,
            gravity=gravity,
            limit_angle_deg=limit_angle_deg,
            max_iterations=max_iterations, 
            pixel_resolution=pixel_resolution
        )
        
        # Slope + Height 생성
        # Use terrain_scales parameter if provided, otherwise random generation
        h_map, s_map = generator.generate(
            terrain_scales=terrain_scales  # Use parameter or random (None)
        )
        
        # 난이도 필터링, 학습을 위한 데이터 생성
        slope_degrees = np.degrees(s_map)
        mean_slope = np.mean(slope_degrees)
        max_slope = np.max(slope_degrees)
        
        # High cost 분포 증가: 더 가파른 지형 허용
        if mean_slope < 3.0 or mean_slope > 25.0:  # 평균 경사 3~25도 (증가)
            continue
        if max_slope > 30.0:  # 최대 경사 30도 이하
            continue
        
        # 가파른 구간 비율 증가 허용
        steep_ratio = np.sum(slope_degrees > 30.0) / slope_degrees.size
        if steep_ratio > 0.3:  # 30도 초과 픽셀이 30% 이하 (증가)
            continue
        
        # 2채널 costmap 생성: [Slope, Height]
        slope_norm = slope_degrees / 90.0  # [0, 90°] → [0, 1] 정규화
        height_norm = (h_map - h_map.min()) / (h_map.max() - h_map.min() + 1e-8)  # [0, 1] 정규화
        costmap_2ch = np.stack([slope_norm, height_norm], axis=0)
        
        # 통계 수집 (terrain 단위)
        slope_stats['mean'].append(mean_slope)
        slope_stats['max'].append(max_slope)
        slope_stats['std'].append(np.std(slope_degrees))
        
        # 2. 🔥 같은 terrain에서 여러 개의 path 생성
        paths_found = 0
        max_total_attempts = paths_per_terrain * 100
        total_attempts = 0
        pca_rejections = 0  # PCA로 거부된 경로 수 추적
        
        while paths_found < paths_per_terrain and total_attempts < max_total_attempts:
            # 랜덤 시작/끝 지점 선택 
            for position_attempt in range(100):
                start = (np.random.randint(margin, img_size - margin),
                        np.random.randint(margin, img_size - margin))
                goal = (np.random.randint(margin, img_size - margin),
                       np.random.randint(margin, img_size - margin))
                
                # Calculate Euclidean distance
                distance = np.sqrt((goal[0] - start[0])**2 + 
                                 (goal[1] - start[1])**2)
                
                if distance >= min_distance:
                    break
            else:
                # No valid position found in 100 attempts, continue to next path attempt
                total_attempts += 1
                continue
            
            # 높이 차이 제한 (방향성 CoT에서 현실적인 경로 생성)
            # Terrain 필터링이 엄격하므로 높이 차이 제한 불필요
            # (30도 초과 구간이 5% 미만이므로 대부분의 경로가 통과 가능)
            
            # 🔥 Weight 선택: 다양한 전략을 학습하기 위해 랜덤 선택
            # w=0.1: 빠른 경로, w=0.9: 안전한 경로
            weight = np.random.choice([0.1, 0.3, 0.5, 0.7, 0.9], p=[0.2, 0.2, 0.2, 0.2, 0.2])
        
            
            # 경로 탐색 (weight 적용)
            path_pixels = generator.find_path(start, goal, weight=weight)
            
            if path_pixels is None or len(path_pixels) == 0:
                # A* 경로 탐색 실패
                total_attempts += 1
                continue
            
            if path_pixels is not None and len(path_pixels) > 0:
                # 경로 길이 체크
                if len(path_pixels) <= 10:
                    total_attempts += 1
                    continue
                
                # PCA 선형성 검사 (너무 직선적인 경로 제외)
                if len(path_pixels) >= 3:  # 최소 3개 점 필요
                    explained_var, is_linear = check_path_linearity(path_pixels, pca_linearity_threshold)
                    if is_linear:
                        # 너무 직선적인 경로는 스킵
                        pca_rejections += 1
                        total_attempts += 1
                        continue
                
                # Path 변환 및 리샘플링
                path_normalized = path_pixels_to_normalized(path_pixels, img_size)
                path_fixed = resample_path(path_normalized, horizon)
                
                # 텍스트 라벨 생성 (weight -> text)
                text_label = weight_to_text_label(weight)
                
                # 데이터 저장 (같은 costmap을 재사용!)
                costmaps_list.append(costmap_2ch.copy())  # 같은 terrain
                paths_list.append(path_fixed)             # 다른 path
                height_maps_list.append(h_map)
                slope_maps_list.append(slope_degrees)
                text_labels_list.append(text_label)       # 텍스트 라벨
                
                paths_found += 1
                samples_generated += 1
                pbar.update(1)
            
            total_attempts += 1
        
        # 이 terrain에서 최소 1개 이상의 path를 찾았으면 terrain 카운트 증가
        total_pca_rejections += pca_rejections
        if paths_found > 0:
            terrain_path_counts.append(paths_found)
            terrains_generated += 1
            if pca_rejections > 0:
                pbar.write(f"  → Terrain {terrains_generated}: {pca_rejections} paths rejected by PCA")
        else:
            # 경로를 찾지 못한 경우 (PCA가 너무 엄격할 수 있음)
            if pca_rejections > max_total_attempts * 0.5:  # 50% 이상이 PCA로 거부됨
                pbar.write(f"  ⚠️  Terrain skipped: {pca_rejections}/{total_attempts} paths rejected by PCA (threshold may be too strict)")
        
        # 혹시 paths_per_terrain개를 못 찾았으면 경고 (조용히 넘어감)
        # 일부 terrain은 어려울 수 있음
    
    pbar.close()
    
    # 무한 루프 방지: 최대 시도 횟수 도달 시 경고
    if terrain_attempts >= max_terrain_attempts and terrains_generated < num_terrains:
        print(f"\n⚠️  Warning: Reached max terrain attempts ({max_terrain_attempts})")
        print(f"   Generated {terrains_generated}/{num_terrains} terrains")
        print(f"   PCA threshold ({pca_linearity_threshold}) may be too strict, or terrain generation is too difficult")
    
    # 통계 출력
    print(f"\n{'='*60}")
    print(f"Data Generation Summary:")
    print(f"{'='*60}")
    print(f"Total Terrains:  {terrains_generated}")
    print(f"Total Paths:     {samples_generated}")
    print(f"Paths/Terrain:   {np.mean(terrain_path_counts):.1f} (avg)")
    print(f"                 Range: [{np.min(terrain_path_counts)}, {np.max(terrain_path_counts)}]")
    print(f"Data Efficiency: {samples_generated / terrains_generated:.1f}x augmentation")
    print(f"\nTerrain Statistics:")
    print(f"Mean Slope:  {np.mean(slope_stats['mean']):.2f}° ± {np.std(slope_stats['mean']):.2f}°")
    print(f"             Range: [{np.min(slope_stats['mean']):.2f}°, {np.max(slope_stats['mean']):.2f}°]")
    print(f"Max Slope:   {np.mean(slope_stats['max']):.2f}° ± {np.std(slope_stats['max']):.2f}°")
    print(f"             Range: [{np.min(slope_stats['max']):.2f}°, {np.max(slope_stats['max']):.2f}°]")
    print(f"Slope Std:   {np.mean(slope_stats['std']):.2f}° ± {np.std(slope_stats['std']):.2f}°")
    print(f"\nPCA Statistics:")
    print(f"Total PCA Rejections: {total_pca_rejections} paths")
    if total_pca_rejections > 0:
        rejection_rate = total_pca_rejections / (samples_generated + total_pca_rejections) * 100 if (samples_generated + total_pca_rejections) > 0 else 0
        print(f"Rejection Rate: {rejection_rate:.1f}%")
        if rejection_rate > 50:
            print(f"⚠️  High rejection rate! Consider lowering pca_linearity_threshold (current: {pca_linearity_threshold})")
    print(f"{'='*60}\n")
    
    return (costmaps_list, paths_list, height_maps_list, slope_maps_list, text_labels_list)

def generate_and_save(config=None, config_path=None):
    """
    데이터를 생성하고 저장합니다.
    
    Args:
        config (dict, optional): 설정 딕셔너리. None이면 config 파일 로드 시도
        config_path (str, optional): config 파일 경로. None이면 기본 경로 사용
    """
    # Config 로드
    if config is None:
        try:
            if config_path is None:
                config_path = "configs/default_config.yaml"
            config = load_config(config_path)
            print(f"Loaded config from: {config_path}")
        except FileNotFoundError:
            print(f"Config file not found: {config_path}, using default settings")
            config = DEFAULT_CONFIG
    
    # Config에서 값 추출
    num_samples = config['data']['num_samples']
    paths_per_terrain = config['data'].get('paths_per_terrain', 5)  # Default: 5
    img_size = config['data']['img_size']
    horizon = config['data']['horizon']
    height_range = tuple(config['gradient']['height_range'])
    mass = config['gradient']['mass']
    gravity = config['gradient']['gravity']
    limit_angle_deg = config['gradient']['limit_angle_deg']
    pixel_resolution = config['gradient'].get('pixel_resolution', 0.5)  # Default: 0.5 m/pixel
    terrain_scales = config['gradient'].get('terrain_scales', None)  # None = random generation
    min_distance_factor = config['data'].get('min_distance_factor', 1.0)  # Default: 1.0 (main.py와 동일)
    pca_linearity_threshold = config['gradient'].get('pca_linearity_threshold', 0.98)  # Default: 0.98
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    print("="*60)
    print("Slope + Height Dataset Generation (Multi-Path Augmentation)")
    print("="*60)
    print(f"Image Size: {img_size}x{img_size}")
    print(f"Map Size: {img_size * pixel_resolution:.1f}m x {img_size * pixel_resolution:.1f}m")
    print(f"Pixel Resolution: {pixel_resolution} m/pixel")
    print(f"Path Horizon: {horizon}")
    print(f"Terrains: {num_samples}")
    print(f"Paths per Terrain: {paths_per_terrain} (Data Augmentation)")
    print(f"Total Paths: {num_samples * paths_per_terrain}")
    print(f"Height Range: {height_range[0]}m ~ {height_range[1]}m")
    print(f"Max Climb Angle: {limit_angle_deg}°")
    print(f"PCA Linearity Threshold: {pca_linearity_threshold} (exclude too linear paths)")
    print(f"Min Distance Factor: {min_distance_factor} (min_distance = img_size/{min_distance_factor} = {int(img_size // min_distance_factor)} pixels)")
    print(f"Input Channels: [Slope map, Height map]")
    print(f"Cost Model: CoT (Cost of Transport) - Directional (A* only)")
    print("="*60)
    print()
    
    # 데이터 생성 (config 파라미터 사용)
    (costmaps_list, paths_list, height_maps_list,
     slope_maps_list, text_labels_list) = generate_slope_cot_data(
        num_samples=num_samples,
        paths_per_terrain=paths_per_terrain,
        img_size=img_size,
        horizon=horizon,
        height_range=height_range,
        mass=mass,
        gravity=gravity,
        limit_angle_deg=limit_angle_deg,
        pixel_resolution=pixel_resolution,
        terrain_scales=terrain_scales,
        min_distance_factor=min_distance_factor,
        pca_linearity_threshold=pca_linearity_threshold
    )
    
    # 텐서로 변환
    costmaps_tensor = torch.from_numpy(np.array(costmaps_list)).float()
    paths_tensor = torch.from_numpy(np.array(paths_list)).float()
    height_maps_tensor = torch.from_numpy(np.array(height_maps_list)).float()
    slope_maps_tensor = torch.from_numpy(np.array(slope_maps_list)).float()
    
    # 🔥 텍스트 라벨 → 토큰 변환
    # 모든 고유한 단어를 수집하여 vocabulary 생성
    all_words = set() 
    for label in text_labels_list:
        words = label.lower().split()
        all_words.update(words)
    
    unique_words = sorted(list(all_words))
    vocab_size = len(unique_words) + 2  # +2 for <PAD> and <UNK>
    
    # Vocabulary 생성: word -> token_id
    vocab = {word: idx + 2 for idx, word in enumerate(unique_words)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    vocab_reverse = {v: k for k, v in vocab.items()}
    
    # 텍스트 라벨을 토큰 인덱스로 변환
    def text_to_tokens(text_label, max_seq_len=32):
        """텍스트 라벨을 토큰 시퀀스로 변환"""
        words = text_label.lower().split()
        tokens = [vocab.get(word, vocab['<UNK>']) for word in words]
        
        # Padding or truncation
        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]
        else:
            tokens = tokens + [vocab['<PAD>']] * (max_seq_len - len(tokens))
        
        return tokens
    
    # 모든 텍스트 라벨을 토큰으로 변환
    text_tokens_list = [text_to_tokens(label) for label in text_labels_list]
    text_tokens_tensor = torch.tensor(text_tokens_list, dtype=torch.long)  # [N, max_seq_len]
    
    # 저장
    save_path = os.path.join(SAVE_DIR, SAVE_NAME)
    torch.save({
        "costmaps": costmaps_tensor,          # [N, 2, H, W] - [Slope, Height] 2채널
        "paths": paths_tensor,                # CoT 기반 최적 경로 [N, HORIZON, 2]
        "height_maps": height_maps_tensor,    # 높이 맵 [N, H, W]
        "slope_maps": slope_maps_tensor,      # 경사각 맵 (도) [N, H, W]
        "text_labels": text_labels_list,      # 원본 텍스트 라벨 리스트
        "text_tokens": text_tokens_tensor,    # 텍스트 토큰 [N, max_seq_len]
        "vocab": vocab,                       # Vocabulary 딕셔너리
        "vocab_size": vocab_size,             # Vocabulary 크기
        "type": "slope_height_2channel_text"  # 텍스트 조건 포함
    }, save_path)
    
    # 결과 출력
    print(f"\n{'='*60}")
    print(f"Dataset saved to: {save_path}")
    print(f"{'='*60}")
    print(f"Costmaps shape:      {costmaps_tensor.shape}  (2-channel: [Slope, Height])")
    print(f"  - Channel 0: Slope map (terrain steepness)")
    print(f"  - Channel 1: Height map (elevation for gradient inference)")
    print(f"Paths shape:         {paths_tensor.shape}     (CoT-based A* - GT)")
    print(f"Text tokens shape:   {text_tokens_tensor.shape}  (Text command tokens)")
    print(f"Vocabulary size:     {vocab_size}")
    print(f"Unique words:        {len(unique_words)}")
    print(f"Sample vocab:        {list(vocab.keys())[:10]}...")
    print(f"Height maps shape:   {height_maps_tensor.shape}")
    print(f"Slope maps shape:    {slope_maps_tensor.shape}")
    print(f"\n💡 Approach:")
    print(f"  - Input: [Slope, Height] + Text command - Language-conditioned navigation")
    print(f"  - GT: A* with weighted CoT (weight=0.1: quickly, weight=0.9: safe route)")
    print(f"  - Text Labels: Weight-based commands (e.g., 'Quickly', 'Safe route', 'Energy efficient')")
    print(f"  - Learning: Model learns to generate paths based on terrain + text command")
    
    file_size_mb = os.path.getsize(save_path) / 1024 / 1024
    print(f"\nFile size: {file_size_mb:.2f} MB")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training data with different config files")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to config file (default: configs/default_config.yaml)')
    
    args = parser.parse_args()
    generate_and_save(config_path=args.config)
