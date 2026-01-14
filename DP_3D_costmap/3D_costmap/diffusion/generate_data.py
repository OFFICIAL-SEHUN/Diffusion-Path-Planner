import os
import torch
import numpy as np
import yaml
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import heapq

"""
Slope + CoT (Cost of Transport) 기반 학습 데이터 생성
경사를 고려한 CoT 효율적 경로를 생성합니다.
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
        'img_size': 64,
        'horizon': 128
    },
    'gradient': {
        'height_range': [0, 10],
        'terrain_scales': [[20, 30], [10, 15], [5, 5]],
        'mass': 10.0,
        'gravity': 9.8,
        'limit_angle_deg': 30
    }
}

# --- 저장 설정 ---
SAVE_DIR = "data"
SAVE_NAME = "dataset_gradient.pt"


class SlopeCotGenerator:
    """Slope + CoT 기반 지형 생성 및 경로 계획"""
    
    def __init__(self, img_size, height_range=(0, 20), mass=10.0, gravity=9.8, limit_angle_deg=35):
        """
        Args:
            img_size (int): 맵 크기 (img_size x img_size)
            height_range (tuple): 높이 범위 (min_m, max_m)
            mass (float): 로봇 질량 (kg)
            gravity (float): 중력 가속도 (m/s^2)
            limit_angle_deg (float): 최대 등반 가능 각도 (도)
        """
        self.img_size = img_size
        self.height_range = height_range
        self.mass = mass
        self.gravity = gravity
        self.limit_angle = np.radians(limit_angle_deg)
        
        # 생성된 데이터 저장
        self.height_map = None
        self.slope_map = None
        self.cot_costmap = None
    
    def generate(self, terrain_scales=[(40, 50), (15, 20), (5, 5)]):
        """
        Slope + CoT 기반 지형과 비용 맵 생성
        
        Args:
            terrain_scales (list): [(scale, weight), ...] 지형 노이즈 파라미터
            
        Returns:
            tuple: (height_map, slope_map, cot_costmap)
        """
        # 1) 높이 맵 생성
        self.height_map = self._generate_height_map(terrain_scales)
        
        # 2) 경사 맵 계산
        self.slope_map = self._calculate_slope_map(self.height_map)
        
        # 3) CoT 비용 맵 생성 (마찰 제외)
        self.cot_costmap = self._calculate_cot_costmap()
        
        return self.height_map, self.slope_map, self.cot_costmap
    
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
            terrain_scales = []
            
            # 1. Large scale (부드러운 베이스) - 항상 포함
            large_scale = np.random.uniform(15, 25)   # 큰 언덕
            large_weight = np.random.uniform(15, 30)  # 중간 높이
            terrain_scales.append((large_scale, large_weight))
            
            # 2. Medium scale (중간 언덕) - 항상 포함
            medium_scale = np.random.uniform(8, 15)   # 중간 언덕
            medium_weight = np.random.uniform(10, 20) # 중간 높이
            terrain_scales.append((medium_scale, medium_weight))
            
            # 3. Small scale (디테일) - 확률적으로 포함
            if np.random.rand() > 0.3:  # 70% 확률
                small_scale = np.random.uniform(3, 8)    # 작은 디테일
                small_weight = np.random.uniform(5, 15)  # 작은 높이
                terrain_scales.append((small_scale, small_weight))
        
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
        """높이 맵으로부터 경사각 맵 계산"""
        gy, gx = np.gradient(height_map)
        slope_map = np.arctan(np.sqrt(gx**2 + gy**2))
        return slope_map.astype(np.float32)
    
    def _calculate_cot_costmap(self):
        """
        CoT (Cost of Transport) 기반 비용 맵 계산
        마찰을 제외하고 경사만 고려
        
        CoT = Energy / (Mass × Distance × Gravity)
        
        경사 기반 모델:
        - 오르막: CoT = sin(slope) + base_cost
        - 내리막: CoT = base_cost (또는 작은 값)
        - 평지: CoT = base_cost
        
        Returns:
            ndarray: CoT 비용 맵 [H, W]
        """
        # 경사각이 한계를 넘으면 무한대
        impossible = self.slope_map > self.limit_angle
        
        # 기본 CoT (평지 이동 비용)
        base_cot = 0.2
        
        # 경사 기반 CoT 계산
        # 오르막: sin(slope)를 CoT에 추가
        # 내리막: 기본 비용만 (음수 경사는 0으로 클램핑)
        slope_component = np.maximum(0, np.sin(self.slope_map))
        
        # 전체 CoT = 기본 비용 + 경사 비용
        cot_cost = base_cot + slope_component
        
        # 가파른 경사일수록 추가 페널티 (비선형성)
        # 한계 각도에 가까울수록 비용 급증
        slope_ratio = np.clip(self.slope_map / self.limit_angle, 0, 0.98)
        steep_penalty = 1.0 / (1.0 - slope_ratio)
        
        cot_cost = cot_cost * steep_penalty
        
        # 최소 비용 보장
        cot_cost = np.maximum(base_cot, cot_cost)
        
        # 등반 불가 지역 처리
        cot_cost[impossible] = np.inf
        
        return cot_cost.astype(np.float32)
    
    def find_path(self, start, goal):
        """
        A* 알고리즘으로 CoT 효율적 경로 탐색
        
        Args:
            start (tuple): 시작 위치 (row, col)
            goal (tuple): 목표 위치 (row, col)
            
        Returns:
            list: 경로 [(row, col), ...] 또는 None
        """
        if self.cot_costmap is None:
            raise RuntimeError("먼저 generate()를 호출하여 맵을 생성해야 합니다.")
        
        return a_star_cot_search(
            self.cot_costmap, 
            self.slope_map,
            start, 
            goal
        )


def a_star_cot_search(cot_costmap, slope_map, start, goal):
    """
    CoT 효율 기반 A* 경로 탐색
    
    Args:
        cot_costmap (ndarray): 단위 거리당 CoT 비용 맵
        slope_map (ndarray): 경사각 맵 (등반 불가 체크용)
        start (tuple): 시작 위치 (row, col)
        goal (tuple): 목표 위치 (row, col)
        
    Returns:
        list: 경로 또는 None
    """
    rows, cols = cot_costmap.shape
    start = tuple(start)
    goal = tuple(goal)
    
    # 시작/끝 지점이 등반 불가능하면 실패
    if not np.isfinite(cot_costmap[start]) or not np.isfinite(cot_costmap[goal]):
        return None
    
    def heuristic(a, b):
        """유클리드 거리 휴리스틱"""
        return np.hypot(a[0] - b[0], a[1] - b[1])
    
    # 우선순위 큐
    counter = 0
    open_heap = [(heuristic(start, goal), counter, start)]
    came_from = {}
    
    # g_score: numpy 배열로 메모리 효율 향상
    g_score = np.full((rows, cols), np.inf, dtype=np.float32)
    g_score[start] = 0.0
    
    # closed set으로 중복 방문 방지
    closed_set = set()
    
    while open_heap:
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
            
            # 등반 불가능 지역 체크
            if not np.isfinite(cot_costmap[nr, nc]):
                continue
            
            # 대각선 이동 시 코너컷 방지
            if abs(dr) + abs(dc) == 2:
                if (not np.isfinite(cot_costmap[cr + dr, cc]) or 
                    not np.isfinite(cot_costmap[cr, cc + dc])):
                    continue
            
            # 이동 거리
            step_distance = np.sqrt(2.0) if (dr != 0 and dc != 0) else 1.0
            
            # CoT 비용 계산 (현재와 다음 셀의 평균)
            avg_cost = 0.5 * (cot_costmap[cr, cc] + cot_costmap[nr, nc])
            tentative_g = g_score[cr, cc] + step_distance * avg_cost
            
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


def normalize_costmap(costmap):
    """
    Costmap 정규화 (Inf 처리 및 [0, 1] 범위로)
    
    Inf 처리 전략:
    - Inf를 max_val보다 충분히 큰 값으로 설정 (max_val * 2)
    - 정규화 후 Inf 영역이 명확히 높은 값(~1.0)을 가지도록 함
    - 모델이 "등반 불가능" vs "매우 어려움"을 구분 가능
    """
    costmap_float = costmap.astype(np.float32)
    inf_mask = np.isinf(costmap_float)
    
    if (~inf_mask).any():
        max_val = np.max(costmap_float[~inf_mask])
    else:
        max_val = 1.0
    
    # ✅ Inf를 max_val의 2배로 설정 (명확한 구분)
    # 정규화 후: 일반 영역 [0, ~0.5], Inf 영역 [~1.0]
    costmap_float[inf_mask] = max_val * 2.0
    
    # 안전한 나눗셈
    costmap_norm = costmap_float / (np.max(costmap_float) + 1e-8)
    
    return costmap_norm

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

def generate_slope_cot_data(num_samples=5000, paths_per_terrain=5, img_size=64, horizon=128,
                            height_range=(0, 10), mass=10.0, gravity=9.8, 
                            limit_angle_deg=30):
    """
    Slope + CoT 기반 데이터 생성 (Multi-path augmentation)
    
    🔥 하나의 terrain에서 여러 개의 path를 생성하여 데이터 효율성 향상!
    
    Args:
        num_samples (int): 생성할 terrain 개수
        paths_per_terrain (int): 각 terrain당 생성할 path 개수 (Data Augmentation)
                                 총 path 개수 = num_samples × paths_per_terrain
        img_size (int): 맵 크기
        horizon (int): 경로 길이
        height_range (tuple): 높이 범위
        mass (float): 로봇 질량
        gravity (float): 중력
        limit_angle_deg (float): 최대 등반 각도
    
    Returns:
        - Costmaps: [Slope map, CoT map] 2채널 (Diffusion 입력으로 모두 제공)
    - Paths: CoT 기반 A* 경로 (GT)
        - 모델은 Slope와 CoT 정보를 모두 활용하여 학습
    """
    num_terrains = num_samples  # num_samples = terrain 개수
    total_paths = num_samples * paths_per_terrain
    print(f"[Slope + CoT Generator] Generating dataset...")
    print(f"  → Terrains: {num_terrains}")
    print(f"  → Paths per terrain: {paths_per_terrain}")
    print(f"  → Total paths: {total_paths}")
    print(f"  → Diffusion Input: [Slope map, CoT map] (2-channel - 모두 사용)")
    print(f"  → GT Generation: CoT-based A* search")
    print(f"  → Data Augmentation: Same terrain, different start/goal positions\n")
    
    costmaps_list = []
    paths_list = []
    height_maps_list = []
    slope_maps_list = []
    
    # 통계 수집
    slope_stats = {'mean': [], 'max': [], 'std': []}
    terrain_path_counts = []  # 각 terrain에서 생성된 path 개수
    
    # 랜덤 시작/끝 지점 범위 (너무 가장자리는 피함)
    margin = img_size // 10
    
    pbar = tqdm(total=total_paths, desc="Generating paths")
    samples_generated = 0  # 총 path 개수
    terrains_generated = 0  # terrain 개수
    
    while terrains_generated < num_terrains:
        # 1. 지형 생성 (Slope + CoT only)
        generator = SlopeCotGenerator(
            img_size=img_size,
            height_range=height_range,
            mass=mass,
            gravity=gravity,
            limit_angle_deg=limit_angle_deg
        )
        
        # Slope + CoT만 사용
        # Data Augmentation: terrain_scales=None으로 랜덤 terrain 생성
        h_map, s_map, cot_costmap = generator.generate(
            terrain_scales=None  # None = 랜덤하게 다양한 지형 생성!
        )
        
        # 난이도 필터링: 너무 극단적인 지형 거부
        slope_degrees = np.degrees(s_map)
        mean_slope = np.mean(slope_degrees)
        max_slope = np.max(slope_degrees)
        
        # 너무 쉽거나 어려운 지형 스킵
        if mean_slope < 3.0 or mean_slope > 20.0:  # 평균 경사 3~20도 유지
            continue
        if max_slope > 45.0:  # 최대 경사 45도 이하
            continue
        
        # 2채널 costmap 미리 생성 (재사용)
        slope_norm = slope_degrees / 90.0  # [0, 90°] → [0, 1] 정규화
        cot_norm = normalize_costmap(cot_costmap)  # CoT 정규화
        costmap_2ch = np.stack([slope_norm, cot_norm], axis=0)
        
        # 통계 수집 (terrain 단위)
        slope_stats['mean'].append(mean_slope)
        slope_stats['max'].append(max_slope)
        slope_stats['std'].append(np.std(slope_degrees))
        
        # 2. 🔥 같은 terrain에서 여러 개의 path 생성
        paths_found = 0
        max_total_attempts = paths_per_terrain * 30  # 전체 시도 제한
        total_attempts = 0
        
        while paths_found < paths_per_terrain and total_attempts < max_total_attempts:
            # 랜덤 시작/끝 지점 선택
            start = (np.random.randint(margin, img_size - margin),
                    np.random.randint(margin, img_size - margin))
            goal = (np.random.randint(margin, img_size - margin),
                   np.random.randint(margin, img_size - margin))
            
            # 시작과 끝이 너무 가까우면 다시 선택
            dist = np.hypot(goal[0] - start[0], goal[1] - start[1])
            if dist < img_size * 0.5:
                total_attempts += 1
                continue
            
            # 경로 탐색
            path_pixels = generator.find_path(start, goal)
            
            if path_pixels is not None and len(path_pixels) > 0:
                # Path 변환 및 리샘플링
                path_normalized = path_pixels_to_normalized(path_pixels, img_size)
                path_fixed = resample_path(path_normalized, horizon)
                
                # 데이터 저장 (같은 costmap을 재사용!)
                costmaps_list.append(costmap_2ch.copy())  # 같은 terrain
                paths_list.append(path_fixed)             # 다른 path
                height_maps_list.append(h_map)
                slope_maps_list.append(slope_degrees)
                
                paths_found += 1
                samples_generated += 1
                pbar.update(1)
            
            total_attempts += 1
        
        # 이 terrain에서 최소 1개 이상의 path를 찾았으면 terrain 카운트 증가
        if paths_found > 0:
            terrain_path_counts.append(paths_found)
            terrains_generated += 1
        
        # 혹시 paths_per_terrain개를 못 찾았으면 경고 (조용히 넘어감)
        # 일부 terrain은 어려울 수 있음
    
    pbar.close()
    
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
    print(f"{'='*60}\n")
    
    return (costmaps_list, paths_list, height_maps_list, slope_maps_list)

def generate_and_save(config=None):
    """
    데이터를 생성하고 저장합니다.
    
    Args:
        config (dict, optional): 설정 딕셔너리. None이면 config 파일 로드 시도
    """
    # Config 로드
    if config is None:
        try:
            config = load_config()
            print("✅ Loaded config from: configs/default_config.yaml")
        except FileNotFoundError:
            print("⚠️  Config file not found, using default settings")
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
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    print("="*60)
    print("Slope + CoT Dataset Generation (Multi-Path Augmentation)")
    print("="*60)
    print(f"Image Size: {img_size}x{img_size}")
    print(f"Path Horizon: {horizon}")
    print(f"Terrains: {num_samples}")
    print(f"Paths per Terrain: {paths_per_terrain} (Data Augmentation)")
    print(f"Total Paths: {num_samples * paths_per_terrain}")
    print(f"Height Range: {height_range[0]}m ~ {height_range[1]}m")
    print(f"Max Climb Angle: {limit_angle_deg}°")
    print(f"Cost Model: CoT (Cost of Transport) - Slope Only")
    print("="*60)
    print()
    
    # 데이터 생성 (config 파라미터 사용)
    (costmaps_list, paths_list, height_maps_list, 
     slope_maps_list) = generate_slope_cot_data(
         num_samples=num_samples,
         paths_per_terrain=paths_per_terrain,
         img_size=img_size,
         horizon=horizon,
         height_range=height_range,
         mass=mass,
         gravity=gravity,
         limit_angle_deg=limit_angle_deg
     )
    
    # 텐서로 변환
    costmaps_tensor = torch.from_numpy(np.array(costmaps_list)).float()
    paths_tensor = torch.from_numpy(np.array(paths_list)).float()
    height_maps_tensor = torch.from_numpy(np.array(height_maps_list)).float()
    slope_maps_tensor = torch.from_numpy(np.array(slope_maps_list)).float()
    
    # 저장
    save_path = os.path.join(SAVE_DIR, SAVE_NAME)
    torch.save({
        "costmaps": costmaps_tensor,          # [N, 2, H, W] - [Slope, CoT] 2채널
        "paths": paths_tensor,                # CoT 기반 최적 경로 [N, HORIZON, 2]
        "height_maps": height_maps_tensor,    # 높이 맵 [N, H, W]
        "slope_maps": slope_maps_tensor,      # 경사각 맵 (도) [N, H, W]
        "type": "slope_cot_2channel"          # 2채널 입력
    }, save_path)
    
    # 결과 출력
    print(f"\n{'='*60}")
    print(f"✅ Dataset saved to: {save_path}")
    print(f"{'='*60}")
    print(f"Costmaps shape:      {costmaps_tensor.shape}  (2-channel: [Slope, CoT])")
    print(f"  - Channel 0: Slope map (physical terrain)")
    print(f"  - Channel 1: CoT map (energy cost)")
    print(f"Paths shape:         {paths_tensor.shape}     (CoT-based A* - GT)")
    print(f"Height maps shape:   {height_maps_tensor.shape}")
    print(f"Slope maps shape:    {slope_maps_tensor.shape}")
    print(f"\n💡 Approach:")
    print(f"  - Explicit information: Both Slope and CoT provided as 2-channel input")
    print(f"  - Learning target: Model learns from both terrain physics and energy cost")
    print(f"  - Generate energy-efficient paths using combined information")
    
    file_size_mb = os.path.getsize(save_path) / 1024 / 1024
    print(f"\nFile size: {file_size_mb:.2f} MB")
    print(f"{'='*60}")

if __name__ == "__main__":
    generate_and_save()
