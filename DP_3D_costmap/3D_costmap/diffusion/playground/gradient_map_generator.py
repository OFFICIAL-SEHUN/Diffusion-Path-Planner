import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import heapq
from mpl_toolkits.mplot3d import Axes3D

"""
물리 기반 지형 생성 및 에너지 효율적 경로 계획
"""

class GradientMapGenerator:
    """물리 기반 지형(경사/마찰)을 생성하고 에너지 효율적 경로를 계획합니다."""
    
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
        self.friction_map = None
        self.energy_costmap = None
        
    def generate(self, 
                 terrain_scales=[(40, 50), (15, 20), (5, 5)],
                 friction_range=(0.05, 0.3),
                 friction_smooth=10):
        """
        물리 기반 지형과 비용 맵을 생성합니다.
        
        Args:
            terrain_scales (list): [(scale, weight), ...] 지형 노이즈 파라미터
            friction_range (tuple): 마찰 계수 범위 (min, max)
            friction_smooth (float): 마찰 맵 스무딩 시그마
            
        Returns:
            tuple: (height_map, slope_map, friction_map, energy_costmap)
        """
        # 1) 높이 맵 생성
        self.height_map = self._generate_height_map(terrain_scales)
        
        # 2) 경사 맵 계산
        self.slope_map = self._calculate_slope_map(self.height_map)
        
        # 3) 마찰 계수 맵 생성
        self.friction_map = self._generate_friction_map(friction_range, friction_smooth)
        
        # 4) 에너지 비용 맵 생성 (벡터화)
        self.energy_costmap = self._calculate_energy_costmap()
        
        return self.height_map, self.slope_map, self.friction_map, self.energy_costmap
    
    def generate_slope_cot(self, terrain_scales=[(40, 50), (15, 20), (5, 5)]):
        """
        Slope + CoT (Cost of Transport) 기반 지형 생성
        마찰을 제외하고 경사만 고려한 CoT 비용 맵을 생성합니다.
        
        CoT = Energy / (Mass × Distance × Gravity)
        경사 기반: CoT ∝ sin(slope) for uphill
        
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
        self.energy_costmap = self._calculate_cot_costmap()
        
        # friction_map은 None으로 설정
        self.friction_map = None
        
        return self.height_map, self.slope_map, self.energy_costmap
    
    def _generate_height_map(self, terrain_scales):
        """다중 스케일 노이즈로 높이 맵 생성"""
        height_map = np.zeros((self.img_size, self.img_size))
        
        # 다중 스케일 노이즈 합성
        for scale, weight in terrain_scales:
            noise = np.random.rand(self.img_size, self.img_size)
            height_map += gaussian_filter(noise, sigma=scale) * weight
        
        # 정규화
        height_map = (height_map - height_map.min()) / (height_map.max() - height_map.min())
        height_map = height_map * (self.height_range[1] - self.height_range[0]) + self.height_range[0]
        
        return height_map.astype(np.float32)
    
    def _calculate_slope_map(self, height_map):
        """높이 맵으로부터 경사각 맵 계산"""
        gy, gx = np.gradient(height_map)
        slope_map = np.arctan(np.sqrt(gx**2 + gy**2))
        return slope_map.astype(np.float32)
    
    def _generate_friction_map(self, friction_range, smooth_sigma):
        """랜덤 마찰 계수 맵 생성 (지형 특성: 늪지대, 모래, 포장도로 등)"""
        roughness = gaussian_filter(np.random.rand(self.img_size, self.img_size), sigma=smooth_sigma)
        roughness = (roughness - roughness.min()) / (roughness.max() - roughness.min())
        friction_map = friction_range[0] + roughness * (friction_range[1] - friction_range[0])
        return friction_map.astype(np.float32)
    
    def _calculate_energy_costmap(self):
        """
        전체 맵에 대해 벡터화된 에너지 비용 계산
        등반 불가능한 지역은 무한대 비용
        """
        # 경사각이 한계를 넘으면 무한대
        impossible = self.slope_map > self.limit_angle
        
        # 1. 등판 저항
        f_grade = self.mass * self.gravity * np.sin(self.slope_map)
        f_grade = np.maximum(0, f_grade)  # 내리막은 0
        
        # 2. 구름 저항
        f_roll = self.mass * self.gravity * self.friction_map * np.cos(self.slope_map)
        
        # 3. 슬립 페널티
        slope_ratio = np.clip(self.slope_map / self.limit_angle, 0, 0.975)
        slip_factor = 1.0 / (1.0 - slope_ratio**2)
        
        # 총 에너지 비용 (단위 거리당)
        energy_cost = (f_grade + f_roll) * slip_factor
        energy_cost = np.maximum(1.0, energy_cost)  # 최소 비용
        
        # 등반 불가 지역 처리
        energy_cost[impossible] = np.inf
        
        return energy_cost.astype(np.float32)
    
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
        A* 알고리즘으로 에너지 효율적 경로 탐색
        
        Args:
            start (tuple): 시작 위치 (row, col)
            goal (tuple): 목표 위치 (row, col)
            
        Returns:
            list: 경로 [(row, col), ...] 또는 None
        """
        if self.energy_costmap is None:
            raise RuntimeError("먼저 generate()를 호출하여 맵을 생성해야 합니다.")
        
        return a_star_energy_search(
            self.energy_costmap, 
            self.slope_map,
            start, 
            goal
        )
    
    def visualize(self, path=None, start=None, goal=None, save_path='results/gradient_map.png'):
        """경사 맵과 경로 시각화"""
        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(1, 1, 1)
        
        # 경사각을 도(degree) 단위로 표시
        im = ax.imshow(np.degrees(self.slope_map), cmap='jet', origin='lower')
        plt.colorbar(im, ax=ax, label="Slope Angle (Degrees)")
        
        # 경로 그리기
        if path is not None and len(path) > 0:
            path_x = [p[1] for p in path]
            path_y = [p[0] for p in path]
            ax.plot(path_x, path_y, c='black', lw=5, alpha=0.8, zorder=9)
            ax.plot(path_x, path_y, c='white', lw=3, label='Energy Efficient Path', zorder=10)
        
        # 시작/목표 지점
        if start is not None:
            ax.scatter(start[1], start[0], c='cyan', s=250, edgecolors='black', 
                      linewidths=3, label='Start', zorder=11, marker='o')
        if goal is not None:
            ax.scatter(goal[1], goal[0], c='black', s=200, edgecolors='black', 
                      linewidths=3, marker='*', label='Goal', zorder=11)
        
        ax.set_title("Gradient Map with Energy Efficient Path", fontsize=16)
        ax.legend(loc='upper right')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig


def a_star_energy_search(energy_costmap, slope_map, start, goal):
    """
    에너지 효율 기반 A* 경로 탐색 (최적화 버전)
    
    Args:
        energy_costmap (ndarray): 단위 거리당 에너지 비용 맵
        slope_map (ndarray): 경사각 맵 (등반 불가 체크용)
        start (tuple): 시작 위치 (row, col)
        goal (tuple): 목표 위치 (row, col)
        
    Returns:
        list: 경로 또는 None
    """
    rows, cols = energy_costmap.shape
    start = tuple(start)
    goal = tuple(goal)
    
    # 시작/끝 지점이 등반 불가능하면 실패
    if not np.isfinite(energy_costmap[start]) or not np.isfinite(energy_costmap[goal]):
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
            if not np.isfinite(energy_costmap[nr, nc]):
                continue
            
            # 대각선 이동 시 코너컷 방지
            if abs(dr) + abs(dc) == 2:
                if (not np.isfinite(energy_costmap[cr + dr, cc]) or 
                    not np.isfinite(energy_costmap[cr, cc + dc])):
                    continue
            
            # 이동 거리
            step_distance = np.sqrt(2.0) if (dr != 0 and dc != 0) else 1.0
            
            # 에너지 비용 계산 (현재와 다음 셀의 평균)
            avg_cost = 0.5 * (energy_costmap[cr, cc] + energy_costmap[nr, nc])
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


# --- 메인 실행 코드 ---
if __name__ == "__main__":
    # 1. Generator 생성
    size = 100
    generator = GradientMapGenerator(
        img_size=size,
        height_range=(0, 10),
        mass=10.0,
        gravity=9.8,
        limit_angle_deg=30
    )
    
    # 2. 지형 생성
    print("지형 생성 중...")
    h_map, s_map, mu_map, e_costmap = generator.generate(
        terrain_scales=[(20, 30), (10, 15), (5, 5)],
        friction_range=(0.05, 0.3),
        friction_smooth=10
    )
    print(f"생성 완료 - 크기: {size}x{size}")
    print(f"높이 범위: {h_map.min():.2f}m ~ {h_map.max():.2f}m")
    print(f"경사각 범위: {np.degrees(s_map.min()):.2f}° ~ {np.degrees(s_map.max()):.2f}°")
    print(f"마찰 계수 범위: {mu_map.min():.3f} ~ {mu_map.max():.3f}")
    
    # 3. 경로 탐색
    start, goal = (10, 10), (90, 90)
    print(f"\n경로 탐색 중: {start} -> {goal}")
    path = generator.find_path(start, goal)
    
    if path:
        print(f"경로 발견! 길이: {len(path)} 포인트")
        
        # 경로의 총 에너지 계산
        total_energy = 0.0
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            dist = np.hypot(r2 - r1, c2 - c1)
            avg_cost = 0.5 * (e_costmap[r1, c1] + e_costmap[r2, c2])
            total_energy += dist * avg_cost
        print(f"총 에너지 소비: {total_energy:.2f} J")
    else:
        print("경로를 찾을 수 없습니다!")
    
    # 4. 시각화
    print("\n시각화 중...")
    generator.visualize(path=path, start=start, goal=goal)
    print("완료!")