"""
Terrain 이미지에 마우스로 path를 그리는 스크립트
생성된 Terrain 이미지(.png) 또는 .pt 파일을 로드하여 마우스로 path를 그리고 좌표를 저장


사용방법
python3 draw_path/draw_path.py --terrain data/raw/terrain_00001.pt --map-type height --a-star-weight 0.5

"""

import os
import sys
import argparse
import numpy as np
import heapq

# RDP 환경을 위한 matplotlib 백엔드 설정
# matplotlib.use()는 matplotlib.pyplot을 import하기 전에 호출해야 함
import matplotlib

# DISPLAY 환경 변수 확인
display = os.environ.get('DISPLAY')
has_display = display is not None and display != ''

# MPLBACKEND 환경 변수 확인 (headless 모드 강제 설정 여부)
mpl_backend = os.environ.get('MPLBACKEND', '').lower()
force_headless = mpl_backend == 'agg'

# 현재 백엔드 확인
current_backend = matplotlib.get_backend().lower()

print(f"DISPLAY: {display if has_display else 'Not set'}")
print(f"MPLBACKEND: {mpl_backend if mpl_backend else 'Not set'}")
print(f"Current matplotlib backend: {current_backend}")

# headless 모드인지 확인
is_headless = force_headless or 'agg' in current_backend

if not has_display:
    print("\n" + "="*60)
    print("WARNING: DISPLAY environment variable is not set!")
    print("="*60)
    print("To use RDP display, please set DISPLAY:")
    print("  export DISPLAY=:0.0")
    print("  or")
    print("  export DISPLAY=:10.0")
    print("="*60 + "\n")

if has_display and not is_headless:
    # DISPLAY가 있고 headless가 아니면 GUI 백엔드 시도
    gui_backends = ['TkAgg', 'Qt5Agg', 'Qt4Agg']
    backend_set = False
    
    for backend in gui_backends:
        try:
            # 백엔드 모듈이 실제로 존재하는지 확인
            backend_module_name = f'matplotlib.backends.backend_{backend.lower().replace("agg", "")}'
            try:
                __import__(backend_module_name)
            except ImportError:
                continue
            
            # 백엔드 설정 시도
            matplotlib.use(backend, force=False)
            backend_set = True
            print(f"✓ Using matplotlib backend: {backend}")
            break
        except (ImportError, ValueError, AttributeError) as e:
            continue
    
    if not backend_set:
        print("Warning: Could not set GUI backend, using default")

# 이제 matplotlib.pyplot을 import
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
import torch
import json
import yaml

# 최종 백엔드 확인
final_backend = matplotlib.get_backend()
print(f"Final backend: {final_backend}")


# -----------------------------------------------------------------------------
# 경로 보간 유틸
# -----------------------------------------------------------------------------

def _resample_path(path, horizon):
    """[N,2] path -> [horizon,2] 고정 길이 리샘플링."""
    path = np.asarray(path, dtype=np.float32)
    n = path.shape[0]
    if n == 0:
        return np.zeros((horizon, 2), dtype=np.float32)
    if n == 1:
        # 점이 하나면 horizon 개수만큼 복제
        return np.tile(path, (horizon, 1))
    t_cur = np.linspace(0, 1, n)
    t_tgt = np.linspace(0, 1, horizon)
    x = np.interp(t_tgt, t_cur, path[:, 0])
    y = np.interp(t_tgt, t_cur, path[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


# -----------------------------------------------------------------------------
# CoT 계산 및 A* 경로 탐색
# -----------------------------------------------------------------------------

def _calculate_paper_cot(slope_deg):
    """논문 기반 CoT (4차 다항식). slope_deg: 도 단위."""
    a, b, c, d, e = -1.53e-06, 2.07e-05, 2.20e-03, -3.24e-02, 0.65
    return (a * slope_deg**4) + (b * slope_deg**3) + (c * slope_deg**2) + (d * slope_deg) + e


def _calculate_directional_cot(height_curr, height_next, distance, limit_angle_deg=35.0):
    """이동 방향 CoT. 등반 불가 시 np.inf."""
    height_diff = height_next - height_curr
    slope_deg = np.degrees(np.arctan2(height_diff, distance))
    if abs(slope_deg) >= limit_angle_deg:
        return np.inf
    cot = _calculate_paper_cot(slope_deg)
    return max(cot, 0.1)


def _a_star_cot_search(slope_map, height_map, start, goal, limit_angle_rad, max_iterations,
                       pixel_resolution=0.5, weight=0.5, verbose=False):
    """CoT 효율 A* (방향성 CoT + weight). 경로 [(row,col), ...] 또는 None."""
    rows, cols = height_map.shape
    start, goal = tuple(start), tuple(goal)
    limit_angle_deg = np.degrees(limit_angle_rad)

    if slope_map[start] >= limit_angle_rad or slope_map[goal] >= limit_angle_rad:
        if verbose:
            print(f"      A* failed: start or goal too steep (start slope: {np.degrees(slope_map[start]):.1f}°, "
                  f"goal slope: {np.degrees(slope_map[goal]):.1f}°, limit: {np.degrees(limit_angle_rad):.1f}°)")
        return None

    map_size = rows * cols
    if max_iterations < map_size * 10:
        max_iterations = int(map_size * 10)

    def heuristic(a, b):
        d = np.hypot(a[0] - b[0], a[1] - b[1]) * pixel_resolution
        return d * 0.1

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
            if verbose:
                print(f"      A* success: found path with {len(path)} points in {iterations} iterations")
            return path[::-1]

        cr, cc = current
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
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
                if slope_map[cr + dr, cc] >= limit_angle_rad or slope_map[cr, cc + dc] >= limit_angle_rad:
                    continue
            step_cost = (1 - weight) * real_d + weight * (cot * real_d)
            g_new = g_score[cr, cc] + step_cost
            if g_new < g_score[nr, nc]:
                came_from[(nr, nc)] = current
                g_score[nr, nc] = g_new
                counter += 1
                heapq.heappush(open_heap, (g_new + heuristic((nr, nc), goal) * (1.0 + 1e-3), counter, (nr, nc)))

    if verbose:
        print(f"      A* failed: max iterations ({max_iterations}) reached or no path found "
              f"(explored {len(closed_set)} nodes)")
    return None


class TerrainPathDrawer:
    def __init__(self, terrain_path, map_type='height', a_star_weight=0.5, limit_angle_deg=30.0):
        """
        Args:
            terrain_path: Terrain 이미지 경로(.png) 또는 .pt 파일 경로
            map_type: 'height' 또는 'slope' - 어떤 맵을 표시할지 선택
            a_star_weight: A* 경로 탐색에서 CoT 가중치 (0.0-1.0, 기본값: 0.5)
            limit_angle_deg: 최대 등반 각도 (도 단위, 기본값: 30.0)
        """
        self.terrain_path = Path(terrain_path)
        self.map_type = map_type
        self.a_star_weight = a_star_weight
        self.limit_angle_rad = np.radians(limit_angle_deg)
        
        # 경로 포인트 저장 (픽셀 좌표) - A*로 계산된 전체 경로
        self.path_points = []
        # 클릭한 waypoint만 저장
        self.waypoints = []
        # 시작점과 끝점
        self.start_point = None
        self.end_point = None
        
        # Config에서 horizon 값 읽기
        self.horizon = self._load_horizon_from_config()
        
        # Terrain 데이터 로드
        if self.terrain_path.suffix == '.pt':
            # .pt 파일에서 로드
            self.load_from_pt()
        elif self.terrain_path.suffix == '.png':
            # 이미지 파일에서 로드
            self.load_from_image()
        else:
            raise ValueError(f"Unsupported file format: {self.terrain_path.suffix}")
        
        # matplotlib figure 설정
        # headless 모드 체크 및 GUI 백엔드 확인
        backend = matplotlib.get_backend()
        backend_lower = backend.lower()
        display = os.environ.get('DISPLAY')
        
        # 'Agg' 백엔드만 headless로 판단 (TkAgg, Qt5Agg 등은 GUI 백엔드)
        # 정확히 'Agg'이거나 'agg'로 끝나면서 GUI가 아닌 경우만 체크
        is_headless_backend = (
            backend_lower == 'agg' or 
            (backend_lower.endswith('agg') and backend_lower not in ['tkagg', 'qt5agg', 'qt4agg', 'gtkagg', 'gtk3agg'])
        )
        
        if is_headless_backend:
            error_msg = (
                "\n" + "="*60 + "\n"
                "ERROR: Cannot create interactive plot in headless mode!\n"
                "="*60 + "\n"
            )
            if not display:
                error_msg += (
                    "DISPLAY environment variable is not set.\n"
                    "Please set it for RDP:\n"
                    "  export DISPLAY=:0.0\n"
                    "  or\n"
                    "  export DISPLAY=:10.0\n"
                )
            else:
                error_msg += (
                    f"DISPLAY is set to: {display}\n"
                    f"But matplotlib is using headless backend: {backend}\n"
                    "Try unsetting MPLBACKEND:\n"
                    "  unset MPLBACKEND\n"
                )
            error_msg += (
                "Then restart the script.\n"
                "="*60 + "\n"
            )
            raise RuntimeError(error_msg)
        
        try:
            # 두 개의 서브플롯 생성: 왼쪽 height map, 오른쪽 slope map
            self.fig, (self.ax_height, self.ax_slope) = plt.subplots(1, 2, figsize=(20, 10))
            self.ax = self.ax_height  # 기본 ax는 height map (클릭 이벤트용)
        except Exception as e:
            if 'headless' in str(e).lower() or 'tk' in str(e).lower():
                raise RuntimeError(
                    f"\nFailed to create plot window: {e}\n"
                    f"Backend: {backend}\n"
                    f"DISPLAY: {display if display else 'Not set'}\n"
                    "\nPlease ensure:\n"
                    "  1. DISPLAY is set: export DISPLAY=:0.0\n"
                    "  2. X server is running\n"
                    "  3. Required packages are installed:\n"
                    "     - For TkAgg: apt-get install python3-tk\n"
                    "     - For Qt5Agg: pip install PyQt5\n"
                ) from e
            raise
        self.setup_plot()
        
        # 마우스 이벤트 연결
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # 현재 그려진 라인과 포인트
        self.line = None
        self.point_artists = []
    
    def _load_horizon_from_config(self):
        """Config 파일에서 horizon 값 읽기"""
        try:
            # 여러 가능한 config 경로 시도
            config_paths = [
                self.terrain_path.parent.parent.parent / "configs" / "default_config.yaml",
                Path(__file__).parent.parent / "configs" / "default_config.yaml",
                Path(__file__).parent.parent.parent / "configs" / "default_config.yaml",
            ]
            
            for config_path in config_paths:
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                        horizon = config.get("data", {}).get("horizon", 120)
                        print(f"Loaded horizon from config: {horizon}")
                        return horizon
            
            # config 파일을 찾을 수 없으면 기본값 사용
            print("Config file not found, using default horizon: 120")
            return 120
        except Exception as e:
            print(f"Error loading config, using default horizon: 120 ({e})")
            return 120
    
    def load_from_pt(self):
        """.pt 파일에서 terrain 데이터 로드 - height_map과 slope_map 모두 로드"""
        data = torch.load(self.terrain_path, map_location="cpu", weights_only=False)
        
        # A* 경로 탐색을 위해 height_map과 slope_map 모두 필요
        if "height_map" in data:
            self.height_map = data["height_map"].numpy()
        else:
            raise ValueError("height_map not found in .pt file")
        
        if "slope_map" in data:
            # slope_map이 라디안이면 도로 변환
            slope_data = data["slope_map"].numpy()
            if np.max(slope_data) < 2.0:  # 라디안으로 보임 (최대 약 1.57)
                self.slope_map = slope_data  # 이미 라디안
            else:  # 도 단위로 보임
                self.slope_map = np.radians(slope_data)  # 도를 라디안으로 변환
        else:
            # slope_map이 없으면 height_map에서 계산
            pixel_res = float(data.get("pixel_resolution", 0.5))
            gy, gx = np.gradient(self.height_map, pixel_res)
            mag = np.sqrt(gx**2 + gy**2)
            self.slope_map = np.arctan(mag).astype(np.float32)
        
        # 표시용 terrain_map (height 또는 slope)
        if self.map_type == 'height':
            self.terrain_map = self.height_map
        elif self.map_type == 'slope':
            self.terrain_map = np.degrees(self.slope_map)  # 표시는 도 단위
        else:
            raise ValueError(f"Unknown map_type: {self.map_type}")
        
        self.img_size = int(data.get("img_size", self.terrain_map.shape[0]))
        self.pixel_resolution = float(data.get("pixel_resolution", 0.5))
        self.max_iterations = int(data.get("max_iterations", self.img_size * self.img_size * 10))
        
    def load_from_image(self):
        """이미지 파일에서 terrain 로드 - A*를 사용할 수 없음 (height/slope 정보 부족)"""
        img = plt.imread(self.terrain_path)
        if len(img.shape) == 3:
            # RGB 이미지를 grayscale로 변환
            self.terrain_map = np.mean(img, axis=2)
        else:
            self.terrain_map = img
        
        self.img_size = self.terrain_map.shape[0]
        self.pixel_resolution = 0.5  # 기본값
        
        # 이미지 파일에서는 height_map과 slope_map을 추정할 수 없음
        # A*는 .pt 파일에서만 사용 가능
        self.height_map = None
        self.slope_map = None
        print("Warning: A* pathfinding requires .pt file with height_map and slope_map.")
        print("Image files only support manual path drawing.")
        
    def setup_plot(self):
        """플롯 설정 - height map과 slope map을 함께 표시"""
        # Height map (왼쪽)
        if self.height_map is not None:
            im_height = self.ax_height.imshow(self.height_map, cmap='terrain', origin='lower')
            title_height = 'Height Map - Click waypoints, press G for random start/end, Enter to compute A* path (R: reset)'
        else:
            # 이미지 파일인 경우
            im_height = self.ax_height.imshow(self.terrain_map, cmap='terrain', origin='lower')
            title_height = 'Height Map [Image mode: no A*]'
        
        self.ax_height.set_title(title_height, fontsize=12, pad=10)
        plt.colorbar(im_height, ax=self.ax_height, shrink=0.8)
        self.ax_height.set_aspect('equal')
        self.ax_height.set_xlabel('X (pixels)')
        self.ax_height.set_ylabel('Y (pixels)')
        
        # Slope map (오른쪽)
        if self.slope_map is not None:
            slope_map_deg = np.degrees(self.slope_map)
            im_slope = self.ax_slope.imshow(slope_map_deg, cmap='jet', origin='lower', vmin=0, vmax=35)
            title_slope = 'Slope Map (degrees)'
        else:
            # 이미지 파일인 경우 slope map이 없음
            im_slope = self.ax_slope.imshow(np.zeros_like(self.terrain_map), cmap='jet', origin='lower', vmin=0, vmax=35)
            title_slope = 'Slope Map [Not available for image files]'
        
        self.ax_slope.set_title(title_slope, fontsize=12, pad=10)
        plt.colorbar(im_slope, ax=self.ax_slope, shrink=0.8)
        self.ax_slope.set_aspect('equal')
        self.ax_slope.set_xlabel('X (pixels)')
        self.ax_slope.set_ylabel('Y (pixels)')
        
    def on_click(self, event):
        """마우스 클릭 이벤트 핸들러 - waypoint만 추가 (height map 또는 slope map에서 클릭 가능)"""
        # height map 또는 slope map에서 클릭 가능
        if event.inaxes != self.ax_height and event.inaxes != self.ax_slope:
            return
        
        if event.button == 1:  # 왼쪽 클릭
            x, y = int(event.xdata), int(event.ydata)
            
            # 경계 체크
            if 0 <= x < self.img_size and 0 <= y < self.img_size:
                self.waypoints.append([x, y])
                print(f"Waypoint added: ({x}, {y}) [Total: {len(self.waypoints)}]")
                
                self.update_display()
    
    def on_key(self, event):
        """키보드 이벤트 핸들러"""
        if event.key == 'enter' or event.key == 'return':
            self.finish_drawing()
        elif event.key == 'r' or event.key == 'R':
            self.reset_path()
        elif event.key == 'g' or event.key == 'G':
            # 'G' 키로 랜덤 시작/끝점 생성
            self.generate_random_start_end()
    
    def update_display(self):
        """화면 업데이트 - waypoint와 A* 경로 표시 (height map과 slope map 모두에 표시)"""
        # 기존 라인과 포인트 제거
        if self.line:
            self.line.remove()
            self.line = None
        if hasattr(self, 'line_slope') and self.line_slope:
            self.line_slope.remove()
            self.line_slope = None
        for artist in self.point_artists:
            artist.remove()
        self.point_artists = []
        
        # 두 개의 서브플롯에 모두 그리기
        for ax in [self.ax_height, self.ax_slope]:
            # 1. 시작점(녹색) 및 끝점(주황색) 표시
            if self.start_point is not None:
                circle = Circle((self.start_point[0], self.start_point[1]), radius=1, 
                               color='green', zorder=11, fill=True, label='Start')
                ax.add_patch(circle)
                self.point_artists.append(circle)
            
            if self.end_point is not None:
                circle = Circle((self.end_point[0], self.end_point[1]), radius=1, 
                               color='orange', zorder=11, fill=True, label='End')
                ax.add_patch(circle)
                self.point_artists.append(circle)
            
            # 2. Waypoint 표시 (빨간 원)
            if len(self.waypoints) > 0:
                waypoints_arr = np.array(self.waypoints)
                for i, waypoint in enumerate(waypoints_arr):
                    label = 'Waypoint' if i == 0 else None
                    circle = Circle((waypoint[0], waypoint[1]), radius=0.75, 
                                   color='red', zorder=10, fill=True, label=label)
                    ax.add_patch(circle)
                    self.point_artists.append(circle)

                # --- 가이드라인(점선) 로직 ---
                if len(self.path_points) == 0: # A* 계산 전일 때만 표시
                    # (1) Start부터 첫 번째 Waypoint까지 점선 연결
                    if self.start_point is not None:
                        pts = np.array([self.start_point, self.waypoints[0]])
                        l, = ax.plot(pts[:, 0], pts[:, 1], 'r--', linewidth=1.0, alpha=0.5, zorder=4)
                        self.point_artists.append(l)
                    
                    # (2) Waypoint들 사이를 순서대로 점선 연결
                    if len(self.waypoints) > 1:
                        l, = ax.plot(waypoints_arr[:, 0], waypoints_arr[:, 1], 
                                     'r--', linewidth=1.0, alpha=0.5, zorder=4)
                        self.point_artists.append(l)
                    
                    # (3) 마지막으로 찍은 Waypoint와 End 포인트만 점선으로 연결
                    if self.end_point is not None:
                        last_wp_to_end = np.array([self.waypoints[-1], self.end_point])
                        l, = ax.plot(last_wp_to_end[:, 0], last_wp_to_end[:, 1], 
                                     'r--', linewidth=1.0, alpha=0.8, zorder=4)
                        self.point_artists.append(l)
            
            # Waypoint가 하나도 없을 때, Start와 End를 연결하는 가이드라인 (선택 사항)
            elif self.start_point is not None and self.end_point is not None and len(self.path_points) == 0:
                pts = np.array([self.start_point, self.end_point])
                l, = ax.plot(pts[:, 0], pts[:, 1], 'r--', linewidth=1.0, alpha=0.5, zorder=4)
                self.point_artists.append(l)

            # 3. A* 경로 표시 (계산 완료 후 파란 실선)
            if len(self.path_points) > 1:
                points = np.array(self.path_points)
                if ax == self.ax_height:
                    self.line, = ax.plot(points[:, 0], points[:, 1], 
                                         'b-', linewidth=1.5, alpha=0.8, zorder=12, label='A* Path')
                else:
                    self.line_slope, = ax.plot(points[:, 0], points[:, 1], 
                                               'b-', linewidth=1.5, alpha=0.8, zorder=12, label='A* Path')
            
            # 범례 표시
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)
        
        self.fig.canvas.draw()
    
    def reset_path(self):
        """경로 초기화"""
        self.path_points = []
        self.waypoints = []
        self.start_point = None
        self.end_point = None
        self.update_display()
        print("Path reset.")
    
    def generate_random_start_end(self, margin=10, min_distance_factor=1.5):
        """랜덤 시작점과 끝점 생성"""
        if self.height_map is None or self.slope_map is None:
            print("Warning: Cannot generate random points without height_map and slope_map")
            return False
        
        margin = max(margin, self.img_size // 10)
        min_distance = int(self.img_size // min_distance_factor)
        max_attempts = 100
        
        for attempt in range(max_attempts):
            # 랜덤 시작점 생성
            start_x = np.random.randint(margin, self.img_size - margin)
            start_y = np.random.randint(margin, self.img_size - margin)
            start = (start_y, start_x)  # (row, col)
            
            # 랜덤 끝점 생성
            end_x = np.random.randint(margin, self.img_size - margin)
            end_y = np.random.randint(margin, self.img_size - margin)
            end = (end_y, end_x)  # (row, col)
            
            # 최소 거리 체크
            distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            if distance < min_distance:
                continue
            
            # 경사도 체크 (너무 가파른 곳은 피함)
            if (self.slope_map[start] >= self.limit_angle_rad or 
                self.slope_map[end] >= self.limit_angle_rad):
                continue
            
            # 유효한 점들 (waypoint에 추가하지 않음, 참고용으로만 표시)
            self.start_point = [start_x, start_y]
            self.end_point = [end_x, end_y]
            
            print(f"Generated random start: ({start_x}, {start_y}), end: ({end_x}, {end_y})")
            print("Click to add waypoints, then press Enter to compute A* path")
            self.update_display()
            return True
        
        print("Warning: Could not generate valid random start/end points after 100 attempts")
        return False
    
    def compute_path_from_waypoints(self):
        """Start -> Waypoints -> End를 모두 연결하는 A* 경로 계산"""
        if self.height_map is None or self.slope_map is None:
            print("Warning: A* requires .pt file.")
            return

        # 모든 방문 지점을 하나의 리스트로 통합
        nodes_to_visit = []
        if self.start_point: nodes_to_visit.append(self.start_point)
        nodes_to_visit.extend(self.waypoints)
        if self.end_point: nodes_to_visit.append(self.end_point)

        if len(nodes_to_visit) < 2:
            print("Warning: Not enough points to compute path.")
            return

        full_path = []
        # 첫 번째 지점을 경로의 시작으로 추가
        full_path.append([nodes_to_visit[0][0], nodes_to_visit[0][1]])

        for i in range(len(nodes_to_visit) - 1):
            curr = nodes_to_visit[i]
            nxt = nodes_to_visit[i+1]
            
            start_node = (int(curr[1]), int(curr[0])) # (row, col)
            goal_node = (int(nxt[1]), int(nxt[0]))
            
            print(f"  Segment {i+1}/{len(nodes_to_visit)-1}: {curr} -> {nxt}")
            print(f"    A* search: start_node={start_node} (row,col), goal_node={goal_node} (row,col)")
            segment = _a_star_cot_search(
                self.slope_map, self.height_map, start_node, goal_node,
                self.limit_angle_rad, self.max_iterations,
                pixel_resolution=self.pixel_resolution, weight=self.a_star_weight,
                verbose=True
            )
            
            if segment:
                # 첫 번째 점은 이미 이전 세그먼트의 마지막(또는 시작점)이므로 제외하고 추가
                num_points = len(segment) - 1  # 첫 번째 점 제외
                for row, col in segment[1:]:
                    full_path.append([col, row])
                print(f"    ✓ A* success: {len(segment)} points found, added {num_points} new points")
            else:
                # A* 실패 시 직선 연결
                print(f"    ✗ A* failed, using straight line for this segment.")
                full_path.append([nxt[0], nxt[1]])

        self.path_points = full_path
        print(f"\n✓ Path computed with {len(self.path_points)} points.")
        self.update_display()
    
    def compute_path_from_start_to_end(self):
        """시작점에서 끝점으로 A* 경로 계산 (레거시 메서드, 사용 안 함)"""
        if self.start_point is None or self.end_point is None:
            return
        
        if self.height_map is None or self.slope_map is None:
            print("Warning: Cannot compute A* path without height_map and slope_map")
            return
        
        start = (self.start_point[1], self.start_point[0])  # (row, col)
        goal = (self.end_point[1], self.end_point[0])  # (row, col)
        
        print(f"Computing A* path from start ({self.start_point[0]}, {self.start_point[1]}) to end ({self.end_point[0]}, {self.end_point[1]})...")
        a_star_path = _a_star_cot_search(
            self.slope_map, self.height_map, start, goal,
            self.limit_angle_rad, self.max_iterations,
            pixel_resolution=self.pixel_resolution,
            weight=self.a_star_weight
        )
        
        if a_star_path is not None and len(a_star_path) > 0:
            # A* 경로를 (x, y) 형식으로 변환
            self.path_points = []
            for row, col in a_star_path:
                self.path_points.append([col, row])  # (x, y) 형식으로 변환
            print(f"✓ A* path found: {len(a_star_path)} points")
            self.update_display()
        else:
            print(f"✗ A* path not found between start and end")
            self.path_points = []
            self.update_display()
    
    def finish_drawing(self):
        """경로 그리기 완료 및 저장 - waypoint 기준으로 A* 경로 계산"""
        # start_point, waypoints, end_point를 모두 포함해서 최소 2개 이상의 점이 있어야 함
        total_points = 0
        if self.start_point: total_points += 1
        total_points += len(self.waypoints)
        if self.end_point: total_points += 1
        
        if total_points < 2:
            print("Warning: At least 2 points (start, waypoints, or end) are required to form a path.")
            return
        
        # waypoint들을 기준으로 A* 경로 계산
        if self.height_map is not None and self.slope_map is not None:
            print("\nComputing A* paths between waypoints...")
            self.compute_path_from_waypoints()
        else:
            # 이미지 파일인 경우: waypoint를 직선으로 연결
            print("\nImage mode: Connecting waypoints with straight lines...")
            # start_point, waypoints, end_point를 모두 포함
            self.path_points = []
            if self.start_point:
                self.path_points.append(self.start_point)
            self.path_points.extend(self.waypoints)
            if self.end_point:
                self.path_points.append(self.end_point)
        
        if len(self.path_points) < 2:
            print("Warning: Could not compute valid path.")
            return
        
        # 텍스트 instruction 입력 받기
        print("\n" + "="*60)
        print("Enter text instruction for this path (or press Enter to skip):")
        print("Examples: 'Find the fastest route', 'Take the safest path', 'Minimize energy cost'")
        print("="*60)
        text_instruction = input("Instruction: ").strip()
        if not text_instruction:
            text_instruction = None
            print("No instruction provided, saving path without text.")
        else:
            print(f"Instruction saved: '{text_instruction}'")
        
        # 좌표 저장
        self.save_path(text_instruction=text_instruction)
        plt.close(self.fig)
    
    def save_path(self, text_instruction=None):
        """경로 좌표를 파일로 저장 - 보간된 경로만 저장
        
        Args:
            text_instruction: Optional text instruction describing the path (e.g., "Find fastest route")
        """
        points = np.array(self.path_points)
        
        # 출력 파일명 생성
        base_name = self.terrain_path.stem
        output_dir = self.terrain_path.parent / "drawn_paths"
        output_dir.mkdir(exist_ok=True)
        
        # 보간된 경로 생성 (horizon 개수만큼)
        resampled_points = _resample_path(points, self.horizon)
        
        # 2. JSON으로 저장 (보간된 경로만)
        json_path = output_dir / f"{base_name}_path.json"
        path_data = {
            "terrain_file": str(self.terrain_path),
            "map_type": self.map_type,
            "img_size": int(self.img_size),
            "pixel_resolution": float(self.pixel_resolution),
            "horizon": int(self.horizon),
            "num_points": int(len(resampled_points)),
            "num_waypoints": int(len(self.waypoints)),
            "waypoints_pixel": self.waypoints,  # 클릭한 waypoint만
            "start_point": self.start_point,  # 시작점
            "end_point": self.end_point,  # 끝점
            "points_pixel": resampled_points.tolist(),  # 보간된 픽셀 좌표 (horizon 개수)
            "points_world": self.pixel_to_world(resampled_points).tolist(),  # 보간된 월드 좌표
            "a_star_weight": float(self.a_star_weight) if self.height_map is not None else None,
            "limit_angle_deg": float(np.degrees(self.limit_angle_rad)) if self.height_map is not None else None,
            "used_a_star": self.height_map is not None and self.slope_map is not None,
            "text_instruction": text_instruction  # 텍스트 instruction 추가
        }
        
        with open(json_path, 'w') as f: 
            json.dump(path_data, f, indent=2)
        print(f"✓ Path saved as JSON: {json_path}")
        
        # metadata 디렉토리에 instruction도 별도로 저장 (VLM 형식과 호환)
        if text_instruction:
            metadata_dir = self.terrain_path.parent.parent / "metadata"
            metadata_dir.mkdir(exist_ok=True)
            metadata_path = metadata_dir / f"{base_name}_instruction.json"
            metadata_data = {
                "map_id": base_name,
                "instruction": text_instruction,
                "path_json": str(json_path)
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata_data, f, indent=2)
            print(f"✓ Instruction saved to metadata: {metadata_path}")
        
    
    def pixel_to_world(self, points_pixel):
        """픽셀 좌표를 월드 좌표(미터)로 변환"""
        # 이미지 중심을 원점으로 하는 좌표계
        # (0, 0) 픽셀 = (-img_size/2 * resolution, -img_size/2 * resolution) 미터
        center = self.img_size / 2.0
        world_points = (points_pixel - center) * self.pixel_resolution
        return world_points
    
    def show(self):
        """인터랙티브 플롯 표시"""
        print("\n" + "="*60)
        print("Terrain Path Drawer")
        print("="*60)
        print("Instructions:")
        print("  - Left click: Add waypoint")
        print("  - Enter: Compute A* path between waypoints and save")
        print("  - R: Reset current path")
        if self.height_map is not None and self.slope_map is not None:
            print("  - G: Generate random start/end points")
            print(f"  - A* weight: {self.a_star_weight:.2f} (CoT weight)")
            print(f"  - Limit angle: {np.degrees(self.limit_angle_rad):.1f}°")
        print("="*60 + "\n")
        print(f"Backend: {matplotlib.get_backend()}")
        print(f"Display: {os.environ.get('DISPLAY', 'Not set')}")
        print("Opening window...\n")
        
        # 인터랙티브 모드 활성화
        plt.ion()
        
        try:
            # 창이 제대로 열리는지 확인
            self.fig.show()
            plt.draw()
            
            # 사용자가 창을 닫을 때까지 대기
            print("Window opened. Start drawing your path!")
            print("(Close the window or press Enter to finish)\n")
            
            # 메인 루프 - 창이 열려있는 동안 대기
            try:
                plt.show(block=True)
            except Exception as e:
                print(f"Error in plt.show(): {e}")
                # 대안: input()으로 대기
                input("Press Enter to finish and save path...")
            
        except Exception as e:
            print(f"Error displaying plot: {e}")
            import traceback
            traceback.print_exc()
            print("\nTrying to save current state...")
            if len(self.path_points) > 0:
                self.save_path()
            raise
        finally:
            plt.ioff()


def main():
    parser = argparse.ArgumentParser(
        description="Draw path on terrain image using mouse clicks"
    )
    parser.add_argument(
        "--terrain", "-t",
        type=str,
        required=True,
        help="Path to terrain image (.png) or terrain data (.pt)"
    )
    parser.add_argument(
        "--map-type", "-m",
        type=str,
        choices=['height', 'slope'],
        default='height',
        help="Type of map to display (default: height)"
    )
    parser.add_argument(
        "--a-star-weight", "-w",
        type=float,
        default=0.5,
        help="A* pathfinding CoT weight (0.0-1.0, default: 0.5)"
    )
    parser.add_argument(
        "--limit-angle",
        type=float,
        default=30.0,
        help="Maximum climbing angle in degrees (default: 30.0)"
    )
    
    args = parser.parse_args()
    
    terrain_path = Path(args.terrain)
    if not terrain_path.exists():
        print(f"Error: Terrain file not found: {terrain_path}")
        return
    
    # DISPLAY 환경 변수 확인
    display = os.environ.get('DISPLAY')
    if display:
        print(f"DISPLAY environment variable: {display}")
    else:
        print("Warning: DISPLAY environment variable not set")
        print("If using RDP, make sure DISPLAY is set correctly")
    
    try:
        print(f"Loading terrain from: {terrain_path}")
        drawer = TerrainPathDrawer(terrain_path, args.map_type, 
                                  a_star_weight=args.a_star_weight,
                                  limit_angle_deg=args.limit_angle)
        print("Terrain loaded successfully")
        
        # 랜덤 시작/끝점 자동 생성 (항상 실행)
        if drawer.height_map is not None and drawer.slope_map is not None:
            print("Generating random start and end points...")
            drawer.generate_random_start_end()
        else:
            print("Warning: Cannot generate random points for image files (requires .pt file)")
        
        drawer.show()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        if len(drawer.path_points) > 0:
            print("Saving current path...")
            drawer.save_path()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()