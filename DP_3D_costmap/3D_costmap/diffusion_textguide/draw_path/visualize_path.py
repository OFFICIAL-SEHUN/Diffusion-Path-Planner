"""
저장된 waypoint와 경로를 시각화하는 스크립트
JSON 파일(.json) 또는 NumPy 파일(.npy)을 로드하여 terrain 위에 표시


# 기본 사용 (JSON 파일)
python3 draw_path/visualize_path.py --path data/raw/drawn_paths/terrain_00001_path.json

# Terrain 파일 직접 지정
python3 visualize_path.py --path drawn_paths/terrain_name_path.json --terrain terrain.pt

# Map type 지정
python3 visualize_path.py --path drawn_paths/terrain_name_path.json --map-type slope

# 이미지로 저장
python3 visualize_path.py --path drawn_paths/terrain_name_path.json --save output.png

# NumPy 파일 사용
python3 visualize_path.py --path drawn_paths/terrain_name_path.npy --terrain terrain.pt

"""

import os
import sys
import argparse
import numpy as np
import json
from pathlib import Path

# RDP 환경을 위한 matplotlib 백엔드 설정
import matplotlib

# DISPLAY 환경 변수 확인
display = os.environ.get('DISPLAY')
has_display = display is not None and display != ''

# MPLBACKEND 환경 변수 확인 (headless 모드 강제 설정 여부)
mpl_backend = os.environ.get('MPLBACKEND', '').lower()
force_headless = mpl_backend == 'agg'

# 현재 백엔드 확인
current_backend = matplotlib.get_backend().lower()

# headless 모드인지 확인
is_headless = force_headless or 'agg' in current_backend

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
import torch


def load_path_data(path_file):
    """경로 데이터 로드 (JSON 또는 NumPy)"""
    path_file = Path(path_file)
    
    if path_file.suffix == '.json':
        with open(path_file, 'r') as f:
            data = json.load(f)
        return data
    elif path_file.suffix == '.npy':
        points = np.load(path_file)
        # NumPy 파일만 있는 경우 기본 데이터 구조 생성
        return {
            "points_pixel": points.tolist(),
            "waypoints_pixel": points.tolist(),  # 전체 경로를 waypoint로 간주
            "start_point": points[0].tolist() if len(points) > 0 else None,
            "end_point": points[-1].tolist() if len(points) > 0 else None,
            "terrain_file": None,
            "map_type": "height"
        }
    else:
        raise ValueError(f"Unsupported file format: {path_file.suffix}")


def load_terrain(terrain_path, return_both=False):
    """Terrain 데이터 로드 - height_map과 slope_map 모두 반환"""
    terrain_path = Path(terrain_path)
    
    if terrain_path.suffix == '.pt':
        data = torch.load(terrain_path, map_location="cpu", weights_only=False)
        
        if "height_map" in data:
            height_map = data["height_map"].numpy()
        else:
            raise ValueError("height_map not found in .pt file")
        
        # slope_map 로드 또는 계산
        if "slope_map" in data:
            slope_data = data["slope_map"].numpy()
            if np.max(slope_data) < 2.0:  # 라디안
                slope_map = slope_data
            else:  # 도 단위
                slope_map = np.radians(slope_data)
        else:
            pixel_res = float(data.get("pixel_resolution", 0.5))
            gy, gx = np.gradient(height_map, pixel_res)
            mag = np.sqrt(gx**2 + gy**2)
            slope_map = np.arctan(mag)
        
        if return_both:
            return height_map, np.degrees(slope_map), height_map.shape[0]
        else:
            return height_map, height_map.shape[0]
    elif terrain_path.suffix == '.png':
        img = plt.imread(terrain_path)
        if len(img.shape) == 3:
            terrain_map = np.mean(img, axis=2)
        else:
            terrain_map = img
        if return_both:
            return terrain_map, None, terrain_map.shape[0]
        else:
            return terrain_map, terrain_map.shape[0]
    else:
        raise ValueError(f"Unsupported terrain file format: {terrain_path.suffix}")


def visualize_path(path_file, terrain_file=None, map_type=None, save_image=None):
    """경로 시각화"""
    # 경로 데이터 로드
    print(f"Loading path data from: {path_file}")
    path_data = load_path_data(path_file)
    
    # Terrain 로드 - height_map과 slope_map 모두 로드
    if terrain_file is None:
        terrain_file = path_data.get("terrain_file")
    
    if terrain_file is None:
        print("Warning: No terrain file specified. Creating visualization without terrain background.")
        height_map = None
        slope_map = None
        img_size = path_data.get("img_size", 256)
    else:
        terrain_file = Path(terrain_file)
        if not terrain_file.exists():
            print(f"Warning: Terrain file not found: {terrain_file}")
            height_map = None
            slope_map = None
            img_size = path_data.get("img_size", 256)
        else:
            print(f"Loading terrain from: {terrain_file}")
            height_map, slope_map, img_size = load_terrain(terrain_file, return_both=True)
    
    # 경로 데이터 추출
    waypoints = path_data.get("waypoints_pixel", [])
    path_points = path_data.get("points_pixel", [])
    start_point = path_data.get("start_point")
    end_point = path_data.get("end_point")
    
    if len(waypoints) == 0 and len(path_points) == 0:
        print("Error: No waypoints or path points found in the file")
        return
    
    # 시각화 - 두 개의 서브플롯 생성 (height map과 slope map)
    fig, (ax_height, ax_slope) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 경로를 두 맵 모두에 그리는 함수
    def draw_path_on_ax(ax, title_suffix=""):
        # Terrain 배경 표시
        if height_map is not None and ax == ax_height:
            im = ax.imshow(height_map, cmap='terrain', origin='lower')
            title = f'Path Visualization - Height Map{title_suffix}'
            plt.colorbar(im, ax=ax, shrink=0.8)
        elif slope_map is not None and ax == ax_slope:
            im = ax.imshow(slope_map, cmap='jet', origin='lower', vmin=0, vmax=35)
            title = f'Path Visualization - Slope Map{title_suffix}'
            plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            title = f'Path Visualization{title_suffix}'
        
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_aspect('equal')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        # 시작점 표시 (녹색)
        if start_point is not None:
            circle = Circle((start_point[0], start_point[1]), radius=1, 
                           color='green', zorder=12, fill=True, label='Start')
            ax.add_patch(circle)
        
        # 끝점 표시 (주황색)
        if end_point is not None:
            circle = Circle((end_point[0], end_point[1]), radius=1, 
                           color='orange', zorder=12, fill=True, label='End')
            ax.add_patch(circle)
        
        # Waypoint 표시 (빨간 원)
        if len(waypoints) > 0:
            waypoints_arr = np.array(waypoints)
            for waypoint in waypoints_arr:
                circle = Circle((waypoint[0], waypoint[1]), radius=1, 
                               color='red', zorder=11, fill=True)
                ax.add_patch(circle)
            
            # start_point가 있고 waypoint가 있으면 start와 첫 waypoint 연결
            if start_point is not None and len(waypoints) > 0:
                start_to_first = np.array([start_point, waypoints[0]])
                ax.plot(start_to_first[:, 0], start_to_first[:, 1], 
                       'r--', linewidth=1.5, alpha=0.6, zorder=6)
            
            # end_point가 있고 waypoint가 있으면 마지막 waypoint와 end 연결
            if end_point is not None and len(waypoints) > 0:
                last_to_end = np.array([waypoints[-1], end_point])
                ax.plot(last_to_end[:, 0], last_to_end[:, 1], 
                       'r--', linewidth=1.5, alpha=0.6, zorder=6)
            
            # Waypoint를 점선으로 연결
            if len(waypoints) > 1:
                ax.plot(waypoints_arr[:, 0], waypoints_arr[:, 1], 
                       'r--', linewidth=1.5, alpha=0.6, zorder=6, label='Waypoints')
        
        # A* 경로 표시 (파란 직선)
        if len(path_points) > 0:
            points_arr = np.array(path_points)
            if len(path_points) > 1:
                ax.plot(points_arr[:, 0], points_arr[:, 1], 
                       'b-', linewidth=2.0, alpha=0.8, zorder=7, label='A* Path')
        
        # 범례 추가
        ax.legend(loc='upper right', fontsize=10)
        
        # 정보 텍스트 추가 (height map에만)
        if ax == ax_height:
            info_text = []
            if len(waypoints) > 0:
                info_text.append(f"Waypoints: {len(waypoints)}")
            if len(path_points) > 0:
                info_text.append(f"Path points: {len(path_points)}")
            if path_data.get("used_a_star"):
                info_text.append(f"A* weight: {path_data.get('a_star_weight', 'N/A')}")
                info_text.append(f"Limit angle: {path_data.get('limit_angle_deg', 'N/A')}°")
            
            if info_text:
                info_str = "\n".join(info_text)
                ax.text(0.02, 0.98, info_str, transform=ax.transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 두 맵 모두에 경로 그리기
    draw_path_on_ax(ax_height)
    draw_path_on_ax(ax_slope)
    
    plt.tight_layout()
    
    # 저장 또는 표시
    if save_image:
        output_path = Path(save_image)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize saved waypoints and paths"
    )
    parser.add_argument(
        "--path", "-p",
        type=str,
        required=True,
        help="Path to saved path file (.json or .npy)"
    )
    parser.add_argument(
        "--terrain", "-t",
        type=str,
        default=None,
        help="Path to terrain file (.pt or .png). If not specified, uses terrain_file from path data"
    )
    parser.add_argument(
        "--map-type", "-m",
        type=str,
        choices=['height', 'slope'],
        default=None,
        help="Type of map to display (default: from path data)"
    )
    parser.add_argument(
        "--save", "-s",
        type=str,
        default=None,
        help="Save visualization to image file (e.g., output.png)"
    )
    
    args = parser.parse_args()
    
    path_file = Path(args.path)
    if not path_file.exists():
        print(f"Error: Path file not found: {path_file}")
        return
    
    terrain_file = None
    if args.terrain:
        terrain_file = Path(args.terrain)
        if not terrain_file.exists():
            print(f"Warning: Terrain file not found: {terrain_file}")
            terrain_file = None
    
    try:
        visualize_path(path_file, terrain_file, args.map_type, args.save)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
