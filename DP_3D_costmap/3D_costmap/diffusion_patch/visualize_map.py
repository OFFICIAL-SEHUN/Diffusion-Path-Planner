"""
맵 시각화 스크립트
생성된 terrain 맵과 다양한 weight 값에 따른 경로를 시각화합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from generate_data import SlopeCotGenerator, load_config
from path_utils import denormalize_path
import argparse


def visualize_terrain_map(config_path='configs/default_config.yaml', num_samples=3):
    """
    Terrain 맵과 다양한 weight에 따른 경로를 시각화
    
    Args:
        config_path: 설정 파일 경로
        num_samples: 생성할 샘플 개수
    """
    # Config 로드
    try:
        config = load_config(config_path)
    except:
        print(f"Config 파일을 찾을 수 없습니다: {config_path}")
        print("기본 설정을 사용합니다.")
        config = {
            'data': {'img_size': 100, 'horizon': 120},
            'gradient': {
                'height_range': [0, 5],
                'terrain_scales': None,
                'mass': 10.0,
                'gravity': 9.8,
                'limit_angle_deg': 30,
                'pixel_resolution': 0.5
            }
        }
    
    img_size = config['data']['img_size']
    height_range = tuple(config['gradient']['height_range'])
    mass = config['gradient']['mass']
    gravity = config['gradient']['gravity']
    limit_angle_deg = config['gradient']['limit_angle_deg']
    pixel_resolution = config['gradient'].get('pixel_resolution', 0.5)
    terrain_scales = config['gradient'].get('terrain_scales', None)
    
    margin = img_size // 10
    min_distance = int(img_size * 0.7)  # 최소 거리
    
    # Weight 값들 (트레이드오프)
    weights = [0.1, 0.3, 0.5, 0.7, 0.9]
    weight_labels = {
        0.1: "Quickly\n(거리 우선)",
        0.3: "Fast route\n(거리 약간 우선)",
        0.5: "Balanced\n(균형)",
        0.7: "Safe route\n(CoT 약간 우선)",
        0.9: "Energy efficient\n(CoT 우선)"
    }
    
    for sample_idx in range(num_samples):
        print(f"\n{'='*60}")
        print(f"Sample {sample_idx + 1}/{num_samples}")
        print(f"{'='*60}")
        
        # 1. 지형 생성
        generator = SlopeCotGenerator(
            img_size=img_size,
            height_range=height_range,
            mass=mass,
            gravity=gravity,
            limit_angle_deg=limit_angle_deg,
            pixel_resolution=pixel_resolution
        )
        
        # 지형 생성 시도 (난이도 필터링 통과할 때까지)
        for attempt in range(50):
            h_map, s_map = generator.generate(terrain_scales=terrain_scales)
            slope_degrees = np.degrees(s_map)
            mean_slope = np.mean(slope_degrees)
            max_slope = np.max(slope_degrees)
            steep_ratio = np.sum(slope_degrees > 30.0) / slope_degrees.size
            
            if (3.0 <= mean_slope <= 25.0 and max_slope <= 35.0 and steep_ratio <= 0.3):
                break
        
        print(f"✓ Terrain generated (mean_slope: {mean_slope:.2f}°, max_slope: {max_slope:.2f}°)")
        
        # 2. 시작/끝 지점 찾기 (slope 체크 포함)
        start_pos = None
        goal_pos = None
        limit_angle_rad = np.radians(limit_angle_deg)
        
        print("Searching for valid start/goal positions...")
        for attempt in range(500):  # 더 많은 시도 허용
            start_pos = (np.random.randint(margin, img_size - margin),
                        np.random.randint(margin, img_size - margin))
            goal_pos = (np.random.randint(margin, img_size - margin),
                       np.random.randint(margin, img_size - margin))
            
            # 시작/끝 지점의 slope 체크
            start_slope_rad = s_map[start_pos]
            goal_slope_rad = s_map[goal_pos]
            
            # 등반 불가능한 지점은 제외
            if start_slope_rad > limit_angle_rad or goal_slope_rad > limit_angle_rad:
                continue
            
            distance = np.sqrt((goal_pos[0] - start_pos[0])**2 + 
                             (goal_pos[1] - start_pos[1])**2)
            
            if distance >= min_distance:
                break
        
        if start_pos is None or goal_pos is None:
            print("⚠️  유효한 시작/끝 지점을 찾지 못했습니다. 다음 샘플로...")
            continue
        
        start_slope_deg = np.degrees(s_map[start_pos])
        goal_slope_deg = np.degrees(s_map[goal_pos])
        print(f"✓ Start: ({start_pos[0]}, {start_pos[1]}, slope: {start_slope_deg:.2f}°)")
        print(f"✓ Goal: ({goal_pos[0]}, {goal_pos[1]}, slope: {goal_slope_deg:.2f}°)")
        
        # 3. 다양한 weight로 경로 생성
        paths_by_weight = {}
        for weight in weights:
            path_pixels = generator.find_path(start_pos, goal_pos, weight=weight)
            if path_pixels is not None and len(path_pixels) > 10:
                paths_by_weight[weight] = path_pixels
                print(f"✓ Weight {weight}: Path found ({len(path_pixels)} points)")
            else:
                print(f"✗ Weight {weight}: No path found")
        
        if len(paths_by_weight) == 0:
            print("⚠️  모든 weight에서 경로를 찾지 못했습니다. 다음 샘플로...")
            continue
        
        # 4. 시각화
        # 레이아웃: 3행 3열 (최대 9개 subplot)
        fig = plt.figure(figsize=(24, 18))
        
        # Weight별 색상 매핑
        weight_color_map = {0.1: 'blue', 0.3: 'cyan', 0.5: 'green', 0.7: 'orange', 0.9: 'red'}
        colors = [weight_color_map.get(w, 'gray') for w in weights]
        
        # 4-1. Height Map
        ax1 = plt.subplot(3, 3, 1)
        im1 = ax1.imshow(h_map, cmap='terrain', origin='lower')
        ax1.set_title('Height Map (Elevation)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        plt.colorbar(im1, ax=ax1, label='Height (m)')
        
        # 시작/끝 지점 표시
        ax1.plot(start_pos[1], start_pos[0], 'go', markersize=12, label='Start', zorder=10)
        ax1.plot(goal_pos[1], goal_pos[0], 'ro', markersize=12, label='Goal', zorder=10)
        ax1.legend()
        
        # 4-2. Slope Map
        ax2 = plt.subplot(3, 3, 2)
        im2 = ax2.imshow(slope_degrees, cmap='jet', origin='lower')
        ax2.set_title('Slope Map (Degrees)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        plt.colorbar(im2, ax=ax2, label='Slope (degrees)')
        ax2.plot(start_pos[1], start_pos[0], 'go', markersize=12, zorder=10)
        ax2.plot(goal_pos[1], goal_pos[0], 'ro', markersize=12, zorder=10)
        
        # 4-3. 모든 경로를 하나의 맵에 표시
        ax3 = plt.subplot(3, 3, 3)
        ax3.imshow(slope_degrees, cmap='jet', alpha=0.6, origin='lower')
        ax3.set_title('All Paths Comparison', fontsize=14, fontweight='bold')
        ax3.set_xlabel('X (pixels)')
        ax3.set_ylabel('Y (pixels)')
        
        # 각 weight별 경로 표시
        for weight, path in paths_by_weight.items():
            path_array = np.array(path)
            color = weight_color_map.get(weight, 'gray')
            ax3.plot(path_array[:, 1], path_array[:, 0], 
                    color=color, linewidth=2, 
                    label=f'w={weight}', alpha=0.8)
        
        ax3.plot(start_pos[1], start_pos[0], 'go', markersize=12, zorder=10)
        ax3.plot(goal_pos[1], goal_pos[0], 'ro', markersize=12, zorder=10)
        ax3.legend(loc='upper right', fontsize=9)
        
        # 4-4~4-9. 각 weight별 개별 경로 (경로가 있는 것만 표시)
        plot_idx = 4  # subplot 인덱스 시작 (4부터 시작)
        for weight in weights:
            if weight not in paths_by_weight:
                continue
            
            ax = plt.subplot(3, 3, plot_idx)
            ax.imshow(slope_degrees, cmap='jet', origin='lower')
            ax.set_title(f'Weight = {weight}\n{weight_labels[weight]}', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            
            # 경로 표시
            path_array = np.array(paths_by_weight[weight])
            color = weight_color_map.get(weight, 'gray')
            ax.plot(path_array[:, 1], path_array[:, 0], 
                   color=color, linewidth=3, alpha=0.9)
            
            # 시작/끝 지점
            ax.plot(start_pos[1], start_pos[0], 'go', markersize=10, zorder=10)
            ax.plot(goal_pos[1], goal_pos[0], 'ro', markersize=10, zorder=10)
            
            # 경로 정보
            path_len = len(paths_by_weight[weight])
            ax.text(0.02, 0.98, f'Path length: {path_len} points', 
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='white', alpha=0.8))
            
            plot_idx += 1
        
        plt.suptitle(f'Terrain Map Visualization - Sample {sample_idx + 1}\n'
                    f'Mean Slope: {mean_slope:.2f}°, Max Slope: {max_slope:.2f}°',
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        # 저장
        save_path = f'results/terrain_map_sample_{sample_idx}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        
        plt.close()
    
    print(f"\n{'='*60}")
    print(f"✓ Visualization complete! ({num_samples} samples)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize terrain maps with different weight paths")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to config file')
    parser.add_argument('--num-samples', type=int, default=3,
                       help='Number of terrain samples to visualize')
    
    args = parser.parse_args()
    visualize_terrain_map(config_path=args.config, num_samples=args.num_samples)
