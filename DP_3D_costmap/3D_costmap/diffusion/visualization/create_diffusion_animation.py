"""
Diffusion Inference Step-by-Step 시각화 생성 스크립트

이 스크립트는 학습된 모델로 inference를 실행하고:
1. diffusion_steps.png - Step-by-step 그리드 이미지 (12 steps)
2. diffusion_frames/*.png - 개별 프레임들
3. diffusion_animation.gif - GIF 애니메이션

을 생성합니다.

사용법:
    python visualization/create_diffusion_animation.py
    
    또는 설정 파일 지정:
    python visualization/create_diffusion_animation.py --config configs/default_config.yaml
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from generate_data import SlopeCotGenerator
from model import ConditionalPathModel
from diffusion import DiffusionScheduler


def load_config(config_path):
    """설정 파일 로드"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_test_terrain(config):
    """테스트용 terrain 생성"""
    img_size = config['data']['img_size']
    
    # Terrain generator 초기화
    terrain_generator = SlopeCotGenerator(
        img_size=img_size,
        height_range=tuple(config['gradient']['height_range']),
        mass=config['gradient']['mass'],
        gravity=config['gradient']['gravity'],
        limit_angle_deg=config['gradient']['limit_angle_deg'],
        pixel_resolution=config['gradient'].get('pixel_resolution', 0.5)
    )
    
    # Terrain 생성 (난이도 필터링)
    config_terrain_scales = config['gradient'].get('terrain_scales', None)
    max_attempts = 50
    
    print("Generating terrain...")
    for attempt in range(max_attempts):
        h_map, s_map = terrain_generator.generate(terrain_scales=config_terrain_scales)
        
        slope_degrees = np.degrees(s_map)
        mean_slope = np.mean(slope_degrees)
        max_slope = np.max(slope_degrees)
        steep_ratio = np.sum(slope_degrees > 30.0) / slope_degrees.size
        
        if (3.0 <= mean_slope <= 25.0 and 
            max_slope <= 35.0 and 
            steep_ratio <= 0.3):
            print(f"✓ Terrain generated (attempt {attempt + 1})")
            print(f"  Mean slope: {mean_slope:.2f}°")
            print(f"  Max slope: {max_slope:.2f}°")
            break
    
    # Random start/goal 선택
    margin = img_size // 10
    min_distance = img_size // 2
    max_slope_for_start_goal = 30.0
    
    print("Finding valid start/goal positions...")
    for _ in range(50):
        for _ in range(100):
            start_pos = (np.random.randint(margin, img_size - margin),
                        np.random.randint(margin, img_size - margin))
            goal_pos = (np.random.randint(margin, img_size - margin),
                       np.random.randint(margin, img_size - margin))
            
            start_slope = slope_degrees[start_pos]
            goal_slope = slope_degrees[goal_pos]
            
            if start_slope > max_slope_for_start_goal or goal_slope > max_slope_for_start_goal:
                continue
            
            distance = np.sqrt((start_pos[0] - goal_pos[0])**2 + 
                             (start_pos[1] - goal_pos[1])**2)
            
            if distance >= min_distance:
                break
        
        path_pixels = terrain_generator.find_path(start_pos, goal_pos)
        if path_pixels is not None and len(path_pixels) > 0:
            print(f"✓ Valid path found")
            break
    
    return h_map, s_map, start_pos, goal_pos


def run_inference_with_intermediates(model, diffusion_scheduler, costmap_tensor, 
                                     start_tensor, goal_tensor, config):
    """중간 step 저장하며 inference 실행"""
    print("\nRunning diffusion sampling...")
    generated_path, intermediates = diffusion_scheduler.sample(
        model=model,
        condition=costmap_tensor,
        shape=(1, config['data']['horizon'], 2),
        start_pos=start_tensor,
        end_pos=goal_tensor,
        show_progress=True,
        save_intermediates=True
    )
    
    print(f"✓ Generated path with {len(intermediates)} intermediate steps")
    return generated_path, intermediates


def create_step_by_step_grid(intermediates, slope_map, config, max_steps=12):
    """Step-by-step 그리드 이미지 생성"""
    img_size = config['data']['img_size']
    
    # 표시할 step 선택
    if len(intermediates) > max_steps:
        indices = np.linspace(0, len(intermediates) - 1, max_steps, dtype=int)
        steps_to_show = [intermediates[i] for i in indices]
    else:
        steps_to_show = intermediates
    
    # Slope map 준비
    slope_degrees = np.degrees(slope_map)
    cmap = cm.get_cmap('jet').copy()
    cmap.set_bad(color='black')
    masked_map = np.ma.masked_invalid(slope_degrees)
    
    # Grid layout
    n_steps = len(steps_to_show)
    n_cols = 4
    n_rows = int(np.ceil(n_steps / n_cols))
    
    fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)
    
    for idx, step_data in enumerate(steps_to_show):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        # Plot terrain
        ax.imshow(masked_map, cmap=cmap, origin='lower', vmin=0, vmax=35)
        
        # Get path
        path_norm = step_data['path'].squeeze().numpy()
        path_scaled = (path_norm + 1) / 2 * img_size
        
        # Plot path
        ax.plot(path_scaled[:, 0], path_scaled[:, 1], 
               'c-', linewidth=2, alpha=0.8)
        
        # Mark start/end
        ax.scatter(path_scaled[0, 0], path_scaled[0, 1],
                  c='cyan', marker='o', s=80, edgecolors='black',
                  linewidths=1.5, zorder=10)
        ax.scatter(path_scaled[-1, 0], path_scaled[-1, 1],
                  c='yellow', marker='*', s=120, edgecolors='black',
                  linewidths=1.5, zorder=10)
        
        # Title
        timestep = step_data['timestep']
        noise_level = step_data['noise_level']
        ax.set_title(f"t={timestep} (noise={noise_level:.3f})", 
                    fontsize=10, fontweight='bold')
        
        ax.set_xlim(0, img_size)
        ax.set_ylim(0, img_size)
        ax.axis('off')
    
    # Overall title
    fig.suptitle("Diffusion Path Generation Process (Step-by-Step)", 
                fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, 'diffusion_steps.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def create_individual_frames(intermediates, slope_map, config):
    """개별 프레임 생성"""
    img_size = config['data']['img_size']
    results_dir = 'results/diffusion_frames'
    os.makedirs(results_dir, exist_ok=True)
    
    # Slope map 준비
    slope_degrees = np.degrees(slope_map)
    cmap = cm.get_cmap('jet').copy()
    cmap.set_bad(color='black')
    masked_map = np.ma.masked_invalid(slope_degrees)
    
    print(f"Creating {len(intermediates)} frames...")
    
    for idx, step_data in enumerate(intermediates):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.imshow(masked_map, cmap=cmap, origin='lower', vmin=0, vmax=35)
        
        path_norm = step_data['path'].squeeze().numpy()
        path_scaled = (path_norm + 1) / 2 * img_size
        
        ax.plot(path_scaled[:, 0], path_scaled[:, 1],
               'c-', linewidth=3, alpha=0.9)
        
        ax.scatter(path_scaled[0, 0], path_scaled[0, 1],
                  c='cyan', marker='o', s=150, edgecolors='black',
                  linewidths=2, zorder=10)
        ax.scatter(path_scaled[-1, 0], path_scaled[-1, 1],
                  c='yellow', marker='*', s=200, edgecolors='black',
                  linewidths=2, zorder=10)
        
        timestep = step_data['timestep']
        noise_level = step_data['noise_level']
        ax.set_title(f"Diffusion Step: t={timestep} (noise={noise_level:.4f})",
                    fontsize=14, fontweight='bold')
        
        ax.set_xlim(0, img_size)
        ax.set_ylim(0, img_size)
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        frame_path = os.path.join(results_dir, f'frame_{idx:04d}_t{timestep:04d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Saved {len(intermediates)} frames to: {results_dir}")


def create_gif(frames_dir='results/diffusion_frames', output_path='results/diffusion_animation.gif'):
    """GIF 애니메이션 생성"""
    try:
        from PIL import Image
        import glob
        
        frame_files = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.png')))
        
        if not frame_files:
            print(f"⚠️  No frames found in {frames_dir}")
            return
        
        print(f"Creating GIF from {len(frame_files)} frames...")
        
        images = [Image.open(f) for f in frame_files]
        
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=100,
            loop=0
        )
        
        file_size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"✓ Saved: {output_path} ({file_size_mb:.2f} MB)")
        
    except ImportError:
        print("⚠️  PIL not installed. Install: pip install Pillow")
        print(f"   Or use: convert -delay 10 -loop 0 {frames_dir}/frame_*.png {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create diffusion step-by-step visualizations")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to config file')
    parser.add_argument('--skip-frames', action='store_true',
                       help='Skip individual frames')
    parser.add_argument('--skip-gif', action='store_true',
                       help='Skip GIF animation')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Diffusion Step-by-Step Visualization")
    print(f"{'='*60}\n")
    
    # Load config
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print("Loading model...")
    model = ConditionalPathModel(config=config)
    model_path = os.path.join(config['training']['checkpoint_dir'], 
                             config['training']['model_name'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✓ Model loaded: {model_path}\n")
    
    # Initialize diffusion scheduler
    diffusion_scheduler = DiffusionScheduler(
        timesteps=config['diffusion']['timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        device=device
    )
    
    # Generate terrain
    h_map, s_map, start_pos, goal_pos = generate_test_terrain(config)
    
    # Prepare inputs
    img_size = config['data']['img_size']
    slope_degrees = np.degrees(s_map)
    slope_norm = slope_degrees / 90.0
    height_norm = (h_map - h_map.min()) / (h_map.max() - h_map.min() + 1e-8)
    costmap_norm = np.stack([slope_norm, height_norm], axis=0)
    
    costmap_tensor = torch.from_numpy(costmap_norm).float().to(device).unsqueeze(0)
    
    start_pos_xy = np.array([start_pos[1], start_pos[0]], dtype=np.float32)
    goal_pos_xy = np.array([goal_pos[1], goal_pos[0]], dtype=np.float32)
    norm_start = (start_pos_xy / img_size) * 2 - 1
    norm_goal = (goal_pos_xy / img_size) * 2 - 1
    
    start_tensor = torch.from_numpy(norm_start).float().to(device).unsqueeze(0)
    goal_tensor = torch.from_numpy(norm_goal).float().to(device).unsqueeze(0)
    
    # Run inference
    generated_path, intermediates = run_inference_with_intermediates(
        model, diffusion_scheduler, costmap_tensor, start_tensor, goal_tensor, config
    )
    
    # Create visualizations
    print(f"\n{'='*60}")
    print("Creating visualizations...")
    print(f"{'='*60}\n")
    
    print("1. Creating step-by-step grid...")
    create_step_by_step_grid(intermediates, s_map, config)
    
    if not args.skip_frames:
        print("\n2. Creating individual frames...")
        create_individual_frames(intermediates, s_map, config)
    
    if not args.skip_gif and not args.skip_frames:
        print("\n3. Creating GIF animation...")
        create_gif()
    
    print(f"\n{'='*60}")
    print("✓ All visualizations complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
