"""
Diffusion 생성 경로 vs A* GT 경로의 CoT 비교 스크립트

이 스크립트는:
1. Diffusion 모델로 경로 생성
2. A* CoT-efficient 경로 생성
3. 각 경로의 실제 CoT 비용 계산 및 비교
4. 경사각, 오르막/내리막 분포 분석

사용법:
    python compare_path_cost.py
"""

import os
import sys
import argparse
import torch
import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from generate_data import SlopeCotGenerator, a_star_cot_search
from model import ConditionalPathModel
from diffusion import DiffusionScheduler

# Import local modules
from config_loader import load_config
from path_utils import denormalize_path
from cost_calculator import calculate_path_cot
from patch_inference import patch_based_inference
from visualization import visualize_comparison



def main():
    parser = argparse.ArgumentParser(description="Compare Diffusion vs A* path costs")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to config file')  
    parser.add_argument('--use-patches', action='store_true',
                       help='Enable patch-based inference (4x4 patches)')
    parser.add_argument('--num-patches', type=int, default=4,
                       help='Number of patches per dimension (default: 4 for 4x4 grid)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Number of patches to process in parallel (default: 16, use >=16 for single-batch processing)')
    parser.add_argument('--min-distance-factor', type=float, default=0.7,
                       help='Minimum distance factor: min_distance = img_size / factor (default: 0.7 → img_size/0.7)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Path Cost Comparison: Diffusion vs A*")
    print(f"{'='*60}\n")
    
    # Load config
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = config['data']['img_size']
    pixel_resolution = config['gradient'].get('pixel_resolution', 0.5)
    
    # Load model
    print("Loading model...")
    model = ConditionalPathModel(config=config)
    model_path = os.path.join(config['training']['checkpoint_dir'], 
                             config['training']['model_name'])
    
    # Load checkpoint with strict=False to handle missing keys (text_encoder, cross_attn)
    # This allows loading old checkpoints that don't have text conditioning
    checkpoint = torch.load(model_path, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    
    if missing_keys:
        print(f"⚠️  Missing keys (will use random initialization): {len(missing_keys)} keys")
        # Filter out only text-related missing keys
        text_missing = [k for k in missing_keys if 'text_encoder' in k or 'cross_attn' in k]
        if text_missing:
            print(f"   → Text encoder/cross-attention not in checkpoint (expected for old models)")
            print(f"   → These will be randomly initialized")
    
    if unexpected_keys:
        print(f"⚠️  Unexpected keys (ignored): {len(unexpected_keys)} keys")
    
    model.to(device)
    model.eval()
    print(f"✓ Model loaded\n")
    
    # Initialize diffusion scheduler
    diffusion_scheduler = DiffusionScheduler(
        timesteps=config['diffusion']['timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        device=device
    )
    
    # Generate terrain
    print("Generating terrain...")
    terrain_generator = SlopeCotGenerator(
        img_size=img_size,
        height_range=tuple(config['gradient']['height_range']),
        mass=config['gradient']['mass'],
        gravity=config['gradient']['gravity'],
        limit_angle_deg=config['gradient']['limit_angle_deg'],
        pixel_resolution=pixel_resolution
    )
    
    config_terrain_scales = config['gradient'].get('terrain_scales', None)
    
    max_terrain_attempts = 50
    terrain_attempt = 0
    h_map, s_map = None, None
    slope_degrees = None
    mean_slope = None
    astar_path_pixels = None
    astar_time = 0
    start_pos = None
    goal_pos = None
    
    # 공통 파라미터: while 루프 밖에서 정의하여 모든 곳에서 사용 가능
    min_distance_factor = 1.0  # min_distance = img_size / factor
    margin = img_size // 10
    min_distance = int(img_size // min_distance_factor)
    
    # 전체 루프: 지형 생성 → A* 경로 찾기 → 통과하면 break
    while terrain_attempt < max_terrain_attempts:
        terrain_attempt += 1
        
        # --- 1. 지형 생성 (기본 조건만 체크) ---
        print(f"\n{'='*60}")
        print(f"Terrain Generation Attempt {terrain_attempt}/{max_terrain_attempts}")
        print(f"{'='*60}")
        print("Generating terrain...")
        
        for terrain_gen_attempt in range(50):
            h_map, s_map = terrain_generator.generate(
                terrain_scales=config_terrain_scales
            )
            
            slope_degrees = np.degrees(s_map)
            mean_slope = np.mean(slope_degrees)
            max_slope = np.max(slope_degrees)
            steep_ratio = np.sum(slope_degrees > 30.0) / slope_degrees.size
            
            if (3.0 <= mean_slope <= 25.0 and max_slope <= 35.0 and steep_ratio <= 0.3):
                break
        
        print(f"✓ Terrain generated (mean_slope: {mean_slope:.2f}°)\n")
        
        # --- 2. A* 경로 찾기 ---
        print("Searching for valid start/goal positions with A* path...")
        
        print(f"  Margin: {margin} pixels")
        print(f"  Min distance: {min_distance} pixels ({min_distance * pixel_resolution:.1f}m)")
        print(f"  Distance factor: {min_distance_factor} (img_size/{min_distance_factor})")
        
        max_path_attempts = 500
        path_found = False
        
        for path_attempt in range(max_path_attempts):
            # Randomly pick start and goal positions (no slope constraint)
            for position_attempt in range(100):
                start_pos = (np.random.randint(margin, img_size - margin),
                            np.random.randint(margin, img_size - margin))
                goal_pos = (np.random.randint(margin, img_size - margin),
                           np.random.randint(margin, img_size - margin))
                
                # Calculate Euclidean distance
                distance = np.sqrt((start_pos[0] - goal_pos[0])**2 + 
                                 (start_pos[1] - goal_pos[1])**2)
                
                if distance >= min_distance:
                    break
            else:
                # No valid position found in 100 attempts, continue to next path attempt
                continue
            
            # Try A* path finding
            start_slope = slope_degrees[start_pos]
            goal_slope = slope_degrees[goal_pos]
            
            print(f"  Attempt {path_attempt + 1}/{max_path_attempts}: "
                  f"Start({start_pos[0]}, {start_pos[1]}, {start_slope:.1f}°) → "
                  f"Goal({goal_pos[0]}, {goal_pos[1]}, {goal_slope:.1f}°), "
                  f"dist={distance:.1f}px", end=' ')
            
            astar_start_time = time.perf_counter()
            astar_path_pixels = terrain_generator.find_path(start_pos, goal_pos)
            astar_time = time.perf_counter() - astar_start_time
            
            if astar_path_pixels is not None and len(astar_path_pixels) > 10:
                # 경로 찾기 성공!
                print(f"✓ SUCCESS!")
                print(f"\n✓ A* path found!")
                print(f"  Start position: ({start_pos[0]}, {start_pos[1]}) - slope: {start_slope:.2f}°")
                print(f"  Goal position:  ({goal_pos[0]}, {goal_pos[1]}) - slope: {goal_slope:.2f}°")
                print(f"  Euclidean distance: {distance:.1f} pixels ({distance * pixel_resolution:.1f}m)")
                print(f"  Path length: {len(astar_path_pixels)} points")
                print(f"  Inference time: {astar_time*1000:.2f} ms\n")
                path_found = True
                break
            else:
                print(f"✗ No path")
        
        # 경로를 찾았으면 전체 루프 종료
        if path_found:
            break
        
        # 경로를 못 찾았으면 지형 재생성
        if not path_found:
            print(f"\n⚠️  Failed to find valid A* path")
            print(f"    Regenerating terrain...\n")
    
    # 최종 결과 처리
    if terrain_attempt >= max_terrain_attempts:
        print(f"⚠️  Reached max terrain attempts ({max_terrain_attempts})")
        if astar_path_pixels is None or len(astar_path_pixels) == 0:
            print("    Continuing with Diffusion only...\n")
            astar_path_pixels = None
            astar_time = 0
            # start/goal이 없으면 랜덤으로 선택 
            if start_pos is None or goal_pos is None:
                for _ in range(100):
                    start_pos = (np.random.randint(margin, img_size - margin),
                                np.random.randint(margin, img_size - margin))
                    goal_pos = (np.random.randint(margin, img_size - margin),
                               np.random.randint(margin, img_size - margin))
                    distance = np.sqrt((start_pos[0] - goal_pos[0])**2 + 
                                   (start_pos[1] - goal_pos[1])**2)
                    if distance >= min_distance:
                        break
    
    # Prepare inputs for Diffusion
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
    
    # Run Diffusion
    print("Running diffusion sampling...")
    diffusion_start_time = time.perf_counter()
    with torch.no_grad():
        generated_path = diffusion_scheduler.sample(
            model=model,
            condition=costmap_tensor,
            shape=(1, config['data']['horizon'], 2),
            start_pos=start_tensor,
            end_pos=goal_tensor,
            show_progress=True
        )
    diffusion_time = time.perf_counter() - diffusion_start_time
    print(f"✓ Diffusion inference time: {diffusion_time*1000:.2f} ms\n")
    
    # Denormalize paths
    diffusion_path_norm = generated_path.squeeze().cpu().numpy()
    diffusion_path_pixels = denormalize_path(diffusion_path_norm, img_size)
    
    # Patch-based inference (optional)
    patch_path_pixels = None
    patch_time = None
    patch_result = None
    
    if args.use_patches:
        print(f"\n{'='*60}")
        print("Running patch-based diffusion sampling...")
        print(f"{'='*60}\n")
        
        patch_start_time = time.perf_counter()
        patch_path_pixels = patch_based_inference(
            model=model,
            diffusion_scheduler=diffusion_scheduler,
            costmap_tensor=costmap_tensor,
            start_pos=start_pos,
            goal_pos=goal_pos,
            img_size=img_size,
            horizon=config['data']['horizon'],
            num_patches=args.num_patches,
            batch_size=args.batch_size,
            device=device,
            show_progress=True
        )
        patch_time = time.perf_counter() - patch_start_time
        print(f"✓ Patch-based inference time: {patch_time*1000:.2f} ms\n")
    
    # Calculate CoT costs
    print(f"\n{'='*60}")
    print("Calculating CoT costs...")
    print(f"{'='*60}\n")
    
    limit_angle = config['gradient']['limit_angle_deg']
    
    # A* 결과 (실패 가능)
    astar_result = None
    if astar_path_pixels is not None and len(astar_path_pixels) > 0:
        print("A* Path CoT...")
        astar_result = calculate_path_cot(astar_path_pixels, h_map, pixel_resolution, limit_angle)
    else:
        print("⚠️  A* failed to find a path (terrain too complex)")
        print("    → Showing Diffusion and Patch results only\n")
        # 더미 A* 결과 생성 (시각화용)
        astar_result = {
            'total_cot': 0,
            'avg_cot': 0,
            'segments': [],
            'stats': {
                'total_distance': 0,
                'uphill_ratio': 0,
                'downhill_ratio': 0,
                'flat_ratio': 0,
                'avg_slope': 0,
                'max_slope': 0,
                'min_slope': 0,
                'uphill_cot': 0,
                'downhill_cot': 0,
            }
        }
        astar_path_pixels = []  # 빈 경로
    
    print("Diffusion Path CoT...")
    diffusion_result = calculate_path_cot(diffusion_path_pixels, h_map, pixel_resolution, limit_angle)
    
    if patch_path_pixels:
        print("Patch Diffusion Path CoT...")
        patch_result = calculate_path_cot(patch_path_pixels, h_map, pixel_resolution, limit_angle)
    
    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    
    has_astar = astar_result['total_cot'] > 0
    
    print(f"\nInference Time:")
    if has_astar:
        print(f"  A* GT:      {astar_time*1000:.2f} ms")
    else:
        print(f"  A* GT:      FAILED (no path found)")
    print(f"  Diffusion:  {diffusion_time*1000:.2f} ms")
    if patch_time:
        print(f"  Patch:      {patch_time*1000:.2f} ms")
    
    if has_astar:
        speedup = diffusion_time / astar_time if astar_time > 0 else 0
        if speedup > 1:
            print(f"  Speed:      A* is {speedup:.2f}x faster than Diffusion")
        else:
            print(f"  Speed:      Diffusion is {1/speedup:.2f}x faster than A*")
        
        if patch_time:
            patch_speedup = patch_time / astar_time if astar_time > 0 else 0
            patch_vs_full = patch_time / diffusion_time if diffusion_time > 0 else 0
            print(f"              A* is {patch_speedup:.2f}x faster than Patch")
            if patch_vs_full > 1:
                print(f"              Patch is {patch_vs_full:.2f}x slower than Full Diffusion")
            else:
                print(f"              Patch is {1/patch_vs_full:.2f}x faster than Full Diffusion")
    elif patch_time:
        patch_vs_full = patch_time / diffusion_time if diffusion_time > 0 else 0
        if patch_vs_full > 1:
            print(f"  Speed:      Patch is {patch_vs_full:.2f}x slower than Full Diffusion")
        else:
            print(f"  Speed:      Patch is {1/patch_vs_full:.2f}x faster than Full Diffusion")
    
    print(f"\nTotal CoT Cost:")
    if has_astar:
        print(f"  A* GT:      {astar_result['total_cot']:.2f}")
        print(f"  Diffusion:  {diffusion_result['total_cot']:.2f}")
        print(f"  Ratio:      {diffusion_result['total_cot'] / astar_result['total_cot']:.2f}x")
        
        if patch_result:
            print(f"  Patch:      {patch_result['total_cot']:.2f}")
            print(f"  Ratio:      {patch_result['total_cot'] / astar_result['total_cot']:.2f}x")
        
        if diffusion_result['total_cot'] < astar_result['total_cot']:
            print(f"  🎉 Diffusion found a BETTER path!")
        elif diffusion_result['total_cot'] < astar_result['total_cot'] * 1.1:
            print(f"  ✅ Diffusion found a comparable path (within 10%)")
        else:
            print(f"  ⚠️  Diffusion path is less efficient")
        
        if patch_result:
            if patch_result['total_cot'] < astar_result['total_cot']:
                print(f"  🎉 Patch Diffusion found a BETTER path!")
            elif patch_result['total_cot'] < astar_result['total_cot'] * 1.1:
                print(f"  ✅ Patch Diffusion found a comparable path (within 10%)")
            else:
                print(f"  ⚠️  Patch Diffusion path is less efficient")
    else:
        print(f"  A* GT:      FAILED")
        print(f"  Diffusion:  {diffusion_result['total_cot']:.2f}")
        if patch_result:
            print(f"  Patch:      {patch_result['total_cot']:.2f}")
            if patch_result['total_cot'] < diffusion_result['total_cot']:
                print(f"  ✅ Patch is better than Full Diffusion")
            else:
                print(f"  ⚠️  Full Diffusion is better than Patch")
    
    # Visualize
    print(f"\n{'='*60}")
    print("Creating visualization...")
    print(f"{'='*60}\n")
    
    visualize_comparison(diffusion_result, astar_result, s_map, h_map,
                        diffusion_path_pixels, astar_path_pixels, img_size,
                        astar_time=astar_time, diffusion_time=diffusion_time,
                        patch_result=patch_result, patch_path_pixels=patch_path_pixels,
                        patch_time=patch_time)
    
    print(f"{'='*60}")
    print("✓ Comparison complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
