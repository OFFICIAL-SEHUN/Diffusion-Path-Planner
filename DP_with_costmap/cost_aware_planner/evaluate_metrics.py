import argparse
import os
import yaml
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

# 프로젝트 모듈
from model import ConditionalPathModel
from guidance_diffusion import DiffusionScheduler

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def check_collision_interpolated(costmap, p1, p2, threshold=0.9, step_size=0.5, margin=2):
    """
    두 점 p1, p2 사이를 촘촘하게(step_size) 검사하여 충돌 여부 확인
    """
    img_size = costmap.shape[0]
    vec = p2 - p1
    dist = np.linalg.norm(vec)
    
    if dist == 0: return [], 0.0
    
    num_steps = int(np.ceil(dist / step_size))
    collision_hits = []
    max_val_segment = 0.0
    
    for i in range(num_steps + 1):
        t = i / num_steps if num_steps > 0 else 0
        p_interp = p1 + vec * t
        
        r_float, c_float = p_interp[0], p_interp[1]
        
        # Margin Check
        if (r_float < margin) or (r_float >= img_size - margin) or \
           (c_float < margin) or (c_float >= img_size - margin):
            continue
            
        r = int(np.clip(r_float, 0, img_size - 1))
        c = int(np.clip(c_float, 0, img_size - 1))
        
        val = costmap[r, c]
        
        if val > max_val_segment:
            max_val_segment = val
            
        if val > threshold:
            collision_hits.append([r_float, c_float])
            
    return collision_hits, max_val_segment

def calculate_path_length(path):
    # path shape: [N, 2]
    diffs = path[1:] - path[:-1]
    distances = np.linalg.norm(diffs, axis=1)
    return np.sum(distances)

def debug_visualize(costmap, gen_pixel, true_pixel, collision_points, idx, scale, is_success):
    plt.figure(figsize=(8, 8))
    
    # 제공해주신 코드와 동일한 컬러맵 설정
    cmap = plt.cm.get_cmap('plasma_r').copy()
    cmap.set_bad(color='black')
    
    # costmap이 텐서일 경우 numpy로 변환
    if torch.is_tensor(costmap):
        costmap = costmap.cpu().numpy()
        
    masked_costmap = np.ma.masked_invalid(costmap)
    plt.imshow(masked_costmap, cmap=cmap, origin='upper', vmin=0, vmax=1.0)
    
    # GT Path (빨간 점선)
    if true_pixel is not None:
        plt.plot(true_pixel[:, 1], true_pixel[:, 0], 'r--', linewidth=2, alpha=0.6, label='GT Path')

    # Generated Path (청록색 실선)
    plt.plot(gen_pixel[:, 1], gen_pixel[:, 0], 'cyan', linewidth=3, alpha=0.9, label='Gen Path')
    
    # Start / Goal 표시
    plt.scatter(gen_pixel[0, 1], gen_pixel[0, 0], c='lime', marker='o', s=80, label='Start', zorder=5)
    plt.scatter(gen_pixel[-1, 1], gen_pixel[-1, 0], c='lime', marker='x', s=80, label='End', zorder=5)

    # 충돌 지점 표시
    if len(collision_points) > 0:
        coll_arr = np.array(collision_points)
        plt.scatter(coll_arr[:, 1], coll_arr[:, 0], c='red', s=60, marker='x', label='Collision', zorder=10)

    status = "SUCCESS" if is_success else "FAIL"
    plt.title(f"Val #{idx} | Scale={scale} | Result={status}")
    plt.legend(loc='lower right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    os.makedirs('debug_val', exist_ok=True)
    plt.savefig(f'debug_val/val_{idx}.png')
    plt.close()

def main(args):
    # --- 1. 설정 및 모델 로드 ---
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = ConditionalPathModel(config=config).to(device)
    scheduler = DiffusionScheduler(
        timesteps=config['diffusion']['timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        device=device
    )
    
    ckpt_path = os.path.join(config['training']['checkpoint_dir'], args.model_name)
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        print(f"Loaded Model: {args.model_name}")
    except Exception as e: 
        print(f"Model load failed: {e}")
        return

    # --- 2. 데이터셋 로드 (수정된 구조) ---
    val_data_path = args.val_path
    if not os.path.exists(val_data_path):
        print(f"Error: Validation dataset not found at {val_data_path}")
        return
        
    print(f"Loading validation dataset from: {val_data_path}")
    # CPU로 먼저 로드
    raw_data = torch.load(val_data_path, map_location='cpu')
    
    # 데이터 구조 확인 (visualize_dataset_sample.py 기반)
    if "costmaps" not in raw_data or "paths" not in raw_data:
        print("Error: Dataset must contain 'costmaps' and 'paths' keys.")
        return
        
    costmaps_all = raw_data["costmaps"] # [N, H, W]
    paths_all = raw_data["paths"]       # [N, Len, 2], Normalized [-1, 1]
    
    num_samples = len(costmaps_all)
    img_size = config['data']['img_size']
    
    print(f"\n--- Validation Start ({num_samples} samples) ---")
    print(f"Scale (Guidance): {args.scale}, Collision Threshold: 0.9")
    
    results = {
        "success_count": 0,
        "success_avg_costs": [],
        "success_max_costs": [],
        "success_length_ratios": []
    }
    
    set_seed(42) # 재현성을 위한 시드 고정
    
    # --- 3. 평가 루프 ---
    for i in tqdm(range(num_samples)):
        # (1) 데이터 준비
        # Costmap: [H, W] -> [1, 1, H, W]
        costmap_raw = costmaps_all[i].numpy() # Numpy for collision check
        costmap_tensor = costmaps_all[i].unsqueeze(0).unsqueeze(0).float().to(device)
        
        # Paths: Normalized [-1, 1] 상태로 존재
        gt_path_norm = paths_all[i] # [Len, 2]
        
        # Start/Goal 추출 (정답 경로의 처음과 끝)
        start_pos_norm = gt_path_norm[0]   # [2]
        end_pos_norm = gt_path_norm[-1]    # [2]
        
        # 모델 입력용 차원 변경: [1, 2]
        start_tensor = start_pos_norm.unsqueeze(0).float().to(device)
        end_tensor = end_pos_norm.unsqueeze(0).float().to(device)

        # (2) 모델 추론
        # Output shape: [1, Horizon, 2] (Normalized -1~1)
        with torch.no_grad():
            generated_path = scheduler.sample(
                model=model,
                condition=costmap_tensor,
                shape=(1, config['data']['horizon'], 2),
                start_pos=start_tensor,
                end_pos=end_tensor,
                cost_guidance_scale=args.scale
            )

        # (3) 좌표 변환 (Normalized -> Pixel)
        gen_np = generated_path.squeeze().cpu().numpy() # [Horizon, 2]
        gt_np = gt_path_norm.numpy()                    # [Len, 2]
        
        # De-normalize: (-1~1) -> (0~img_size)
        gen_pixel = (gen_np + 1) / 2 * img_size
        true_pixel = (gt_np + 1) / 2 * img_size
        
        # (4) 충돌 검사 & 메트릭 계산
        # Costmap 처리 (inf -> 1.0)
        costmap_clean = costmap_raw.copy()
        costmap_clean[np.isinf(costmap_clean)] = 1.0
        
        all_collisions = []
        path_costs = [] 
        max_path_cost = 0.0
        
        for k in range(len(gen_pixel) - 1):
            p1 = gen_pixel[k]
            p2 = gen_pixel[k+1]
            
            # 충돌 체크
            hits, max_val = check_collision_interpolated(costmap_clean, p1, p2, threshold=0.9, step_size=0.5)
            all_collisions.extend(hits)
            
            if max_val > max_path_cost:
                max_path_cost = max_val
                
            r = int(np.clip(p1[0], 0, img_size-1))
            c = int(np.clip(p1[1], 0, img_size-1))
            path_costs.append(costmap_clean[r, c])

        is_success = (len(all_collisions) == 0)
        avg_path_cost = np.mean(path_costs) if len(path_costs) > 0 else 0.0
        
        # 경로 길이 비율 (Efficiency)
        len_gen = calculate_path_length(gen_pixel)
        len_gt = calculate_path_length(true_pixel)
        length_ratio = len_gen / len_gt if len_gt > 0 else 1.0
        
        # 결과 저장
        if is_success:
            results["success_count"] += 1
            results["success_avg_costs"].append(avg_path_cost)
            results["success_max_costs"].append(max_path_cost)
            results["success_length_ratios"].append(length_ratio)
            
        # (5) 디버그 이미지 저장 (앞쪽 20개만)
        if i < 20:
            debug_visualize(costmap_raw, gen_pixel, true_pixel, all_collisions, i, args.scale, is_success)

    # --- 4. 최종 결과 출력 ---
    if num_samples == 0:
        print("No samples found.")
        return

    success_rate = (results["success_count"] / num_samples) * 100
    
    if results["success_count"] > 0:
        final_avg_cost = np.mean(results["success_avg_costs"])
        final_max_cost = np.mean(results["success_max_costs"])
        final_efficiency = np.mean(results["success_length_ratios"])
    else:
        final_avg_cost = 0.0
        final_max_cost = 0.0
        final_efficiency = 0.0

    print("\n" + "="*60)
    print(f"  Validation Result")
    print(f"  Input: {args.val_path} | Scale: {args.scale}")
    print("="*60)
    print(f"1. Success Rate (SR)      : {success_rate:.2f}% ({results['success_count']}/{num_samples})")
    print("-" * 60)
    print(f"2. Mean Path Cost (Succ)  : {final_avg_cost:.4f}")
    print(f"3. Avg Max Cost (Succ)    : {final_max_cost:.4f}")
    print(f"4. Path Efficiency (Succ) : {final_efficiency:.2f}x (Lower is better)")
    print("="*60 + "\n")
    print(f"Debug images saved in 'debug_val/' directory.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')
    parser.add_argument('--model_name', type=str, default='epoch_2000_model.pt')
    parser.add_argument('--val_path', type=str, default='data/dataset_validation.pt')
    parser.add_argument('--scale', type=float, default=2.0) 
    args = parser.parse_args()
    main(args)