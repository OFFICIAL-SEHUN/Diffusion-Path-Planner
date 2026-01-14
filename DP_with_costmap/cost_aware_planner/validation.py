import argparse
import os
import yaml
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from tqdm import tqdm

# 프로젝트 모듈
from maze import MazeGenerator, a_star_search
from model import ConditionalPathModel
from guidance_diffusion import DiffusionScheduler

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_and_infer_one_sample(config, device, model, scheduler, guidance_scale):
    """
    단 하나의 샘플을 생성하고 추론하여 시각화용 데이터를 반환합니다.
    """
    img_size = config['data']['img_size']
    scale = config['maze']['scale']
    maze_gen = MazeGenerator(img_size, scale)

    # 1. 맵 생성 (A* 성공할 때까지)
    while True:
        costmap, _, start_pos, end_pos = maze_gen.generate()
        if isinstance(costmap, np.ndarray): costmap[costmap == np.inf] = 1.0
        
        # A* (GT)
        true_path_pixels = a_star_search(
            costmap, tuple(start_pos), tuple(end_pos), config['data']['cost_weight']
        )
        if true_path_pixels is not None:
            break

    # 2. 전처리 (main.py 로직 준수)
    # Costmap (0~1 범위 그대로 사용)
    costmap_tensor = torch.from_numpy(costmap).float().to(device).unsqueeze(0).unsqueeze(0)
    
    # 좌표 변환 (Row, Col -> X, Y)
    start_xy = start_pos[::-1]
    end_xy = end_pos[::-1]
    
    norm_start = (start_xy / img_size) * 2 - 1
    norm_end = (end_xy / img_size) * 2 - 1
    
    start_tensor = torch.from_numpy(norm_start).float().to(device).unsqueeze(0)
    end_tensor = torch.from_numpy(norm_end).float().to(device).unsqueeze(0)

    # 3. Diffusion Sampling
    generated_path = scheduler.sample(
        model=model,
        condition=costmap_tensor,
        shape=(1, config['data']['horizon'], 2),
        start_pos=start_tensor,
        end_pos=end_tensor,
        cost_guidance_scale=guidance_scale # [수정] 주석 해제 및 적용
    )

    # 4. 시각화용 데이터 변환 (Pixel 좌표계)
    gen_np = generated_path.squeeze().cpu().detach().numpy()
    gen_pixel = (gen_np + 1) / 2 * img_size
    
    true_pixel = np.array(true_path_pixels)
    
    return costmap, true_pixel, gen_pixel

def main(args):
    # --- 설정 로드 ---
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # --- 모델 로드 ---
    model = ConditionalPathModel(config=config).to(device)
    scheduler = DiffusionScheduler(
        timesteps=config['diffusion']['timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        device=device
    )
    
    ckpt_path = os.path.join(config['training']['checkpoint_dir'], args.model_name)
    print(f"Loading model from {ckpt_path}...")
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
    except FileNotFoundError:
        print("Model file not found! Please check the path.")
        return

    # --- 설정: 100개 샘플, 9개씩 저장 ---
    TOTAL_SAMPLES = 100
    SAMPLES_PER_PAGE = 9
    TOTAL_PAGES = math.ceil(TOTAL_SAMPLES / SAMPLES_PER_PAGE)
    
    print(f"Starting Validation: {TOTAL_SAMPLES} samples -> {TOTAL_PAGES} pages.")
    
    # 결과 저장 폴더 생성
    save_dir = f'results_validation_{args.scale}'
    os.makedirs(save_dir, exist_ok=True)

    sample_counter = 0
    
    # 페이지 단위 루프
    for page_idx in range(TOTAL_PAGES):
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle(f'Validation Batch {page_idx + 1}/{TOTAL_PAGES} (Scale: {args.scale})', fontsize=16)
        
        # 페이지 내 9개 그리드 루프
        for i in range(SAMPLES_PER_PAGE):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            # 100개를 다 채웠으면 나머지 칸은 비우고 종료
            if sample_counter >= TOTAL_SAMPLES:
                ax.axis('off')
                continue
                
            # 진행상황 출력
            print(f"Processing Sample {sample_counter + 1}/{TOTAL_SAMPLES}...", end='\r')
            
            # 랜덤 시드로 다양성 확보
            seed = random.randint(0, 100000)
            set_seed(seed)
            
            costmap, true_path, gen_path = generate_and_infer_one_sample(
                config, device, model, scheduler, args.scale
            )
            
            # --- Plotting ---
            # 1. Costmap
            cmap = cm.get_cmap('plasma_r').copy()
            cmap.set_bad(color='black')
            ax.imshow(costmap, cmap=cmap, origin='upper', vmin=0, vmax=1.0)
            
            # 2. GT Path (Red Dashed) 
            # A* Output: [Row, Col] -> Plot: [x, y] = [Col, Row]
            ax.plot(true_path[:, 1], true_path[:, 0], 'r--', linewidth=1.5, alpha=0.6, label='GT')
            
            # 3. Gen Path (Cyan)
            ax.plot(gen_path[:, 1], gen_path[:, 0], 'k', linewidth=2.5, alpha=0.9, label='Gen')
            
            # 4. Start/End Markers
            s, e = gen_path[0], gen_path[-1]
            ax.scatter(s[0], s[1], c='purple', s=50, marker='o', zorder=5)
            ax.scatter(e[0], e[1], c='purple', s=50, marker='x', zorder=5)
            
            ax.set_title(f"Sample {sample_counter + 1}")
            ax.axis('off')
            
            sample_counter += 1

        # 범례 표시 (첫 번째 그래프)
        axes[0, 0].legend(loc='upper right', fontsize='small')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # 파일 저장
        filename = f'validation_page_{page_idx + 1:02d}.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        plt.close() # 메모리 해제
        
        print(f"\nSaved {filename}")

    print(f"\nAll validation images saved to {save_dir}/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')
    parser.add_argument('--model_name', type=str, default='epoch_2000_model.pt')
    parser.add_argument('--scale', type=float, default=0.0)
    args = parser.parse_args()
    main(args)