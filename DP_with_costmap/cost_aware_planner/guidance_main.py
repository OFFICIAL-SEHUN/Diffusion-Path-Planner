import argparse
import os
import yaml
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

# 프로젝트 모듈
from maze import MazeGenerator, a_star_search
from model import ConditionalPathModel
from guidance_diffsion import DiffusionScheduler

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_new_sample(config, device):
    """
    main.py의 infer 모드와 100% 동일한 로직으로 데이터를 생성합니다.
    """
    img_size = config['data']['img_size']
    scale = config['maze']['scale']
    maze_gen = MazeGenerator(img_size, scale)
    
    # 1. 맵 생성 Loop
    while True:
        costmap, _, start_pos, end_pos = maze_gen.generate()
        
        # [main.py 동일] INF 제거
        if isinstance(costmap, np.ndarray):
            costmap[costmap == np.inf] = 1.0

        # [main.py 동일] A* (True Path) 계산
        # costmap: 0~1 범위, start_pos: (Row, Col)
        true_path_pixels = a_star_search(
            costmap, 
            tuple(start_pos), 
            tuple(end_pos), 
            config['data']['cost_weight']
        )
        if true_path_pixels is not None:
            break

    # 2. 데이터 전처리
    
    # (A) 시각화용 데이터 (Numpy)
    costmap_vis = costmap.copy()
    true_path_vis = np.array(true_path_pixels) if true_path_pixels else None

    # (B) 모델 입력용 Costmap Tensor 
    # [main.py 동일] -1~1 정규화 없이 0~1 범위를 그대로 Tensor로 변환
    costmap_tensor = torch.from_numpy(costmap).float().to(device)
    if costmap_tensor.dim() == 2:
        costmap_tensor = costmap_tensor.unsqueeze(0).unsqueeze(0)
    elif costmap_tensor.dim() == 3:
        costmap_tensor = costmap_tensor.unsqueeze(0)
    
    # (C) Start/End 좌표 전처리
    # [main.py 동일] 좌표 반전 수행 (Row, Col -> X, Y)
    start_pos_xy = start_pos[::-1]
    end_pos_xy = end_pos[::-1]
    
    norm_start = (start_pos_xy / img_size) * 2 - 1
    norm_end = (end_pos_xy / img_size) * 2 - 1
    
    start_tensor = torch.from_numpy(norm_start).float().to(device).unsqueeze(0)
    end_tensor = torch.from_numpy(norm_end).float().to(device).unsqueeze(0)
    
    return costmap_tensor, start_tensor, end_tensor, costmap_vis, true_path_vis

def plot_comparison(costmap_vis, path_vanilla, path_guided, true_path_vis, config, scale):
    """
    utils.py의 plot_results 스타일(Kalman 제외)을 유지하면서 비교 결과를 시각화합니다.
    """
    img_size = config['data']['img_size']
    
    # Tensor -> Pixel 변환 (utils.py 로직)
    def to_pixel(path_tensor):
        if path_tensor is None: return None
        p = path_tensor.squeeze().cpu().detach().numpy()
        # (-1~1) -> (0~64)
        return (p + 1) / 2 * img_size

    vanilla_np = to_pixel(path_vanilla)
    guided_np = to_pixel(path_guided)
    
    plt.figure(figsize=(10, 10))
    
    # 1. Costmap Plotting
    cmap = cm.get_cmap('plasma_r').copy()
    cmap.set_bad(color='black')
    plt.imshow(costmap_vis, cmap=cmap, origin='upper', vmin=0, vmax=1.0)
    
    # 2. Paths Plotting
    # utils.py에서는 plt.plot(path[:, 1], path[:, 0]) 순서로 그립니다.
    
    # GT Path (A*)
    if true_path_vis is not None:
        # A*는 [Row, Col] -> Plot [Col, Row] ([:, 1], [:, 0])
        plt.plot(true_path_vis[:, 1], true_path_vis[:, 0], 'r--', linewidth=2, alpha=0.6, label='GT (A*)')

    # Vanilla Path
    if vanilla_np is not None:
        # utils.py 스타일: plt.plot(gen_path[:, 1], gen_path[:, 0])
        plt.plot(vanilla_np[:, 1], vanilla_np[:, 0], color='gray', linewidth=3, alpha=0.6, label='Vanilla (No Guidance)')
    
    # Guided Path
    if guided_np is not None:
        plt.plot(guided_np[:, 1], guided_np[:, 0], color='cyan', linewidth=4, alpha=0.9, label=f'Guided (Scale={scale})')
        
        # Start/End Markers (Guided 기준)
        # utils.py 기준: scatter(path[:, 1], path[:, 0])
        start = guided_np[0]
        end = guided_np[-1]
        plt.scatter(start[1], start[0], c='purple', s=150, marker='o', label='Start', zorder=10)
        plt.scatter(end[1], end[0], c='purple', s=150, marker='x', label='End', zorder=10)

    plt.legend(loc='upper right')
    plt.title(f"Comparison: Vanilla vs Guided (Scale: {scale})")
    
    # utils.py 스타일 설정
    plt.xlim(0, img_size)
    plt.ylim(img_size, 0) # 상단이 0 (이미지 좌표계)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # 저장
    os.makedirs('results_guidance', exist_ok=True)
    # timestamp = int(time.time())
    save_path = f"results_guidance/guidance_{scale}.png"
    plt.savefig(save_path)
    print(f"Comparison result saved to: {save_path}")
    plt.close()

def main(args):
    # 1. 설정 및 모델 로드
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running Guidance Check (Scale={args.scale}) ---")

    model = ConditionalPathModel(config=config).to(device)
    diffusion_scheduler = DiffusionScheduler(
        timesteps=config['diffusion']['timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        device=device
    )

    model_path = os.path.join(config['training']['checkpoint_dir'], config['training']['model_name'])
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model loaded: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. 새로운 맵 생성 (Online)
    print("Generating new sample...")
    costmap_tensor, start_tensor, end_tensor, costmap_vis, true_path_vis = generate_new_sample(config, device)
    
    # 3. 비교 추론 (동일 시드)
    # 동일한 노이즈에서 출발해야 공정한 비교가 됩니다.
    compare_seed = random.randint(0, 10000)
    print(f"Comparison Seed: {compare_seed}")

    # Case A: Vanilla (No Guidance)
    print("Running Vanilla...")
    set_seed(compare_seed) # 시드 리셋
    path_vanilla = diffusion_scheduler.sample(
        model=model,
        condition=costmap_tensor,
        shape=(1, config['data']['horizon'], 2),
        start_pos=start_tensor,
        end_pos=end_tensor,
        cost_guidance_scale=0.0
    )

    # Case B: Guided
    print(f"Running Guided (Scale={args.scale})...")
    set_seed(compare_seed) # 시드 리셋
    path_guided = diffusion_scheduler.sample(
        model=model,
        condition=costmap_tensor,
        shape=(1, config['data']['horizon'], 2),
        start_pos=start_tensor,
        end_pos=end_tensor,
        cost_guidance_scale=args.scale
    )

    # 4. 결과 저장
    plot_comparison(costmap_vis, path_vanilla, path_guided, true_path_vis, config, args.scale)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')
    parser.add_argument('--scale', type=float, default=0.01)
    args = parser.parse_args()
    main(args)