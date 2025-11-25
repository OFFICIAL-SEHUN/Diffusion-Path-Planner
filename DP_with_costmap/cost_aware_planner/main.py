import yaml
import argparse
import torch
import os
import random
import numpy as np
import time

from maze import MazeGenerator, a_star_search
from data_loader import FixedDataset
from model import ConditionalPathModel
from diffusion import DiffusionScheduler
from trainer import Trainer
from utils import plot_results

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    # --- Load Configuration ---
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # --- Set Seed for Reproducibility ---
    set_seed(config['seed'])
    
    # --- Setup Device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Initialize Components ---
    dataset = FixedDataset(config)
    model = ConditionalPathModel(config=config)
    diffusion_scheduler = DiffusionScheduler(
        timesteps=config['diffusion']['timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        device=device
    )

    if args.mode == 'train':
        print("--- Starting Training Mode ---")
        trainer = Trainer(
            model=model,
            dataset=dataset,
            diffusion_scheduler=diffusion_scheduler,
            config=config
        )
        trainer.train()

    elif args.mode == 'infer':
        print("--- Starting Inference Mode ---")
        
        print(f"Config Image Size: {config['data']['img_size']}")
        
        seed = int(time.time())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # --- Load Model Checkpoint ---
        model_path = os.path.join(config['training']['checkpoint_dir'], config['training']['model_name'])
        try:
            # map_location을 사용하여 CPU/GPU 호환성 확보
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            print(f"Model loaded from {model_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}.")
            print("Please run in 'train' mode first to generate a model checkpoint.")
            return
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            return
            
        # --- Generate a New Test Sample ---
        print("Generating a new costmap for inference...")
        maze_gen = MazeGenerator(config['data']['img_size'], config['maze']['scale'])
        costmap, _, start_pos, end_pos = maze_gen.generate()
        
        #np.inf가 있다면 1.0으로 치환
        if isinstance(costmap, np.ndarray):
            costmap[costmap == np.inf] = 1.0
        

        # --- Calculate the Optimal Path using A* ---
        print(f"Calculating A* path from {start_pos} to {end_pos}...")
        true_path_pixels = a_star_search(
            costmap, 
            tuple(start_pos), 
            tuple(end_pos), 
            config['data']['cost_weight']
        )
        
        true_path = None
        if true_path_pixels is not None:
            # Convert path to tensor and normalize for visualization
            true_path_np = np.array(true_path_pixels, dtype=np.float32)
            true_path_norm = (true_path_np / config['data']['img_size']) * 2 - 1
            true_path = torch.from_numpy(true_path_norm).float().to(device).unsqueeze(0)
        else:
            print("A* could not find a path.")


        # 픽셀 좌표(0~64)를 모델 좌표(-1~1)로 변환해야 함
        img_size = config['data']['img_size']
        
        #Tensor 변환 [H, W] -> [1, 1, H, W]
        costmap_tensor = torch.from_numpy(costmap).float().to(device)
        if costmap_tensor.dim() == 2:
            costmap_tensor = costmap_tensor.unsqueeze(0).unsqueeze(0)
        elif costmap_tensor.dim() == 3:
            costmap_tensor = costmap_tensor.unsqueeze(0)
            
        start_pos_xy = start_pos[::-1]  
        end_pos_xy = end_pos[::-1]

        
        norm_start = (start_pos_xy / img_size) * 2 - 1
        norm_end = (end_pos_xy / img_size) * 2 - 1
        
        # Tensor로 변환하고 배치 차원 추가 [1, 2]
        start_tensor = torch.from_numpy(norm_start).float().to(device).unsqueeze(0)
        end_tensor = torch.from_numpy(norm_end).float().to(device).unsqueeze(0)
        
        # --- Run Sampling (In-painting 적용) ---
        print("Running Diffusion Sampling with In-painting...")
        generated_path = diffusion_scheduler.sample(
            model=model,
            condition=costmap_tensor,
            shape=(1, config['data']['horizon'], 2),
            start_pos=start_tensor, 
            end_pos=end_tensor
            # cost_guidance_scale=args.scale # on/off
        )
        
        print("Generated path shape:", generated_path.shape)
        
        # --- Visualize the Result ---
        plot_results(costmap_tensor, generated_path, true_path, config)
        
    elif args.mode == 'check_train':
        print("\n--- Starting Check-Train Mode (Overfitting Diagnosis) ---")
        
        # 1. 시드 설정 & 모델 로드
        seed = int(time.time())
        random.seed(seed)
        torch.manual_seed(seed)
        
        model_path = os.path.join(config['training']['checkpoint_dir'], config['training']['model_name'])
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Model loaded from {model_path}")

        # 2. 데이터셋 로드
        try:
            train_dataset = FixedDataset(config)
        except Exception as e:
            print(f"Error loading FixedDataset: {e}")
            return
            
        # 3. 랜덤 샘플 추출
        idx = random.randint(0, len(train_dataset) - 1)
        print(f"Testing with Training Sample Index: {idx}")
        
        # [핵심 수정] 반환값이 2개(costmap, path)이므로 이에 맞게 받습니다.
        try:
            sample = train_dataset[idx]
            costmap_data, true_path_data = sample
        except ValueError:
            print(f"Error: Dataset returned unexpected number of items. Check data_loader.py.")
            return

        # 4. Start / End Position 추출
        # Path의 첫 점과 끝 점을 가져옵니다.
        # true_path_data는 [Horizon, 2] 형태일 것입니다.
        start_data = true_path_data[0]
        end_data = true_path_data[-1]

        # 5. Tensor 차원 확장 (Batch Dimension 추가)
        
        # Costmap: [H, W] -> [1, 1, H, W] (채널이 이미 있다면 [1, C, H, W])
        costmap_tensor = costmap_data.to(device)
        if costmap_tensor.dim() == 2:
            costmap_tensor = costmap_tensor.unsqueeze(0).unsqueeze(0)
        elif costmap_tensor.dim() == 3:
            costmap_tensor = costmap_tensor.unsqueeze(0)
            
        # Path / Start / End
        true_path_tensor = true_path_data.to(device).unsqueeze(0) # [1, L, 2]
        start_tensor = start_data.to(device).unsqueeze(0)         # [1, 2]
        end_tensor = end_data.to(device).unsqueeze(0)             # [1, 2]

        # 6. 추론 실행
        print(f"Start: {start_tensor[0].cpu().numpy()}")
        print(f"End:   {end_tensor[0].cpu().numpy()}")
        print("Running Diffusion Sampling on Training Data...")
        
        generated_path = diffusion_scheduler.sample(
            model=model,
            condition=costmap_tensor,
            shape=(1, config['data']['horizon'], 2),
            start_pos=start_tensor,
            end_pos=end_tensor
        )
        
        # 7. 결과 시각화
        # Training Data는 이미 정규화되어 있을 것이므로 그대로 그립니다.
        plot_results(costmap_tensor, generated_path, true_path_tensor, config)

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Diffusion-Based Path Planner for Cost-Aware Environments")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer', 'check_train'],
                        help="Mode to run the script in: 'train' or 'infer' or 'check_train'")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help="Path to the configuration file")
    
    args = parser.parse_args()
    main(args)
