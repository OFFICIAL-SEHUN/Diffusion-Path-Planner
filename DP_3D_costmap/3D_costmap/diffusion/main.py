import yaml
import argparse
import torch
import os
import random
import numpy as np
import time

from generate_data import SlopeCotGenerator, normalize_costmap
from data_loader import GradientDataset
from model import ConditionalPathModel
from diffusion import DiffusionScheduler
from trainer import Trainer
from utils import plot_results

"""
Diffusion-based Path Planning with Slope + CoT
Main entry point for training and inference.
Uses Cost of Transport (CoT) based on terrain slope only.
"""

def set_seed(seed: int):
    """
    재현성을 위한 시드 설정
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")

def main(args):
    """
    Main function for training and inference
    
    Args:
        args: Command line arguments
    """
    print("\n" + "="*60)
    print("Diffusion Path Planning with Slope + CoT")
    print("="*60 + "\n")
    
    # --- Load Configuration ---
    print(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # --- Set Seed for Reproducibility ---
    set_seed(config['seed'])
    
    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- Initialize Components ---
    print("Initializing components...")
    
    # Dataset (GradientDataset for gradient terrain)
    dataset = GradientDataset(config, load_auxiliary=False)
    
    # Model (U-Net with visual encoder)
    model = ConditionalPathModel(config=config)
    
    # Diffusion Scheduler
    diffusion_scheduler = DiffusionScheduler(
        timesteps=config['diffusion']['timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        device=device
    )
    
    print("Initialization complete\n")

    # === Mode Selection ===
    
    if args.mode == 'train':
        print("="*60)
        print("TRAINING MODE")
        print("="*60 + "\n")
        
        trainer = Trainer(
            model=model,
            dataset=dataset,
            diffusion_scheduler=diffusion_scheduler,
            config=config
        )
        trainer.train()

    elif args.mode == 'infer':
        print("="*60)
        print("INFERENCE MODE")
        print("="*60 + "\n")
        
        # Random seed for varied terrain generation
        seed = int(time.time())
        set_seed(seed)
        
        # --- Load Model Checkpoint ---
        model_path = os.path.join(config['training']['checkpoint_dir'], config['training']['model_name'])
        print(f"Loading model from: {model_path}")
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            print(f"Model loaded successfully\n")
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}.")
            print("Please run in 'train' mode first to generate a model checkpoint.")
            return
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            return
            
        # --- Generate New Slope + CoT Terrain ---
        print("Generating new slope-based CoT terrain for inference...")
        print("  → Diffusion input: Slope + CoT (2-channel map)")
        print("  → GT generation: CoT-based A* search")
        
        img_size = config['data']['img_size']
        terrain_generator = SlopeCotGenerator(
            img_size=img_size,
            height_range=(0, 10),
            mass=10.0,
            gravity=9.8,
            limit_angle_deg=30
        )
        
        # Generate terrain (slope map for input, CoT for A* search)
        # ✅ Random terrain generation for robustness testing
        # Use None to generate diverse terrains (same as training augmentation)
        h_map, s_map, cot_costmap = terrain_generator.generate(
            terrain_scales=None  # Random terrain for better generalization test
        )
        
        # Select random start/end positions with minimum distance guarantee
        margin = img_size // 10
        min_distance = img_size // 2  # Minimum distance: 1/2 of map size
        
        max_attempts = 100
        for attempt in range(max_attempts):
            start_pos = (np.random.randint(margin, img_size - margin),
                        np.random.randint(margin, img_size - margin))
            goal_pos = (np.random.randint(margin, img_size - margin),
                       np.random.randint(margin, img_size - margin))
            
            # Calculate Euclidean distance
            distance = np.sqrt((start_pos[0] - goal_pos[0])**2 + 
                             (start_pos[1] - goal_pos[1])**2)
            
            if distance >= min_distance:
                break
        
        print(f"Start position: {start_pos}")
        print(f"Goal position: {goal_pos}")
        print(f"Distance between start and goal: {distance:.1f} pixels")
        
        # --- Calculate CoT-Efficient Path using A* ---
        print(f"Calculating A* CoT-efficient path...")
        true_path_pixels = terrain_generator.find_path(start_pos, goal_pos)
        
        # Prepare ground truth path
        true_path = None
        if true_path_pixels is not None and len(true_path_pixels) > 0:
            print(f"A* found path with {len(true_path_pixels)} points")
            # Convert path to tensor and normalize [-1, 1]
            true_path_np = np.array(true_path_pixels, dtype=np.float32)
            # ✅ (row, col) → (x, y) = (col, row) 반전
            true_path_np = true_path_np[:, [1, 0]]
            true_path_norm = (true_path_np / img_size) * 2 - 1
            true_path = torch.from_numpy(true_path_norm).float().to(device).unsqueeze(0)
        else:
            print("Warning: A* could not find a path.")

        # --- Prepare Inputs for Model ---
        
        # Create 2-channel input: [Slope map, CoT map]
        slope_degrees = np.degrees(s_map)  # 라디안 → 도
        slope_norm = slope_degrees / 90.0  # [0, 90°] → [0, 1] 정규화
        
        # Normalize CoT costmap (같은 방식으로 Inf 처리)
        cot_norm = normalize_costmap(cot_costmap)  # ✅ Training과 동일한 정규화
        
        # Stack as 2 channels: [Slope, CoT]
        costmap_norm = np.stack([slope_norm, cot_norm], axis=0)  # [2, H, W]
        
        print(f"\n2-channel input statistics:")
        print(f"  Slope - Min: {slope_degrees.min():.2f}°, Max: {slope_degrees.max():.2f}°")
        print(f"  CoT   - Min: {cot_costmap.min():.2f}, Max: {cot_costmap.max():.2f}")
        
        # Check for NaN values
        if np.isnan(costmap_norm).any():
            print("Warning: NaN detected in costmap normalization. Replacing with zeros.")
            costmap_norm = np.nan_to_num(costmap_norm, nan=0.0)
        
        # Costmap tensor: [2, H, W] -> [1, 2, H, W]
        costmap_tensor = torch.from_numpy(costmap_norm).float().to(device)
        if costmap_tensor.dim() == 3:
            costmap_tensor = costmap_tensor.unsqueeze(0)  # [2, H, W] → [1, 2, H, W]
        
        # Convert (row, col) to (x, y) and normalize [-1, 1]
        start_pos_xy = np.array([start_pos[1], start_pos[0]], dtype=np.float32)  # (col, row)
        goal_pos_xy = np.array([goal_pos[1], goal_pos[0]], dtype=np.float32)     # (col, row)
        
        norm_start = (start_pos_xy / img_size) * 2 - 1
        norm_goal = (goal_pos_xy / img_size) * 2 - 1
        
        # Tensors [1, 2]
        start_tensor = torch.from_numpy(norm_start).float().to(device).unsqueeze(0)
        goal_tensor = torch.from_numpy(norm_goal).float().to(device).unsqueeze(0)
        
        print(f"\nNormalized coordinates:")
        print(f"  Start: {norm_start}")
        print(f"  Goal:  {norm_goal}")
        
        # --- Run Diffusion Sampling ---
        print(f"\nRunning diffusion sampling with inpainting...")
        generated_path = diffusion_scheduler.sample(
            model=model,
            condition=costmap_tensor,
            shape=(1, config['data']['horizon'], 2),
            start_pos=start_tensor, 
            end_pos=goal_tensor,
            show_progress=True
        )
        
        print(f"Generated path shape: {generated_path.shape}")
        
        # Check for NaN in generated path
        if torch.isnan(generated_path).any():
            print("Warning: NaN detected in generated path. Attempting to fix...")
            # Replace NaN with linear interpolation between start and goal
            generated_path = torch.nan_to_num(generated_path, nan=0.0)
            # If still problematic, create a simple linear interpolation
            if torch.isnan(generated_path).any() or (generated_path.abs() > 1e6).any():
                print("Creating fallback linear path...")
                t = torch.linspace(0, 1, config['data']['horizon'], device=device).unsqueeze(0).unsqueeze(-1)
                generated_path = start_tensor.unsqueeze(1) * (1 - t) + goal_tensor.unsqueeze(1) * t
        
        print()
        
        # --- Visualize Results ---
        print("Creating visualization...")
        plot_results(costmap_tensor, generated_path, true_path, config, slope_map=s_map)
        print("Inference complete!\n")
        
    elif args.mode == 'check_train':
        print("="*60)
        print("CHECK-TRAIN MODE (Overfitting Diagnosis)")
        print("="*60 + "\n")
        
        # Load model
        model_path = os.path.join(config['training']['checkpoint_dir'], config['training']['model_name'])
        print(f"Loading model from: {model_path}")
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            print(f"Model loaded successfully\n")
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        # Load training dataset with auxiliary data (for slope_map visualization)
        print("Loading training dataset...")
        try:
            train_dataset = GradientDataset(config, load_auxiliary=True)
        except Exception as e:
            print(f"Error loading GradientDataset: {e}")
            return
            
        # Random sample selection
        idx = random.randint(0, len(train_dataset) - 1)
        print(f"Testing with Training Sample Index: {idx}\n")
        
        # Get sample (with auxiliary=True, returns dict)
        try:
            sample_data = train_dataset[idx]
            costmap_data = sample_data['costmap']
            true_path_data = sample_data['path']
            slope_map_data = sample_data.get('slope_map', None)
        except (ValueError, KeyError) as e:
            print(f"Error: Dataset returned unexpected format. Check data_loader.py.")
            print(f"Details: {e}")
            return

        # Extract start/end positions
        # 데이터가 이미 (x, y) 형식으로 저장되어 있음 ✅
        start_data = true_path_data[0]    # [2] - (x, y)
        end_data = true_path_data[-1]     # [2] - (x, y)

        # Prepare tensors
        costmap_tensor = costmap_data.to(device)
        if costmap_tensor.dim() == 2:
            costmap_tensor = costmap_tensor.unsqueeze(0).unsqueeze(0)
        elif costmap_tensor.dim() == 3:
            costmap_tensor = costmap_tensor.unsqueeze(0)
            
        true_path_tensor = true_path_data.to(device).unsqueeze(0)  # [1, Horizon, 2]
        start_tensor = start_data.to(device).unsqueeze(0)          # [1, 2]
        end_tensor = end_data.to(device).unsqueeze(0)              # [1, 2]

        # Inference
        print(f"Start: {start_tensor[0].cpu().numpy()}")
        print(f"End:   {end_tensor[0].cpu().numpy()}")
        print(f"\nRunning diffusion sampling on training data...\n")
        
        generated_path = diffusion_scheduler.sample(
            model=model,
            condition=costmap_tensor,
            shape=(1, config['data']['horizon'], 2),
            start_pos=start_tensor,
            end_pos=end_tensor,
            show_progress=True
        )
        
        # Visualization
        print(f"\nCreating visualization...")
        # Convert slope_map to numpy if available
        slope_map_np = slope_map_data.cpu().numpy() if slope_map_data is not None else None
        plot_results(costmap_tensor, generated_path, true_path_tensor, config, slope_map=slope_map_np)
        print("Check-train complete!\n")

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Diffusion-Based Path Planner for Cost-Aware Environments")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer', 'check_train'],
                        help="Mode to run the script in: 'train' or 'infer' or 'check_train'")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help="Path to the configuration file")
    
    args = parser.parse_args()
    main(args)
