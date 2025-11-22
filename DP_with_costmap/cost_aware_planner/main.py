import yaml
import argparse
import torch
import os
import random
import numpy as np

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
        
        # --- Load Model Checkpoint ---
        model_path = os.path.join(config['training']['checkpoint_dir'], config['training']['model_name'])
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
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
        
        # --- Calculate the Optimal Path using A* ---
        print(f"Calculating A* path from {start_pos} to {end_pos}...")
        true_path_pixels = a_star_search(
            costmap, 
            tuple(start_pos), 
            tuple(end_pos), 
            config['data']['cost_weight']
        )
        
        if true_path_pixels is None:
            print("A* could not find a path. Skipping visualization of the true path.")
            true_path_pixels = [] # Empty path

        # Convert path to tensor and normalize for visualization
        true_path_np = np.array(true_path_pixels, dtype=np.float32)
        true_path_norm = (true_path_np / config['data']['img_size']) * 2 - 1
        true_path = torch.from_numpy(true_path_norm).float().to(device)

        if costmap is None:
            print("Failed to generate a test map. Please try again.")
            return
            
        costmap_tensor = torch.from_numpy(costmap).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # --- Run Sampling ---
        generated_path = diffusion_scheduler.sample(
            model=model,
            condition=costmap_tensor,
            shape=(1, config['data']['horizon'], 2)
        )
        
        print("Generated path from diffusion sampler:")
        print(generated_path)
        
        # --- Visualize the Result ---
        plot_results(costmap_tensor, generated_path, true_path.unsqueeze(0), config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Diffusion-Based Path Planner for Cost-Aware Environments")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer'],
                        help="Mode to run the script in: 'train' or 'infer'")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help="Path to the configuration file")
    
    args = parser.parse_args()
    main(args)
