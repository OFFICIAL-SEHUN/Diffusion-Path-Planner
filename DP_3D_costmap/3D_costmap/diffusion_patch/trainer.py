import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import wandb
from matplotlib import pyplot as plt
import io
from PIL import Image
from datetime import datetime
from typing import Dict, Any

"""
Trainer for Diffusion-based Path Planning
Manages training loop, optimization, logging, and checkpointing.
Trains model with Slope + Height 2-channel input to generate energy-efficient paths.
"""

class Trainer:
    """
    Diffusion Model Trainer for Slope + Height Path Planning
    
    Features:
    - WandB logging with time-based run naming
    - Periodic visualization during training
    - Model checkpointing
    - Gradient clipping for stability
    """
    
    def __init__(self, 
                 model: torch.nn.Module, 
                 dataset: torch.utils.data.Dataset, 
                 diffusion_scheduler, 
                 config: Dict[str, Any]):
        """
        Args:
            model (torch.nn.Module): ConditionalPathModel
            dataset (Dataset): GradientDataset
            diffusion_scheduler (DiffusionScheduler): Diffusion scheduler
            config (dict): Configuration dictionary
        """
        self.model = model
        self.diffusion_scheduler = diffusion_scheduler
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model.to(self.device)

        # Setup dataloader
        self.dataset = dataset
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )

        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate']
        )

        # Checkpoint directory
        self.checkpoint_dir = config['training']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f"Trainer initialized on device: {self.device}")
        print(f"Dataset size: {len(self.dataset)}")
        print(f"Batch size: {config['training']['batch_size']}")
        print(f"Total batches per epoch: {len(self.dataloader)}")

    def train(self):
        """
        메인 학습 루프 실행
        
        Features:
        - DDPM 학습 (노이즈 예측)
        - WandB 로깅 (step/epoch 단위)
        - 주기적 시각화 및 체크포인트 저장
        """
        # Training config
        epochs = self.config['training']['epochs']
        log_interval = self.config['training']['log_interval']
        model_name = self.config['training']['model_name']

        # WandB initialization with time-based naming
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{model_name}_{current_time}"

        wandb.init(
            project="diffusion-slope-height-path-planning",  # Updated project name
            config=self.config,
            name=run_name 
        )
        
        print("="*60)
        print(f"Starting Training Run: {run_name}")
        print("="*60)
        
        global_step = 0 

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            epoch_loss_sum = 0.0 

            # Iterate over batches
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                # Handle both old format (costmaps, paths) and new format (costmaps, paths, text_tokens)
                if len(batch) == 2:
                    costmaps, paths = batch
                    text_tokens = None  # No text tokens
                else:
                    costmaps, paths, text_tokens = batch
                
                # Move data to device
                costmaps = costmaps.to(self.device, non_blocking=True).float()
                paths = paths.to(self.device, non_blocking=True).float()
                if text_tokens is not None:
                    text_tokens = text_tokens.to(self.device, non_blocking=True).long()

                # Costmaps should be [B, 2, H, W] (2-channel: Slope + Height)
                # Paths should be [B, Horizon, 2] - 저장 형식: (x, y) ✅
                # Text tokens should be [B, max_seq_len] (optional)
                
                # Extract start and goal positions from paths
                start_pos = paths[:, 0, :]   # [B, 2] - First waypoint (x, y)
                goal_pos = paths[:, -1, :]   # [B, 2] - Last waypoint (x, y)

                self.optimizer.zero_grad()
                
                # === DDPM Training Step ===
                
                # 1. Sample random timesteps
                t = torch.randint(
                    0,
                    self.config['diffusion']['timesteps'],
                    (paths.shape[0],),
                    device=self.device,
                    dtype=torch.long
                )

                # 2. Forward diffusion: Add noise to GT paths
                noisy_paths, noise = self.diffusion_scheduler.forward_process(paths, t)
    
                # 3. Model prediction: Predict the added noise
                # 🔥 Now with start/goal + text conditioning!
                predicted_noise = self.model(noisy_paths, t, costmaps, start_pos, goal_pos, text_tokens)

                # 4. Compute loss: MSE between true noise and predicted noise
                loss = F.mse_loss(predicted_noise, noise)

                # 5. Backpropagation
                loss.backward()
                
                # Gradient clipping for training stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()

                # Logging
                current_loss = loss.item()
                wandb.log({"train_step_loss": current_loss, "global_step": global_step})
                
                epoch_loss_sum += current_loss
                global_step += 1
                
                # Update progress bar
                pbar.set_postfix({"loss": f"{current_loss:.4f}"})

            # Epoch metrics
            avg_loss = epoch_loss_sum / len(self.dataloader)
            wandb.log({"train_epoch_loss": avg_loss, "epoch": epoch + 1})
            
            print(f"\nEpoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.6f}")

            # === Visualization and Checkpointing ===
            if (epoch + 1) % log_interval == 0:
                print(f"\nSaving checkpoint and generating visualization...")
                
                # Generate sample for visualization
                self.model.eval()
                with torch.no_grad():
                    # Use first sample from last batch
                    test_costmap = costmaps[0:1]  # [1, 2, H, W] - 2 channels
                    
                    # Generate path using diffusion sampling
                    generated_path = self.diffusion_scheduler.sample(
                        model=self.model, 
                        condition=test_costmap, 
                        shape=(1, paths.shape[1], 2),
                        show_progress=False  # No progress bar during training
                    )
                    
                    # Create visualization (use Height channel for display)
                    fig, ax = plt.subplots(figsize=(6, 6))
                    
                    # Plot Height costmap (Channel 1)
                    costmap_np = test_costmap.squeeze().cpu().numpy()  # [2, H, W]
                    height_channel = costmap_np[1]  # Channel 1: Height
                    ax.imshow(height_channel, cmap='terrain', origin='lower', vmin=0, vmax=1)
                    ax.set_title(f"Epoch {epoch+1} - Generated Path on Height Map")
                    
                    # Plot generated path
                    h, w = height_channel.shape
                    gen_pts = generated_path.squeeze().cpu().numpy()
                    # Convert from [-1, 1] to pixel coordinates
                    gen_x = (gen_pts[:, 1] + 1) / 2 * w
                    gen_y = (gen_pts[:, 0] + 1) / 2 * h
                    ax.plot(gen_x, gen_y, 'c-', linewidth=2, label='Generated', alpha=0.8)
                    ax.scatter(gen_x[0], gen_y[0], c='cyan', s=100, 
                             edgecolors='black', linewidths=2, marker='o', label='Start', zorder=10)
                    ax.scatter(gen_x[-1], gen_y[-1], c='yellow', s=100,
                             edgecolors='black', linewidths=2, marker='*', label='Goal', zorder=10)
                    
                    ax.legend()
                    ax.axis('off')
                    plt.tight_layout()
                    
                    # Save to buffer and log to WandB
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                    buf.seek(0)
                    image = Image.open(buf)
                    
                    wandb.log({"inference_sample": wandb.Image(image), "epoch": epoch+1})
                    plt.close(fig)
                
                self.model.train()

                # Save checkpoint
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, 
                    f"epoch_{epoch+1}_{model_name}"
                )
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")

        # Save final model
        final_model_path = os.path.join(self.checkpoint_dir, model_name)
        torch.save(self.model.state_dict(), final_model_path)
        
        print("\n" + "="*60)
        print(f"Training Complete!")
        print(f"Final model saved: {final_model_path}")
        print("="*60)
        
        wandb.finish()