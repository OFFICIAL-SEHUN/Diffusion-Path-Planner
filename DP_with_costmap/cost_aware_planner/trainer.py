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

class Trainer:
    """
    Manages the training loop, optimization, logging, and checkpointing of the model.
    """
    def __init__(self, model, dataset, diffusion_scheduler, config):
        self.model = model
        self.diffusion_scheduler = diffusion_scheduler
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(self.device)

        self.dataset = dataset
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=4
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate']
        )

        self.checkpoint_dir = config['training']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self):
        """
        Executes the main training loop with WandB logging (Time-based naming).
        """
        epochs = self.config['training']['epochs']
        log_interval = self.config['training']['log_interval']
        model_name = self.config['training']['model_name']

        # [수정 1] 현재 시간 문자열 생성 (예: 2023-10-25_14-30-00)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{model_name}_{current_time}"

        # [수정 2] WandB 초기화 시 name에 시간 포함
        wandb.init(
            project="diffusion-path-planning", 
            config=self.config,
            name=run_name 
        )
        
        print(f"Starting training run: {run_name}")
        
        global_step = 0 

        for epoch in range(epochs):
            self.model.train()
            epoch_loss_sum = 0.0 

            # Dataset에서 2개(costmap, path)만 받도록 수정된 상태
            for costmaps, paths in tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                
                costmaps = costmaps.to(self.device).float()
                paths = paths.to(self.device).float()

                if costmaps.dim() == 3:
                    costmaps = costmaps.unsqueeze(1)

                self.optimizer.zero_grad()
                
                # 1. Timestep 샘플링
                t = torch.randint(
                    0,
                    self.config['diffusion']['timesteps'],
                    (paths.shape[0],),
                    device=self.device
                ).long()

                # 2. Forward Process
                noisy_paths, noise = self.diffusion_scheduler.forward_process(paths, t)
    
                # 3. 모델 예측
                predicted_noise = self.model(noisy_paths, t, costmaps)

                # 4. Loss 계산
                loss = F.mse_loss(noise, predicted_noise)

                # 5. 역전파
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # WandB Logging (Step)
                current_loss = loss.item()
                wandb.log({"train_step_loss": current_loss, "global_step": global_step})
                
                epoch_loss_sum += current_loss
                global_step += 1

            # WandB Logging (Epoch)
            avg_loss = epoch_loss_sum / len(self.dataloader)
            wandb.log({"train_epoch_loss": avg_loss, "epoch": epoch + 1})

            # --- 시각화 및 저장 ---
            if (epoch + 1) % log_interval == 0:
                print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.6f}")
                
                # WandB에 이미지 올리기
                self.model.eval()
                with torch.no_grad():
                    test_costmap = costmaps[0].unsqueeze(0)
                    
                    generated_path = self.diffusion_scheduler.sample(
                        self.model, 
                        test_costmap, 
                        (1, paths.shape[1], 2)
                    )
                    
                    fig, ax = plt.subplots(figsize=(5, 5))
                    
                    cmap_img = test_costmap.squeeze().cpu().numpy()
            
                    ax.imshow(cmap_img, cmap='plasma_r') 
                    
                    h, w = cmap_img.shape
                    gen_pts = generated_path.squeeze().cpu().numpy()
                    gen_x = (gen_pts[:, 1] + 1) / 2 * w
                    gen_y = (gen_pts[:, 0] + 1) / 2 * h
                    ax.plot(gen_x, gen_y, 'r-', linewidth=2, label='Gen Path')
                    
                    ax.legend()
                    ax.set_title(f"Epoch {epoch+1}")
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    image = Image.open(buf)
                    
                    wandb.log({"inference_sample": wandb.Image(image), "epoch": epoch+1})
                    plt.close(fig)
                
                self.model.train()

                checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch+1}_{model_name}")
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

        final_model_path = os.path.join(self.checkpoint_dir, model_name)
        torch.save(self.model.state_dict(), final_model_path)
        print(f"Training finished. Final model saved to {final_model_path}")
        
        wandb.finish()