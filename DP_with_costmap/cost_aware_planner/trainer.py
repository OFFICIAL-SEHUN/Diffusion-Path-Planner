import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

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
        Executes the main training loop for the specified number of epochs.
        """
        epochs = self.config['training']['epochs']
        log_interval = self.config['training']['log_interval']
        model_name = self.config['training']['model_name']

        print("Starting training...")
        for epoch in range(epochs):
            self.model.train()
            for costmaps, paths in tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                costmaps = costmaps.to(self.device)
                paths = paths.to(self.device)

                self.optimizer.zero_grad()
                
                # 1. Sample a random timestep
                t = torch.randint(
                    0,
                    self.config['diffusion']['timesteps'],
                    (paths.shape[0],),
                    device=self.device
                ).long()

                # 2. Apply forward diffusion process
                noisy_paths, noise = self.diffusion_scheduler.forward_process(paths, t)
    
                # 3. Predict the noise using the model
                predicted_noise = self.model(noisy_paths, t, costmaps)

                # 4. Calculate the loss
                loss = F.mse_loss(noise, predicted_noise)

                # 5. Backpropagate and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # Gradient Clipping
                self.optimizer.step()

            if (epoch + 1) % log_interval == 0:
                print(f"Epoch {epoch+1} | Loss: {loss.item():.6f}")
                # Save a checkpoint
                checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch+1}_{model_name}")
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

        # Save the final model
        final_model_path = os.path.join(self.checkpoint_dir, model_name)
        torch.save(self.model.state_dict(), final_model_path)
        print(f"Training finished. Final model saved to {final_model_path}")
