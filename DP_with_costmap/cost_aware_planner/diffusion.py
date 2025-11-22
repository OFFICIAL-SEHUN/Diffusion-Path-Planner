import torch
from tqdm import tqdm

class DiffusionScheduler:
    """
    Manages the diffusion process (forward noising and reverse sampling).
    """
    def __init__(self, timesteps, beta_start, beta_end, device):
        """
        Initializes the scheduler and pre-computes diffusion parameters.
        """
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def forward_process(self, x0, t):
        """
        Applies noise to a clean data sample x0 to get xt for a given timestep t.
        This uses the closed-form solution.
        Args:
            x0 (torch.Tensor): The initial clean data (e.g., a batch of trajectories).
            t (torch.Tensor): A tensor of timesteps for each sample in the batch.
        Returns:
            tuple (torch.Tensor, torch.Tensor): The noised data xt and the noise added.
        """
        noise = torch.randn_like(x0)
        
        # Gather the appropriate sqrt_alphas_cumprod for each timestep
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        
        xt = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        return xt, noise

    @torch.no_grad()
    def sample(self, model, condition, shape):
        """
        Performs the full reverse diffusion process (sampling) to generate a new sample.
        Args:
            model (nn.Module): The trained noise prediction model.
            condition (torch.Tensor): The conditioning information (e.g., a costmap).
            shape (tuple): The shape of the desired output tensor (batch, horizon, 2).
        Returns:
            torch.Tensor: The generated data sample (e.g., a trajectory).
        """
        model.eval()
        
        # Start with pure Gaussian noise
        x = torch.randn(shape, device=self.device)
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc="Sampling", total=self.timesteps):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = model(x, t, condition)
            
            # --- Debug: Check for NaNs from the model ---
            if torch.isnan(predicted_noise).any():
                print(f"!!! NaN DETECTED in predicted_noise at timestep {i} !!!")
                # The model is likely unstable. Further sampling is pointless.
                return x # Return the current tensor to show where it failed
            
            # Denoise one step using the DDPM formula
            alpha_t = self.alphas[t][:, None, None]
            alpha_cumprod_t = self.alphas_cumprod[t][:, None, None]
            beta_t = self.betas[t][:, None, None]
            
            # --- Modified Denoising Step for Stability ---
            # The original formula can be unstable when alpha_cumprod_t is close to 1.
            # We add a small epsilon to the denominator to prevent division by zero.
            epsilon = 1e-8
            term1 = 1 / torch.sqrt(alpha_t)
            term2 = (beta_t / torch.sqrt(1 - alpha_cumprod_t + epsilon)) * predicted_noise
            x = term1 * (x - term2)
            
            # Add noise back in if not the last step
            if i > 0:
                z = torch.randn_like(x)
                x += torch.sqrt(beta_t) * z
                
        return x
