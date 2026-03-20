"""
DDPM Diffusion Scheduler

- Linear beta schedule
- Forward process: q(x_t | x_0)
- Reverse sampling with start/goal inpainting
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from tqdm import tqdm


class DiffusionScheduler:
    def __init__(self, timesteps: int = 200,
                 beta_start: float = 0.0001, beta_end: float = 0.02,
                 device: torch.device = torch.device("cpu")):
        self.timesteps = timesteps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def to(self, device: torch.device):
        self.device = device
        for attr in ["betas", "alphas", "alphas_cumprod",
                      "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod"]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def forward_process(self, x0: torch.Tensor,
                        t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """q(x_t | x_0) = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε"""
        noise = torch.randn_like(x0)
        sqrt_a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_a = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_a * x0 + sqrt_one_minus_a * noise, noise

    @torch.no_grad()
    def sample(self, model: nn.Module, condition: torch.Tensor,
               shape: Tuple[int, ...],
               start_pos: Optional[torch.Tensor] = None,
               end_pos: Optional[torch.Tensor] = None,
               text_tokens: Optional[torch.Tensor] = None,
               show_progress: bool = True) -> torch.Tensor:
        """DDPM reverse sampling with optional start/goal inpainting."""
        B = shape[0]
        device = self.device
        x = torch.randn(shape, device=device)

        steps = reversed(range(self.timesteps))
        if show_progress:
            steps = tqdm(steps, total=self.timesteps, desc="Sampling")

        for t_val in steps:
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            alpha_t = self.alphas[t_val]
            alpha_bar_t = self.alphas_cumprod[t_val]
            beta_t = self.betas[t_val]

            eps_pred = model(x, t, condition,
                             start_pos=start_pos, goal_pos=end_pos,
                             text_tokens=text_tokens)

            mean = (1.0 / alpha_t.sqrt()) * (
                x - (beta_t / (1.0 - alpha_bar_t).sqrt()) * eps_pred
            )

            if t_val > 0:
                z = torch.randn_like(x)
                x = mean + beta_t.sqrt() * z
            else:
                x = mean

            if start_pos is not None:
                x[:, 0, :] = start_pos
            if end_pos is not None:
                x[:, -1, :] = end_pos

        return x
