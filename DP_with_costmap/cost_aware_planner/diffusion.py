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
    def sample(self, model, condition, shape, start_pos=None, end_pos=None):
        """
        Args:
            ...
            start_pos (torch.Tensor): [B, 2] Start coordinates (normalized)
            end_pos (torch.Tensor): [B, 2] End coordinates (normalized)
        """
        model.eval()
        
        # 1. 랜덤 노이즈로 시작
        x = torch.randn(shape, device=self.device)
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc="Sampling", total=self.timesteps):
            # [수정] 매 스텝 시작 전에 Start/End를 강제로 고정 (In-painting)
            if start_pos is not None and end_pos is not None:
                # 현재 시점(i)에 맞는 노이즈 레벨을 계산해서 Start/End에 섞어줌
                # (x0 상태의 start/end를 그냥 박으면 안됨, 노이즈가 껴있어야 함)
                
                # 원본 start/end (Batch, 1, 2)
                current_start = start_pos.unsqueeze(1) 
                current_end = end_pos.unsqueeze(1)
                
                if i > 0: # 마지막 스텝이 아니면 노이즈 추가
                    noise_s = torch.randn_like(current_start)
                    noise_e = torch.randn_like(current_end)
                    
                    # q_sample 공식 (forward process) 사용
                    # sqrt_alpha_cumprod * x0 + sqrt(1-alpha_cumprod) * epsilon
                    
                    # scalar 값 추출
                    s_alpha = self.sqrt_alphas_cumprod[i]
                    s_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[i]
                    
                    current_start = s_alpha * current_start + s_one_minus_alpha * noise_s
                    current_end = s_alpha * current_end + s_one_minus_alpha * noise_e

                # x의 첫번째 점과 마지막 점을 강제로 교체
                x[:, 0, :] = current_start.squeeze(1)
                x[:, -1, :] = current_end.squeeze(1)

            # ---------------------------------------------------------
            
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            predicted_noise = model(x, t, condition)
            
            # (기존 Denoising 로직 유지)
            alpha_t = self.alphas[t][:, None, None]
            alpha_cumprod_t = self.alphas_cumprod[t][:, None, None]
            beta_t = self.betas[t][:, None, None]
            
            epsilon = 1e-8
            term1 = 1 / torch.sqrt(alpha_t)
            term2 = (beta_t / torch.sqrt(1 - alpha_cumprod_t + epsilon)) * predicted_noise
            x = term1 * (x - term2)
            
            if i > 0:
                z = torch.randn_like(x)
                x += torch.sqrt(beta_t) * z
        
        # 마지막으로 한번 더 고정 (노이즈 없는 원본 좌표)
        if start_pos is not None and end_pos is not None:
            x[:, 0, :] = start_pos
            x[:, -1, :] = end_pos
            
        return x

