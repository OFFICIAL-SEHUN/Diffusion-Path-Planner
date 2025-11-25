import torch
import torch.nn.functional as F
from tqdm import tqdm

class DiffusionScheduler:
    """
    Manages the diffusion process (forward noising and reverse sampling).
    """
    def __init__(self, timesteps, beta_start, beta_end, device):
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def forward_process(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        xt = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        return xt, noise

    # [수정 1] @torch.no_grad() 제거 (Gradient 계산 필요)
    def sample(self, model, condition, shape, start_pos=None, end_pos=None, cost_guidance_scale=20.0):
        """
        Args:
            cost_guidance_scale (float): 벽을 피하려는 힘의 세기. (보통 10.0 ~ 100.0 사이 권장)
        """
        model.eval()
        
        # 1. 랜덤 노이즈로 시작
        x = torch.randn(shape, device=self.device)
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc="Sampling", total=self.timesteps):
            
            # [기존 로직] In-painting (Start/End 고정)
            if start_pos is not None and end_pos is not None:
                current_start = start_pos.unsqueeze(1) 
                current_end = end_pos.unsqueeze(1)
                
                if i > 0: 
                    noise_s = torch.randn_like(current_start)
                    noise_e = torch.randn_like(current_end)
                    s_alpha = self.sqrt_alphas_cumprod[i]
                    s_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[i]
                    
                    current_start = s_alpha * current_start + s_one_minus_alpha * noise_s
                    current_end = s_alpha * current_end + s_one_minus_alpha * noise_e

                x[:, 0, :] = current_start.squeeze(1)
                x[:, -1, :] = current_end.squeeze(1)

            # ============================================================
            # [추가] Cost Guidance (벽 밀어내기)
            # ============================================================
            if cost_guidance_scale > 0:
                with torch.enable_grad():
                    # x를 복제하여 Gradient 추적 가능하게 만듦
                    x_in = x.detach().requires_grad_(True)
                    
                    # 1. 현재 경로(x_in)가 Costmap의 어디에 위치하는지 샘플링
                    # x_in: [B, Horizon, 2] -> grid_sample을 위해 [B, Horizon, 1, 2]로 변환
                    # grid_sample은 좌표가 [-1, 1] 범위여야 함 (이미 정규화되어 있다고 가정)
                    grid = x_in.unsqueeze(2) 
                    
                    # Costmap 샘플링 (Bilinear Interpolation)
                    # condition(costmap): [B, 1, H, W]
                    cost_sample = F.grid_sample(condition, grid, align_corners=True) # 결과: [B, 1, Horizon, 1]
                    
                    # 2. Loss 계산
                    # 벽(1.0)에 가까울수록 Loss가 커짐 -> 이를 최소화하는 방향으로 미분
                    # 벽이 아닌 곳(-1.0 근처)은 무시하기 위해 ReLU 등을 쓸 수도 있지만,
                    # 단순 Sum으로도 효과가 있습니다. (높은 값을 낮추려 하므로)
                    
                    # 팁: 벽(값 > 0)인 부분만 강하게 밀어내기 위해 ReLU 사용 권장
                    # Costmap이 -1~1 범위라면 0보다 큰 값(벽)만 타겟팅
                    cost_loss = torch.sum(torch.relu(cost_sample)) 
                    
                    # 3. Gradient 계산
                    grad = torch.autograd.grad(cost_loss, x_in)[0]
                    
                    # 4. 경로 수정 (Gradient Descent)
                    # Loss를 줄이는 방향(벽 반대 방향)으로 이동
                    # 노이즈가 많은 초기 단계보다 후반부에 더 정교하게 작용함
                    
                    # Gradient 폭주 방지를 위한 클리핑 (선택 사항)
                    grad = torch.clamp(grad, -1.0, 1.0)
                    
                    x = x - cost_guidance_scale * grad
                    
                    # 좌표가 -1~1 범위를 벗어나지 않게 클램핑
                    x = torch.clamp(x, -1.0, 1.0)
            # ============================================================

            # 2. Denoising Step (모델 예측)
            # 모델 추론 시에는 Gradient 필요 없으므로 no_grad 처리 (메모리 절약)
            with torch.no_grad():
                t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
                predicted_noise = model(x, t, condition)
            
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
        
        # 마지막으로 Start/End 원본 고정
        if start_pos is not None and end_pos is not None:
            x[:, 0, :] = start_pos
            x[:, -1, :] = end_pos
            
        return x