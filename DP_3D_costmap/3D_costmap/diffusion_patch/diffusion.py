import torch
from tqdm import tqdm
from typing import Tuple, Optional

"""
Diffusion Scheduler for Path Planning
Implements DDPM (Denoising Diffusion Probabilistic Models) for trajectory generation
with Slope + Height 2-channel costmap conditioning.
Model receives both Slope and Height information as input.
"""

class DiffusionScheduler:
    """
    Diffusion 기반 경로 계획을 위한 스케줄러
    
    Forward process: 깨끗한 경로에 점진적으로 노이즈 추가
    Reverse process: 노이즈로부터 경로 복원 (costmap 조건부)
    """
    
    def __init__(self, timesteps: int, beta_start: float, beta_end: float, device: torch.device):
        """
        Diffusion 스케줄러 초기화 및 파라미터 사전 계산
        
        Args:
            timesteps (int): Diffusion 스텝 수 (T)
            beta_start (float): 시작 노이즈 스케일 (e.g., 0.0001)
            beta_end (float): 끝 노이즈 스케일 (e.g., 0.02)
            device (torch.device): 계산 디바이스 (cuda/cpu)
        """
        self.timesteps = timesteps
        self.device = device

        # Beta schedule: 선형 증가
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        
        # Alpha 계산: α_t = 1 - β_t
        self.alphas = 1. - self.betas
        
        # Cumulative product: ᾱ_t = ∏(α_i) for i=1 to t
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        
        # Forward process용 계수 사전 계산
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def forward_process(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: 깨끗한 경로에 노이즈 추가 (학습용)
        
        Closed-form solution 사용:
        q(x_t | x_0) = N(x_t; √ᾱ_t * x_0, (1 - ᾱ_t) * I)
        x_t = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε, where ε ~ N(0, I) #DDPM
        
        Args:
            x0 (torch.Tensor): 깨끗한 경로 데이터 [B, H, 2]
            t (torch.Tensor): 타임스텝 [B] (0 ~ timesteps-1)
            
        Returns:
            tuple: (x_t, noise)
                - x_t (torch.Tensor): 노이즈가 추가된 경로 [B, H, 2]
                - noise (torch.Tensor): 추가된 노이즈 [B, H, 2]
        """
        # 가우시안 노이즈 샘플링
        noise = torch.randn_like(x0)
        
        # 각 타임스텝에 맞는 계수 추출
        # [B] -> [B, 1, 1] for broadcasting
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        
        # Forward process 공식 적용
        x_t = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t, noise

    @torch.no_grad()
    def sample(self, 
               model: torch.nn.Module, 
               condition: torch.Tensor, 
               shape: Tuple[int, int, int],
               start_pos: Optional[torch.Tensor] = None, 
               end_pos: Optional[torch.Tensor] = None,
               text_tokens: Optional[torch.Tensor] = None,
               show_progress: bool = True,
               save_intermediates: bool = False,
               intermediate_steps: Optional[list] = None) -> torch.Tensor:
        """
        Reverse diffusion process: 노이즈로부터 경로 생성 (추론용)
        
        DDPM sampling with optional inpainting (start/end position fixing)
        
        Args:
            model (torch.nn.Module): Denoising model (UNet 등)
            condition (torch.Tensor): 조건 (2-channel costmap) [B, 2, H, W] - [Slope, Height]
            shape (tuple): 생성할 경로 shape (B, Horizon, 2)
            start_pos (torch.Tensor, optional): 시작 위치 [B, 2], 범위 [-1, 1]
            end_pos (torch.Tensor, optional): 목표 위치 [B, 2], 범위 [-1, 1]
            show_progress (bool): tqdm 진행바 표시 여부
            save_intermediates (bool): 중간 step들을 저장할지 여부
            intermediate_steps (list, optional): 저장할 timestep 리스트 (None이면 균등하게 선택)
            
        Returns:
            torch.Tensor: 생성된 경로 [B, Horizon, 2]
            or tuple: (생성된 경로, 중간 결과들) if save_intermediates=True
        """
        model.eval()
        
        # 중간 결과 저장을 위한 리스트
        intermediates = []
        if save_intermediates:
            # 저장할 timestep 결정 (None이면 20개 균등하게)
            if intermediate_steps is None:
                num_saves = min(20, self.timesteps)
                intermediate_steps = [int(i * self.timesteps / num_saves) for i in range(num_saves)]
            intermediate_steps = set(intermediate_steps)  # 빠른 검색을 위해 set으로 변환
        
        # 1. 순수 가우시안 노이즈로 시작 (x_T ~ N(0, I))
        x = torch.randn(shape, device=self.device)
        
        # 2. Reverse diffusion: T -> 0
        iterator = reversed(range(0, self.timesteps))
        if show_progress:
            iterator = tqdm(iterator, desc="Sampling Path", total=self.timesteps)
        
        for i in iterator:
            # === Inpainting: Start/End 위치 고정 ===
            if start_pos is not None and end_pos is not None:
                # 타임스텝 i에 맞는 노이즈 레벨로 start/end 조정
                # 이렇게 하면 모델이 일관된 경로를 생성할 수 있음
                
                current_start = start_pos.unsqueeze(1)  # [B, 1, 2] -> [B, horizon, 2]
                current_end = end_pos.unsqueeze(1)      # [B, 1, 2] -> [B, horizon, 2]
                
                if i > 0:  # 마지막 스텝 아니면 노이즈 추가
                    noise_start = torch.randn_like(current_start)
                    noise_end = torch.randn_like(current_end)
                    
                    # Forward process 공식: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
                    sqrt_alpha = self.sqrt_alphas_cumprod[i]
                    sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[i]
                    
                    current_start = sqrt_alpha * current_start + sqrt_one_minus_alpha * noise_start
                    current_end = sqrt_alpha * current_end + sqrt_one_minus_alpha * noise_end
                
                # 경로의 첫/마지막 점을 고정
                x[:, 0, :] = current_start.squeeze(1)
                x[:, -1, :] = current_end.squeeze(1)
            
            # === Denoising Step ===
            # 타임스텝 텐서 생성
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            
            # 모델로 노이즈 예측 (start/goal/text 전달)
            predicted_noise = model(x, t, condition, start_pos, end_pos, text_tokens)
            
            # Reverse process 공식 적용
            # p(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
            # μ_θ = 1/√α_t * (x_t - β_t/√(1-ᾱ_t) * ε_θ(x_t, t))
            
            alpha_t = self.alphas[t][:, None, None]
            alpha_cumprod_t = self.alphas_cumprod[t][:, None, None]
            beta_t = self.betas[t][:, None, None]
            
            # 수치 안정성을 위한 작은 값
            eps = 1e-8
            
            # Mean 계산
            coeff1 = 1.0 / torch.sqrt(alpha_t)
            coeff2 = beta_t / torch.sqrt(1.0 - alpha_cumprod_t + eps)
            mean = coeff1 * (x - coeff2 * predicted_noise)
            
            # 다음 스텝으로 이동
            if i > 0:
                # Variance 추가 (stochastic sampling)
                z = torch.randn_like(x)
                sigma = torch.sqrt(beta_t)
                x = mean + sigma * z
            else:
                # 마지막 스텝: 노이즈 없이 mean만 사용
                x = mean
            
            # 중간 결과 저장
            if save_intermediates and i in intermediate_steps:
                intermediates.append({
                    'timestep': i,
                    'path': x.clone().cpu(),
                    'noise_level': self.sqrt_one_minus_alphas_cumprod[i].item()
                })
        
        # 3. 최종 결과에서 start/end 정확히 고정
        if start_pos is not None and end_pos is not None:
            x[:, 0, :] = start_pos
            x[:, -1, :] = end_pos
        
        if save_intermediates:
            # 최종 결과도 추가
            intermediates.append({
                'timestep': 0,
                'path': x.clone().cpu(),
                'noise_level': 0.0
            })
            return x, intermediates
        
        return x

