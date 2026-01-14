import torch
import torch.nn as nn
from typing import Tuple

"""
Diffusion Loss Functions for Path Planning
Standard DDPM objective with Slope + CoT costmap conditioning.
"""

class StandardDiffusionLoss(nn.Module):
    """
    Standard DDPM Loss for CoT-Efficient Path Planning
    
    학습 목표: Slope + CoT map을 보고 에너지 효율적 경로를 생성하도록 학습
    - Input: Slope + CoT 2채널 map (물리적 경사 정보 + 에너지 비용 정보)
    - GT 경로: A* 알고리즘이 CoT 비용으로 찾은 에너지 효율적 경로
    - 모델: Slope와 CoT 정보를 모두 활용하여 최적 경로의 패턴을 학습
    """
    
    def __init__(self, scheduler, device: torch.device):
        """
        Args:
            scheduler (DiffusionScheduler): Forward/reverse diffusion 관리
            device (torch.device): 계산 디바이스
        """
        super().__init__()
        self.scheduler = scheduler
        self.device = device
        self.mse = nn.MSELoss()

    def forward(self, 
                model: nn.Module, 
                x0: torch.Tensor, 
                condition: torch.Tensor) -> torch.Tensor:
        """
        DDPM 학습 손실 계산
        
        목표: ε_θ(x_t, t, c) ≈ ε
        모델이 노이즈를 정확히 예측하도록 학습
        
        Args:
            model (nn.Module): 노이즈 예측 모델 (ConditionalPathModel)
            x0 (torch.Tensor): Ground Truth 경로 [B, H, 2]
                              CoT 기반 A*가 찾은 에너지 효율적 경로
            condition (torch.Tensor): 2-channel map [B, 2, H, W]
                                     Channel 0: Slope map (정규화)
                                     Channel 1: CoT map (정규화)
        
        Returns:
            torch.Tensor: MSE Loss (scalar)
        """
        batch_size = x0.shape[0]

        # 1. Timestep 랜덤 샘플링
        # 각 배치에 대해 0 ~ T-1 사이의 랜덤한 시점 선택
        # "이 경로는 T 단계 중 t 단계만큼 노이즈가 추가된 상태"
        t = torch.randint(
            0, self.scheduler.timesteps, 
            (batch_size,), 
            device=self.device,
            dtype=torch.long
        )

        # 2. Forward Diffusion Process
        # GT 경로(x0)에 가우시안 노이즈(ε)를 추가하여 x_t 생성
        # 
        # 수식: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
        #       where ε ~ N(0, I)
        # 
        # 이는 closed-form으로 한 번에 계산 가능 (효율적)
        x_t, noise = self.scheduler.forward_process(x0, t)

        # 3. 모델 예측 (Denoising Network)
        # 입력: 노이즈가 섞인 경로(x_t), 타임스텝(t), 조건(costmap)
        # 출력: 추가된 노이즈(ε) 예측
        # 
        # "이 정도 망가진 경로와 지형 정보를 보고, 
        #  어떤 노이즈가 섞였는지 맞춰봐"
        predicted_noise = model(x_t, t, condition)

        # 4. Loss 계산
        # L_simple = ||ε - ε_θ(x_t, t, c)||²
        # 
        # 실제 노이즈와 예측 노이즈의 MSE
        # 모델이 노이즈를 정확히 예측할 수 있다면
        # → Reverse process에서 노이즈를 제거하여 GT 경로 복원 가능
        loss = self.mse(predicted_noise, noise)

        return loss