import torch
import torch.nn as nn

class StandardDiffusionLoss(nn.Module):
    def __init__(self, scheduler, device):
        """
        Standard DDPM Loss for Path Planning.
        GT(A*) 경로를 따라가도록 학습합니다.
        """
        super().__init__()
        self.scheduler = scheduler
        self.device = device
        self.mse = nn.MSELoss()

    def forward(self, model, x0, condition):
        """
        Args:
            model: 노이즈 예측 모델 (ConditionalPathModel)
            x0: [Batch, N, 2] (Ground Truth Path, A* 경로)
            condition: [Batch, 1, H, W] (Visual Encoder 입력용 Costmap)
        Returns:
            loss: Scalar Tensor
        """
        batch_size = x0.shape[0]

        # 1. Timestep t 샘플링 (랜덤한 시점 고르기)
        # "이 경로는 100단계 중 35단계만큼 망가진 상태야" 라고 정해줌
        t = torch.randint(
            0, self.scheduler.timesteps, 
            (batch_size,), device=self.device
        ).long()

        # 2. Forward Process (Reparameterization Trick)
        # GT 경로(x0)에 가우시안 노이즈(epsilon)를 섞어서 망가진 경로(xt)를 만듦
        # 공식: xt = sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * epsilon
        
        # 여기서 xt, noise(epsilon)를 받아옴
        xt, noise = self.scheduler.forward_process(x0, t)

        # 3. 모델 예측 (Denoising)
        # "망가진 경로(xt)와 지도(condition)를 줄 테니, 섞인 노이즈(epsilon)가 뭔지 맞춰봐"
        # 모델이 노이즈를 정확히 맞춘다는 건 = 노이즈를 걷어내고 원본 GT로 돌아가는 법을 안다는 뜻
        noise_pred = model(xt, t, condition)

        # 4. Loss 계산 (MSE)
        # 정답 노이즈(noise)와 예측 노이즈(noise_pred)의 차이
        loss = self.mse(noise_pred, noise)

        return loss