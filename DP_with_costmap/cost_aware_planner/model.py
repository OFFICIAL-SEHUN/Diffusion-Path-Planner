import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Expects a 1D or 2D tensor of integers.
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # Store original shape and flatten time
        original_shape = time.shape
        time = time.flatten()
        
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        # Reshape to original shape with embedding dim
        return embeddings.view(*original_shape, self.dim)

class Block1D(nn.Module):
    """ResNet Block for 1D Data"""
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x

class ResnetBlock1D(nn.Module):
    def __init__(self, dim, dim_out, time_cond_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_cond_dim, dim_out * 2)
        ) if time_cond_dim is not None else None

        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_cond_emb):
        scale_shift = None
        if self.mlp is not None and time_cond_emb is not None:
            time_cond_emb = self.mlp(time_cond_emb)
            time_cond_emb = time_cond_emb.unsqueeze(-1) 
            scale_shift = time_cond_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class VisualEncoder(nn.Module):
    """
    Costmap(이미지)을 처리하여 특징 벡터로 만드는 CNN
    Input: [B, 1, 64, 64] -> Output: [B, 512]
    """
    def __init__(self, input_channels=1, feature_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1), # 64x64
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                             # 32x32
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                             # 16x16
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                             # 8x8
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),                # [B, 256, 1, 1] -> 1x1로 압축
            nn.Flatten(),                                # [B, 256]
            nn.Linear(256, feature_dim),
            nn.SiLU()
        )

    def forward(self, x):
        return self.net(x)

class ConditionalPathModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.transition_dim = 2
        dim = 64
        time_dim = dim * 4
        
        # [수정] 시각 정보 처리를 위한 CNN Encoder 추가
        self.visual_encoder = VisualEncoder(input_channels=1, feature_dim=dim*4)
        
        # Embedding Setup
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        self.pos_embed = SinusoidalPositionEmbeddings(dim)
        
        # [수정] cond_proj는 이제 필요 없거나, CNN 출력과 Time을 합치는 용도로 변경 가능
        # 여기서는 visual_encoder가 이미 차원을 맞춰주므로 제거하거나 Identity 처리
        
        # ... (나머지 U-Net 구조 down1_block 등은 그대로 유지) ...
        # 아래 부분은 그대로 복사해서 쓰시면 됩니다 (단, init_conv 등 정의 필요)
        self.init_conv = nn.Conv1d(self.transition_dim, dim, 7, padding=3)
        self.down1_block = ResnetBlock1D(dim, dim, time_dim)
        self.down1_pool = nn.Conv1d(dim, dim*2, 3, 2, 1)
        self.down2_block = ResnetBlock1D(dim*2, dim*2, time_dim)
        self.down2_pool = nn.Conv1d(dim*2, dim*4, 3, 2, 1)
        self.mid_block1 = ResnetBlock1D(dim*4, dim*4, time_dim)
        self.mid_block2 = ResnetBlock1D(dim*4, dim*4, time_dim)
        self.up2_upsample = nn.ConvTranspose1d(dim*4, dim*2, 4, 2, 1)
        self.up2_block = ResnetBlock1D(dim*2 + dim*2, dim*2, time_dim)
        self.up1_upsample = nn.ConvTranspose1d(dim*2, dim, 4, 2, 1)
        self.up1_block = ResnetBlock1D(dim + dim, dim, time_dim)
        self.final_conv = nn.Conv1d(dim, self.transition_dim, 1)

    def forward(self, x, time, condition):
        # x: [B, H, 2] (Noisy Path)
        # time: [B] (Timestep)
        # condition: [B, H, W] 또는 [B, 1, H, W] (Costmap)

        # ========== [여기부터 수정하세요] ==========
        # 1. 입력 차원 검사 및 자동 수정
        # 입력이 [Batch, Height, Width] (3차원)으로 들어오면 -> [Batch, 1, Height, Width] (4차원)으로 변경
        if len(condition.shape) == 3:
            condition = condition.unsqueeze(1)
            
        # 이제 차원이 4차원인지 확인 (안전장치)
        if len(condition.shape) != 4:
             raise ValueError(f"Condition shape mismatch. Expected [B, 1, H, W] or [B, H, W], got {condition.shape}")
        # =========================================

        # 2. 이미지 인코딩 (Visual Encoder 통과)
        c = self.visual_encoder(condition) 

        # 3. Data & Time Embedding
        x = x.transpose(1, 2) # [B, 2, H]
        t = self.time_mlp(time)
        
        # Time Embedding + Image Feature 합치기
        emb = t + c 


        # 3. Positional Embedding
        horizon = x.shape[2]
        pos = torch.arange(horizon, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embed(pos).transpose(1, 2) # [1, dim, H]

        x = self.init_conv(x)
        x = x + pos_emb # [B, dim, H]

        # ... (이후 U-Net 통과 과정은 기존 코드와 동일) ...
        h1 = self.down1_block(x, emb)
        x = self.down1_pool(h1)
        h2 = self.down2_block(x, emb)
        x = self.down2_pool(h2)
        x = self.mid_block1(x, emb)
        x = self.mid_block2(x, emb)
        x = self.up2_upsample(x)
        x = torch.cat((x, h2), dim=1)
        x = self.up2_block(x, emb)
        x = self.up1_upsample(x)
        x = torch.cat((x, h1), dim=1)
        x = self.up1_block(x, emb)
        x = self.final_conv(x)
        
        return x.transpose(1, 2)