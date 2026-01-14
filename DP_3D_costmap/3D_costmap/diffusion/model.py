import torch
import torch.nn as nn
import math
from typing import Optional

"""
Conditional Diffusion Model for Path Planning
U-Net architecture with visual encoding for Slope + CoT costmap conditioning.
"""

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal Position Embeddings (from Transformer)
    
    타임스텝 t를 연속적인 임베딩 벡터로 변환
    sin/cos 주기 함수 조합으로 각 타임스텝을 고유하게 표현
    """
    
    def __init__(self, dim: int):
        """
        Args:
            dim (int): 임베딩 차원 (짝수여야 함)
        """
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        타임스텝을 sinusoidal 임베딩으로 변환
        
        공식: PE(t, 2i) = sin(t / 10000^(2i/d))
             PE(t, 2i+1) = cos(t / 10000^(2i/d))
        
        Args:
            time (torch.Tensor): 타임스텝 [B] 또는 [B, ...] 정수 텐서
            
        Returns:
            torch.Tensor: 임베딩 [..., dim]
        """
        device = time.device
        half_dim = self.dim // 2
        
        # 주파수 계산
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # 원래 shape 저장 및 flatten
        original_shape = time.shape
        time = time.flatten()
        
        # Sinusoidal 계산
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        # 원래 shape 복원
        return embeddings.view(*original_shape, self.dim)

class Block1D(nn.Module):
    """
    기본 Conv1D 블록
    Conv → GroupNorm → SiLU 구조
    """
    
    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        """
        Args:
            dim (int): 입력 채널 수
            dim_out (int): 출력 채널 수
            groups (int): GroupNorm 그룹 수
        """
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, scale_shift: Optional[tuple] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 입력 [B, C, L]
            scale_shift (tuple, optional): (scale, shift) for adaptive norm
            
        Returns:
            torch.Tensor: 출력 [B, C_out, L]
        """
        x = self.proj(x)
        x = self.norm(x)
        
        # Adaptive GroupNorm (FiLM)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
            
        x = self.act(x)
        return x


class ResnetBlock1D(nn.Module):
    """
    1D ResNet Block with Time Conditioning
    
    Skip connection과 time embedding을 사용한 residual block
    U-Net의 기본 구성 요소
    """
    
    def __init__(self, dim: int, dim_out: int, time_cond_dim: Optional[int] = None, groups: int = 8):
        """
        Args:
            dim (int): 입력 채널 수
            dim_out (int): 출력 채널 수
            time_cond_dim (int, optional): Time embedding 차원
            groups (int): GroupNorm 그룹 수
        """
        super().__init__()
        
        # Time conditioning MLP
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_cond_dim, dim_out * 2)
        ) if time_cond_dim is not None else None

        # Main blocks
        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)
        
        # Residual connection
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_cond_emb: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 입력 [B, C, L]
            time_cond_emb (torch.Tensor, optional): Time + condition embedding [B, D]
            
        Returns:
            torch.Tensor: 출력 [B, C_out, L]
        """
        scale_shift = None
        
        # Time conditioning을 scale/shift로 변환
        if self.mlp is not None and time_cond_emb is not None:
            time_cond_emb = self.mlp(time_cond_emb)
            time_cond_emb = time_cond_emb.unsqueeze(-1)  # [B, 2*C_out, 1]
            scale_shift = time_cond_emb.chunk(2, dim=1)

        # Forward through blocks
        h = self.block1(x, scale_shift)
        h = self.block2(h)
        
        # Residual connection
        return h + self.res_conv(x)

class BasicBlock(nn.Module):
    """
    ResNet Basic Block (for ResNet18/34)
    
    Two 3x3 convolutions with residual connection
    """
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        """
        Args:
            in_channels (int): 입력 채널 수
            out_channels (int): 출력 채널 수
            stride (int): Conv stride
            downsample (nn.Module, optional): Residual connection downsampling
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(32, out_channels // 4), out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(32, out_channels // 4), out_channels)
        
        self.downsample = downsample
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.gn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class VisualEncoder(nn.Module):
    """
    ResNet18-based Visual Encoder for Slope + CoT Map (2-channel)
    
    2채널 입력 (Slope + CoT)을 ResNet18 구조로 처리하여 특징 벡터로 변환
    - Channel 0: Slope map (물리적 지형 정보)
    - Channel 1: CoT map (에너지 비용 정보)
    
    Architecture: ResNet18 (without pre-trained weights)
    - 구조적으로 검증된 ResNet18 아키텍처 사용
    - Pre-trained weights 없이 처음부터 학습
    - Residual connections로 깊은 네트워크 안정적 학습
    """
    
    def __init__(self, input_channels: int = 2, feature_dim: int = 512):
        """
        Args:
            input_channels (int): 입력 채널 수 (2 - Slope + CoT)
            feature_dim (int): 출력 특징 벡터 차원
        """
        super().__init__()
        
        # Initial convolution (7x7 conv, modified for 2-channel input)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.gn1 = nn.GroupNorm(8, 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet18 layers (2 blocks each)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global Average Pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, feature_dim)
        self.act = nn.SiLU()
        
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        """
        Create a ResNet layer with multiple BasicBlocks
        
        Args:
            in_channels (int): 입력 채널
            out_channels (int): 출력 채널
            blocks (int): BasicBlock 개수
            stride (int): 첫 번째 block의 stride
            
        Returns:
            nn.Sequential: Layer
        """
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(min(32, out_channels // 4), out_channels),
            )
        
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 2-channel map [B, 2, H, W]
                            - Channel 0: Slope map
                            - Channel 1: CoT map
            
        Returns:
            torch.Tensor: Feature vector [B, feature_dim]
        """
        # Initial conv
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling and FC
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.act(x)
        
        return x

class ConditionalPathModel(nn.Module):
    """
    Conditional U-Net for Path Planning with Start/Goal Conditioning
    
    Slope + CoT 2채널 map + Start/Goal positions을 조건으로 받아 경로 생성
    노이즈가 섞인 경로에서 노이즈를 예측하는 U-Net 모델
    
    Architecture:
        - Visual Encoder: Slope + CoT Map (2-channel) → Feature Vector
        - Time Embedding: Timestep → Embedding
        - Global Conditioning: [Time, Visual, Start, Goal] concatenated
        - U-Net: Noisy Path → Predicted Noise
    
    Conditioning:
        - Terrain: Slope + CoT 2채널 map (physical terrain + energy cost)
        - Start: Start position [x, y] normalized to [-1, 1]
        - Goal: Goal position [x, y] normalized to [-1, 1]
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config (dict): 설정 딕셔너리
        """
        super().__init__()
        
        # Model hyperparameters
        self.transition_dim = 2  # (x, y) 좌표
        dim = 64                  # Base channel dimension
        time_dim = dim * 4        # Time embedding dimension
        visual_dim = time_dim     # Visual feature dimension
        
        # Start/Goal position dimension
        start_dim = 2
        goal_dim = 2
        
        # Global conditioning dimension (time + visual + start + goal)
        self.global_cond_dim = time_dim + visual_dim + start_dim + goal_dim
        
        # === 1. Conditioning Networks ===
        
        # Visual Encoder: 2-channel map [B, 2, H, W] → Feature [B, visual_dim]
        # Channel 0: Slope map, Channel 1: CoT map
        self.visual_encoder = VisualEncoder(
            input_channels=2,  # Slope + CoT
            feature_dim=visual_dim
        )
        
        # Time Embedding: Timestep [B] → Embedding [B, time_dim]
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Positional Embedding for sequence
        self.pos_embed = SinusoidalPositionEmbeddings(dim)
        
        # === 2. U-Net Architecture ===
        
        # Initial projection
        self.init_conv = nn.Conv1d(self.transition_dim, dim, 7, padding=3)
        
        # Encoder (Downsampling)
        self.down1_block = ResnetBlock1D(dim, dim, self.global_cond_dim)
        self.down1_pool = nn.Conv1d(dim, dim*2, 3, 2, 1)
        
        self.down2_block = ResnetBlock1D(dim*2, dim*2, self.global_cond_dim)
        self.down2_pool = nn.Conv1d(dim*2, dim*4, 3, 2, 1)
        
        # Bottleneck
        self.mid_block1 = ResnetBlock1D(dim*4, dim*4, self.global_cond_dim)
        self.mid_block2 = ResnetBlock1D(dim*4, dim*4, self.global_cond_dim)
        
        # Decoder (Upsampling with skip connections)
        self.up2_upsample = nn.ConvTranspose1d(dim*4, dim*2, 4, 2, 1)
        self.up2_block = ResnetBlock1D(dim*2 + dim*2, dim*2, self.global_cond_dim)  # + skip
        
        self.up1_upsample = nn.ConvTranspose1d(dim*2, dim, 4, 2, 1)
        self.up1_block = ResnetBlock1D(dim + dim, dim, self.global_cond_dim)  # + skip
        
        # Output projection
        self.final_conv = nn.Conv1d(dim, self.transition_dim, 1)

    def forward(self, 
                x: torch.Tensor, 
                time: torch.Tensor, 
                condition: torch.Tensor,
                start_pos: Optional[torch.Tensor] = None,
                goal_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass: 노이즈 예측
        
        Args:
            x (torch.Tensor): Noisy path [B, Horizon, 2]
            time (torch.Tensor): Timestep [B]
            condition (torch.Tensor): 2-channel map [B, 2, H, W]
                                    - Channel 0: Slope map (정규화)
                                    - Channel 1: CoT map (정규화)
            start_pos (torch.Tensor, optional): Start position [B, 2], normalized to [-1, 1]
            goal_pos (torch.Tensor, optional): Goal position [B, 2], normalized to [-1, 1]
            
        Returns:
            torch.Tensor: Predicted noise [B, Horizon, 2]
        """
        
        # === 1. Input Preprocessing ===
        
        # 2-channel map 차원 확인
        if condition.dim() != 4:
            raise ValueError(
                f"Condition shape mismatch. "
                f"Expected [B, 2, H, W], got {condition.shape}"
            )
        
        if condition.shape[1] != 2:
            raise ValueError(
                f"Expected 2 channels (Slope + CoT), got {condition.shape[1]} channels"
            )
        
        batch_size = x.shape[0]
        
        # === 2. Global Conditioning ===
        
        # Visual encoding: Costmap → Feature vector
        visual_features = self.visual_encoder(condition)  # [B, visual_dim]
        
        # Time embedding
        time_emb = self.time_mlp(time)  # [B, time_dim]
        
        # 🔥 Concatenate all conditioning signals (GitHub 방식)
        # [Time, Visual, Start, Goal] → Rich global context
        if start_pos is not None and goal_pos is not None:
            # Start/Goal positions provided
            global_cond = torch.cat([
                time_emb,        # [B, time_dim]
                visual_features, # [B, visual_dim]
                start_pos,       # [B, 2]
                goal_pos         # [B, 2]
            ], dim=-1)  # [B, time_dim + visual_dim + 2 + 2]
        else:
            # Fallback: Start/Goal 없으면 zero padding
            device = x.device
            zero_start = torch.zeros(batch_size, 2, device=device)
            zero_goal = torch.zeros(batch_size, 2, device=device)
            global_cond = torch.cat([
                time_emb,
                visual_features,
                zero_start,
                zero_goal
            ], dim=-1)
        
        conditioning = global_cond  # [B, global_cond_dim]
        
        # === 3. Path Encoding ===
        
        # Transpose for Conv1d: [B, Horizon, 2] → [B, 2, Horizon]
        x = x.transpose(1, 2)
        
        # Positional embedding
        horizon = x.shape[2]
        pos = torch.arange(horizon, device=x.device).unsqueeze(0)  # [1, Horizon]
        pos_emb = self.pos_embed(pos).transpose(1, 2)  # [1, dim, Horizon]
        
        # Initial conv + positional encoding
        x = self.init_conv(x)  # [B, dim, Horizon]
        x = x + pos_emb        # Add positional information
        
        # === 4. U-Net Forward Pass ===
        
        # Encoder
        h1 = self.down1_block(x, conditioning)    # [B, dim, Horizon]
        x = self.down1_pool(h1)                   # [B, dim*2, Horizon/2]
        
        h2 = self.down2_block(x, conditioning)    # [B, dim*2, Horizon/2]
        x = self.down2_pool(h2)                   # [B, dim*4, Horizon/4]
        
        # Bottleneck
        x = self.mid_block1(x, conditioning)      # [B, dim*4, Horizon/4]
        x = self.mid_block2(x, conditioning)      # [B, dim*4, Horizon/4]
        
        # Decoder with skip connections
        x = self.up2_upsample(x)                  # [B, dim*2, Horizon/2]
        x = torch.cat((x, h2), dim=1)             # [B, dim*4, Horizon/2]
        x = self.up2_block(x, conditioning)       # [B, dim*2, Horizon/2]
        
        x = self.up1_upsample(x)                  # [B, dim, Horizon]
        x = torch.cat((x, h1), dim=1)             # [B, dim*2, Horizon]
        x = self.up1_block(x, conditioning)       # [B, dim, Horizon]
        
        # Output projection
        x = self.final_conv(x)                    # [B, 2, Horizon]
        
        # Transpose back: [B, 2, Horizon] → [B, Horizon, 2]
        return x.transpose(1, 2)