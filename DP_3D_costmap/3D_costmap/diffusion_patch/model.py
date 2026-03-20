import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

"""
Conditional Diffusion Model for Path Planning
U-Net architecture with visual encoding for Slope + Height costmap conditioning.
Language-conditioned navigation with text encoder and cross-attention.
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
    ResNet18-based Visual Encoder for Slope + Height Map (2-channel)
    
    2채널 입력 (Slope + Height)을 ResNet18 구조로 처리하여 특징 벡터로 변환
    - Channel 0: Slope map (물리적 지형 정보)
    - Channel 1: Height map (고도 정보)
    
    Architecture: ResNet18 (without pre-trained weights)
    - 구조적으로 검증된 ResNet18 아키텍처 사용
    - Pre-trained weights 없이 처음부터 학습
    - Residual connections로 깊은 네트워크 안정적 학습
    """
    
    def __init__(self, input_channels: int = 2, feature_dim: int = 512):
        """
        Args:
            input_channels (int): 입력 채널 수 (2 - Slope + Height)
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
                            - Channel 1: Height map
            
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

class TextEncoder(nn.Module):
    """
    Text Encoder for Language-Conditioned Navigation
    
    텍스트 명령을 임베딩 벡터로 변환하여 추상적인 개념 학습
    예: "Quickly", "Safe route", "Energy efficient" 등
    
    Architecture:
    - Token Embedding: 텍스트 → 토큰 임베딩
    - Transformer Encoder: Self-attention으로 문맥 이해
    - Pooling: 시퀀스 → 고정 크기 벡터
    """
    
    def __init__(self, vocab_size: int = 1000, embed_dim: int = 256, num_layers: int = 2, max_seq_len: int = 32):
        """
        Args:
            vocab_size (int): 어휘 크기 (단어 수)
            embed_dim (int): 임베딩 차원
            num_layers (int): Transformer 레이어 수
            max_seq_len (int): 최대 시퀀스 길이
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm for stability
        self.ln = nn.LayerNorm(embed_dim)
        
    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_tokens (torch.Tensor): 텍스트 토큰 [B, L] (L ≤ max_seq_len)
            
        Returns:
            torch.Tensor: 텍스트 임베딩 [B, embed_dim]
        """
        batch_size, seq_len = text_tokens.shape
        
        # Token embeddings
        x = self.token_embedding(text_tokens)  # [B, L, embed_dim]
        
        # Positional embeddings
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Transformer encoding
        x = self.transformer(x)  # [B, L, embed_dim]
        
        # Global average pooling (또는 CLS token 방식)
        x = x.mean(dim=1)  # [B, embed_dim]
        
        # Layer norm
        x = self.ln(x)
        
        return x

class CrossAttention(nn.Module):
    """
    Cross-Attention Module
    
    U-Net의 path feature와 text embedding 간의 상호작용
    모델이 지형(Slope Map)을 보면서 동시에 사용자의 명령(Text Vector)을 참고
    """
    
    def __init__(self, query_dim: int, context_dim: int, heads: int = 8, dim_head: int = 64):
        """
        Args:
            query_dim (int): Query 차원 (path feature dimension)
            context_dim (int): Context 차원 (text embedding dimension)
            heads (int): Attention head 수
            dim_head (int): 각 head의 차원
        """
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        
        # Query, Key, Value projections
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Query [B, C, L] (path features)
            context (torch.Tensor): Context [B, D] (text embedding)
            
        Returns:
            torch.Tensor: Attended features [B, C, L]
        """
        batch_size, C, L = x.shape
        
        # Convert to [B, L, C] for attention
        x_flat = x.transpose(1, 2)  # [B, L, C]
        
        # Expand context for attention: [B, D] -> [B, 1, D] -> [B, L, D]
        context_expanded = context.unsqueeze(1).expand(-1, L, -1)  # [B, L, D]
        
        # Compute Q, K, V
        q = self.to_q(x_flat)  # [B, L, inner_dim]
        k = self.to_k(context_expanded)  # [B, L, inner_dim]
        v = self.to_v(context_expanded)  # [B, L, inner_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, L, self.heads, -1).transpose(1, 2)  # [B, heads, L, dim_head]
        k = k.view(batch_size, L, self.heads, -1).transpose(1, 2)  # [B, heads, L, dim_head]
        v = v.view(batch_size, L, self.heads, -1).transpose(1, 2)  # [B, heads, L, dim_head]
        
        # Attention
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, heads, L, L]
        attn = F.softmax(dots, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, heads, L, dim_head]
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, L, -1)  # [B, L, inner_dim]
        out = self.to_out(out)  # [B, L, C]
        
        # Residual connection
        out = out + x_flat
        
        # Convert back to [B, C, L]
        out = out.transpose(1, 2)  # [B, C, L]
        
        return out

class ConditionalPathModel(nn.Module):
    """
    Language-Conditioned U-Net for Path Planning
    
    Slope + Height 2채널 map + Start/Goal positions + Text command을 조건으로 받아 경로 생성
    노이즈가 섞인 경로에서 노이즈를 예측하는 U-Net 모델
    
    Architecture:
        - Visual Encoder: Slope + Height Map (2-channel) → Feature Vector
        - Text Encoder: Text command → Text Embedding
        - Time Embedding: Timestep → Embedding
        - Global Conditioning: [Time, Visual, Start, Goal] concatenated
        - Cross-Attention: U-Net bottleneck에서 text embedding과 path feature 상호작용
        - U-Net: Noisy Path → Predicted Noise
    
    Conditioning:
        - Terrain: Slope + Height 2채널 map (physical terrain + height)
        - Text: Text command (e.g., "Quickly", "Safe route", "Energy efficient")
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
        text_dim = 256            # Text embedding dimension
        
        # Start/Goal position dimension
        start_dim = 2
        goal_dim = 2
        
        # Global conditioning dimension (time + visual + start + goal)
        # Note: Text is handled separately via cross-attention
        self.global_cond_dim = time_dim + visual_dim + start_dim + goal_dim
        
        # === 1. Conditioning Networks ===
        
        # Visual Encoder: 2-channel map [B, 2, H, W] → Feature [B, visual_dim]
        # Channel 0: Slope map, Channel 1: Height map
        self.visual_encoder = VisualEncoder(
            input_channels=2,  # Slope + Height
            feature_dim=visual_dim
        )
        
        # Text Encoder: Text tokens [B, L] → Embedding [B, text_dim]
        vocab_size = config.get('text_encoder', {}).get('vocab_size', 1000)
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=text_dim,
            num_layers=2,
            max_seq_len=32
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
        
        # Bottleneck with Cross-Attention
        self.mid_block1 = ResnetBlock1D(dim*4, dim*4, self.global_cond_dim)
        
        # Cross-Attention: Path features attend to text embedding
        self.cross_attn = CrossAttention(
            query_dim=dim*4,      # Path feature dimension
            context_dim=text_dim, # Text embedding dimension
            heads=8,
            dim_head=64
        )
        
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
                goal_pos: Optional[torch.Tensor] = None,
                text_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass: 노이즈 예측
        
        Args:
            x (torch.Tensor): Noisy path [B, Horizon, 2]
            time (torch.Tensor): Timestep [B]
            condition (torch.Tensor): 2-channel map [B, 2, H, W]
                                    - Channel 0: Slope map (정규화)
                                    - Channel 1: Height map (정규화)
            start_pos (torch.Tensor, optional): Start position [B, 2], normalized to [-1, 1]
            goal_pos (torch.Tensor, optional): Goal position [B, 2], normalized to [-1, 1]
            text_tokens (torch.Tensor, optional): Text command tokens [B, L] (L ≤ 32)
            
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
                f"Expected 2 channels (Slope + Height), got {condition.shape[1]} channels"
            )
        
        batch_size = x.shape[0]
        
        # === 2. Conditioning ===
        
        # Visual encoding: Costmap → Feature vector
        visual_features = self.visual_encoder(condition)  # [B, visual_dim]
        
        # Text encoding: Text tokens → Text embedding
        if text_tokens is not None:
            text_emb = self.text_encoder(text_tokens)  # [B, text_dim]
        else:
            # Fallback: zero text embedding if no text provided
            device = x.device
            text_emb = torch.zeros(batch_size, 256, device=device)  # [B, text_dim]
        
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
        
        # Bottleneck with Cross-Attention
        x = self.mid_block1(x, conditioning)      # [B, dim*4, Horizon/4]
        
        # Cross-Attention: Path features attend to text embedding
        x = self.cross_attn(x, text_emb)          # [B, dim*4, Horizon/4]
        
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