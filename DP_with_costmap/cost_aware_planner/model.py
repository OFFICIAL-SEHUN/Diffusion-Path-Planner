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

class ConditionalPathModel(nn.Module):
    """
    Fixed 1D U-Net with correct Upsampling order
    """
    def __init__(self, config):
        super().__init__()
        
        self.transition_dim = 2 # (x, y)
        
        # [설정] 64x64 이미지를 펼치면 4096이 됩니다.
        # 나중에 Visual Encoder를 쓰면 512 등으로 바꾸세요.
        visual_input_dim = config.get('visual_emb_dim', 4096) 
        
        dim = 64 
        time_dim = dim * 4

        # 1. Embedding Setup
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Waypoint 순서를 위한 Positional Embedding 추가
        self.pos_embed = SinusoidalPositionEmbeddings(dim)
        
        self.cond_proj = nn.Sequential(
            nn.Linear(visual_input_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # 2. Initial Conv
        self.init_conv = nn.Conv1d(self.transition_dim, dim, 7, padding=3)

        # 3. Downsampling Path (Encoder)
        # Level 1: 64 -> 64 (Save h1)
        self.down1_block = ResnetBlock1D(dim, dim, time_dim)
        self.down1_pool = nn.Conv1d(dim, dim*2, 3, 2, 1) # Size / 2 (64->32)

        # Level 2: 32 -> 32 (Save h2)
        self.down2_block = ResnetBlock1D(dim*2, dim*2, time_dim)
        self.down2_pool = nn.Conv1d(dim*2, dim*4, 3, 2, 1) # Size / 2 (32->16)

        # 4. Middle Path (Bottleneck)
        self.mid_block1 = ResnetBlock1D(dim*4, dim*4, time_dim)
        self.mid_block2 = ResnetBlock1D(dim*4, dim*4, time_dim)

        # 5. Upsampling Path (Decoder)
        # Level 2: 16 -> 32 -> Concat(h2) -> Block
        self.up2_upsample = nn.ConvTranspose1d(dim*4, dim*2, 4, 2, 1) # Size * 2
        self.up2_block = ResnetBlock1D(dim*2 + dim*2, dim*2, time_dim)

        # Level 1: 32 -> 64 -> Concat(h1) -> Block
        self.up1_upsample = nn.ConvTranspose1d(dim*2, dim, 4, 2, 1) # Size * 2
        self.up1_block = ResnetBlock1D(dim + dim, dim, time_dim)

        # 6. Final Output
        self.final_conv = nn.Conv1d(dim, self.transition_dim, 1)

    def forward(self, x, time, condition):
        # [자동 수정] 입력 Condition이 이미지(3차원)라면 벡터(2차원)로 펼쳐줌
        if len(condition.shape) > 2:
            condition = condition.view(condition.size(0), -1)

        # Waypoint 순서를 위한 Positional Embedding 생성
        horizon = x.shape[1]
        pos = torch.arange(horizon, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embed(pos)
        pos_emb = pos_emb.transpose(1, 2)

        # 1. Data & Embedding Prep
        x = x.transpose(1, 2) # [B, H, 2] -> [B, 2, H]
        t = self.time_mlp(time)
        c = self.cond_proj(condition)
        emb = t + c 

        # 2. Initial
        x = self.init_conv(x)

        # Positional Embedding 더하기
        x = x + pos_emb

        # 3. Downstream
        # Level 1
        h1 = self.down1_block(x, emb) # Save for skip connection (Size 64)
        x = self.down1_pool(h1)       # Downsample (Size 32)

        # Level 2
        h2 = self.down2_block(x, emb) # Save for skip connection (Size 32)
        x = self.down2_pool(h2)       # Downsample (Size 16)

        # 4. Middle
        x = self.mid_block1(x, emb)
        x = self.mid_block2(x, emb)

        # 5. Upstream (반드시 Up -> Concat -> Block 순서여야 함)
        # Level 2 Recovery
        x = self.up2_upsample(x)      # 16 -> 32
        x = torch.cat((x, h2), dim=1) # Concat with h2 (Size 32)
        x = self.up2_block(x, emb)

        # Level 1 Recovery
        x = self.up1_upsample(x)      # 32 -> 64
        x = torch.cat((x, h1), dim=1) # Concat with h1 (Size 64)
        x = self.up1_block(x, emb)

        # 6. Final
        x = self.final_conv(x)
        return x.transpose(1, 2) # [B, 2, H] -> [B, H, 2]