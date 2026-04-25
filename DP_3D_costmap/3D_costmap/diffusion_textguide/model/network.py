"""
Text-conditioned Diffusion Path Model

Architecture (from diffusion_patch, extended):
  - Visual encoder: timm only — set `model.timm_model_name` in YAML (e.g. convnext_tiny / resnet18 / efficientnet_b0 / swin / vit…); scratch = pretrained=False [B,2,H,W] → [B, visual_dim]
  - TextEncoder:   Embedding + Transformer, [B,L] → [B, text_dim]
  - CrossAttention: path features attend to text embedding
  - ConditionalPathModel: 1-D U-Net with FiLM conditioning

Conditioning flow:
  global_cond = concat(time_emb, visual_feat, start, goal)  → FiLM into ResBlocks
  text_emb                                                   → CrossAttention at bottleneck
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, Tuple

import timm


def _require_timm_model_name(timm_model_name: Optional[str]) -> str:
    """YAML must set `model.timm_model_name` (no code defaults)."""
    name = (timm_model_name or "").strip()
    if not name:
        raise ValueError(
            "model.timm_model_name is required in the config (e.g. convnext_tiny, resnet18, "
            "efficientnet_b0, swin_tiny_patch4_window7_224, vit_tiny_patch16_224)."
        )
    return name


# ============================================================================
# Sinusoidal Position Embeddings
# ============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__() 
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = time.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ============================================================================
# 1-D ResNet Blocks
# ============================================================================

class Block1D(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor,
                scale_shift: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock1D(nn.Module):
    def __init__(self, dim: int, dim_out: int,
                 time_cond_dim: Optional[int] = None, groups: int = 8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_cond_dim, dim_out * 2))
            if time_cond_dim else None
        )
        self.block1 = Block1D(dim, dim_out, groups)
        self.block2 = Block1D(dim_out, dim_out, groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor,
                time_cond_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        scale_shift = None
        if self.mlp is not None and time_cond_emb is not None:
            t = self.mlp(time_cond_emb).unsqueeze(-1)
            scale, shift = t.chunk(2, dim=1)
            scale_shift = (scale, shift)
        h = self.block1(x, scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


# ============================================================================
# Visual encoders  [B, 2, H, W] → [B, feature_dim]
# ============================================================================

class VisualEncoderTimm(nn.Module):
    """Generic timm visual encoder wrapper (e.g., ConvNeXt/EfficientNet/ResNet/Swin/ViT)."""

    def __init__(
        self,
        input_channels: int = 2,
        feature_dim: int = 256,
        pretrained: bool = True,
        model_name: str = "convnext_tiny",
        input_img_size: Optional[int] = None,
    ):
        super().__init__()
        create_kwargs = dict(
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=input_channels,
        )
        # Swin in timm validates input spatial size against model img_size.
        # Set it from config so non-224 terrains (e.g., 100x100) are supported.
        if input_img_size is not None and "swin" in model_name.lower():
            create_kwargs["img_size"] = input_img_size
        # ViT defaults to 224; enable flexible H×W (must be compatible with patch size).
        if model_name.lower().startswith("vit_"):
            create_kwargs["dynamic_img_size"] = True
        self.backbone = timm.create_model(
            model_name,
            **create_kwargs,
        )
        self._vit_patch_hw: Optional[Tuple[int, int]] = None
        if model_name.lower().startswith("vit_"):
            pe = getattr(self.backbone, "patch_embed", None)
            ps = getattr(pe, "patch_size", None) if pe is not None else None
            if ps is not None:
                if isinstance(ps, int):
                    self._vit_patch_hw = (ps, ps)
                else:
                    self._vit_patch_hw = (int(ps[0]), int(ps[1]))
        in_features = self.backbone.num_features
        self.fc = nn.Linear(in_features, feature_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._vit_patch_hw is not None:
            ph, pw = self._vit_patch_hw
            _, _, H, W = x.shape
            pad_h = (ph - (H % ph)) % ph
            pad_w = (pw - (W % pw)) % pw
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.backbone(x)
        if x.dim() == 4:
            x = x.flatten(1)
        return self.act(self.fc(x))


class VisualEncoderResNet(VisualEncoderTimm):
    """Backward-compatible alias; pass ``timm_model_name`` (e.g. resnet18) like ``ConditionalPathModel``."""

    def __init__(
        self,
        input_channels: int = 2,
        feature_dim: int = 256,
        pretrained: bool = False,
        timm_model_name: Optional[str] = None,
        input_img_size: Optional[int] = None,
    ):
        name = _require_timm_model_name(timm_model_name)
        super().__init__(
            input_channels=input_channels,
            feature_dim=feature_dim,
            pretrained=pretrained,
            model_name=name,
            input_img_size=input_img_size,
        )


class VisualEncoderConvNeXt(VisualEncoderTimm):
    """Backward-compatible alias for older imports."""
    pass


# Backward compatibility: historical module name (generic timm wrapper)
VisualEncoder = VisualEncoderTimm


# ============================================================================
# TextEncoder  [B, L] → [B, text_dim]
# ============================================================================

class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 200, embed_dim: int = 256,
                 num_layers: int = 2, max_seq_len: int = 16):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=512,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, L] → [B, embed_dim]"""
        B, L = tokens.shape
        x = self.token_emb(tokens) + self.pos_emb[:, :L, :]
        pad_mask = tokens == 0
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        lengths = (~pad_mask).sum(dim=1, keepdim=True).clamp(min=1).float()
        x = (x * (~pad_mask).unsqueeze(-1).float()).sum(dim=1) / lengths.squeeze(1).unsqueeze(-1)
        return self.ln(x)


# ============================================================================
# CrossAttention
# ============================================================================

class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int,
                 heads: int = 8, dim_head: int = 64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """x: [B, C, L], context: [B, D] → [B, C, L]"""
        B, C, L = x.shape
        x_flat = x.permute(0, 2, 1)

        if context.dim() == 2:
            context = context.unsqueeze(1).expand(-1, L, -1)

        q = self.to_q(x_flat).view(B, L, self.heads, -1).transpose(1, 2)
        k = self.to_k(context).view(B, -1, self.heads, q.shape[-1]).transpose(1, 2)
        v = self.to_v(context).view(B, -1, self.heads, q.shape[-1]).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, -1)
        out = self.to_out(out).permute(0, 2, 1)
        return out + x


# ============================================================================
# ConditionalPathModel (1-D U-Net)
# ============================================================================

class ConditionalPathModel(nn.Module):
    """Text-conditioned diffusion path denoiser.

    Input:  noisy path [B, Horizon, 2]
    Output: predicted noise [B, Horizon, 2]

    Conditioning:
      - costmap [B, 2, H, W]  → VisualEncoder → [B, 256]
      - text_tokens [B, L]    → TextEncoder   → [B, 256]
      - start_pos [B, 2]
      - goal_pos  [B, 2]
      - timestep  [B]
    """

    def __init__(self, transition_dim: int = 2, dim: int = 64,
                 horizon: int = 120, visual_dim: int = 256,
                 text_dim: int = 256, vocab_size: int = 200,
                 max_seq_len: int = 16,
                 visual_backbone: Literal[
                     "convnext", "resnet", "efficientnet_b0", "swin_tiny", "vit_tiny"
                 ] = "convnext",
                 visual_pretrained: bool = True,
                 timm_model_name: Optional[str] = None,
                 timm_pretrained: Optional[bool] = None,
                 convnext_pretrained: Optional[bool] = None,
                 input_img_size: Optional[int] = None):
        super().__init__()
        time_dim = dim * 4

        # Legacy support: older configs may still carry `convnext_pretrained`.
        base_pretrained = visual_pretrained if convnext_pretrained is None else convnext_pretrained
        use_timm_pretrained = base_pretrained if timm_pretrained is None else timm_pretrained
        backbone_name = _require_timm_model_name(timm_model_name)
        self.visual_encoder = VisualEncoderTimm(
            input_channels=2,
            feature_dim=visual_dim,
            pretrained=use_timm_pretrained,
            model_name=backbone_name,
            input_img_size=input_img_size,
        )
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=text_dim,
                                        max_seq_len=max_seq_len)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        global_cond_dim = time_dim + visual_dim + 2 + 2  # time + visual + start + goal

        self.init_conv = nn.Conv1d(transition_dim, dim, 1)
        self.pos_embed = nn.Parameter(torch.randn(1, dim, horizon) * 0.02)

        # Down
        self.down1_block = ResnetBlock1D(dim, dim, time_cond_dim=global_cond_dim)
        self.down1_pool = nn.Conv1d(dim, dim * 2, 3, stride=2, padding=1)
        self.down2_block = ResnetBlock1D(dim * 2, dim * 2, time_cond_dim=global_cond_dim)
        self.down2_pool = nn.Conv1d(dim * 2, dim * 4, 3, stride=2, padding=1)

        # Bottleneck
        self.mid_block1 = ResnetBlock1D(dim * 4, dim * 4, time_cond_dim=global_cond_dim)
        self.cross_attn = CrossAttention(query_dim=dim * 4, context_dim=text_dim)
        self.mid_block2 = ResnetBlock1D(dim * 4, dim * 4, time_cond_dim=global_cond_dim)

        # Up
        self.up2_upsample = nn.ConvTranspose1d(dim * 4, dim * 2, 4, stride=2, padding=1)
        self.up2_block = ResnetBlock1D(dim * 4, dim * 2, time_cond_dim=global_cond_dim)
        self.up1_upsample = nn.ConvTranspose1d(dim * 2, dim, 4, stride=2, padding=1)
        self.up1_block = ResnetBlock1D(dim * 2, dim, time_cond_dim=global_cond_dim)

        self.final_conv = nn.Conv1d(dim, transition_dim, 1)

    def forward(self, x: torch.Tensor, time: torch.Tensor,
                condition: torch.Tensor,
                start_pos: Optional[torch.Tensor] = None,
                goal_pos: Optional[torch.Tensor] = None,
                text_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:           [B, Horizon, 2] noisy path
        time:        [B]
        condition:   [B, 2, H, W] costmap
        start_pos:   [B, 2]
        goal_pos:    [B, 2]
        text_tokens: [B, L]
        Returns:     [B, Horizon, 2] predicted noise
        """
        B = x.shape[0]
        device = x.device

        time_emb = self.time_mlp(time)
        visual_feat = self.visual_encoder(condition)

        if start_pos is None:
            start_pos = torch.zeros(B, 2, device=device)
        if goal_pos is None:
            goal_pos = torch.zeros(B, 2, device=device)

        global_cond = torch.cat([time_emb, visual_feat, start_pos, goal_pos], dim=-1)

        if text_tokens is not None:
            text_emb = self.text_encoder(text_tokens)
        else:
            text_emb = torch.zeros(B, self.text_encoder.ln.normalized_shape[0], device=device)

        # U-Net
        h = self.init_conv(x.permute(0, 2, 1)) + self.pos_embed

        h1 = self.down1_block(h, global_cond)
        h = self.down1_pool(h1)
        h2 = self.down2_block(h, global_cond)
        h = self.down2_pool(h2)

        h = self.mid_block1(h, global_cond)
        h = self.cross_attn(h, text_emb)
        h = self.mid_block2(h, global_cond)

        h = self.up2_upsample(h)
        h = torch.cat([h, h2], dim=1)
        h = self.up2_block(h, global_cond)
        h = self.up1_upsample(h)
        h = torch.cat([h, h1], dim=1)
        h = self.up1_block(h, global_cond)

        out = self.final_conv(h)
        return out.permute(0, 2, 1)
