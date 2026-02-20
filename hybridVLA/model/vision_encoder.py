"""
Quantized Vision Encoder with Window Attention and M-RoPE.

Combines:
- SigLIP-style ViT architecture (from BitVLA)
- Window attention with sparse global layers (from Qwen2.5-VL)
- 2D M-RoPE position encoding (from Qwen2-VL)
- BitNet 1.58-bit quantization on all linear layers (from BitVLA)
- SwiGLU activation (from Qwen2.5-VL)
- 2x2 token merging MLP connector (from Qwen2-VL)

Supports dynamic resolution: images are processed at native resolution,
producing a variable number of visual tokens proportional to pixel count.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .quantization import BitLinear158
from .mrope import MultimodalRoPE, apply_rotary_emb


class SwiGLU(nn.Module):
    """SwiGLU activation: SiLU(xW1) * (xW2), used in modern transformers."""

    def __init__(self, in_features: int, hidden_features: int, quantize: bool = True):
        super().__init__()
        Linear = BitLinear158 if quantize else nn.Linear
        self.w1 = Linear(in_features, hidden_features, bias=False)
        self.w2 = Linear(in_features, hidden_features, bias=False)
        self.w3 = Linear(hidden_features, in_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class WindowAttention(nn.Module):
    """
    Multi-head attention with optional window partitioning.

    When window_size > 0, attention is computed within local windows
    for O(n * w^2) complexity instead of O(n^2), enabling efficient
    processing of high-resolution images.

    Uses M-RoPE for 2D spatial position encoding.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 0,
        quantize: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        Linear = BitLinear158 if quantize else nn.Linear
        self.qkv = Linear(dim, 3 * dim, bias=False)
        self.proj = Linear(dim, dim, bias=False)
        self.mrope = MultimodalRoPE(self.head_dim, interleave=True)

    def _window_partition(self, x: torch.Tensor, h: int, w: int) -> tuple[torch.Tensor, int, int]:
        """Partition spatial tokens into non-overlapping windows."""
        B, N, C = x.shape
        ws = self.window_size

        # Pad to multiple of window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = x.view(B, h, w, C)
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            h, w = h + pad_h, w + pad_w
            x = x.view(B, h * w, C)

        nH, nW = h // ws, w // ws
        x = x.view(B, nH, ws, nW, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * nH * nW, ws * ws, C)
        return x, nH, nW

    def _window_unpartition(
        self, x: torch.Tensor, nH: int, nW: int, B: int, orig_h: int, orig_w: int
    ) -> torch.Tensor:
        """Reverse window partitioning."""
        ws = self.window_size
        x = x.view(B, nH, nW, ws, ws, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, nH * ws * nW * ws, -1)
        # Remove padding
        if nH * ws != orig_h or nW * ws != orig_w:
            x = x.view(B, nH * ws, nW * ws, -1)[:, :orig_h, :orig_w, :]
            x = x.reshape(B, orig_h * orig_w, -1)
        return x

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        grid_h: int | None = None,
        grid_w: int | None = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        use_window = self.window_size > 0 and grid_h is not None and grid_w is not None

        if use_window:
            x, nH, nW = self._window_partition(x, grid_h, grid_w)
            # Rebuild position IDs for windows
            ws = self.window_size
            win_pos = self.mrope._build_position_ids_image(ws, ws).to(x.device)
            cos, sin = self.mrope(win_pos, dtype=x.dtype)
            B_win = x.shape[0]
        else:
            cos, sin = self.mrope(position_ids.to(x.device), dtype=x.dtype)
            B_win = B

        qkv = self.qkv(x).reshape(B_win, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q, k, v = qkv.unbind(0)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_win, -1, C)
        out = self.proj(out)

        if use_window:
            out = self._window_unpartition(out, nH, nW, B, grid_h, grid_w)

        return out


class VisionBlock(nn.Module):
    """
    Single vision transformer block with:
    - Pre-norm (RMSNorm, from Qwen2.5-VL)
    - Window or global attention with M-RoPE
    - SwiGLU FFN
    - All linear layers optionally quantized to 1.58-bit
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        window_size: int = 0,
        quantize: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size, quantize)
        self.norm2 = nn.RMSNorm(dim)
        mlp_hidden = int(dim * mlp_ratio * 2 / 3)  # SwiGLU conventional sizing
        self.mlp = SwiGLU(dim, mlp_hidden, quantize)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        grid_h: int | None = None,
        grid_w: int | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), position_ids, grid_h, grid_w)
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """
    Image to patch embedding with dynamic resolution support.
    Converts [B, C, H, W] -> [B, num_patches, embed_dim].
    """

    def __init__(self, patch_size: int = 14, in_channels: int = 3, embed_dim: int = 1024):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, D, H/P, W/P]
        grid_h, grid_w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, D]
        return x, grid_h, grid_w


class TokenMerger(nn.Module):
    """
    2x2 spatial token merging MLP (from Qwen2-VL).

    Merges adjacent 2x2 patches into a single token, reducing the visual
    token count by 4x before feeding into the LLM backbone.
    """

    def __init__(self, vis_dim: int, llm_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(vis_dim * 4, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, x: torch.Tensor, grid_h: int, grid_w: int) -> tuple[torch.Tensor, int, int]:
        """
        Args:
            x: [B, grid_h * grid_w, vis_dim]
        Returns:
            merged: [B, (grid_h//2) * (grid_w//2), llm_dim]
        """
        B, N, D = x.shape
        # Pad if grid dimensions are odd
        pad_h = grid_h % 2
        pad_w = grid_w % 2
        if pad_h or pad_w:
            x = x.view(B, grid_h, grid_w, D)
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            grid_h += pad_h
            grid_w += pad_w
            x = x.view(B, grid_h * grid_w, D)

        x = x.view(B, grid_h // 2, 2, grid_w // 2, 2, D)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, (grid_h // 2) * (grid_w // 2), 4 * D)

        new_h, new_w = grid_h // 2, grid_w // 2
        return self.mlp(x), new_h, new_w


class QuantizedVisionEncoder(nn.Module):
    """
    Complete quantized vision encoder combining:

    - Patch embedding (full precision, following BitVLA convention)
    - Stacked ViT blocks with window attention on most layers and
      global attention on select layers (Qwen2.5-VL pattern)
    - M-RoPE 2D position encoding
    - 1.58-bit quantization on all linear layers in ViT blocks
    - 2x2 token merger MLP (full precision connector)

    Architecture choices:
    - Default: 24 layers, window attention on all except layers [5, 11, 17, 23]
      which use global attention (following Qwen2.5-VL's 4-global-layer pattern)
    - Window size 8x8 = 64 tokens per window
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1024,
        llm_dim: int = 2048,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        window_size: int = 8,
        global_layers: list[int] | None = None,
        quantize: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(patch_size, in_channels, embed_dim)

        if global_layers is None:
            # Place global attention at evenly spaced layers (Qwen2.5-VL: 4 global layers)
            global_layers = [depth // 4 - 1, depth // 2 - 1, 3 * depth // 4 - 1, depth - 1]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            ws = 0 if i in global_layers else window_size
            self.blocks.append(
                VisionBlock(embed_dim, num_heads, mlp_ratio, ws, quantize)
            )

        self.norm = nn.RMSNorm(embed_dim)
        self.merger = TokenMerger(embed_dim, llm_dim)
        self.mrope = MultimodalRoPE(embed_dim // num_heads, interleave=True)

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        """
        Args:
            pixel_values: [B, C, H, W] - supports dynamic resolution.

        Returns:
            visual_tokens: [B, num_merged_tokens, llm_dim]
            merged_h, merged_w: spatial dimensions after 2x2 merging
        """
        x, grid_h, grid_w = self.patch_embed(pixel_values)

        # Build 2D position IDs for patches
        position_ids = self.mrope._build_position_ids_image(grid_h, grid_w).to(x.device)

        for block in self.blocks:
            x = block(x, position_ids, grid_h, grid_w)

        x = self.norm(x)

        # 2x2 merge: reduce token count by 4x
        visual_tokens, merged_h, merged_w = self.merger(x, grid_h, grid_w)

        return visual_tokens, merged_h, merged_w

    def get_intermediate_features(
        self,
        pixel_values: torch.Tensor,
        layer_indices: list[int] | None = None,
    ) -> list[torch.Tensor]:
        """
        Extract features from intermediate layers for DeepStack injection
        into the LLM backbone (Qwen3-VL concept).

        Args:
            pixel_values: [B, C, H, W]
            layer_indices: Which layers to extract from. If None,
                          extracts from each quarter of the network.

        Returns:
            List of feature tensors [B, N, D] from requested layers.
        """
        x, grid_h, grid_w = self.patch_embed(pixel_values)
        position_ids = self.mrope._build_position_ids_image(grid_h, grid_w).to(x.device)

        if layer_indices is None:
            depth = len(self.blocks)
            layer_indices = [depth // 4, depth // 2, 3 * depth // 4, depth - 1]

        features = []
        for i, block in enumerate(self.blocks):
            x = block(x, position_ids, grid_h, grid_w)
            if i in layer_indices:
                features.append(self.norm(x))

        return features
