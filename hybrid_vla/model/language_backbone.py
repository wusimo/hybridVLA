"""
Quantized Language Model Backbone for HybridVLA.

A decoder-only transformer with:
- BitNet 1.58-bit quantized linear layers (from BitVLA's use of BitNet b1.58)
- M-RoPE for unified multimodal position encoding (from Qwen2-VL)
- SwiGLU FFN (from Qwen2.5-VL)
- RMSNorm (from Qwen2.5-VL)
- DeepStack visual feature injection at multiple layers (from Qwen3-VL)
- Grouped-Query Attention (GQA) for efficiency

This backbone processes interleaved visual tokens and text tokens,
producing contextual representations for both language generation
and action prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .quantization import BitLinear158
from .mrope import MultimodalRoPE, apply_rotary_emb


class GQAttention(nn.Module):
    """
    Grouped-Query Attention with M-RoPE and optional causal/bidirectional masking.

    GQA uses fewer KV heads than Q heads for memory efficiency,
    which is critical for 1.58-bit models targeting edge deployment.

    The attention mask mode switches between:
    - Causal: for autoregressive language generation
    - Bidirectional: for parallel action chunk decoding (BitVLA)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        quantize: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = dim // num_heads
        self.kv_dim = self.head_dim * self.num_kv_heads
        self.scale = self.head_dim ** -0.5
        self.num_kv_groups = num_heads // self.num_kv_heads

        Linear = BitLinear158 if quantize else nn.Linear
        self.q_proj = Linear(dim, dim, bias=True)
        self.k_proj = Linear(dim, self.kv_dim, bias=True)
        self.v_proj = Linear(dim, self.kv_dim, bias=True)
        self.o_proj = Linear(dim, dim, bias=False)

        self.mrope = MultimodalRoPE(self.head_dim, interleave=True)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal: bool = True,
    ) -> torch.Tensor:
        B, N, C = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply M-RoPE
        cos, sin = self.mrope(position_ids.to(x.device), dtype=x.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Expand KV heads for GQA
        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            k = k.reshape(B, self.num_heads, N, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            v = v.reshape(B, self.num_heads, N, self.head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if causal:
            causal_mask = torch.triu(
                torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        if attention_mask is not None:
            attn = attn + attention_mask

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.o_proj(out)


class LLMBlock(nn.Module):
    """
    Single transformer decoder block with:
    - Pre-norm RMSNorm
    - GQA with M-RoPE
    - SwiGLU FFN
    - Optional DeepStack visual feature injection
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        mlp_ratio: float = 4.0,
        quantize: bool = True,
        deepstack_inject: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = GQAttention(dim, num_heads, num_kv_heads, quantize)
        self.norm2 = nn.RMSNorm(dim)

        mlp_hidden = int(dim * mlp_ratio * 2 / 3)
        Linear = BitLinear158 if quantize else nn.Linear
        self.gate_proj = Linear(dim, mlp_hidden, bias=False)
        self.up_proj = Linear(dim, mlp_hidden, bias=False)
        self.down_proj = Linear(mlp_hidden, dim, bias=False)

        self.deepstack_inject = deepstack_inject
        if deepstack_inject:
            # Learnable gate for blending injected visual features
            self.vis_gate = nn.Parameter(torch.zeros(1))
            self.vis_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal: bool = True,
        visual_injection: torch.Tensor | None = None,
        visual_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self-attention
        x = x + self.attn(self.norm1(x), position_ids, attention_mask, causal)

        # DeepStack: inject multi-level visual features (Qwen3-VL)
        # visual_injection may have fewer tokens than x (only visual positions),
        # so we pad with zeros to match the full sequence length.
        if self.deepstack_inject and visual_injection is not None:
            gate = torch.sigmoid(self.vis_gate)
            vis = self.vis_proj(visual_injection)
            # Pad visual injection to match x's sequence length if needed
            if vis.shape[1] < x.shape[1]:
                pad = torch.zeros(
                    vis.shape[0], x.shape[1] - vis.shape[1], vis.shape[2],
                    device=vis.device, dtype=vis.dtype,
                )
                vis = torch.cat([vis, pad], dim=1)
            elif vis.shape[1] > x.shape[1]:
                vis = vis[:, :x.shape[1]]
            if visual_mask is not None:
                mask = visual_mask
                if mask.shape[1] < vis.shape[1]:
                    mask_pad = torch.zeros(
                        mask.shape[0], vis.shape[1] - mask.shape[1],
                        device=mask.device, dtype=mask.dtype,
                    )
                    mask = torch.cat([mask, mask_pad], dim=1)
                vis = vis * mask[:, :vis.shape[1]].unsqueeze(-1)
            x = x + gate * vis

        # SwiGLU FFN
        h = self.norm2(x)
        x = x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return x


class QuantizedLLMBackbone(nn.Module):
    """
    Full quantized decoder-only transformer backbone.

    Features:
    - 1.58-bit quantized linear layers throughout
    - M-RoPE multimodal position encoding
    - GQA for memory efficiency
    - DeepStack injection at configurable layers
    - Supports both causal (language) and bidirectional (action) modes

    Architecture follows the Qwen2 family structure, scaled down and
    quantized following the BitVLA approach.
    """

    def __init__(
        self,
        vocab_size: int = 151936,  # Qwen2 tokenizer size
        dim: int = 2048,
        depth: int = 24,
        num_heads: int = 16,
        num_kv_heads: int = 4,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 4096,
        quantize: bool = True,
        deepstack_layers: list[int] | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size

        # Token embedding stays full precision (following BitVLA/BitNet convention)
        self.tok_emb = nn.Embedding(vocab_size, dim)

        if deepstack_layers is None:
            # Inject at 4 evenly-spaced layers
            deepstack_layers = [depth // 4, depth // 2, 3 * depth // 4, depth - 1]
        self.deepstack_layers = deepstack_layers

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                LLMBlock(
                    dim, num_heads, num_kv_heads, mlp_ratio,
                    quantize=quantize,
                    deepstack_inject=(i in deepstack_layers),
                )
            )

        self.norm = nn.RMSNorm(dim)

        # LM head stays full precision (following BitVLA/BitNet convention)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying between embedding and LM head
        self.lm_head.weight = self.tok_emb.weight

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        causal: bool = True,
        visual_injections: list[torch.Tensor] | None = None,
        visual_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: [B, N] token IDs (for text tokens)
            inputs_embeds: [B, N, D] pre-computed embeddings (for mixed visual+text)
            position_ids: [N, 3] M-RoPE position IDs
            attention_mask: [B, 1, N, N] additive mask
            causal: Whether to apply causal attention mask
            visual_injections: List of [B, N, D] features for DeepStack injection.
                              Length should match number of deepstack layers.
            visual_mask: [B, N] binary mask indicating visual token positions

        Returns:
            logits: [B, N, vocab_size] language model logits
            hidden: [B, N, D] final hidden states (for action head)
        """
        if inputs_embeds is None:
            x = self.tok_emb(input_ids)
        else:
            x = inputs_embeds

        if position_ids is None:
            N = x.shape[1]
            position_ids = torch.arange(N, device=x.device).unsqueeze(-1).expand(-1, 3)

        ds_idx = 0
        for i, layer in enumerate(self.layers):
            vis_inj = None
            if layer.deepstack_inject and visual_injections is not None and ds_idx < len(visual_injections):
                vis_inj = visual_injections[ds_idx]
                ds_idx += 1

            x = layer(x, position_ids, attention_mask, causal, vis_inj, visual_mask)

        hidden = self.norm(x)
        logits = self.lm_head(hidden)
        return logits, hidden
