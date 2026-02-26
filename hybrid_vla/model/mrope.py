"""
Multimodal Rotary Position Embedding (M-RoPE) from Qwen2-VL.

Decomposes RoPE dimensions into three components:
- Temporal: frame index (constant=0 for images, incrementing for video frames)
- Height: vertical patch position in the image grid
- Width: horizontal patch position in the image grid

For text tokens, all three components receive identical position IDs,
making M-RoPE equivalent to standard 1D RoPE.

Qwen3-VL improvement: interleaves t/h/w components across frequency bands
rather than assigning contiguous blocks, improving temporal reasoning.
"""

import torch
import torch.nn as nn
import math


def _compute_rope_freqs(dim: int, base: float = 10000.0, device: str = "cpu") -> torch.Tensor:
    """Compute rotary embedding frequency bands: theta_i = base^(-2i/d)."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    return freqs


class MultimodalRoPE(nn.Module):
    """
    Multimodal Rotary Position Embedding supporting text, image, and video.

    Each attention head dimension is split into three equal parts for
    temporal, height, and width position encoding. For text, all three
    parts use the same sequential position ID.

    Args:
        head_dim: Dimension per attention head (must be divisible by 6
                  since we need pairs for sin/cos across 3 components).
        base: RoPE frequency base.
        interleave: If True, interleave t/h/w across frequency bands
                    (Qwen3-VL style) instead of contiguous blocks.
    """

    def __init__(self, head_dim: int, base: float = 10000.0, interleave: bool = True):
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
        self.head_dim = head_dim
        self.base = base
        half = head_dim // 2

        if head_dim % 6 == 0:
            # Equal split across t/h/w
            sec = half // 3
            self.sections = [sec, sec, sec]
            self.interleave = interleave
        else:
            # Unequal split: fewer bands for temporal, more for spatial
            t = half // 2
            h = (half - t) // 2
            w = half - t - h
            self.sections = [t, h, w]
            self.interleave = False  # can't interleave unequal sizes
        self.mrope_sections = ['freqs_0', 'freqs_1', 'freqs_2']
        # Backward-compat aliases
        self.component_dim = self.sections[0] * 2
        self.half_component = self.sections[0]

        for i, sec in enumerate(self.sections):
            self.register_buffer(f"freqs_{i}", _compute_rope_freqs(sec * 2, base), persistent=False)

    def _build_position_ids_text(self, seq_len: int, offset: int = 0) -> torch.Tensor:
        """Build position IDs for text: shape [seq_len, 3] with identical t/h/w."""
        pos = torch.arange(offset, offset + seq_len)
        return pos.unsqueeze(-1).expand(-1, 3)  # [seq_len, 3]

    def _build_position_ids_image(
        self,
        grid_h: int,
        grid_w: int,
        temporal_id: int = 0,
        offset: int = 0,
    ) -> torch.Tensor:
        """
        Build position IDs for image patches: shape [grid_h * grid_w, 3].

        Temporal component is constant (single frame).
        Height/width components are spatial grid coordinates.
        """
        h_ids = torch.arange(grid_h).unsqueeze(1).expand(-1, grid_w).reshape(-1)
        w_ids = torch.arange(grid_w).unsqueeze(0).expand(grid_h, -1).reshape(-1)
        t_ids = torch.full_like(h_ids, temporal_id)
        return torch.stack([t_ids + offset, h_ids + offset, w_ids + offset], dim=-1)

    def _build_position_ids_video(
        self,
        num_frames: int,
        grid_h: int,
        grid_w: int,
        offset: int = 0,
    ) -> torch.Tensor:
        """
        Build position IDs for video: shape [num_frames * grid_h * grid_w, 3].

        Temporal component increments per frame.
        """
        ids_list = []
        for t in range(num_frames):
            frame_ids = self._build_position_ids_image(grid_h, grid_w, temporal_id=t, offset=offset)
            ids_list.append(frame_ids)
        return torch.cat(ids_list, dim=0)

    def forward(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary embeddings from position IDs.

        Args:
            position_ids: [seq_len, 3] tensor with (temporal, height, width) positions.
            dtype: Output dtype.

        Returns:
            cos, sin: [seq_len, head_dim] rotary embeddings.
        """
        device = position_ids.device

        cos_parts = []
        sin_parts = []
        for i in range(3):
            pos = position_ids[:, i].float().unsqueeze(-1)  # [seq_len, 1]
            angles = pos * getattr(self, self.mrope_sections[i]).unsqueeze(0)  # [seq_len, section_size]
            cos_parts.append(angles.cos())
            sin_parts.append(angles.sin())

        if self.interleave:
            # Interleave t/h/w across frequency bands (Qwen3-VL style)
            cos_out = torch.stack(cos_parts, dim=-1).reshape(-1, self.head_dim // 2)
            sin_out = torch.stack(sin_parts, dim=-1).reshape(-1, self.head_dim // 2)
        else:
            # Contiguous blocks (Qwen2-VL style)
            cos_out = torch.cat(cos_parts, dim=-1)  # [seq_len, head_dim // 2]
            sin_out = torch.cat(sin_parts, dim=-1)

        # Duplicate for full head_dim (pairs of [cos, cos] for complex rotation)
        cos_out = torch.cat([cos_out, cos_out], dim=-1).to(dtype)
        sin_out = torch.cat([sin_out, sin_out], dim=-1).to(dtype)
        return cos_out, sin_out


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensor.

    Args:
        x: [batch, heads, seq_len, head_dim]
        cos: [seq_len, head_dim]
        sin: [seq_len, head_dim]

    Returns:
        Rotated tensor with same shape as x.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[:x.shape[2]].unsqueeze(0).unsqueeze(0)  # [1, 1, seq, head_dim]
    sin = sin[:x.shape[2]].unsqueeze(0).unsqueeze(0)

    cos1, cos2 = cos[..., :half], cos[..., half:]
    sin1, sin2 = sin[..., :half], sin[..., half:]

    out1 = x1 * cos1 - x2 * sin1
    out2 = x2 * cos2 + x1 * sin2
    return torch.cat([out1, out2], dim=-1)


def build_multimodal_position_ids(
    text_len: int,
    image_grids: list[tuple[int, int]] | None = None,
    video_specs: list[tuple[int, int, int]] | None = None,
    modality_order: list[str] | None = None,
) -> torch.Tensor:
    """
    Build concatenated position IDs for a mixed-modality sequence.

    Args:
        text_len: Number of text tokens.
        image_grids: List of (grid_h, grid_w) for each image.
        video_specs: List of (num_frames, grid_h, grid_w) for each video.
        modality_order: Order of modalities, e.g. ["text", "image_0", "text", "video_0"].
                        If None, defaults to images first, then text.

    Returns:
        position_ids: [total_seq_len, 3] tensor.
    """
    image_grids = image_grids or []
    video_specs = video_specs or []

    if modality_order is None:
        modality_order = []
        for i in range(len(image_grids)):
            modality_order.append(f"image_{i}")
        for i in range(len(video_specs)):
            modality_order.append(f"video_{i}")
        modality_order.append("text")

    mrope = MultimodalRoPE(head_dim=6, interleave=False)  # minimal for ID building
    parts = []
    offset = 0
    text_consumed = 0
    img_idx = 0
    vid_idx = 0

    for mod in modality_order:
        if mod == "text":
            ids = mrope._build_position_ids_text(text_len - text_consumed, offset=offset)
            parts.append(ids)
            offset += text_len - text_consumed
            text_consumed = text_len
        elif mod.startswith("image_"):
            i = int(mod.split("_")[1])
            gh, gw = image_grids[i]
            ids = mrope._build_position_ids_image(gh, gw, offset=offset)
            parts.append(ids)
            offset = max(offset + gh, offset + gw)
        elif mod.startswith("video_"):
            i = int(mod.split("_")[1])
            nf, gh, gw = video_specs[i]
            ids = mrope._build_position_ids_video(nf, gh, gw, offset=offset)
            parts.append(ids)
            offset = max(offset + nf, offset + gh, offset + gw)

    return torch.cat(parts, dim=0)
