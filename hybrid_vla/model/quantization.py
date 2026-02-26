"""
BitNet-style 1.58-bit ternary quantization utilities.

Implements the quantization scheme from BitVLA where weights are constrained to
{-1, 0, +1} (ternary) using absmean scaling, and activations are quantized to 8-bit
using per-token absmax scaling. Straight-through estimator (STE) enables gradient
flow through non-differentiable quantization operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round with straight-through estimator for gradient flow."""
    return x + (x.round() - x).detach()


def weight_quant_ternary(w: torch.Tensor) -> torch.Tensor:
    """
    Absmean ternary quantization for weights -> {-1, 0, +1}.

    From BitNet b1.58: scale = mean(|w|), then round(w / scale) clamped to [-1, 1].
    """
    scale = w.abs().mean().clamp(min=1e-8)
    w_scaled = w / scale
    w_quant = ste_round(w_scaled).clamp(-1, 1)
    return w_quant * scale


def activation_quant_int8(x: torch.Tensor) -> torch.Tensor:
    """
    Per-token absmax quantization for activations -> INT8 range [-127, 127].

    Each token (last dim) is independently scaled to maximize dynamic range.
    """
    Qb = 127.0
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    x_scaled = x / scale * Qb
    x_quant = ste_round(x_scaled).clamp(-Qb, Qb)
    return x_quant / Qb * scale


class BitLinear158(nn.Linear):
    """
    Drop-in replacement for nn.Linear with BitNet 1.58-bit quantization.

    During training: weights quantized to ternary {-1,0,+1} via absmean,
    activations quantized to INT8 via per-token absmax. STE used for gradients.
    Full-precision weights maintained in optimizer for stable training.

    During inference: uses quantized weights directly for memory efficiency.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self._quantize_inference = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or not self._quantize_inference:
            w_q = weight_quant_ternary(self.weight)
            x_q = activation_quant_int8(x)
        else:
            w_q = self.weight
            x_q = activation_quant_int8(x)
        return F.linear(x_q, w_q, self.bias)

    def freeze_quantized(self):
        """Freeze weights to quantized ternary values for inference deployment."""
        with torch.no_grad():
            scale = self.weight.abs().mean().clamp(min=1e-8)
            self.weight.data = (self.weight.data / scale).round().clamp(-1, 1) * scale
        self._quantize_inference = True


class DistillationLoss(nn.Module):
    """
    Distillation-aware training loss from BitVLA.

    Combines task loss with a representation alignment loss between
    a full-precision teacher and quantized student vision encoder.

    L_total = L_task + gamma * L_align
    where L_align = MSE(student_features, teacher_features.detach())
    """

    def __init__(self, gamma: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.mse = nn.MSELoss()

    def forward(
        self,
        task_loss: torch.Tensor,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        align_loss = self.mse(student_features, teacher_features.detach())
        return task_loss + self.gamma * align_loss


def replace_linear_with_bitlinear(
    module: nn.Module,
    exclude_names: set[str] | None = None,
) -> nn.Module:
    """
    Recursively replace all nn.Linear layers with BitLinear158,
    optionally excluding layers by name (e.g. embedding projections).
    """
    exclude_names = exclude_names or set()
    for name, child in module.named_children():
        if name in exclude_names:
            continue
        if isinstance(child, nn.Linear):
            bit_linear = BitLinear158(
                child.in_features, child.out_features, bias=child.bias is not None
            )
            bit_linear.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                bit_linear.bias.data.copy_(child.bias.data)
            setattr(module, name, bit_linear)
        else:
            replace_linear_with_bitlinear(child, exclude_names)
    return module


def compute_model_size_bits(model: nn.Module) -> dict:
    """Compute effective model size assuming ternary weights = 1.58 bits each."""
    total_params = 0
    ternary_params = 0
    fp_params = 0

    for name, param in model.named_parameters():
        n = param.numel()
        total_params += n
        is_bit_linear = False
        parts = name.split(".")
        for i, part in enumerate(parts):
            parent = model
            for p in parts[:i]:
                if p.isdigit():
                    parent = parent[int(p)]
                else:
                    parent = getattr(parent, p, parent)
            if isinstance(parent, BitLinear158):
                is_bit_linear = True
                break
        if is_bit_linear and "weight" in name:
            ternary_params += n
        else:
            fp_params += n

    ternary_bits = ternary_params * 1.58
    fp_bits = fp_params * 16  # assume fp16/bf16
    total_bits = ternary_bits + fp_bits
    total_bytes = total_bits / 8

    return {
        "total_params": total_params,
        "ternary_params": ternary_params,
        "fp_params": fp_params,
        "estimated_size_mb": total_bytes / (1024 * 1024),
    }
