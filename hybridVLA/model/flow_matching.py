"""
Conditional Flow Matching for continuous action generation.

From LingBot-VLA and π0/π0.5: replaces L1 action loss with a learned
velocity field that transports Gaussian noise to the ground-truth action
distribution along a linear probability path.

Why flow matching over L1:
- L1 assumes unimodal actions. If two valid grasps exist (left vs right
  approach), L1 averages them into an invalid middle trajectory.
- Flow matching models the full distribution and can sample diverse,
  valid trajectories via iterative denoising.
- Linear path flow matching is simpler than DDPM diffusion (no noise
  schedule tuning) and trains faster.

Training:
    σ ~ U(0, 1)                    # noise level
    ε ~ N(0, I)                    # Gaussian noise
    a_σ = (1 - σ)·ε + σ·a_gt      # interpolated action
    v_pred = model(a_σ, σ, obs)    # predicted velocity
    loss = ||v_pred - (a_gt - ε)||²

Inference (iterative denoising):
    a_0 ~ N(0, I)
    for i in 0..K-1:
        σ = i / K
        v = model(a_σ, σ, obs)
        a_{σ+Δ} = a_σ + v · (1/K)
    return a_1
"""

import math
import torch
import torch.nn as nn


class FlowMatchingScheduler:
    """
    Manages the noise schedule and denoising steps for flow matching.

    Supports linear and cosine schedules for the interpolation parameter σ.
    """

    def __init__(
        self,
        num_inference_steps: int = 10,
        schedule: str = "linear",
    ):
        self.num_inference_steps = num_inference_steps
        self.schedule = schedule

    def sample_sigma(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample noise levels for training. Returns σ ~ U(0, 1)."""
        return torch.rand(batch_size, device=device)

    def get_inference_sigmas(self, device: torch.device) -> torch.Tensor:
        """Get the sigma schedule for inference denoising steps."""
        K = self.num_inference_steps
        if self.schedule == "linear":
            return torch.linspace(0, 1, K + 1, device=device)
        elif self.schedule == "cosine":
            # Cosine schedule: more steps near σ=0 and σ=1
            t = torch.linspace(0, 1, K + 1, device=device)
            return 0.5 * (1 - torch.cos(t * math.pi))
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def add_noise(
        self,
        actions: torch.Tensor,
        noise: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Interpolate between noise and ground-truth actions.

        a_σ = (1 - σ) · ε + σ · a_gt

        Args:
            actions: [B, T, D] ground-truth action chunk
            noise: [B, T, D] Gaussian noise
            sigma: [B] noise levels in [0, 1]

        Returns:
            noised_actions: [B, T, D]
        """
        sigma = sigma[:, None, None]  # [B, 1, 1]
        return (1 - sigma) * noise + sigma * actions

    def get_velocity_target(
        self,
        actions: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the target velocity field: v = a_gt - ε.

        The model learns to predict this velocity, which points from
        the noise sample toward the ground-truth action.
        """
        return actions - noise


class SinusoidalTimestepEmbedding(nn.Module):
    """
    Sinusoidal embedding for the noise level σ, injected into action tokens.

    Same approach as diffusion timestep embedding: encodes the continuous
    noise level as a high-dimensional vector for the transformer to condition on.
    """

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sigma: [B] noise levels in [0, 1]

        Returns:
            embedding: [B, dim]
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=sigma.device, dtype=sigma.dtype)
            / half
        )
        args = sigma[:, None] * freqs[None, :]
        embedding = torch.cat([args.cos(), args.sin()], dim=-1)
        if self.dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return self.mlp(embedding)


class FlowMatchingLoss(nn.Module):
    """
    Flow matching training loss.

    Computes MSE between predicted and target velocity fields:
        loss = ||v_pred - (a_gt - ε)||²
    """

    def __init__(self):
        super().__init__()
        self.scheduler = FlowMatchingScheduler()

    def forward(
        self,
        model_output: torch.Tensor,
        actions: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            model_output: [B, T, D] predicted velocity from the action expert
            actions: [B, T, D] ground-truth actions (normalized)
            noise: [B, T, D] the noise that was added

        Returns:
            Scalar MSE loss.
        """
        target = self.scheduler.get_velocity_target(actions, noise)
        return (model_output - target).pow(2).mean()
