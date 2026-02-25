"""
Depth perception via distillation from a pretrained monocular depth model.

From LingBot-VLA: integrates metric depth understanding by aligning visual
tokens with features from a pretrained depth estimator (Depth Anything V2
or LingBot-Depth). The depth model is frozen — we only train a small
projection layer and distillation alignment loss.

Why depth matters for manipulation:
- RGB alone makes depth ambiguous (is the object 10cm or 30cm away?)
- Grasping requires precise depth to position the gripper correctly.
- LingBot-VLA showed +2.3% absolute SR improvement from depth integration.

Two integration modes:
- "token": Extract depth features, project to VLM dim, add to observation
  sequence. More expressive but adds tokens. (Recommended)
- "channel": Render depth map, concatenate as 4th channel to RGB. Simpler
  but the ViT patch embedding needs modification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthTokenProjector(nn.Module):
    """
    Projects depth model features into the VLM's token space.

    Takes intermediate features from a frozen depth encoder, pools them
    spatially, and projects to the VLM hidden dimension. The resulting
    depth tokens are added to the observation sequence.
    """

    def __init__(
        self,
        depth_dim: int = 384,
        output_dim: int = 2048,
        num_tokens: int = 16,
    ):
        super().__init__()
        self.num_tokens = num_tokens

        # Adaptive pool to fixed number of tokens regardless of depth map size
        self.pool = nn.AdaptiveAvgPool1d(num_tokens)

        # Project depth features to VLM dimension
        self.proj = nn.Sequential(
            nn.Linear(depth_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.norm = nn.RMSNorm(output_dim)

    def forward(self, depth_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth_features: [B, N_depth, depth_dim] from frozen depth encoder

        Returns:
            depth_tokens: [B, num_tokens, output_dim]
        """
        # Pool to fixed token count: [B, N, D] -> [B, D, N] -> pool -> [B, D, K]
        x = depth_features.transpose(1, 2)
        x = self.pool(x).transpose(1, 2)  # [B, num_tokens, depth_dim]
        return self.norm(self.proj(x))


class DepthEncoder(nn.Module):
    """
    Wrapper around a frozen pretrained depth model + trainable projector.

    Loads a depth estimation model (e.g., Depth Anything V2) and extracts
    intermediate features for each camera view. The features are then
    projected to the VLM's token space via DepthTokenProjector.

    The depth model itself is frozen — only the projector trains.
    """

    def __init__(
        self,
        depth_dim: int = 384,
        output_dim: int = 2048,
        num_tokens_per_view: int = 16,
    ):
        super().__init__()
        self.depth_dim = depth_dim
        self.projector = DepthTokenProjector(depth_dim, output_dim, num_tokens_per_view)

        # The actual depth model backbone is loaded externally via
        # pretrained_loader.load_depth_model() and set as self.backbone.
        # We keep it as None here and set it during initialization.
        self.backbone: nn.Module | None = None

    def set_backbone(self, backbone: nn.Module):
        """Set the frozen depth model backbone."""
        self.backbone = backbone
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    @torch.no_grad()
    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract depth features from the frozen backbone.

        Args:
            pixel_values: [B, C, H, W] RGB image

        Returns:
            features: [B, N, depth_dim]
        """
        if self.backbone is None:
            # Fallback: return zero features (useful for testing without depth model)
            B = pixel_values.shape[0]
            return torch.zeros(
                B, 64, self.depth_dim,
                device=pixel_values.device, dtype=pixel_values.dtype,
            )

        self.backbone.eval()
        # Use the backbone's vision encoder to get intermediate features
        # Compatible with DPT/DepthAnything style models from HuggingFace
        outputs = self.backbone(pixel_values, output_hidden_states=True)
        # Use the last hidden state
        if hasattr(outputs, "last_hidden_state"):
            features = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states:
            features = outputs.hidden_states[-1]
        else:
            # Fallback: just use whatever the model returns
            features = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        # Ensure correct dim
        if features.shape[-1] != self.depth_dim:
            # Simple linear projection if dims don't match
            if not hasattr(self, "_dim_proj"):
                self._dim_proj = nn.Linear(
                    features.shape[-1], self.depth_dim, bias=False
                ).to(features.device)
                self._dim_proj.requires_grad_(False)
            features = self._dim_proj(features)

        return features

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Full forward: extract depth features and project to VLM tokens.

        Args:
            pixel_values: [B, C, H, W] RGB image

        Returns:
            depth_tokens: [B, num_tokens, output_dim]
        """
        features = self.extract_features(pixel_values)
        return self.projector(features)


class DepthDistillationLoss(nn.Module):
    """
    Alignment loss between VLM visual tokens and depth tokens.

    Encourages the VLM's visual representations to encode depth
    information by aligning them with the depth encoder's features
    via a projection + MSE loss.
    """

    def __init__(self, vis_dim: int, depth_dim: int, gamma: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.align_proj = nn.Linear(vis_dim, depth_dim, bias=False)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        depth_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual_tokens: [B, N_vis, vis_dim] from VLM
            depth_features: [B, N_depth, depth_dim] from depth encoder

        Returns:
            Scalar alignment loss.
        """
        # Pool both to same length
        min_len = min(visual_tokens.shape[1], depth_features.shape[1])
        vis = F.adaptive_avg_pool1d(
            visual_tokens.transpose(1, 2), min_len
        ).transpose(1, 2)
        dep = F.adaptive_avg_pool1d(
            depth_features.transpose(1, 2), min_len
        ).transpose(1, 2)

        vis_proj = self.align_proj(vis)
        return self.gamma * F.mse_loss(vis_proj, dep.detach())
