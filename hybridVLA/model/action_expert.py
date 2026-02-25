"""
Mixture-of-Transformers (MoT) Action Expert.

From LingBot-VLA and π0/π0.5/π0.6: a dedicated transformer branch for
action prediction that shares self-attention with the VLM backbone but
has its own feed-forward networks (FFN).

Architecture:
    For each transformer layer l:
        # Joint self-attention (shared Q/K/V projections)
        [obs_hidden, act_hidden] = SharedAttention([obs_tokens, act_tokens])

        # Modality-specific FFNs
        obs_tokens = VLM_FFN(obs_hidden)     # in the LLM backbone
        act_tokens = ActionExpert_FFN(act_hidden)  # in this module

The Action Expert processes:
- Proprioception token (current joint state)
- Past action tokens (previous chunk)
- Noised action tokens (flow matching input during training, or
  iteratively denoised during inference)
- Noise level embedding (σ for flow matching)

Key design choices:
- Shared attention enables cross-modal information flow at every layer
- Separate FFNs prevent cross-modal interference
- Knowledge insulation: obs_hidden.detach() before action FFN prevents
  action gradients from degrading VLM representations
- Blockwise causal attention: obs tokens attend bidirectionally,
  action tokens attend to all obs but only past action tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow_matching import SinusoidalTimestepEmbedding


class ActionExpertFFN(nn.Module):
    """
    SwiGLU feed-forward network for the Action Expert branch.

    Same architecture as the VLM's FFN but with separate weights.
    Initialized from VLM FFN weights for warm start (optional).
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        mlp_hidden = int(dim * mlp_ratio * 2 / 3)
        self.norm = nn.RMSNorm(dim)
        self.w1 = nn.Linear(dim, mlp_hidden, bias=False)  # gate
        self.w2 = nn.Linear(dim, mlp_hidden, bias=False)  # up
        self.w3 = nn.Linear(mlp_hidden, dim, bias=False)  # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + self.w3(F.silu(self.w1(h)) * self.w2(h))


class BlockwiseCausalMask:
    """
    Builds the blockwise causal attention mask for MoT.

    Mask rules (from LingBot-VLA):
    - Observation tokens attend to each other BIDIRECTIONALLY
    - Action tokens attend to ALL observation tokens
    - Action tokens attend to PAST action tokens only (causal)
    - Observation tokens do NOT attend to action tokens

    This prevents information leakage from future actions into
    observation representations.
    """

    @staticmethod
    def build(
        num_obs_tokens: int,
        num_action_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build the attention mask for the joint [obs | action] sequence.

        Returns:
            mask: [1, 1, total, total] additive attention mask
                  (0 = attend, -inf = block)
        """
        total = num_obs_tokens + num_action_tokens
        mask = torch.zeros(total, total, device=device)

        # Observation tokens: bidirectional (already 0 = attend)

        # Action tokens attend to all observation tokens (already 0)

        # Observation tokens do NOT attend to action tokens
        mask[:num_obs_tokens, num_obs_tokens:] = float("-inf")

        # Action tokens: causal among themselves
        action_mask = torch.triu(
            torch.ones(num_action_tokens, num_action_tokens, device=device),
            diagonal=1,
        ) * float("-inf")
        mask[num_obs_tokens:, num_obs_tokens:] = action_mask

        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, total, total]


class ActionExpertLayer(nn.Module):
    """
    One layer of the Action Expert: receives hidden states from the
    shared attention and applies an action-specific FFN.

    In the MoT architecture, this runs in parallel with the VLM's FFN
    at each transformer layer. The shared attention is computed in the
    LLM backbone; this module only handles the action-side FFN.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.ffn = ActionExpertFFN(dim, mlp_ratio)

    def forward(self, action_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_hidden: [B, N_action, D] hidden states from shared attention

        Returns:
            action_tokens: [B, N_action, D] after action-specific FFN
        """
        return self.ffn(action_hidden)


class ActionExpert(nn.Module):
    """
    Full Action Expert module for the MoT architecture.

    Contains:
    1. Action token embeddings (noised actions + σ embedding)
    2. Proprioception and past-action token projections
    3. Per-layer FFN blocks (one per VLM layer that participates in MoT)
    4. Velocity prediction head (for flow matching)
    5. Action normalization statistics

    The shared attention is computed externally (in the LLM backbone).
    This module handles action-side processing: embedding input tokens,
    applying action FFNs after each shared attention, and predicting
    the velocity field for flow matching.
    """

    def __init__(
        self,
        dim: int,
        action_dim: int = 14,
        chunk_size: int = 50,
        num_layers: int = 24,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.dim = dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

        # Embed raw action values to hidden dim
        self.action_proj = nn.Linear(action_dim, dim)
        self.action_pos_embed = nn.Embedding(chunk_size, dim)

        # Noise level embedding for flow matching
        self.sigma_embed = SinusoidalTimestepEmbedding(dim)

        # Per-layer FFN blocks
        self.layers = nn.ModuleList([
            ActionExpertLayer(dim, mlp_ratio) for _ in range(num_layers)
        ])

        # Final norm + velocity prediction head
        self.final_norm = nn.RMSNorm(dim)
        self.velocity_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, action_dim),
        )

        # Action statistics for normalization
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))

    def set_action_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """Set action normalization statistics from training data."""
        self.action_mean.copy_(mean)
        self.action_std.copy_(std.clamp(min=1e-6))

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions - self.action_mean) / self.action_std

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions * self.action_std + self.action_mean

    def embed_action_tokens(
        self,
        noised_actions: torch.Tensor,
        sigma: torch.Tensor,
        proprio_tokens: torch.Tensor | None = None,
        past_action_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Build the action expert's input token sequence.

        Sequence: [σ_embed] [proprio] [past_actions] [noised_actions]

        Args:
            noised_actions: [B, chunk_size, action_dim] noised action chunk
            sigma: [B] noise levels
            proprio_tokens: [B, 1, D] encoded proprioception (optional)
            past_action_tokens: [B, chunk_size, D] encoded past actions (optional)

        Returns:
            action_tokens: [B, N_action, D] embedded action sequence
        """
        B = noised_actions.shape[0]
        device = noised_actions.device
        parts = []

        # Sigma embedding as first token
        sigma_token = self.sigma_embed(sigma).unsqueeze(1)  # [B, 1, D]
        parts.append(sigma_token)

        # Proprioception token
        if proprio_tokens is not None:
            parts.append(proprio_tokens)

        # Past action tokens
        if past_action_tokens is not None:
            parts.append(past_action_tokens)

        # Noised action tokens with positional embedding
        pos = torch.arange(self.chunk_size, device=device)
        action_tokens = self.action_proj(noised_actions) + self.action_pos_embed(pos).unsqueeze(0)
        parts.append(action_tokens)

        return torch.cat(parts, dim=1)

    def apply_layer(self, layer_idx: int, action_hidden: torch.Tensor) -> torch.Tensor:
        """Apply the action FFN for a specific layer index."""
        if layer_idx < len(self.layers):
            return self.layers[layer_idx](action_hidden)
        return action_hidden

    def predict_velocity(self, action_hidden: torch.Tensor) -> torch.Tensor:
        """
        Predict the velocity field from the final action hidden states.

        Only uses the last `chunk_size` tokens (the noised action positions),
        stripping off the sigma/proprio/past-action prefix tokens.

        Args:
            action_hidden: [B, N_action, D] final hidden states

        Returns:
            velocity: [B, chunk_size, action_dim]
        """
        # Extract only the action chunk tokens (last chunk_size tokens)
        action_states = action_hidden[:, -self.chunk_size:]
        h = self.final_norm(action_states)
        return self.velocity_head(h)
