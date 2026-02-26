"""
Action Head with Chunking and Parallel Decoding.

From BitVLA / OpenVLA-OFT:
- Replaces autoregressive token-by-token action generation with parallel
  decoding of an entire action trajectory ("action chunk") in one forward pass.
- Uses bidirectional attention mask (not causal) for the action tokens so
  all chunk positions can attend to each other.
- L1 loss between predicted and ground-truth continuous action trajectories.

This enables real-time control throughput since we don't need sequential
decoding steps for each action dimension.

Action space: 7-DoF robot manipulation (x, y, z, roll, pitch, yaw, gripper)
configurable to other action spaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionChunkHead(nn.Module):
    """
    Parallel action chunk prediction head.

    Takes hidden states from the LLM backbone and projects them
    to continuous robot action trajectories. Predicts an entire
    chunk of future actions simultaneously.

    Architecture:
    1. Learnable action query tokens (one per future timestep)
    2. Cross-attention from action queries to LLM hidden states
    3. MLP projection to continuous action space
    4. Optional action normalization/denormalization

    Args:
        dim: Hidden dimension from LLM backbone.
        action_dim: Dimensionality of the action space (default: 7 for 6-DoF + gripper).
        chunk_size: Number of future timesteps to predict simultaneously.
        num_heads: Attention heads for cross-attention.
    """

    def __init__(
        self,
        dim: int,
        action_dim: int = 7,
        chunk_size: int = 10,
        num_heads: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.scale = self.head_dim ** -0.5

        # Learnable action query tokens
        self.action_queries = nn.Parameter(torch.randn(1, chunk_size, dim) * 0.02)

        # Positional encoding for chunk timesteps
        self.chunk_pos_embed = nn.Embedding(chunk_size, dim)

        # Cross-attention: action queries attend to LLM hidden states
        self.cross_q = nn.Linear(dim, dim, bias=False)
        self.cross_k = nn.Linear(dim, dim, bias=False)
        self.cross_v = nn.Linear(dim, dim, bias=False)
        self.cross_out = nn.Linear(dim, dim, bias=False)
        self.cross_norm_q = nn.RMSNorm(dim)
        self.cross_norm_kv = nn.RMSNorm(dim)

        # Self-attention among action queries (bidirectional for parallel decoding)
        self.self_q = nn.Linear(dim, dim, bias=False)
        self.self_k = nn.Linear(dim, dim, bias=False)
        self.self_v = nn.Linear(dim, dim, bias=False)
        self.self_out = nn.Linear(dim, dim, bias=False)
        self.self_norm = nn.RMSNorm(dim)

        # FFN after attention
        self.ffn_norm = nn.RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

        # Final projection to action space
        self.action_proj = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, action_dim),
        )

        # Action statistics for normalization (set during training)
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))

    def set_action_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """Set action normalization statistics from training data."""
        self.action_mean.copy_(mean)
        self.action_std.copy_(std.clamp(min=1e-6))

    def _cross_attention(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, Nq, D = queries.shape
        Nk = context.shape[1]

        q = self.cross_q(self.cross_norm_q(queries)).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.cross_k(self.cross_norm_kv(context)).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.cross_v(self.cross_norm_kv(context)).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, Nq, D)
        return self.cross_out(out)

    def _self_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Bidirectional self-attention among action queries."""
        B, N, D = x.shape

        h = self.self_norm(x)
        q = self.self_q(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.self_k(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.self_v(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Bidirectional: no causal mask
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.self_out(out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predict an action chunk from LLM hidden states.

        Args:
            hidden_states: [B, N, D] from LLM backbone (full sequence)
            context_mask: [B, N] optional mask for valid context positions

        Returns:
            actions: [B, chunk_size, action_dim] predicted action trajectory
                     in normalized space. Call denormalize() for real actions.
        """
        B = hidden_states.shape[0]
        device = hidden_states.device

        # Initialize action queries with positional encoding
        chunk_pos = torch.arange(self.chunk_size, device=device)
        queries = self.action_queries.expand(B, -1, -1) + self.chunk_pos_embed(chunk_pos).unsqueeze(0)

        # Cross-attend to LLM context
        queries = queries + self._cross_attention(queries, hidden_states)

        # Bidirectional self-attention among action queries
        queries = queries + self._self_attention(queries)

        # FFN
        queries = queries + self.ffn(self.ffn_norm(queries))

        # Project to action space
        actions = self.action_proj(queries)  # [B, chunk_size, action_dim]

        return actions

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize raw actions using stored statistics."""
        return (actions - self.action_mean) / self.action_std

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Denormalize predicted actions to real action space."""
        return actions * self.action_std + self.action_mean

    def compute_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        L1 loss between predicted and target action chunks.

        Args:
            predicted: [B, chunk_size, action_dim] model predictions
            target: [B, chunk_size, action_dim] ground truth (already normalized)

        Returns:
            Scalar L1 loss.
        """
        return F.l1_loss(predicted, target)
