"""
Proprioceptive state and past-action encoding for the Action Expert.

From LingBot-VLA and π0: the action expert receives not just visual/text
context from the VLM backbone, but also the robot's current joint state
(proprioception) and the previous action chunk. This gives the action
expert direct access to the robot's kinematic state without needing to
infer it from images.

Why proprioception matters:
- Actions are relative to current state. Without joint angles, the model
  must infer arm configuration from pixel observations, which is noisy.
- Gripper state (open/closed) is critical for grasp planning but often
  occluded in camera views.
- Past actions provide momentum/velocity information for smooth trajectories.
"""

import torch
import torch.nn as nn


class ProprioceptionEncoder(nn.Module):
    """
    Encodes the robot's proprioceptive state into a single token
    for the action expert's input sequence.

    Maps raw joint state (positions, velocities, gripper) to a
    D-dimensional embedding via a 2-layer MLP.
    """

    def __init__(self, proprio_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.RMSNorm(hidden_dim)

    def forward(self, proprio_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proprio_state: [B, proprio_dim] raw joint state vector

        Returns:
            token: [B, 1, hidden_dim] proprioception token
        """
        return self.norm(self.mlp(proprio_state)).unsqueeze(1)


class PastActionEncoder(nn.Module):
    """
    Encodes the previous action chunk into tokens for the action expert.

    Each past action step is projected to an embedding, giving the action
    expert access to recent motor history for temporal coherence.
    """

    def __init__(self, action_dim: int, hidden_dim: int, chunk_size: int):
        super().__init__()
        self.proj = nn.Linear(action_dim, hidden_dim)
        self.pos_embed = nn.Embedding(chunk_size, hidden_dim)
        self.norm = nn.RMSNorm(hidden_dim)
        self.chunk_size = chunk_size

    def forward(self, past_actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            past_actions: [B, chunk_size, action_dim] previous action chunk.
                          Zero-padded if no history available.

        Returns:
            tokens: [B, chunk_size, hidden_dim] past action tokens
        """
        B = past_actions.shape[0]
        device = past_actions.device
        pos = torch.arange(self.chunk_size, device=device)
        tokens = self.proj(past_actions) + self.pos_embed(pos).unsqueeze(0)
        return self.norm(tokens)
