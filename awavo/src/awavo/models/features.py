from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn


class PsiEncoder(nn.Module):
    """Feature map Ψ(s,a) used by the feature-based A-GSWD variant.

    The paper assumes access to a feature vector Ψ(s,a) (see Assumption 4.2),
    and A-GSWD operates on the induced distributions in that feature space.

    This module provides a simple, task-agnostic instantiation:
        Ψ(s,a) = MLP([s,a])

    It is intentionally lightweight and can be trained jointly with the actor
    through the WVI objective.
    """

    def __init__(self, obs_dim: int, act_rep_dim: int, feature_dim: int = 32, hidden: int = 256):
        super().__init__()
        in_dim = obs_dim + act_rep_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, feature_dim),
        )

    def forward(self, s: torch.Tensor, a_rep: torch.Tensor) -> torch.Tensor:
        if s.dim() != 2 or a_rep.dim() != 2:
            raise ValueError("PsiEncoder expects batched inputs: s=(B,obs_dim), a_rep=(B,act_dim)")
        x = torch.cat([s, a_rep], dim=-1)
        return self.net(x)
