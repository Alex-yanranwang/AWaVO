from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn

class QuantileCritic(nn.Module):
    """Quantile critics for reward and multiple constraints.

    Input: concat([obs, action_rep])
      - continuous: action_rep = action vector
      - discrete: action_rep = one-hot
    Output:
      - reward quantiles: (B, N)
      - constraint quantiles: (B, C, N)
    """
    def __init__(self, obs_dim: int, act_rep_dim: int, num_constraints: int, hidden: int = 256, n_quantiles: int = 32):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.num_constraints = num_constraints
        in_dim = obs_dim + act_rep_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head_r = nn.Linear(hidden, n_quantiles)
        self.head_g = nn.Linear(hidden, num_constraints * n_quantiles)

    def forward(self, s: torch.Tensor, a_rep: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = torch.cat([s, a_rep], dim=-1)
        h = self.net(x)
        qr = self.head_r(h)
        qg = self.head_g(h).view(-1, self.num_constraints, self.n_quantiles)
        return {"qr": qr, "qg": qg}

    @torch.no_grad()
    def mean_values(self, s: torch.Tensor, a_rep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(s, a_rep)
        jr = out["qr"].mean(dim=-1)    # (B,)
        jg = out["qg"].mean(dim=-1)    # (B,C)
        return jr, jg
