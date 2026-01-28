from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from awavo.algo.gswd import poly_feature_dim
from awavo.utils.torch_utils import normalize_unit


class ActorContinuous(nn.Module):
    """Gaussian policy with adaptive slicing heads (theta_e, l).

    theta_e parameterizes the defining function α(·, theta_e) used in A-GSWD.
    In the paper, α can be chosen as an odd-degree homogeneous polynomial; we
    use degree=3 by default.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, poly_degree: int = 3, theta_input_dim: int | None = None):
        super().__init__()
        self.act_dim = act_dim
        self.poly_degree = poly_degree
        theta_input_dim = act_dim if theta_input_dim is None else int(theta_input_dim)
        theta_dim = poly_feature_dim(theta_input_dim, poly_degree)

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.logstd = nn.Linear(hidden, act_dim)

        self.theta_e = nn.Linear(hidden, theta_dim)
        self.l = nn.Linear(hidden, 1)

    def forward(self, s: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.net(s)
        mu = self.mu(h)
        logstd = torch.clamp(self.logstd(h), -5.0, 2.0)
        std = torch.exp(logstd)

        theta_e = normalize_unit(self.theta_e(h))
        l = self.l(h)
        return {"mu": mu, "std": std, "theta_e": theta_e, "l": l}

    def sample(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.forward(s)
        dist = torch.distributions.Normal(out["mu"], out["std"])
        a = dist.rsample()
        logp = dist.log_prob(a).sum(-1, keepdim=True)
        return a, logp, out


class ActorDiscrete(nn.Module):
    """Categorical policy with adaptive slicing heads (theta_e, l).

    For A-GSWD we represent actions as one-hot vectors so that the defining
    function operates on a continuous vector space.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256, poly_degree: int = 3, theta_input_dim: int | None = None):
        super().__init__()
        self.n_actions = n_actions
        self.poly_degree = poly_degree
        theta_input_dim = n_actions if theta_input_dim is None else int(theta_input_dim)
        theta_dim = poly_feature_dim(theta_input_dim, poly_degree)

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.logits = nn.Linear(hidden, n_actions)

        self.theta_e = nn.Linear(hidden, theta_dim)
        self.l = nn.Linear(hidden, 1)

    def forward(self, s: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.net(s)
        logits = self.logits(h)
        theta_e = normalize_unit(self.theta_e(h))
        l = self.l(h)
        return {"logits": logits, "theta_e": theta_e, "l": l}

    @torch.no_grad()
    def act(self, s: torch.Tensor) -> int:
        """Greedy action for environment stepping."""
        out = self.forward(s)
        return int(torch.argmax(out["logits"], dim=-1).item())

    def sample(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.forward(s)
        dist = torch.distributions.Categorical(logits=out["logits"])
        a_idx = dist.sample()
        logp = dist.log_prob(a_idx).unsqueeze(-1)
        a_oh = F.one_hot(a_idx, num_classes=self.n_actions).float()
        return a_oh, logp, out
