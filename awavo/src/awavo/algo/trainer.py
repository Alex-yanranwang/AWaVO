from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from awavo.dist.quantile import quantile_huber_loss
from awavo.models.actors import ActorContinuous, ActorDiscrete
from awavo.models.critics import QuantileCritic
from awavo.models.features import PsiEncoder
from awavo.algo.gswd import a_gswd
from awavo.algo.optimality import optimality_weights
from awavo.utils.torch_utils import to_torch

@dataclass
class AWaVOConfig:
    gamma: float = 0.99
    tau_c: float = 0.5
    n_quantiles: int = 32
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    critic_tau: float = 0.005
    wvi_coef: float = 0.1
    k_wass: int = 2
    beta_r: float = 1.0
    beta_g: float = 5.0
    poly_degree: int = 3
    use_psi: bool = True
    feature_dim: int = 32
    psi_hidden: int = 256
    # Trajectory-level WVI: build p(O|τ) from short rollout prefixes.
    traj_horizon: int = 4
    traj_bootstrap: bool = False
    grad_clip: float = 1.0

class AWaVOTrainer:
    """AWaVO reference trainer.

    - Distributional quantile critics for reward and constraints
    - ORPO-DR direction choice
    - WVI loss using A-GSWD between policy samples and optimality-resampled samples
    """
    def __init__(
        self,
        obs_dim: int,
        act_space: str,                 # "continuous" or "discrete"
        act_dim_or_n: int,
        num_constraints: int,
        b: np.ndarray,
        device: torch.device,
        cfg: AWaVOConfig,
    ):
        self.device = device
        self.cfg = cfg
        self.b = to_torch(b.astype(np.float32), device)

        # Action representation dimension used by critics and Ψ(s,a).
        # - continuous: raw action vector
        # - discrete: one-hot vector
        self.act_rep_dim = act_dim_or_n

        # Feature map Ψ(s,a) for the feature-based A-GSWD variant (Assumption 4.2).
        self.psi: Optional[PsiEncoder] = None
        theta_input_dim = self.act_rep_dim
        if self.cfg.use_psi:
            self.psi = PsiEncoder(obs_dim, self.act_rep_dim, feature_dim=self.cfg.feature_dim, hidden=self.cfg.psi_hidden).to(device)
            theta_input_dim = self.cfg.feature_dim

        if act_space == "continuous":
            self.actor = ActorContinuous(
                obs_dim, act_dim_or_n, poly_degree=self.cfg.poly_degree, theta_input_dim=theta_input_dim
            ).to(device)
        elif act_space == "discrete":
            self.actor = ActorDiscrete(
                obs_dim, act_dim_or_n, poly_degree=self.cfg.poly_degree, theta_input_dim=theta_input_dim
            ).to(device)
        else:
            raise ValueError(f"Unknown act_space={act_space}")

        self.critic = QuantileCritic(obs_dim, self.act_rep_dim, num_constraints, n_quantiles=cfg.n_quantiles).to(device)
        self.critic_tgt = QuantileCritic(obs_dim, self.act_rep_dim, num_constraints, n_quantiles=cfg.n_quantiles).to(device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        actor_params = list(self.actor.parameters())
        if self.psi is not None:
            actor_params += list(self.psi.parameters())
        self.opt_actor = torch.optim.Adam(actor_params, lr=cfg.actor_lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.taus = (torch.arange(cfg.n_quantiles, device=device, dtype=torch.float32) + 0.5) / cfg.n_quantiles

        self.act_space = act_space

    @torch.no_grad()
    def act(self, s_np: np.ndarray) -> np.ndarray | int:
        s = to_torch(s_np, self.device).unsqueeze(0)
        if self.act_space == "continuous":
            a, _, _ = self.actor.sample(s)
            return a.squeeze(0).cpu().numpy()
        else:
            # return discrete action index
            return self.actor.act(s)

    def _soft_update(self):
        tau = self.cfg.critic_tau
        for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
            pt.data.mul_(1 - tau).add_(tau * p.data)

    def update(self, batch, *, rb=None) -> Dict[str, float]:
        """Update from a list of Transition objects.

        If `rb` (ReplayBuffer) is provided and supports trajectory sampling,
        we build p(O|τ) weights from short rollout prefixes (trajectory-level
        view) for the WVI objective.
        """
        s = to_torch(np.stack([t.s for t in batch]), self.device)
        r = to_torch(np.array([t.r for t in batch], dtype=np.float32), self.device)
        g = to_torch(np.stack([t.g for t in batch]).astype(np.float32), self.device)
        s2 = to_torch(np.stack([t.s2 for t in batch]), self.device)
        done = to_torch(np.array([t.done for t in batch], dtype=np.float32), self.device)

        # action representation: for discrete, store one-hot in buffer
        a_rep = to_torch(np.stack([t.a for t in batch]).astype(np.float32), self.device)

        # -----------------------
        # Critic update (distributional TD)
        # -----------------------
        with torch.no_grad():
            a2_rep, _, _ = self.actor.sample(s2)  # already rep (continuous vector or one-hot)
            out_next = self.critic_tgt(s2, a2_rep)
            qr_next = out_next["qr"]
            qg_next = out_next["qg"]

            y_r = r.unsqueeze(-1) + self.cfg.gamma * (1.0 - done).unsqueeze(-1) * qr_next
            y_g = g.unsqueeze(-1) + self.cfg.gamma * (1.0 - done).unsqueeze(-1).unsqueeze(-1) * qg_next

        out = self.critic(s, a_rep)
        loss_r = quantile_huber_loss(out["qr"], y_r, self.taus)
        loss_g = 0.0
        for i in range(out["qg"].shape[1]):
            loss_g = loss_g + quantile_huber_loss(out["qg"][:, i, :], y_g[:, i, :], self.taus)
        loss_critic = loss_r + loss_g

        self.opt_critic.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
        self.opt_critic.step()

        # -----------------------
        # Actor update: ORPO-DR + WVI(A-GSWD)
        # -----------------------
        a_pi_rep, _, aux = self.actor.sample(s)
        jr, jg = self.critic.mean_values(s, a_pi_rep)

        feasible = (jg <= (self.b.view(1, -1) + self.cfg.tau_c)).all(dim=-1)

        # ORPO-DR direction choice (Algorithm 1 style):
        # feasible -> maximize reward (minimize -jr), else minimize constraint return (minimize sum(jg))
        loss_orpo = torch.where(feasible, -jr, jg.sum(dim=-1)).mean()

        # WVI(A-GSWD): match q_theta(·) against a proxy for p(O|τ)
        # Trajectory-level variant: estimate r_e(τ), g_e(τ) from short rollout prefixes.
        if rb is not None and hasattr(rb, "sample_trajectories") and self.cfg.traj_horizon > 0:
            traj = rb.sample_trajectories(
                batch_size=len(batch),
                horizon=self.cfg.traj_horizon,
                gamma=self.cfg.gamma,
                bootstrap=self.cfg.traj_bootstrap,
            )

            s0 = to_torch(traj["s0"], self.device)
            a0 = to_torch(traj["a0"], self.device)
            r_e = to_torch(traj["r_e"], self.device)
            g_e = to_torch(traj["g_e"], self.device)

            with torch.no_grad():
                w = optimality_weights(r_e, g_e, beta_r=self.cfg.beta_r, beta_g=self.cfg.beta_g)
                w = (w / (w.sum() + 1e-8)).cpu().numpy()
                idx = np.random.choice(len(batch), size=len(batch), replace=True, p=w)

            # x: current policy samples at s0
            a_pi0_rep, _, aux0 = self.actor.sample(s0)
            theta_e = aux0["theta_e"]
            l = aux0["l"]

            # y: behavior start-actions from high-likelihood trajectory prefixes
            s_star = s0[idx]
            a_star = a0[idx].detach()

            if self.psi is not None:
                x = self.psi(s0, a_pi0_rep)
                y = self.psi(s_star, a_star)
            else:
                x, y = a_pi0_rep, a_star
        else:
            # Fallback: batch-level proxy using one-step value estimates.
            with torch.no_grad():
                w = optimality_weights(jr, jg, beta_r=self.cfg.beta_r, beta_g=self.cfg.beta_g)
                w = (w / (w.sum() + 1e-8)).cpu().numpy()
                idx = np.random.choice(len(batch), size=len(batch), replace=True, p=w)

            a_star = a_pi_rep[idx].detach()
            theta_e = aux["theta_e"]
            l = aux["l"]

            if self.psi is not None:
                x = self.psi(s, a_pi_rep)
                y = self.psi(s, a_star)
            else:
                x, y = a_pi_rep, a_star

        loss_wvi = a_gswd(x, y, theta_e, l, k=self.cfg.k_wass, degree=self.cfg.poly_degree)
        loss_actor = loss_orpo + self.cfg.wvi_coef * loss_wvi

        self.opt_actor.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
        self.opt_actor.step()

        self._soft_update()

        return {
            "loss/critic": float(loss_critic.item()),
            "loss/actor": float(loss_actor.item()),
            "loss/orpo": float(loss_orpo.item()),
            "loss/wvi": float(loss_wvi.item()),
            "stats/jr_mean": float(jr.mean().item()),
            "stats/jg_sum_mean": float(jg.sum(dim=-1).mean().item()),
            "stats/feasible_rate": float(feasible.float().mean().item()),
        }
