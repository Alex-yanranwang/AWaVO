from __future__ import annotations

import torch


@torch.no_grad()
def optimality_likelihood(
    r_hat: torch.Tensor,
    g_hat: torch.Tensor,
    *,
    beta_r: float = 1.0,
    beta_g: float = 1.0,
) -> torch.Tensor:
    """Compute an explicit optimality likelihood p(O|τ).

    The paper introduces binary optimality variables O = {O_r, O_{g_i}} and the
    optimality likelihood p(O|τ). A standard instantiation (consistent with
    probabilistic optimality in control) is:

        p(O_r=1 | τ)  ∝ exp(beta_r * r_e(τ))
        p(O_{g_i}=1 | τ) ∝ exp(-beta_g * g_{e,i}(τ))

    and we combine them as:
        p(O|τ) ∝ p(O_r|τ) * Π_i p(O_{g_i}|τ)

    Args:
        r_hat: (B,) estimated accumulated reward r_e(τ)
        g_hat: (B, C) estimated accumulated utilities/costs g_{e,i}(τ)
        beta_r, beta_g: temperatures

    Returns:
        p: (B,) unnormalized likelihood values (non-negative)
    """
    if r_hat.dim() != 1:
        raise ValueError(f"r_hat must be (B,), got {tuple(r_hat.shape)}")
    if g_hat.dim() != 2:
        raise ValueError(f"g_hat must be (B,C), got {tuple(g_hat.shape)}")

    # Stabilize exponentials by subtracting max per batch.
    log_p_r = beta_r * r_hat
    log_p_g = -beta_g * g_hat.sum(dim=-1)
    log_p = log_p_r + log_p_g
    log_p = log_p - log_p.max()
    return torch.exp(log_p)


@torch.no_grad()
def optimality_weights(
    r_hat: torch.Tensor,
    g_hat: torch.Tensor,
    *,
    beta_r: float = 1.0,
    beta_g: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalized resampling weights proportional to p(O|τ)."""
    p = optimality_likelihood(r_hat, g_hat, beta_r=beta_r, beta_g=beta_g)
    return p / (p.sum() + eps)
