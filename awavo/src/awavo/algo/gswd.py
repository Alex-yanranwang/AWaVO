from __future__ import annotations

import itertools
from math import comb
from typing import List, Tuple

import torch


def poly_feature_dim(d: int, degree: int) -> int:
    """Number of monomials of total degree `degree` in `d` variables (with repetition)."""
    return comb(d + degree - 1, degree)


def monomial_index_tuples(d: int, degree: int) -> List[Tuple[int, ...]]:
    """All index tuples (i1<=i2<=...<=idegree) defining monomials x[i1]*...*x[idegree]."""
    return list(itertools.combinations_with_replacement(range(d), degree))


def alpha_homogeneous_poly(
    x: torch.Tensor,
    coeffs: torch.Tensor,
    l: torch.Tensor | None = None,
    degree: int = 3,
) -> torch.Tensor:
    """Hypersurface defining function α(x, θ_e) as a homogeneous polynomial.

    The paper suggests defining α(·, θ_e) as a homogeneous polynomial of odd degree m
    so that the associated transform can be injective (and thus yield a true metric)
    following Kolouri et al. (2019). In their experiments they set m=3.

    We implement:
        α(x, θ_e) = Σ_{|κ|=m} θ_{e,κ} x^κ  -  l
    where κ ranges over all multi-indices of total degree m.

    Args:
        x:      (B, D) samples
        coeffs: (B, M) polynomial coefficients, M = C(D+m-1, m)
        l:      (B, 1) or (B,) offset
        degree: odd polynomial degree (default 3)

    Returns:
        (B,) α-values
    """
    if x.dim() != 2:
        raise ValueError(f"x must be (B,D), got shape {tuple(x.shape)}")
    if coeffs.dim() != 2:
        raise ValueError(f"coeffs must be (B,M), got shape {tuple(coeffs.shape)}")

    bsz, d = x.shape
    idx = monomial_index_tuples(d, degree)
    m = len(idx)
    if coeffs.shape[1] != m:
        raise ValueError(
            f"coeffs has wrong dim: expected M={m} for D={d}, degree={degree}, got {coeffs.shape[1]}"
        )

    # Build monomial features (B, M)
    feats = []
    for t in idx:
        prod = x[:, t[0]]
        for j in t[1:]:
            prod = prod * x[:, j]
        feats.append(prod)
    phi = torch.stack(feats, dim=1)

    alpha = (phi * coeffs).sum(dim=1)
    if l is not None:
        if l.dim() == 2 and l.shape[1] == 1:
            l = l.squeeze(1)
        alpha = alpha - l
    return alpha


def wasserstein_1d(u: torch.Tensor, v: torch.Tensor, p: int = 2) -> torch.Tensor:
    """Empirical 1D Wasserstein W_p between two sets of samples."""
    u_sorted, _ = torch.sort(u)
    v_sorted, _ = torch.sort(v)
    if u_sorted.numel() != v_sorted.numel():
        raise ValueError("u and v must have the same number of samples for empirical W_p.")
    if p == 1:
        return (u_sorted - v_sorted).abs().mean()
    return ((u_sorted - v_sorted).abs().pow(p).mean()).pow(1.0 / p)


def a_gswd(
    x: torch.Tensor,
    y: torch.Tensor,
    theta_e: torch.Tensor,
    l: torch.Tensor,
    *,
    k: int = 2,
    degree: int = 3,
) -> torch.Tensor:
    """Sample-based A-GSWD using a single adaptive slice.

    In the paper, A-GSWD integrates a generalized Radon transform (push-forward
    through α(x, θ_e)) with adaptive slicing where (θ_e, l) are actor outputs.

    This implementation uses one slice per batch step:
        u = α(x, θ_e) ,  v = α(y, θ_e)
        A-GSWD_k ≈ W_k(u, v)

    Args:
        x, y:     (B, D) sample sets (same B and D)
        theta_e:  (B, M) coefficients for homogeneous polynomial α (degree=3)
        l:        (B, 1) or (B,) offset
        k:        Wasserstein order
        degree:   polynomial degree (odd)

    Returns:
        scalar distance
    """
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {tuple(x.shape)} vs {tuple(y.shape)}")
    u = alpha_homogeneous_poly(x, theta_e, l, degree=degree)
    v = alpha_homogeneous_poly(y, theta_e, l, degree=degree)
    return wasserstein_1d(u, v, p=k)
