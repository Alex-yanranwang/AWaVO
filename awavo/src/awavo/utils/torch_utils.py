from __future__ import annotations
import torch

def to_torch(x, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)

def normalize_unit(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)
