from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # fallback
    import gym


def make_guard_env(key: str):
    """Create GUARD environments if GUARD is installed.

    Expected keys:
      - guard-walker
      - guard-drone

    Returns:
        (env, cost_fn, b)

    The GUARD benchmark typically provides costs via the `info` dict. This adapter
    reads either `info["cost"]` (scalar) or `info["costs"]` (vector). If neither is
    present, it returns zero cost.
    """
    try:
        import guard_rl  # type: ignore  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "GUARD is not installed or not importable. Install GUARD and ensure the "
            "environments are registered with Gym/Gymnasium."
        ) from e

    # NOTE: The exact env IDs depend on your GUARD installation.
    # Update these strings to match your local GUARD registry if needed.
    if key == "guard-walker":
        env = gym.make("GUARDWalker-v0")
        b = np.array([1.0], dtype=np.float32)
    elif key == "guard-drone":
        env = gym.make("GUARDDrone-v0")
        b = np.array([1.0], dtype=np.float32)
    else:
        raise ValueError(f"Unknown GUARD env key: {key}")

    def cost_fn(obs, action, reward, next_obs, done, info) -> np.ndarray:
        if isinstance(info, dict):
            if "costs" in info and info["costs"] is not None:
                c = np.asarray(info["costs"], dtype=np.float32).reshape(-1)
                return c
            if "cost" in info and info["cost"] is not None:
                return np.array([float(info["cost"])], dtype=np.float32)
        return np.zeros_like(b, dtype=np.float32)

    return env, cost_fn, b
