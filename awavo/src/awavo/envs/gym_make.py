from __future__ import annotations
from typing import Tuple
import numpy as np
try:
    import gymnasium as gym
except ImportError:  # fallback
    import gym

from awavo.envs.gym_constraints import cartpole_cost, acrobot_cost

def make_gym_env(name: str):
    if name == "cartpole":
        env = gym.make("CartPole-v1")
        cost_fn = cartpole_cost
        b = np.array([1.0], dtype=np.float32)  # constraint limit (tunable)
        return env, cost_fn, b
    if name == "acrobot":
        env = gym.make("Acrobot-v1")
        cost_fn = acrobot_cost
        b = np.array([0.5], dtype=np.float32)  # constraint limit (tunable)
        return env, cost_fn, b
    raise ValueError(f"Unknown gym env key: {name}")
