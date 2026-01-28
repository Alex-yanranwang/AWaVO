from __future__ import annotations
import numpy as np
try:
    import gymnasium as gym
except ImportError:  # fallback
    import gym

def infer_act_space(env: gym.Env):
    if isinstance(env.action_space, gym.spaces.Discrete):
        return "discrete", int(env.action_space.n)
    if isinstance(env.action_space, gym.spaces.Box):
        return "continuous", int(np.prod(env.action_space.shape))
    raise ValueError(f"Unsupported action space: {env.action_space}")
