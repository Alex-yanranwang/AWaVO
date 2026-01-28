from __future__ import annotations
import numpy as np

def cartpole_cost(obs, action_idx: int, reward: float, next_obs, done: bool) -> np.ndarray:
    """Synthetic constraint cost for CartPole.

    cost = 1 if action == 1 else 0 (action-energy proxy)
    You can replace this with a task-specific safety cost.
    """
    return np.array([float(action_idx == 1)], dtype=np.float32)

def acrobot_cost(obs, action_idx: int, reward: float, next_obs, done: bool) -> np.ndarray:
    """Synthetic constraint cost for Acrobot.

    cost = 1 if action changes direction sharply (proxy for wear/energy).
    Since this function doesn't know previous action, the wrapper provides prev_action via obs[...]
    In this minimal version we approximate using action magnitude:
      actions are {0,1,2} -> cost proportional to |a-1|
    """
    return np.array([abs(float(action_idx) - 1.0)], dtype=np.float32)
