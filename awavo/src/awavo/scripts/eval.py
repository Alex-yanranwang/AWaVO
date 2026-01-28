from __future__ import annotations

import argparse
import numpy as np
try:
    import gymnasium as gym
except ImportError:  # fallback
    import gym

from awavo.algo.trainer import AWaVOTrainer, AWaVOConfig
from awavo.envs.gym_make import make_gym_env
from awavo.envs.guard_adapter import make_guard_env
from awavo.envs.space import infer_act_space
from awavo.utils.torch_utils import to_torch

def parse_args():
    p = argparse.ArgumentParser("AWaVO evaluation")
    p.add_argument("--env", type=str, required=True, choices=["cartpole","acrobot","guard-walker","guard-drone"])
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()

def main():
    args = parse_args()

    if args.env in ["cartpole","acrobot"]:
        env, cost_fn, b = make_gym_env(args.env)
    else:
        env, cost_fn, b = make_guard_env(args.env)

    obs_dim = int(np.prod(env.observation_space.shape))
    act_space, act_dim_or_n = infer_act_space(env)

    import torch
    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = AWaVOConfig(**ckpt.get("cfg", {}))

    trainer = AWaVOTrainer(
        obs_dim=obs_dim,
        act_space=act_space,
        act_dim_or_n=act_dim_or_n,
        num_constraints=int(b.shape[0]),
        b=b,
        device=device,
        cfg=cfg,
    )
    trainer.actor.load_state_dict(ckpt["actor"])
    trainer.critic.load_state_dict(ckpt["critic"])
    if ckpt.get("psi") is not None and trainer.psi is not None:
        trainer.psi.load_state_dict(ckpt["psi"])

    returns, costs = [], []
    for _ in range(args.episodes):
        s, _ = env.reset()
        done = False
        ep_r, ep_g = 0.0, 0.0
        while not done:
            if act_space == "discrete":
                a_idx = trainer.act(s)
                s2, r, terminated, truncated, info = env.step(int(a_idx))
                done = bool(terminated or truncated)
                g = cost_fn(s, int(a_idx), r, s2, done)
            else:
                a = trainer.act(s)
                s2, r, terminated, truncated, info = env.step(np.asarray(a, dtype=np.float32))
                done = bool(terminated or truncated)
                g = cost_fn(s, a, r, s2, done, info)
            ep_r += float(r)
            ep_g += float(np.sum(g))
            s = s2
        returns.append(ep_r)
        costs.append(ep_g)

    print(f"Return: mean={np.mean(returns):.3f} std={np.std(returns):.3f}")
    print(f"Cost:   mean={np.mean(costs):.3f} std={np.std(costs):.3f}")
