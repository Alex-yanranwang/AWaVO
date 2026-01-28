from __future__ import annotations

import argparse
import os
from typing import Dict
import numpy as np
try:
    import gymnasium as gym
except ImportError:  # fallback
    import gym
from tqdm import trange

from awavo.algo.trainer import AWaVOTrainer, AWaVOConfig
from awavo.envs.gym_make import make_gym_env
from awavo.envs.guard_adapter import make_guard_env
from awavo.envs.space import infer_act_space
from awavo.utils.replay import ReplayBuffer, Transition
from awavo.utils.seed import set_seed
from awavo.utils.logging import TBLogger

def parse_args():
    p = argparse.ArgumentParser("AWaVO training")
    p.add_argument("--env", type=str, required=True, choices=["cartpole","acrobot","guard-walker","guard-drone"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--logdir", type=str, default="runs/awavo")

    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--start-steps", type=int, default=5_000)
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--update-every", type=int, default=1)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=5)

    # algo cfg
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau-c", type=float, default=0.5)
    p.add_argument("--beta-r", type=float, default=1.0)
    p.add_argument("--beta-g", type=float, default=5.0)
    p.add_argument("--wvi-coef", type=float, default=0.1)
    p.add_argument("--k-wass", type=int, default=2)
    p.add_argument("--n-quantiles", type=int, default=32)
    p.add_argument("--poly-degree", type=int, default=3)
    p.add_argument("--no-psi", action="store_true", help="Disable Ψ(s,a) features for A-GSWD (use actions directly).")
    p.add_argument("--feature-dim", type=int, default=32)
    p.add_argument("--psi-hidden", type=int, default=256)

    # Trajectory-level p(O|τ) construction for WVI
    p.add_argument("--traj-horizon", type=int, default=4, help="Prefix length H for r_e(τ), g_e(τ) in WVI.")
    p.add_argument("--traj-bootstrap", action="store_true", help="Include a bootstrap term after the prefix (experimental).")

    p.add_argument("--save-every", type=int, default=50_000)
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    return p.parse_args()

def evaluate(env: gym.Env, trainer: AWaVOTrainer, cost_fn, n_episodes: int, act_space: str):
    rets, costs = [], []
    for _ in range(n_episodes):
        s, _ = env.reset()
        done = False
        ep_r = 0.0
        ep_g = 0.0
        while not done:
            if act_space == "discrete":
                a_idx = trainer.act(s)
                s2, r, terminated, truncated, info = env.step(a_idx)
                g = cost_fn(s, a_idx, r, s2, terminated or truncated)
                a_rep = np.eye(trainer.act_rep_dim, dtype=np.float32)[a_idx]
            else:
                a = trainer.act(s)
                s2, r, terminated, truncated, info = env.step(a)
                g = cost_fn(s, a, r, s2, terminated or truncated, info)
            done = bool(terminated or truncated)
            ep_r += float(r)
            ep_g += float(np.sum(g))
            s = s2
        rets.append(ep_r)
        costs.append(ep_g)
    return float(np.mean(rets)), float(np.mean(costs))

def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    logger = TBLogger(args.logdir)

    if args.env in ["cartpole","acrobot"]:
        env, cost_fn, b = make_gym_env(args.env)
    else:
        env, cost_fn, b = make_guard_env(args.env)

    obs_dim = int(np.prod(env.observation_space.shape))
    act_space, act_dim_or_n = infer_act_space(env)

    cfg = AWaVOConfig(
        gamma=args.gamma,
        tau_c=args.tau_c,
        n_quantiles=args.n_quantiles,
        beta_r=args.beta_r,
        beta_g=args.beta_g,
        wvi_coef=args.wvi_coef,
        k_wass=args.k_wass,
        poly_degree=args.poly_degree,
        use_psi=(not args.no_psi),
        feature_dim=args.feature_dim,
        psi_hidden=args.psi_hidden,
        traj_horizon=args.traj_horizon,
        traj_bootstrap=args.traj_bootstrap,
    )

    device = __import__("torch").device(args.device)
    trainer = AWaVOTrainer(
        obs_dim=obs_dim,
        act_space=act_space,
        act_dim_or_n=act_dim_or_n,
        num_constraints=int(b.shape[0]),
        b=b,
        device=device,
        cfg=cfg,
    )

    rb = ReplayBuffer(args.buffer_size)

    s, _ = env.reset(seed=args.seed)
    ep_r, ep_g = 0.0, 0.0
    ep_len = 0
    ep_id = 0

    for t in trange(1, args.total_steps + 1):
        # sample action
        if t < args.start_steps:
            a = env.action_space.sample()
        else:
            a = trainer.act(s)

        if act_space == "discrete":
            a_idx = int(a)
            s2, r, terminated, truncated, info = env.step(a_idx)
            done = bool(terminated or truncated)
            g = cost_fn(s, a_idx, r, s2, done)
            a_rep = np.eye(act_dim_or_n, dtype=np.float32)[a_idx]
        else:
            a_vec = np.asarray(a, dtype=np.float32)
            s2, r, terminated, truncated, info = env.step(a_vec)
            done = bool(terminated or truncated)
            g = cost_fn(s, a_vec, r, s2, done, info)
            a_rep = a_vec.astype(np.float32)

        rb.add(Transition(
            s=np.asarray(s, dtype=np.float32),
            a=a_rep,
            r=float(r),
            g=np.asarray(g, dtype=np.float32),
            s2=np.asarray(s2, dtype=np.float32),
            done=float(done),
            ep_id=int(ep_id),
            t=int(ep_len),
        ))

        ep_r += float(r)
        ep_g += float(np.sum(g))
        ep_len += 1

        s = s2
        if done:
            logger.log({"train/ep_return": ep_r, "train/ep_cost": ep_g, "train/ep_len": ep_len}, t)
            ep_r, ep_g, ep_len = 0.0, 0.0, 0
            ep_id += 1
            s, _ = env.reset()

        # updates
        if len(rb) >= args.batch_size and t >= args.start_steps and (t % args.update_every == 0):
            for _ in range(args.updates_per_step):
                batch = rb.sample(args.batch_size)
                metrics = trainer.update(batch, rb=rb)
                logger.log(metrics, t)

        # eval
        if args.eval_every > 0 and (t % args.eval_every == 0):
            mean_r, mean_g = evaluate(env, trainer, cost_fn, args.eval_episodes, act_space)
            logger.log({"eval/return": mean_r, "eval/cost": mean_g}, t)

        # checkpoints
        if args.save_every > 0 and (t % args.save_every == 0):
            ckpt_path = os.path.join(args.ckpt_dir, f"{args.env}_seed{args.seed}_step{t}.pt")
            import torch
            torch.save({
                "actor": trainer.actor.state_dict(),
                "critic": trainer.critic.state_dict(),
                "psi": None if trainer.psi is None else trainer.psi.state_dict(),
                "cfg": cfg.__dict__,
                "b": b,
                "act_space": act_space,
                "act_dim_or_n": act_dim_or_n,
                "obs_dim": obs_dim,
            }, ckpt_path)

    logger.close()
