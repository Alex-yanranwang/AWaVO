# AWaVO (arXiv:2307.07084)

A runnable PyTorch implementation of **AWaVO** (Adaptive Wasserstein Variational Optimization) for **safe RL / CMDP** settings.

**Author:** Yanran Wang

## What’s included

- **ORPO-DR** policy update (paper Algorithm 1) implemented as a practical “feasible → maximize reward, infeasible → reduce cost” switch
- **WVI** objective with a sample-based **A-GSWD** estimator (paper Definition 4.1)
- **Distributional (quantile) critics** for both reward and constraint returns
- **Gymnasium** examples:
  - `CartPole-v1` (discrete actions)
  - `Acrobot-v1` (discrete actions)
- **Optional GUARD** adapters (continuous actions):
  - `Walker`
  - `Drone`

> The paper leaves a few engineering degrees of freedom (e.g., the exact instantiation of the optimality likelihood \(p(O|\tau)\) and the hypersurface function \(\alpha(\cdot,\theta_e)\)). This repo makes explicit, documented choices so you can run experiments end-to-end and iterate.

---

## Installation

```bash
# Python 3.9+
pip install -r requirements.txt
pip install -e .
```

For Gymnasium classic-control environments, you may need:

```bash
pip install "gymnasium[classic-control]"
```

---

## Quick start (Gymnasium)

### CartPole

```bash
awavo-train --env cartpole
awavo-train --env cartpole --no-psi
# control feature dimensionality
awavo-train --env cartpole --feature-dim 64 --total-steps 200000 --start-steps 5000
```

### Trajectory-level p(O|τ) for WVI

By default, the WVI resampling weights are built from **short trajectory prefixes** of length `H`:

\[
r_e(\tau)=\sum_{k=0}^{H-1}\gamma^k r_{t+k},\qquad
g_e(\tau)=\sum_{k=0}^{H-1}\gamma^k g_{t+k}
\]

You can control the prefix length with:

```bash
awavo-train --env cartpole --traj-horizon 4
```

### Acrobot

```bash
awavo-train --env acrobot --total-steps 200000 --start-steps 5000
```

Logs go to `--logdir` (default: `runs/awavo`). Use TensorBoard:

```bash
tensorboard --logdir runs/awavo
```

---

## Optional: GUARD (Walker / Drone)

This repo does **not** vendor GUARD. Install GUARD separately so its environments are registered with Gymnasium on your machine, then run:

```bash
awavo-train --env guard-walker
awavo-train --env guard-drone
```

Environment IDs are configured in:

- `src/awavo/envs/guard_adapter.py`

If your local GUARD uses different IDs than the defaults (`GUARDWalker-v0`, `GUARDDrone-v0`), update them there.

---

## Checkpoints & evaluation

Training saves checkpoints to `--ckpt-dir` if `--save-every` is set.

```bash
awavo-eval --env cartpole --ckpt checkpoints/cartpole_seed0_step50000.pt --episodes 20
```

---

## Repository layout

```text
awavo/
  pyproject.toml
  requirements.txt
  README.md
  LICENSE
  src/awavo/
    __init__.py
    algo/
      trainer.py        # AWaVOTrainer: ORPO-DR + WVI(A-GSWD)
      gswd.py           # A-GSWD sample estimator (single adaptive slice)
      optimality.py     # practical p(O|tau) weights
    dist/
      quantile.py       # quantile regression loss (distributional TD)
    envs/
      gym_make.py       # CartPole/Acrobot factory + budgets
      gym_constraints.py# example cost functions (replace with your own)
      guard_adapter.py  # optional GUARD factory + costs from info dict
      space.py          # detect discrete/continuous action spaces
    models/
      actors.py         # discrete + continuous actors, with theta_e and l heads
      critics.py        # quantile critic for reward + constraints
    scripts/
      train.py          # awavo-train CLI
      eval.py           # awavo-eval CLI
    utils/
      replay.py         # replay buffer
      logging.py        # TensorBoard logger
      seed.py           # seeding utilities
      torch_utils.py    # small torch helpers
```

---

## Notes on the “optimality likelihood” choice

The paper introduces binary optimality variables and an optimality likelihood \(p(O|\tau)\). In this repo, the WVI resampling weights are proportional to

\[
p(O|\tau) \propto \exp(\beta_r\, r_e(\tau))\;\exp\Big(-\beta_g \sum_i g_{e,i}(\tau)\Big)
\]

where \(r_e(\tau)\) and \(g_{e,i}(\tau)\) are estimated from short trajectory prefixes (see `--traj-horizon`).

---

## License

MIT. See `LICENSE`.

### Notes on p(O|τ) and α(x, θe)

This implementation follows the paper's suggested choices:
- **α(x, θe)** is implemented as an odd-degree (default **m=3**) homogeneous polynomial, as discussed in the paper's Appendix E.1.
- **p(O|τ)** is instantiated as an exponential optimality likelihood: p(Or=1|τ) ∝ exp(βr·re(τ)), p(Ogi=1|τ) ∝ exp(−βg·ge,i(τ)), combined multiplicatively.

