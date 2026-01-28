from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random

import numpy as np


@dataclass
class Transition:
    """A single environment step.

    `a` is the action representation used by the algorithm:
    - continuous: raw action vector
    - discrete: one-hot vector (the environment receives the index)
    """

    s: np.ndarray
    a: np.ndarray
    r: float
    g: np.ndarray
    s2: np.ndarray
    done: float
    ep_id: int
    t: int


class ReplayBuffer:
    """Ring replay buffer with lightweight episode indexing.

    To support trajectory-level optimality likelihood p(O|τ), we keep an index
    from (episode_id, step_in_episode) -> buffer slot.

    The mapping is maintained under overwrites so short rollout segments can be
    reconstructed reliably.
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buf: List[Optional[Transition]] = [None] * self.capacity
        self.i = 0
        self.size = 0
        self._index: Dict[Tuple[int, int], int] = {}

    def add(self, tr: Transition) -> None:
        # Remove old index entry if overwriting.
        old = self.buf[self.i]
        if old is not None:
            self._index.pop((old.ep_id, old.t), None)

        self.buf[self.i] = tr
        self._index[(tr.ep_id, tr.t)] = self.i

        self.i = (self.i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> List[Transition]:
        if self.size < batch_size:
            raise ValueError(f"Not enough samples: size={self.size} batch_size={batch_size}")
        idxs = random.sample(range(self.size), batch_size)
        out: List[Transition] = []
        for j in idxs:
            tr = self.buf[j]
            if tr is None:
                continue
            out.append(tr)
        if len(out) != batch_size:
            # Rare corner: overwritten Nones in early steps.
            return self.sample(batch_size)
        return out

    def sample_trajectories(
        self,
        batch_size: int,
        horizon: int,
        gamma: float,
        *,
        bootstrap: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Sample trajectory prefixes τ of length up to `horizon`.

        Returns arrays describing the *start* (s0, a0) plus discounted
        accumulated reward/cost over the prefix:

            r_e(τ) = Σ_{k=0}^{H-1} γ^k r_{t+k}
            g_e(τ) = Σ_{k=0}^{H-1} γ^k g_{t+k}

        If `bootstrap=True`, also returns the final (sH, doneH) so callers can
        optionally add a bootstrap term outside this buffer.
        """

        if horizon <= 0:
            raise ValueError("horizon must be >= 1")
        batch = self.sample(batch_size)

        s0 = []
        a0 = []
        r_e = []
        g_e = []
        sH = []
        doneH = []

        for tr0 in batch:
            ep_id, t0 = tr0.ep_id, tr0.t
            disc = 1.0
            ret_r = 0.0
            ret_g = np.zeros_like(tr0.g, dtype=np.float32)

            last_tr = tr0
            for k in range(horizon):
                idx = self._index.get((ep_id, t0 + k))
                if idx is None:
                    break
                trk = self.buf[idx]
                if trk is None:
                    break
                ret_r += disc * float(trk.r)
                ret_g += disc * np.asarray(trk.g, dtype=np.float32)
                last_tr = trk
                if bool(trk.done):
                    break
                disc *= gamma

            s0.append(np.asarray(tr0.s, dtype=np.float32))
            a0.append(np.asarray(tr0.a, dtype=np.float32))
            r_e.append(ret_r)
            g_e.append(ret_g)

            if bootstrap:
                sH.append(np.asarray(last_tr.s2, dtype=np.float32))
                doneH.append(float(last_tr.done))

        out = {
            "s0": np.stack(s0),
            "a0": np.stack(a0),
            "r_e": np.asarray(r_e, dtype=np.float32),
            "g_e": np.stack(g_e).astype(np.float32),
        }
        if bootstrap:
            out["sH"] = np.stack(sH)
            out["doneH"] = np.asarray(doneH, dtype=np.float32)
        return out

    def __len__(self) -> int:
        return self.size
