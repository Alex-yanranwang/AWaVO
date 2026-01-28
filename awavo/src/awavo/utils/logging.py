from __future__ import annotations
import os
from torch.utils.tensorboard import SummaryWriter

class TBLogger:
    def __init__(self, logdir: str):
        os.makedirs(logdir, exist_ok=True)
        self.writer = SummaryWriter(logdir)

    def log(self, metrics: dict, step: int):
        for k, v in metrics.items():
            self.writer.add_scalar(k, float(v), step)

    def close(self):
        self.writer.close()
