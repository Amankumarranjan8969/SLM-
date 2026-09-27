"""training/scheduler.py

Linear warmup followed by cosine (default), linear, or constant decay
down to `min_learning_rate` -- implemented as a plain `LambdaLR` multiplier
so it composes with any optimizer without needing a custom step() call.
"""

from __future__ import annotations

import math

import torch

from model.config import TrainingConfig


def build_lr_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainingConfig) -> torch.optim.lr_scheduler.LambdaLR:
    base_lr = cfg.learning_rate
    min_lr = cfg.min_learning_rate
    warmup_steps = max(1, cfg.warmup_steps)
    max_steps = max(warmup_steps + 1, cfg.max_steps)
    min_ratio = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps  # linear warmup, avoids lr=0 at step 0

        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(progress, 1.0)

        if cfg.lr_scheduler == "constant":
            return 1.0
        elif cfg.lr_scheduler == "linear":
            return max(min_ratio, 1.0 - progress * (1.0 - min_ratio))
        elif cfg.lr_scheduler == "cosine":
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine_decay
        else:
            raise ValueError(f"Unknown lr_scheduler: {cfg.lr_scheduler}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
