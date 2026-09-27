"""training/optimizer.py

Builds AdamW with the standard two-group split: weight decay is applied
to every >=2D parameter (linear/embedding weight matrices), and NOT
applied to 1D parameters (RMSNorm gains) -- decaying a norm's gain toward
zero has no principled justification and empirically hurts.

Uses PyTorch's `fused=True` AdamW kernel when running on CUDA (fuses the
whole parameter-update math into one kernel launch per group instead of
one per parameter per op), which reduces both kernel-launch overhead and
the number of temporary tensors materialized during the optimizer step --
a real, if modest, memory saving that matters at a 6GB budget. Falls back
to the default (non-fused) implementation elsewhere (CPU, or a CUDA build
without fused-AdamW support).
"""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn

from model.config import TrainingConfig


def build_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.AdamW:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    use_fused = torch.cuda.is_available() and "fused" in inspect.signature(torch.optim.AdamW).parameters
    extra_kwargs = {"fused": True} if use_fused else {}

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        **extra_kwargs,
    )
    return optimizer
