"""model/normalization.py

RMSNorm (Root Mean Square Layer Normalization), as used in Llama.

Unlike LayerNorm, RMSNorm does not subtract the mean or use a bias term --
it only rescales by the root-mean-square of the activations, then applies
a learned per-channel gain. This is both simpler and cheaper than
LayerNorm (fewer reduction passes, no bias parameters), which matters on
a 6GB-VRAM budget where every saved buffer counts.

Reference: Zhang & Sennrich, 2019, "Root Mean Square Layer Normalization".
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """y = x / rms(x) * weight, computed in fp32 for numerical stability
    regardless of the surrounding autocast dtype (bf16/fp16), then cast
    back -- this is the standard Llama trick to avoid RMSNorm being a
    precision bottleneck under mixed precision.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # mean(x^2) over the last (feature) dimension, keepdim for broadcast
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.float()
        normed = self._norm(x_fp32).to(input_dtype)
        return normed * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.weight.shape[0]}, eps={self.eps}"
