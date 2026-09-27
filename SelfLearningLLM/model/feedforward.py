"""model/feedforward.py

SwiGLU feed-forward network, as used in Llama / PaLM.

Standard Transformer FFNs are `down(activation(up(x)))`. SwiGLU instead
gates the "up" projection with a second, separately-learned projection
passed through SiLU (Swish):

    FFN(x) = down_proj( SiLU(gate_proj(x)) * up_proj(x) )

This costs 3 weight matrices instead of 2, which is why `d_ffn` in
Llama-style configs (here 1728 for the 56M model) is chosen smaller than
the naive `4 * d_model` used by GPT-2-style FFNs, to keep the total
parameter/FLOP budget comparable.

Reference: Shazeer, 2020, "GLU Variants Improve Transformer".
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.up_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.down_proj = nn.Linear(d_ffn, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))
