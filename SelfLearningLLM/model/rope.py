"""model/rope.py

Rotary Position Embeddings (RoPE), as used in Llama.

Instead of adding a learned or sinusoidal position vector to the token
embedding, RoPE rotates each consecutive pair of dimensions in the
query/key vectors by an angle that depends on the token's position and
the pair's frequency. The dot product of two rotated vectors then
depends only on their *relative* position, which is what gives RoPE its
extrapolation-friendly, relative-position property.

Reference: Su et al., 2021, "RoFormer: Enhanced Transformer with Rotary
Position Embedding".

Performance notes:
  - `precompute_rope_cache` builds the cos/sin tables ONCE up to
    `max_seq_len` and registers them as buffers (see model/llm.py), so
    `apply_rotary_emb` is called during every forward pass without
    recomputing any trig functions -- it is pure elementwise multiply/add.
  - The cache is sliced to the current sequence length (or the current
    KV-cache position range during incremental decoding) rather than
    rebuilt, so generation with a KV cache pays for one row of the table
    per new token, not the whole table.
"""

from __future__ import annotations

import torch


def precompute_rope_cache(
    head_dim: int, max_seq_len: int, theta: float = 10000.0, device=None, dtype=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin), each of shape (max_seq_len, head_dim).

    Follows the common "interleaved-as-halves" RoPE layout: the head_dim
    channels are split into two halves, and the second half is treated as
    the "rotated" counterpart of the first half (this is the layout used
    by Llama's reference implementation and is equivalent, after a fixed
    permutation, to the original interleaved-pairs formulation in the
    RoFormer paper).
    """
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")

    # inv_freq[i] = theta ** (-2i / head_dim), for i in [0, head_dim/2)
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)  # (max_seq_len, head_dim/2)
    freqs = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, head_dim) -- duplicated for both halves

    cos = freqs.cos()
    sin = freqs.sin()
    if dtype is not None:
        cos = cos.to(dtype)
        sin = sin.to(dtype)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split the last dim in half and return (-x2, x1) concatenated."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply RoPE to a query or key tensor.

    Args:
        x:   (batch, n_heads, seq_len, head_dim)
        cos: (seq_len, head_dim) -- already sliced to the positions of `x`
        sin: (seq_len, head_dim) -- same

    Returns:
        Rotated tensor of the same shape as `x`.
    """
    # Broadcast cos/sin over batch and head dims: (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (_rotate_half(x) * sin)
