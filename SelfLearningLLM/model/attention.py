"""model/attention.py

Grouped Query Attention (GQA) with Rotary Position Embeddings and a causal
mask.

GQA sits between Multi-Head Attention (MHA, one KV head per query head)
and Multi-Query Attention (MQA, a single shared KV head): query heads are
split into `n_kv_heads` groups, and every query head in a group attends
using the same, shared K/V projection. This cuts the KV-cache memory
(critical during autoregressive generation) and the K/V projection matrix
size versus MHA, while keeping more representational capacity than MQA --
a reasonable trade-off at 6GB VRAM.

Performance notes:
  - Uses `torch.nn.functional.scaled_dot_product_attention` with
    `enable_gqa=True` rather than hand-writing `softmax(QK^T / sqrt(d))V`.
    This dispatches to PyTorch's fused attention kernels (flash-attention /
    memory-efficient attention on CUDA when available) instead of
    materializing the full (seq_len x seq_len) attention matrix in a
    separate softmax step -- both faster and lower peak memory, which
    matters directly for fitting training in 6GB VRAM.
  - K/V projections are the smaller `kv_dim = n_kv_heads * head_dim`
    (not `d_model`), so GQA's memory savings apply to the projection
    weights too, not just the attention computation.
  - Supports an optional KV cache (`past_key_value`) for O(1)-per-token
    incremental decoding during generation, instead of recomputing K/V
    for the whole prefix on every new token.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rope import apply_rotary_emb

KVCache = tuple[torch.Tensor, torch.Tensor]  # (key, value), each (B, n_kv_heads, T, head_dim)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_query_heads: int,
        n_kv_heads: int,
        head_dim: int,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        if n_query_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_query_heads ({n_query_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
            )
        self.n_query_heads = n_query_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.attn_dropout = attn_dropout

        q_dim = n_query_heads * head_dim
        kv_dim = n_kv_heads * head_dim

        # Llama-style: no biases on any linear projection.
        self.q_proj = nn.Linear(d_model, q_dim, bias=False)
        self.k_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        """
        Args:
            x:   (batch, seq_len, d_model) -- only the NEW tokens for this
                 step (the full prefix if past_key_value is None, or just
                 the newest token(s) if decoding incrementally).
            cos, sin: RoPE tables already sliced to the position range of
                 the NEW tokens in `x` (see model/llm.py).
            past_key_value: cached (k, v) from previous steps, each shaped
                 (batch, n_kv_heads, past_len, head_dim), or None.
            use_cache: if True, return the updated (k, v) cache alongside
                 the output.

        Returns:
            (output, present_key_value) where output is (batch, seq_len, d_model)
            and present_key_value is the concatenated cache if use_cache else None.
        """
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_query_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # q: (B, n_query_heads, T, head_dim); k, v: (B, n_kv_heads, T, head_dim)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_key_value = (k, v) if use_cache else None
        past_len = past_key_value[0].shape[2] if past_key_value is not None else 0
        total_len = past_len + seq_len

        if past_len == 0:
            # No cache: a single fused causal-attention call over the
            # whole sequence -- the common case (training, and the initial
            # prompt "prefill" step during generation).
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=(seq_len > 1),
                dropout_p=self.attn_dropout if self.training else 0.0,
                enable_gqa=(self.n_kv_heads != self.n_query_heads),
            )
        elif seq_len == 1:
            # Decoding exactly one new token against a cached prefix: that
            # token may attend to the entire (already-causal) cache plus
            # itself -- no masking needed at all.
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=False,
                dropout_p=self.attn_dropout if self.training else 0.0,
                enable_gqa=(self.n_kv_heads != self.n_query_heads),
            )
        else:
            # Appending a CHUNK of new tokens to a non-empty cache (e.g.
            # a multi-token prompt fed in after a partial cache exists).
            # PyTorch's built-in `is_causal` assumes query and key
            # sequences start at the same offset, which is wrong once a
            # cache offset exists (it would incorrectly hide most of the
            # past context) -- so we build the offset-aware mask
            # explicitly instead of taking the fused fast path.
            device = q.device
            row = torch.arange(seq_len, device=device).unsqueeze(1) + past_len  # (seq_len, 1)
            col = torch.arange(total_len, device=device).unsqueeze(0)  # (1, total_len)
            attn_mask = (col <= row)  # (seq_len, total_len), True = attend
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                is_causal=False,
                dropout_p=self.attn_dropout if self.training else 0.0,
                enable_gqa=(self.n_kv_heads != self.n_query_heads),
            )

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.n_query_heads * self.head_dim)
        out = self.o_proj(out)
        return out, present_key_value
