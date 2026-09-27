"""model/transformer.py

A single pre-norm Llama-style decoder block, and the full stack of
`n_layers` such blocks plus a final RMSNorm.

Block structure (pre-norm, i.e. norm BEFORE the sublayer, not after --
this is what makes very deep Transformers trainable without a learning
rate warmup carrying all the weight):

    x = x + Attention(RMSNorm(x))
    x = x + SwiGLU(RMSNorm(x))
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint

from model.attention import GroupedQueryAttention, KVCache
from model.feedforward import SwiGLU
from model.normalization import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_query_heads: int,
        n_kv_heads: int,
        head_dim: int,
        d_ffn: int,
        norm_eps: float = 1e-5,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        self.attn = GroupedQueryAttention(
            d_model=d_model,
            n_query_heads=n_query_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            attn_dropout=attn_dropout,
        )
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        self.ffn = SwiGLU(d_model, d_ffn, dropout=dropout)
        self.resid_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        attn_out, present_kv = self.attn(
            self.attn_norm(x), cos, sin, past_key_value=past_key_value, use_cache=use_cache
        )
        x = x + self.resid_dropout(attn_out)
        x = x + self.resid_dropout(self.ffn(self.ffn_norm(x)))
        return x, present_kv


class Transformer(nn.Module):
    """Stack of `n_layers` TransformerBlocks + a final RMSNorm.

    Does NOT include the token embedding or the LM head -- those live in
    model/llm.py, which is the module users actually instantiate.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_query_heads: int,
        n_kv_heads: int,
        head_dim: int,
        d_ffn: int,
        norm_eps: float = 1e-5,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_query_heads=n_query_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    d_ffn=d_ffn,
                    norm_eps=norm_eps,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model, eps=norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_values: list[KVCache | None] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[KVCache] | None]:
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        # Gradient checkpointing only makes sense while training, without
        # a KV cache (generation is one-token-at-a-time and already tiny
        # in activation memory; there is nothing worth recomputing there).
        use_checkpointing = self.gradient_checkpointing and self.training and not use_cache

        present_key_values = [] if use_cache else None
        for layer, past_kv in zip(self.layers, past_key_values):
            if use_checkpointing:
                # `use_reentrant=False` lets us checkpoint a closure over
                # `layer`/`past_kv` directly (no restrictions on what the
                # wrapped function returns or captures, unlike the legacy
                # reentrant implementation). Recomputes this block's
                # forward pass during backward instead of keeping its
                # activations resident for the whole forward pass -- see
                # TrainingConfig.gradient_checkpointing for the tradeoff.
                def _run_layer(x, cos, sin, layer=layer, past_kv=past_kv):
                    out, _ = layer(x, cos, sin, past_key_value=past_kv, use_cache=False)
                    return out

                x = torch.utils.checkpoint.checkpoint(_run_layer, x, cos, sin, use_reentrant=False)
                present_kv = None
            else:
                x, present_kv = layer(x, cos, sin, past_key_value=past_kv, use_cache=use_cache)

            if use_cache:
                present_key_values.append(present_kv)

        x = self.final_norm(x)
        return x, present_key_values
