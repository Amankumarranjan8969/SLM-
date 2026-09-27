"""Llama-style decoder-only Transformer, implemented from scratch.

Phase 3 status: fully implemented (RMSNorm, RoPE, GQA via fused SDPA,
SwiGLU, weight tying, KV-cached generation). Import the top-level model
via `from model.llm import SLMForCausalLM`.
"""

from model.llm import SLMForCausalLM  # noqa: F401
