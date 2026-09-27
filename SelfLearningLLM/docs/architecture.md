# Architecture

**Status: Phase 1.** This document will be filled in as Phase 3 (7M
Transformer bring-up) is implemented. For now it records the *design*,
which is already fixed in `model/config.py`.

## Overview

Decoder-only, Llama-style Transformer:

```text
tokens -> embedding (tied w/ LM head)
        -> [ RMSNorm -> GQA self-attn (RoPE, causal mask) -> residual ] x N
        -> [ RMSNorm -> SwiGLU FFN -> residual ]                        x N
        -> RMSNorm
        -> LM head (tied embedding)
        -> logits
```

## Component notes (to be expanded once implemented)

- **`normalization.py` (RMSNorm)**: pre-norm placement, weight-only (no
  bias, no mean-subtraction), `eps=model.norm_eps`.
- **`rope.py` (Rotary Position Embeddings)**: precomputed frequency table
  up to `model.context_length`, applied to Q/K per head at `head_dim=64`.
- **`attention.py` (Grouped Query Attention)**: `n_query_heads=10`,
  `n_kv_heads=5` -> group size 2 query heads share each KV head; causal
  mask; KV heads repeated (not projected) to match query head count at
  attention time.
- **`feedforward.py` (SwiGLU)**: `down_proj(SiLU(gate_proj(x)) * up_proj(x))`,
  `d_ffn=1728`.
- **`transformer.py`**: assembles the pre-norm attention + FFN blocks with
  residual connections, stacks `n_layers=10`.
- **`embeddings.py`**: token embedding table, tied to the output projection
  (`model.tie_embeddings=True`).
- **`llm.py`**: top-level `nn.Module` wiring embeddings -> blocks -> final
  norm -> LM head; will expose `count_parameters()` for a real (not
  analytic) parameter count to cross-check against
  `ModelConfig.analytic_param_count()`.

## Parameter budget

See `model/config.py::ModelConfig.analytic_param_count` for the exact
closed-form derivation, and the README's parameter table for the three
presets (7m / 19m / 56m). These are analytic estimates from the
architecture's equations; Phase 3 will report the *actual* parameter
count from the instantiated `nn.Module` for comparison.
