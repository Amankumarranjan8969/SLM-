"""model/llm.py

Top-level decoder-only causal language model: wires together
TokenEmbedding -> RoPE cache -> Transformer stack -> LM head, with
weight tying, loss computation, a real (not analytic) parameter counter,
and KV-cached autoregressive generation.

This is the module the rest of the project actually imports and trains
(`SLMForCausalLM.from_config(model_cfg)`); model/config.py::ModelConfig
only describes the shape, it doesn't build anything itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig
from model.embeddings import TokenEmbedding
from model.rope import precompute_rope_cache
from model.transformer import KVCache, Transformer


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: list[KVCache] | None = None


class SLMForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = TokenEmbedding(config.vocab_size, config.d_model)
        self.transformer = Transformer(
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_query_heads=config.n_query_heads,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            d_ffn=config.d_ffn,
            norm_eps=config.norm_eps,
            dropout=config.dropout,
            attn_dropout=config.attn_dropout,
        )

        if config.tie_embeddings:
            # Reuse the embedding's own nn.Linear-less weight matrix
            # directly: no separate parameter is allocated for the head.
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

        # RoPE cos/sin tables, precomputed once up to context_length and
        # registered as (non-persistent) buffers so they move with
        # `.to(device)` / `.to(dtype)` automatically but are never saved
        # into checkpoints (they're deterministic given head_dim/theta).
        cos, sin = precompute_rope_cache(
            head_dim=config.head_dim,
            max_seq_len=config.context_length,
            theta=config.rope_theta,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        self._apply_residual_scaling()

    # ------------------------------------------------------------------ #
    # Construction / initialization
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(cls, config: ModelConfig) -> "SLMForCausalLM":
        return cls(config)

    def gradient_checkpointing_enable(self) -> None:
        """Trade ~30% more compute for much lower peak activation memory
        during training. See TrainingConfig.gradient_checkpointing and
        model/transformer.py::Transformer.forward for the mechanism.
        """
        self.transformer.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.transformer.gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        # TokenEmbedding and lm_head (if untied) already set their own
        # std=0.02 init in __init__; this pass covers every plain Linear
        # inside attention/FFN with the same standard small-init.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _apply_residual_scaling(self) -> None:
        """Scale down the output projection of every residual sublayer
        (o_proj, down_proj) by 1/sqrt(2*n_layers), the standard GPT-2/
        Llama init trick that keeps the variance of the residual stream
        from growing with depth. Without this, deeper stacks (our 10-
        layer 56M config included) are noticeably harder to get training
        off the ground with, especially at the higher learning rates a
        small model wants.
        """
        scale = 1.0 / math.sqrt(2 * self.config.n_layers)
        for block in self.transformer.layers:
            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=0.02 * scale)
            nn.init.normal_(block.ffn.down_proj.weight, mean=0.0, std=0.02 * scale)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_key_values: list[KVCache | None] | None = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        """
        Args:
            input_ids: (batch, seq_len) token ids. If `past_key_values` is
                given, `seq_len` should be just the NEW tokens (usually 1
                during generation), not the whole sequence.
            targets: (batch, seq_len) next-token targets, same shape as
                `input_ids`, or None for inference-only. Positions equal
                to -100 are ignored by the loss (PyTorch convention),
                which lets the instruction-tuning loss mask
                (tokenizer.encode_instruction_example) be applied by the
                caller as `targets = input_ids.clone(); targets[loss_mask == 0] = -100`.
            past_key_values: per-layer KV cache from a previous call, or None.
            use_cache: whether to return an updated KV cache.
        """
        batch, seq_len = input_ids.shape
        if past_key_values is not None and past_key_values[0] is not None:
            past_len = past_key_values[0][0].shape[2]
        else:
            past_len = 0

        if past_len + seq_len > self.config.context_length:
            raise ValueError(
                f"Sequence position {past_len + seq_len} exceeds context_length="
                f"{self.config.context_length}. Truncate input or increase context_length."
            )

        x = self.token_embedding(input_ids)

        cos = self.rope_cos[past_len : past_len + seq_len].to(x.dtype)
        sin = self.rope_sin[past_len : past_len + seq_len].to(x.dtype)

        x, present_key_values = self.transformer(
            x, cos, sin, past_key_values=past_key_values, use_cache=use_cache
        )

        if self.config.tie_embeddings:
            logits = F.linear(x, self.token_embedding.weight)
        else:
            logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(logits=logits, loss=loss, past_key_values=present_key_values)

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling with a KV cache.

        Only the FIRST forward pass processes the whole prompt; every
        subsequent step feeds exactly one new token and reuses the cached
        K/V from every previous step, so generation cost is O(new_tokens)
        forward passes over a single token each, not O(new_tokens) forward
        passes over the whole growing sequence -- the difference matters a
        lot on a 6GB card generating anything longer than a few tokens.
        """
        self.eval()
        device = input_ids.device
        batch = input_ids.shape[0]

        # Prefill: encode the whole prompt at once, build the initial cache.
        out = self.forward(input_ids, use_cache=True)
        past_key_values = out.past_key_values
        next_token_logits = out.logits[:, -1, :]

        generated = input_ids
        finished = torch.zeros(batch, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            next_token = _sample_next_token(next_token_logits, temperature, top_k, top_p)
            next_token = torch.where(finished, torch.full_like(next_token, eos_id or 0), next_token)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            if eos_id is not None:
                finished = finished | (next_token == eos_id)
                if finished.all():
                    break

            if generated.shape[1] >= self.config.context_length:
                break

            out = self.forward(next_token.unsqueeze(1), past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            next_token_logits = out.logits[:, -1, :]

        return generated

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def count_parameters(self, trainable_only: bool = False) -> int:
        """The REAL parameter count from the instantiated module -- meant
        to be cross-checked against `ModelConfig.analytic_param_count()`
        (see tests/test_model.py). If these two ever disagree, the
        architecture code has drifted from the documented design and that
        is a bug, not an acceptable discrepancy.
        """
        params = self.parameters() if not trainable_only else (p for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in params)


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    if temperature <= 0.0:
        return logits.argmax(dim=-1)

    logits = logits / temperature

    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        kth_value = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
        logits = torch.where(logits < kth_value, torch.full_like(logits, float("-inf")), logits)

    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > top_p
        # keep at least one token: shift the mask right so the first
        # token that crosses top_p is still included
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, sorted_idx, sorted_mask)
        logits = logits.masked_fill(mask, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
