"""
model/config.py

Central configuration for the Self-Learning SLM project.

This module defines:
  - ModelConfig: the Llama-style decoder-only Transformer architecture
  - TrainingConfig: pretraining/finetuning hyperparameters
  - SelfLearningConfig: experience-replay / regression-gate hyperparameters
  - RuntimeConfig: hardware / device / VRAM-budget settings
  - ExperimentConfig: the top-level container loaded from configs/*.yaml

Design note (Phase 1):
  The Transformer itself is implemented in Phase 3. This file only defines
  the *shapes* the model will be built with, plus an analytic parameter
  counter so we can verify the "~7M / ~19M / ~56M" targets BEFORE spending
  any GPU time. Once the real nn.Module exists (model/llm.py), we will
  cross-check `count_parameters(model)` (actual, from
  `sum(p.numel() for p in model.parameters())`) against this analytic
  estimate in tests/test_model.py. Nothing here is a fabricated result --
  it is a closed-form parameter count derived from the architecture
  equations below, and it will be re-verified against the real
  instantiated model in Phase 3.

All defaults here are placeholders until overridden by a YAML file in
configs/. Nothing is hardcoded into the training scripts themselves --
scripts always go through `ExperimentConfig.from_yaml(path)`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, Optional

import yaml

# --------------------------------------------------------------------------- #
# Model architecture
# --------------------------------------------------------------------------- #


@dataclass
class ModelConfig:
    """Shape/architecture hyperparameters for the Llama-style decoder.

    Invariants that model/transformer.py will assert at build time:
      - n_query_heads * head_dim == d_model   (so O-proj is d_model -> d_model)
      - n_query_heads % n_kv_heads == 0        (GQA group size must be integer)
      - context_length is a hard cap enforced by RoPE table size
    """

    name: str = "56m"

    vocab_size: int = 16384
    d_model: int = 640
    n_layers: int = 10
    n_query_heads: int = 10
    n_kv_heads: int = 5
    head_dim: int = 64
    d_ffn: int = 1728
    context_length: int = 512

    # Norm / activation
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Regularization
    dropout: float = 0.0
    attn_dropout: float = 0.0

    # Weight tying between input embedding and output (LM head) projection
    tie_embeddings: bool = True

    # Mixed precision dtype used for autocast during training/inference.
    # "bf16" preferred on Ampere+/Ada (RTX 4050 supports bf16); falls back
    # to "fp16" if the training script detects no bf16 support.
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"

    def __post_init__(self) -> None:
        if self.n_query_heads * self.head_dim != self.d_model:
            raise ValueError(
                f"[{self.name}] n_query_heads * head_dim must equal d_model: "
                f"{self.n_query_heads} * {self.head_dim} != {self.d_model}"
            )
        if self.n_kv_heads > self.n_query_heads:
            raise ValueError(
                f"[{self.name}] n_kv_heads ({self.n_kv_heads}) cannot exceed "
                f"n_query_heads ({self.n_query_heads})"
            )
        if self.n_query_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"[{self.name}] n_query_heads ({self.n_query_heads}) must be "
                f"divisible by n_kv_heads ({self.n_kv_heads}) for GQA"
            )

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def group_size(self) -> int:
        """Number of query heads sharing each KV head."""
        return self.n_query_heads // self.n_kv_heads

    def analytic_param_count(self) -> dict[str, int]:
        """Closed-form parameter count from the architecture equations.

        Per-layer:
          attn = d_model*q_dim (Q) + d_model*kv_dim (K) + d_model*kv_dim (V)
                 + q_dim*d_model (O)
          ffn  = 3 * d_model * d_ffn         (SwiGLU: gate, up, down)
          norm = 2 * d_model                  (attn RMSNorm + ffn RMSNorm)

        Global:
          embed      = vocab_size * d_model
          final_norm = d_model
          total = n_layers * per_layer + embed + final_norm
                  (+ another `embed` if tie_embeddings is False, for an
                   untied LM head)

        This does NOT include RoPE (no learned parameters) or biases
        (all linear layers are bias-free, matching Llama).
        """
        q_dim = self.n_query_heads * self.head_dim
        kv_dim = self.kv_dim

        attn = (self.d_model * q_dim) + 2 * (self.d_model * kv_dim) + (q_dim * self.d_model)
        ffn = 3 * self.d_model * self.d_ffn
        norm = 2 * self.d_model
        per_layer = attn + ffn + norm
        all_layers = per_layer * self.n_layers

        embed = self.vocab_size * self.d_model
        final_norm = self.d_model

        total = all_layers + embed + final_norm
        if not self.tie_embeddings:
            total += embed  # separate LM head

        return {
            "attn_per_layer": attn,
            "ffn_per_layer": ffn,
            "per_layer_total": per_layer,
            "all_layers_total": all_layers,
            "embedding_params": embed,
            "final_norm_params": final_norm,
            "total_params": total,
        }

    def summary(self) -> str:
        counts = self.analytic_param_count()
        total = counts["total_params"]
        lines = [
            f"ModelConfig(name={self.name!r})",
            f"  vocab_size={self.vocab_size}  d_model={self.d_model}  n_layers={self.n_layers}",
            f"  n_query_heads={self.n_query_heads}  n_kv_heads={self.n_kv_heads}  "
            f"head_dim={self.head_dim}  (GQA group size={self.group_size})",
            f"  d_ffn={self.d_ffn}  context_length={self.context_length}",
            f"  tie_embeddings={self.tie_embeddings}  dtype={self.dtype}",
            f"  --> analytic total params: {total:,} (~{total / 1e6:.2f}M)",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


@dataclass
class TrainingConfig:
    # Optimization
    optimizer: Literal["adamw"] = "adamw"
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Schedule
    warmup_steps: int = 200
    max_steps: int = 20000
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"

    # Batching (VRAM-critical: see RuntimeConfig for OOM auto-backoff)
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    sequence_length: int = 512  # must be <= ModelConfig.context_length

    # Checkpointing / eval cadence
    checkpoint_every: int = 500
    eval_every: int = 250
    log_every: int = 10
    keep_last_n_checkpoints: int = 3

    # Memory: recompute each block's activations during the backward pass
    # instead of keeping them all resident -- trades ~30% more compute time
    # for a large cut in peak activation memory (roughly O(n_layers) ->
    # O(1) in the number of layers' worth of saved activations). This is
    # usually the single biggest lever for fitting a given batch size /
    # sequence length into 6GB VRAM once the model itself is large enough
    # that activations, not weights, dominate memory.
    gradient_checkpointing: bool = False

    # Data
    dataset_path: str = "data/train"
    val_dataset_path: str = "data/val"
    max_data_size_mb: Optional[int] = None  # MAX_DATA_SIZE cap, None = no cap
    max_tokens: Optional[int] = None  # MAX_TOKENS cap, None = no cap

    # Reproducibility
    seed: int = 1337


@dataclass
class InstructionTuningConfig:
    dataset_path: str = "data/instruction"
    learning_rate: float = 1e-4
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_steps: int = 2000
    mask_prompt_loss: bool = True  # only compute loss on <ASSISTANT> tokens
    user_token: str = "<USER>"
    assistant_token: str = "<ASSISTANT>"


@dataclass
class SelfLearningConfig:
    buffer_size: int = 500
    replay_old_data_fraction: float = 0.3  # % old data mixed in vs. new experiences
    regression_threshold: float = 0.98  # reject if new_score < old_score * threshold
    finetune_steps: int = 300
    finetune_learning_rate: float = 5e-5
    experiences_path: str = "data/experiences"
    regression_benchmark_path: str = "evaluation/regression_benchmark.jsonl"
    min_verified_before_training: int = 500


@dataclass
class MemoryConfig:
    use_faiss: bool = True
    embedding_dim: int = 256
    short_term_max_turns: int = 20
    long_term_index_path: str = "self_learning/long_term_index"
    short_term_index_path: str = "self_learning/short_term_index"


@dataclass
class RuntimeConfig:
    device: Literal["auto", "cuda", "cpu"] = "auto"
    vram_budget_gb: float = 6.0
    ram_budget_gb: float = 16.0
    auto_reduce_batch_on_oom: bool = True
    min_micro_batch_size: int = 1
    num_workers: int = 2
    pin_memory: bool = True
    compile_model: bool = False  # torch.compile - off by default (Windows support varies)


@dataclass
class ExperimentConfig:
    """Top-level config loaded from a single YAML file, e.g. configs/56m.yaml."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    instruction: InstructionTuningConfig = field(default_factory=InstructionTuningConfig)
    self_learning: SelfLearningConfig = field(default_factory=SelfLearningConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {path}\n"
                f"Available configs: {list(Path('configs').glob('*.yaml')) if Path('configs').exists() else 'configs/ not found'}"
            )
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        def build(dc_type, key):
            sub = raw.get(key, {}) or {}
            valid_keys = {f.name for f in fields(dc_type)}
            unknown = set(sub.keys()) - valid_keys
            if unknown:
                raise ValueError(
                    f"Unknown key(s) in '{key}' section of {path}: {unknown}. "
                    f"Valid keys: {sorted(valid_keys)}"
                )
            return dc_type(**sub)

        return cls(
            model=build(ModelConfig, "model"),
            training=build(TrainingConfig, "training"),
            instruction=build(InstructionTuningConfig, "instruction"),
            self_learning=build(SelfLearningConfig, "self_learning"),
            memory=build(MemoryConfig, "memory"),
            runtime=build(RuntimeConfig, "runtime"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# --------------------------------------------------------------------------- #
# Presets (used by tests + as a sanity reference; the source of truth for
# actually running an experiment is always the matching configs/*.yaml file)
# --------------------------------------------------------------------------- #

MODEL_PRESETS: dict[str, ModelConfig] = {
    "7m": ModelConfig(
        name="7m", vocab_size=16384, d_model=256, n_layers=5,
        n_query_heads=4, n_kv_heads=2, head_dim=64, d_ffn=512,
        context_length=512,
    ),
    "19m": ModelConfig(
        name="19m", vocab_size=16384, d_model=384, n_layers=8,
        n_query_heads=6, n_kv_heads=2, head_dim=64, d_ffn=1024,
        context_length=512,
    ),
    "56m": ModelConfig(
        name="56m", vocab_size=16384, d_model=640, n_layers=10,
        n_query_heads=10, n_kv_heads=5, head_dim=64, d_ffn=1728,
        context_length=512,
    ),
}


def get_model_preset(name: str) -> ModelConfig:
    name = name.lower()
    if name not in MODEL_PRESETS:
        raise ValueError(f"Unknown model preset '{name}'. Choose from: {list(MODEL_PRESETS)}")
    return MODEL_PRESETS[name]


if __name__ == "__main__":
    # `python -m model.config` -> print analytic param counts for all presets
    for preset_name in MODEL_PRESETS:
        print(get_model_preset(preset_name).summary())
        print()
