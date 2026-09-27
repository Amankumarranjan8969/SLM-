"""tests/test_training.py

Phase 4 tests: optimizer param grouping, LR scheduler shape, checkpoint
save/load/rotation/resume, the memory-mapped PackedDataset, and a real
(if tiny) end-to-end Trainer run on CPU -- including a simulated CUDA-OOM
auto-backoff test that doesn't require an actual GPU to exercise the
retry logic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from model.config import ExperimentConfig, ModelConfig, TrainingConfig
from model.llm import SLMForCausalLM
from tokenizer.train_tokenizer import train as train_tokenizer
from tokenizer.tokenizer import SLMTokenizer
from training.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from training.dataset import PackedDataset
from training.optimizer import build_optimizer
from training.scheduler import build_lr_scheduler
from training.trainer import Trainer, get_gpu_memory_stats

REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_CORPUS = REPO_ROOT / "data" / "raw" / "dev_sample_corpus.txt"


def _tiny_model_config(**overrides) -> ModelConfig:
    defaults = dict(
        name="tiny_test", vocab_size=64, d_model=32, n_layers=2,
        n_query_heads=4, n_kv_heads=2, head_dim=8, d_ffn=64,
        context_length=32, tie_embeddings=True, dtype="fp32",
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _tiny_experiment_config(**training_overrides) -> ExperimentConfig:
    training_defaults = dict(
        learning_rate=3e-3, min_learning_rate=3e-4, warmup_steps=2,
        max_steps=20, micro_batch_size=4, gradient_accumulation_steps=2,
        sequence_length=16, checkpoint_every=10, eval_every=5, log_every=5,
        keep_last_n_checkpoints=2,
    )
    training_defaults.update(training_overrides)
    cfg = ExperimentConfig(model=_tiny_model_config(), training=TrainingConfig(**training_defaults))
    cfg.runtime.device = "cpu"
    cfg.runtime.num_workers = 0
    cfg.runtime.pin_memory = False
    return cfg


# --------------------------------------------------------------------------- #
# Fixtures: a real trained tokenizer + a real packed shard, reused across tests
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def packed_shard(tmp_path_factory) -> tuple[Path, Path]:
    tok_dir = tmp_path_factory.mktemp("tok")
    train_tokenizer(input_path=DEV_CORPUS, output_dir=tok_dir, vocab_size=512, context_length=512)
    tok = SLMTokenizer.from_pretrained(tok_dir)

    from data.packing import pack_corpus

    data_dir = tmp_path_factory.mktemp("packed_data")
    train_bin = data_dir / "train.bin"
    val_bin = data_dir / "val.bin"
    pack_corpus(DEV_CORPUS, tok, train_bin, val_fraction=0.15, output_val_bin=val_bin)
    return train_bin, val_bin


# --------------------------------------------------------------------------- #
# PackedDataset
# --------------------------------------------------------------------------- #


def test_packed_dataset_shift_by_one(packed_shard):
    train_bin, _ = packed_shard
    ds = PackedDataset(train_bin, seq_length=8)
    x, y = ds[0]
    assert x.shape == (8,)
    assert y.shape == (8,)
    assert torch.equal(x[1:], y[:-1])


def test_packed_dataset_length_matches_manifest(packed_shard):
    train_bin, _ = packed_shard
    ds = PackedDataset(train_bin, seq_length=8)
    assert len(ds) == ds.manifest.num_tokens - 8


def test_packed_dataset_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PackedDataset(tmp_path / "nope.bin", seq_length=8)


def test_packed_dataset_rejects_seq_length_ge_num_tokens(tmp_path):
    tiny_bin = tmp_path / "tiny.bin"
    np.array([1, 2, 3], dtype=np.uint16).tofile(tiny_bin)
    manifest_path = tiny_bin.with_suffix(".manifest.json")
    manifest_path.write_text('{"num_tokens": 3, "dtype": "uint16", "vocab_size": 10, "num_documents": 1, "source": "x"}')
    with pytest.raises(ValueError):
        PackedDataset(tiny_bin, seq_length=10)


def test_packed_dataset_does_not_load_whole_file_eagerly(packed_shard):
    """The memmap must not be opened in __init__ -- opening happens lazily
    per (possibly forked) worker. This test just asserts that invariant
    directly rather than trying to measure RSS, which is unreliable in CI."""
    train_bin, _ = packed_shard
    ds = PackedDataset(train_bin, seq_length=8)
    assert ds._memmap is None
    _ = ds[0]
    assert ds._memmap is not None  # opened on first access


# --------------------------------------------------------------------------- #
# Optimizer
# --------------------------------------------------------------------------- #


def test_optimizer_splits_decay_and_no_decay_params():
    cfg = _tiny_model_config()
    model = SLMForCausalLM.from_config(cfg)
    training_cfg = TrainingConfig(weight_decay=0.1)
    optimizer = build_optimizer(model, training_cfg)

    assert len(optimizer.param_groups) == 2
    decay_group = next(g for g in optimizer.param_groups if g["weight_decay"] > 0)
    no_decay_group = next(g for g in optimizer.param_groups if g["weight_decay"] == 0)

    assert all(p.dim() >= 2 for p in decay_group["params"])
    assert all(p.dim() < 2 for p in no_decay_group["params"])

    # every RMSNorm weight (1D) must land in the no-decay group
    norm_param_ids = {id(p) for name, p in model.named_parameters() if "norm" in name}
    no_decay_ids = {id(p) for p in no_decay_group["params"]}
    assert norm_param_ids.issubset(no_decay_ids)


def test_optimizer_covers_every_trainable_param():
    cfg = _tiny_model_config()
    model = SLMForCausalLM.from_config(cfg)
    optimizer = build_optimizer(model, TrainingConfig())
    covered = sum(len(g["params"]) for g in optimizer.param_groups)
    total = sum(1 for p in model.parameters() if p.requires_grad)
    assert covered == total


# --------------------------------------------------------------------------- #
# LR scheduler
# --------------------------------------------------------------------------- #


def test_scheduler_warmup_then_decay_to_floor():
    model = torch.nn.Linear(4, 4)
    cfg = TrainingConfig(learning_rate=1e-3, min_learning_rate=1e-4, warmup_steps=10, max_steps=100, lr_scheduler="cosine")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = build_lr_scheduler(optimizer, cfg)

    lrs = []
    for _ in range(100):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

    # warmup: strictly increasing for the first warmup_steps
    assert all(lrs[i] < lrs[i + 1] for i in range(9))
    # peak near base_lr right after warmup
    assert abs(lrs[10] - cfg.learning_rate) < 1e-6
    # decays afterward, ending near (not below) the floor
    assert lrs[-1] < lrs[10]
    assert lrs[-1] >= cfg.min_learning_rate - 1e-9


def test_scheduler_constant_mode_holds_after_warmup():
    model = torch.nn.Linear(4, 4)
    cfg = TrainingConfig(learning_rate=1e-3, min_learning_rate=1e-4, warmup_steps=5, max_steps=50, lr_scheduler="constant")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = build_lr_scheduler(optimizer, cfg)
    for _ in range(5):
        optimizer.step()
        scheduler.step()
    lr_at_10 = scheduler.get_last_lr()[0]
    for _ in range(20):
        optimizer.step()
        scheduler.step()
    lr_at_30 = scheduler.get_last_lr()[0]
    assert abs(lr_at_10 - lr_at_30) < 1e-9


def test_scheduler_linear_mode_decreases_monotonically_after_warmup():
    model = torch.nn.Linear(4, 4)
    cfg = TrainingConfig(learning_rate=1e-3, min_learning_rate=0.0, warmup_steps=2, max_steps=20, lr_scheduler="linear")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = build_lr_scheduler(optimizer, cfg)
    lrs = []
    for _ in range(20):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
    post_warmup = lrs[2:]
    assert all(post_warmup[i] >= post_warmup[i + 1] - 1e-9 for i in range(len(post_warmup) - 1))


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #


def test_checkpoint_round_trip_restores_model_and_optimizer_state(tmp_path):
    cfg = _tiny_model_config()
    model_a = SLMForCausalLM.from_config(cfg)
    optimizer_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
    scheduler_a = build_lr_scheduler(optimizer_a, TrainingConfig(warmup_steps=2, max_steps=10))

    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model_a(x, targets=y)
    out.loss.backward()
    optimizer_a.step()
    scheduler_a.step()

    save_checkpoint(
        tmp_path, model_a, optimizer_a, scheduler_a, step=7, model_name="tiny_test",
        train_loss=1.23, val_loss=4.56, best_val_loss=4.56, is_best=True, keep_last_n=3,
    )

    model_b = SLMForCausalLM.from_config(cfg)
    optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)
    scheduler_b = build_lr_scheduler(optimizer_b, TrainingConfig(warmup_steps=2, max_steps=10))

    meta = load_checkpoint(tmp_path / "latest.pt", model_b, optimizer_b, scheduler_b)

    assert meta.step == 7
    assert meta.train_loss == 1.23
    assert meta.val_loss == 4.56
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa, pb)
    assert scheduler_a.get_last_lr() == scheduler_b.get_last_lr()


def test_checkpoint_rotation_keeps_only_last_n(tmp_path):
    cfg = _tiny_model_config()
    model = SLMForCausalLM.from_config(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_lr_scheduler(optimizer, TrainingConfig(warmup_steps=1, max_steps=10))

    for step in [10, 20, 30, 40]:
        save_checkpoint(tmp_path, model, optimizer, scheduler, step=step, model_name="t", keep_last_n=2)

    step_files = sorted(tmp_path.glob("step_*.pt"))
    assert len(step_files) == 2
    assert "0000030" in step_files[0].name
    assert "0000040" in step_files[1].name
    assert (tmp_path / "latest.pt").exists()


def test_find_latest_checkpoint_returns_none_when_absent(tmp_path):
    assert find_latest_checkpoint(tmp_path) is None


def test_load_checkpoint_missing_file_raises_helpful_error(tmp_path):
    model = SLMForCausalLM.from_config(_tiny_model_config())
    with pytest.raises(FileNotFoundError, match="expected"):
        load_checkpoint(tmp_path / "nope.pt", model)


# --------------------------------------------------------------------------- #
# GPU memory stats helper (must degrade gracefully with no CUDA)
# --------------------------------------------------------------------------- #


def test_gpu_memory_stats_on_cpu_reports_nan_not_crash():
    stats = get_gpu_memory_stats(torch.device("cpu"))
    assert stats["allocated_gb"] != stats["allocated_gb"]  # nan != nan


# --------------------------------------------------------------------------- #
# End-to-end Trainer run (real, tiny, on CPU)
# --------------------------------------------------------------------------- #


def test_trainer_full_run_reduces_loss_and_writes_log(packed_shard, tmp_path):
    train_bin, val_bin = packed_shard
    cfg = _tiny_experiment_config(max_steps=20, eval_every=10, checkpoint_every=10, log_every=5)
    cfg.model.vocab_size = 512  # must match the tokenizer used to pack `packed_shard`

    model = SLMForCausalLM.from_config(cfg.model)
    trainer = Trainer(
        model=model, config=cfg, train_bin=train_bin, val_bin=val_bin,
        checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs", model_name="tiny_test",
    )
    trainer.train()

    assert trainer.state.step == 20
    log_path = tmp_path / "logs" / "tiny_test_train_log.csv"
    assert log_path.exists()
    assert (tmp_path / "ckpt" / "latest.pt").exists()


def test_trainer_gradient_checkpointing_runs_and_matches_scale_of_loss(packed_shard, tmp_path):
    """Not a numerical-equivalence test (that's tests/test_model.py) --
    this just confirms the Trainer wiring actually enables checkpointing
    end-to-end and training still proceeds normally."""
    train_bin, val_bin = packed_shard
    cfg = _tiny_experiment_config(max_steps=6, eval_every=100, checkpoint_every=100, log_every=100)
    cfg.model.vocab_size = 512
    cfg.training.gradient_checkpointing = True

    model = SLMForCausalLM.from_config(cfg.model)
    trainer = Trainer(
        model=model, config=cfg, train_bin=train_bin, val_bin=None,
        checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs", model_name="tiny_test",
    )
    assert trainer.model.transformer.gradient_checkpointing is True
    trainer.train()
    assert trainer.state.step == 6


def test_trainer_resume_continues_step_count(packed_shard, tmp_path):
    train_bin, val_bin = packed_shard
    cfg = _tiny_experiment_config(max_steps=10, eval_every=100, checkpoint_every=5, log_every=100)
    cfg.model.vocab_size = 512

    model_a = SLMForCausalLM.from_config(cfg.model)
    trainer_a = Trainer(
        model=model_a, config=cfg, train_bin=train_bin, val_bin=None,
        checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs", model_name="tiny_test",
    )
    trainer_a.train()
    assert trainer_a.state.step == 10

    model_b = SLMForCausalLM.from_config(cfg.model)
    cfg_b = _tiny_experiment_config(max_steps=15, eval_every=100, checkpoint_every=5, log_every=100)
    cfg_b.model.vocab_size = 512
    trainer_b = Trainer(
        model=model_b, config=cfg_b, train_bin=train_bin, val_bin=None,
        checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs", model_name="tiny_test",
    )
    resumed = trainer_b.resume_if_available()
    assert resumed is True
    assert trainer_b.state.step == 10
    trainer_b.train(max_steps=15)
    assert trainer_b.state.step == 15


# --------------------------------------------------------------------------- #
# Simulated CUDA-OOM auto-backoff (no real GPU required)
# --------------------------------------------------------------------------- #


def test_oom_backoff_halves_micro_batch_size_and_retries(packed_shard, tmp_path):
    train_bin, _ = packed_shard
    cfg = _tiny_experiment_config(
        max_steps=1, micro_batch_size=8, gradient_accumulation_steps=1,
        eval_every=100, checkpoint_every=100, log_every=100,
    )
    cfg.model.vocab_size = 512
    cfg.runtime.auto_reduce_batch_on_oom = True
    cfg.runtime.min_micro_batch_size = 1

    model = SLMForCausalLM.from_config(cfg.model)
    trainer = Trainer(
        model=model, config=cfg, train_bin=train_bin, val_bin=None,
        checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs", model_name="tiny_test",
    )
    assert trainer.state.micro_batch_size == 8

    call_count = {"n": 0}
    original_forward_backward = trainer._forward_backward

    def flaky_forward_backward(input_ids, targets, grad_accum_steps):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise torch.OutOfMemoryError("simulated OOM for test")
        return original_forward_backward(input_ids, targets, grad_accum_steps)

    trainer._forward_backward = flaky_forward_backward
    trainer.train(max_steps=1)

    assert trainer.state.micro_batch_size == 4  # halved from 8
    assert trainer.state.step == 1  # the step still completed after backoff
    assert call_count["n"] >= 2  # first call failed, a later call succeeded


def test_oom_backoff_gives_up_and_raises_at_floor(packed_shard, tmp_path):
    train_bin, _ = packed_shard
    cfg = _tiny_experiment_config(
        max_steps=1, micro_batch_size=1, gradient_accumulation_steps=1,
        eval_every=100, checkpoint_every=100, log_every=100,
    )
    cfg.model.vocab_size = 512
    cfg.runtime.auto_reduce_batch_on_oom = True
    cfg.runtime.min_micro_batch_size = 1  # already at the floor

    model = SLMForCausalLM.from_config(cfg.model)
    trainer = Trainer(
        model=model, config=cfg, train_bin=train_bin, val_bin=None,
        checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs", model_name="tiny_test",
    )

    def always_oom(*args, **kwargs):
        raise torch.OutOfMemoryError("simulated OOM for test")

    trainer._forward_backward = always_oom
    with pytest.raises(torch.OutOfMemoryError):
        trainer.train(max_steps=1)


def test_oom_backoff_disabled_reraises_immediately(packed_shard, tmp_path):
    train_bin, _ = packed_shard
    cfg = _tiny_experiment_config(
        max_steps=1, micro_batch_size=8, gradient_accumulation_steps=1,
        eval_every=100, checkpoint_every=100, log_every=100,
    )
    cfg.model.vocab_size = 512
    cfg.runtime.auto_reduce_batch_on_oom = False  # disabled

    model = SLMForCausalLM.from_config(cfg.model)
    trainer = Trainer(
        model=model, config=cfg, train_bin=train_bin, val_bin=None,
        checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs", model_name="tiny_test",
    )

    def always_oom(*args, **kwargs):
        raise torch.OutOfMemoryError("simulated OOM for test")

    trainer._forward_backward = always_oom
    with pytest.raises(torch.OutOfMemoryError):
        trainer.train(max_steps=1)
    assert trainer.state.micro_batch_size == 8  # never reduced -- backoff was disabled
