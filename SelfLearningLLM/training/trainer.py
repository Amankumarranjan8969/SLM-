"""training/trainer.py

The training loop itself. Every design choice here is in service of one
constraint: fit training within 6GB of VRAM, and degrade gracefully
(automatically, not by crashing) if it doesn't quite fit at the
configured batch size.

Memory-efficiency techniques used, all independently toggleable via
config:
  1. Mixed precision (bf16 preferred, fp16 fallback with GradScaler):
     halves activation memory versus fp32 and roughly halves matmul
     time on tensor-core hardware.
  2. Gradient accumulation: decouples the EFFECTIVE batch size (for
     gradient quality) from the MICRO batch size (for peak memory) --
     `training.micro_batch_size * training.gradient_accumulation_steps`
     is the effective batch size, but activation memory only ever has to
     hold one micro-batch's worth of activations at a time.
  3. Gradient checkpointing (model/transformer.py): recompute instead of
     store block activations, when `training.gradient_checkpointing=True`.
  4. Automatic CUDA-OOM backoff: if a step still raises
     `torch.OutOfMemoryError`, the trainer clears the failed step, calls
     `torch.cuda.empty_cache()`, halves the micro-batch size (down to
     `runtime.min_micro_batch_size`), rebuilds the dataloader with the
     smaller batch size, and retries -- instead of crashing the whole run.
  5. Gradient clipping: bounds the memory AND numerical blowup risk from
     any single bad batch (also just good optimization hygiene).

Logging: every `log_every` steps, a row is appended to
`logs/train_log.csv` (step, loss, lr, tokens/sec, GPU memory, ETA) and,
if `tensorboard` is importable, mirrored to a TensorBoard event file. If
TensorBoard isn't installed, that half of logging is silently skipped
(CSV/JSON logging always works and is the source of truth either way).
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.config import ExperimentConfig
from training.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from training.dataset import PackedDataset
from training.optimizer import build_optimizer
from training.scheduler import build_lr_scheduler


def detect_device(preferred: str = "auto") -> torch.device:
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("runtime.device='cuda' requested but CUDA is not available on this machine.")
        return torch.device("cuda")
    # "auto"
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_autocast_dtype(requested: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32  # autocast on CPU with bf16 works but offers no benefit here; keep it simple/correct
    if requested == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print("[trainer] bf16 requested but not supported on this GPU -- falling back to fp16.")
        return torch.float16
    if requested == "fp16":
        return torch.float16
    return torch.float32


def get_gpu_memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {"allocated_gb": float("nan"), "reserved_gb": float("nan"), "max_allocated_gb": float("nan")}
    return {
        "allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
        "reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
    }


@dataclass
class TrainState:
    step: int = 0
    best_val_loss: float = float("inf")
    micro_batch_size: int = 0  # may shrink at runtime via OOM backoff
    tokens_seen: int = 0


class CSVLogger:
    """Appends one row per log event; writes the header exactly once."""

    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = path
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._wrote_header = self.path.exists() and self.path.stat().st_size > 0

    def log(self, row: dict) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._wrote_header:
                writer.writeheader()
                self._wrote_header = True
            writer.writerow(row)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        train_bin: str | Path,
        val_bin: Optional[str | Path] = None,
        checkpoint_dir: str | Path = "checkpoints",
        log_dir: str | Path = "logs",
        model_name: str = "model",
    ):
        self.cfg = config
        self.model_name = model_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)

        self.device = detect_device(config.runtime.device)
        self.model = model.to(self.device)
        if config.training.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.autocast_dtype = resolve_autocast_dtype(config.model.dtype, self.device)
        # GradScaler is only meaningful (and only safe) for fp16; bf16 has
        # enough dynamic range that loss scaling isn't needed, and fp32
        # obviously doesn't need it either.
        self.use_grad_scaler = self.autocast_dtype == torch.float16
        self.scaler = torch.amp.GradScaler(enabled=self.use_grad_scaler)

        self.optimizer = build_optimizer(self.model, config.training)
        self.scheduler = build_lr_scheduler(self.optimizer, config.training)

        self.train_bin = Path(train_bin)
        self.val_bin = Path(val_bin) if val_bin else None
        self.state = TrainState(micro_batch_size=config.training.micro_batch_size)

        self.train_loader = self._build_dataloader(self.train_bin, self.state.micro_batch_size)
        self.val_loader = self._build_dataloader(self.val_bin, self.state.micro_batch_size) if self.val_bin else None

        self.csv_logger = CSVLogger(
            self.log_dir / f"{model_name}_train_log.csv",
            fieldnames=[
                "step", "train_loss", "val_loss", "lr", "tokens_per_sec",
                "micro_batch_size", "grad_accum_steps", "gpu_allocated_gb",
                "gpu_reserved_gb", "gpu_max_allocated_gb", "eta_seconds", "elapsed_seconds",
            ],
        )
        self.tb_writer = self._maybe_build_tensorboard_writer()

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #

    def _build_dataloader(self, bin_path: Path, micro_batch_size: int) -> DataLoader:
        dataset = PackedDataset(bin_path, seq_length=self.cfg.training.sequence_length)
        return DataLoader(
            dataset,
            batch_size=micro_batch_size,
            shuffle=True,
            num_workers=self.cfg.runtime.num_workers,
            pin_memory=self.cfg.runtime.pin_memory and self.device.type == "cuda",
            drop_last=True,
        )

    def _maybe_build_tensorboard_writer(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print("[trainer] tensorboard not installed -- skipping TensorBoard logging (CSV/JSON logging still active).")
            return None
        return SummaryWriter(log_dir=str(self.log_dir / f"{self.model_name}_tensorboard"))

    # ------------------------------------------------------------------ #
    # Checkpointing / resume
    # ------------------------------------------------------------------ #

    def resume_if_available(self) -> bool:
        latest = find_latest_checkpoint(self.checkpoint_dir)
        if latest is None:
            return False
        meta = load_checkpoint(latest, self.model, self.optimizer, self.scheduler, map_location=self.device)
        self.state.step = meta.step
        self.state.best_val_loss = meta.best_val_loss or float("inf")
        print(f"[trainer] Resumed from {latest} at step {meta.step} (best_val_loss={self.state.best_val_loss}).")
        return True

    def save(self, train_loss: float, val_loss: float | None, is_best: bool) -> Path:
        return save_checkpoint(
            self.checkpoint_dir,
            self.model,
            self.optimizer,
            self.scheduler,
            step=self.state.step,
            model_name=self.model_name,
            train_loss=train_loss,
            val_loss=val_loss,
            best_val_loss=self.state.best_val_loss,
            is_best=is_best,
            keep_last_n=self.cfg.training.keep_last_n_checkpoints,
        )

    # ------------------------------------------------------------------ #
    # Core loop
    # ------------------------------------------------------------------ #

    def _forward_backward(self, input_ids: torch.Tensor, targets: torch.Tensor, grad_accum_steps: int) -> float:
        input_ids = input_ids.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        with torch.autocast(
            device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.autocast_dtype != torch.float32
        ):
            out = self.model(input_ids, targets=targets)
            loss = out.loss / grad_accum_steps

        if self.use_grad_scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return out.loss.item()  # un-scaled-by-accum loss, for logging

    def _optimizer_step(self) -> None:
        if self.use_grad_scaler:
            self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip)

        if self.use_grad_scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _reduce_batch_size_and_rebuild(self) -> bool:
        """Halve micro_batch_size (down to runtime.min_micro_batch_size)
        and rebuild the train/val dataloaders around the new size. Returns
        False if already at the floor (nothing more can be done).
        """
        floor = self.cfg.runtime.min_micro_batch_size
        if self.state.micro_batch_size <= floor:
            return False
        new_size = max(floor, self.state.micro_batch_size // 2)
        print(
            f"[trainer] CUDA OOM at micro_batch_size={self.state.micro_batch_size}. "
            f"Reducing to {new_size} and retrying (runtime.auto_reduce_batch_on_oom)."
        )
        self.state.micro_batch_size = new_size
        self.train_loader = self._build_dataloader(self.train_bin, new_size)
        if self.val_bin:
            self.val_loader = self._build_dataloader(self.val_bin, new_size)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return True

    @torch.no_grad()
    def evaluate(self, max_batches: int = 20) -> float:
        if self.val_loader is None:
            return float("nan")
        self.model.eval()
        losses = []
        for i, (input_ids, targets) in enumerate(self.val_loader):
            if i >= max_batches:
                break
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            with torch.autocast(
                device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.autocast_dtype != torch.float32
            ):
                out = self.model(input_ids, targets=targets)
            losses.append(out.loss.item())
        self.model.train()
        return sum(losses) / len(losses) if losses else float("nan")

    def train(self, max_steps: int | None = None) -> TrainState:
        max_steps = max_steps if max_steps is not None else self.cfg.training.max_steps
        self.model.train()
        train_iter = iter(self.train_loader)
        t_start = time.time()

        while self.state.step < max_steps:
            grad_accum = self.cfg.training.gradient_accumulation_steps
            step_loss_sum = 0.0
            step_tokens = 0
            oom_this_step = False

            for _ in range(grad_accum):
                try:
                    input_ids, targets = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    input_ids, targets = next(train_iter)

                try:
                    micro_loss = self._forward_backward(input_ids, targets, grad_accum)
                    step_loss_sum += micro_loss
                    step_tokens += input_ids.numel()
                except torch.OutOfMemoryError:
                    if not self.cfg.runtime.auto_reduce_batch_on_oom or not self._reduce_batch_size_and_rebuild():
                        raise
                    self.optimizer.zero_grad(set_to_none=True)
                    oom_this_step = True
                    break  # restart this training step from scratch at the smaller batch size

            if oom_this_step:
                continue  # retry the same step index with the new (smaller) micro_batch_size

            self._optimizer_step()
            self.state.step += 1
            self.state.tokens_seen += step_tokens
            train_loss = step_loss_sum / grad_accum

            if self.state.step % self.cfg.training.log_every == 0:
                self._log(train_loss, t_start)

            if self.val_loader is not None and self.state.step % self.cfg.training.eval_every == 0:
                val_loss = self.evaluate()
                is_best = val_loss < self.state.best_val_loss
                if is_best:
                    self.state.best_val_loss = val_loss
                self._log(train_loss, t_start, val_loss=val_loss)
                if self.state.step % self.cfg.training.checkpoint_every == 0:
                    self.save(train_loss, val_loss, is_best=is_best)
            elif self.state.step % self.cfg.training.checkpoint_every == 0:
                self.save(train_loss, None, is_best=False)

        return self.state

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #

    def _log(self, train_loss: float, t_start: float, val_loss: float | None = None) -> None:
        elapsed = time.time() - t_start
        tokens_per_sec = self.state.tokens_seen / elapsed if elapsed > 0 else 0.0
        remaining_steps = max(0, self.cfg.training.max_steps - self.state.step)
        steps_per_sec = self.state.step / elapsed if elapsed > 0 else 0.0
        eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else float("nan")
        mem = get_gpu_memory_stats(self.device)
        lr = self.scheduler.get_last_lr()[0]

        row = {
            "step": self.state.step,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5) if val_loss is not None else "",
            "lr": lr,
            "tokens_per_sec": round(tokens_per_sec, 1),
            "micro_batch_size": self.state.micro_batch_size,
            "grad_accum_steps": self.cfg.training.gradient_accumulation_steps,
            "gpu_allocated_gb": round(mem["allocated_gb"], 3) if mem["allocated_gb"] == mem["allocated_gb"] else "N/A (CPU)",
            "gpu_reserved_gb": round(mem["reserved_gb"], 3) if mem["reserved_gb"] == mem["reserved_gb"] else "N/A (CPU)",
            "gpu_max_allocated_gb": round(mem["max_allocated_gb"], 3) if mem["max_allocated_gb"] == mem["max_allocated_gb"] else "N/A (CPU)",
            "eta_seconds": round(eta_seconds, 1) if eta_seconds == eta_seconds else "",
            "elapsed_seconds": round(elapsed, 1),
        }
        self.csv_logger.log(row)

        print(
            f"step {self.state.step}/{self.cfg.training.max_steps} | "
            f"loss {train_loss:.4f}" + (f" | val_loss {val_loss:.4f}" if val_loss is not None else "") +
            f" | lr {lr:.2e} | {tokens_per_sec:.0f} tok/s | "
            f"gpu_mem {row['gpu_allocated_gb']}"
        )

        if self.tb_writer is not None:
            self.tb_writer.add_scalar("train/loss", train_loss, self.state.step)
            if val_loss is not None:
                self.tb_writer.add_scalar("val/loss", val_loss, self.state.step)
            self.tb_writer.add_scalar("train/lr", lr, self.state.step)
            self.tb_writer.add_scalar("train/tokens_per_sec", tokens_per_sec, self.state.step)
