"""training/checkpoint.py

Saves/loads full training state (model + optimizer + scheduler + step +
config + best-val-loss-so-far) so training can resume EXACTLY where it
left off -- including optimizer momentum/variance buffers, which matters
a lot for AdamW (resuming with fresh optimizer state after a crash causes
a visible loss spike).

Memory note: `torch.save`/`torch.load` are used with the model already
moved to CPU state_dicts only at save time (`.state_dict()` naturally
holds tensors on whatever device they're already on; we don't force an
extra CPU copy here since checkpoint writes are infrequent and disk I/O,
not RAM, is the bottleneck at that point). On load, `map_location` lets a
checkpoint saved on GPU be restored on a CPU-only machine (or vice versa)
without manually moving every tensor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch


@dataclass
class CheckpointMetadata:
    step: int
    model_name: str
    train_loss: Optional[float]
    val_loss: Optional[float]
    best_val_loss: Optional[float]
    extra: dict[str, Any]


def save_checkpoint(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    model_name: str,
    train_loss: float | None = None,
    val_loss: float | None = None,
    best_val_loss: float | None = None,
    is_best: bool = False,
    keep_last_n: int = 3,
    extra: dict[str, Any] | None = None,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": step,
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "extra": extra or {},
    }

    step_path = checkpoint_dir / f"step_{step:07d}.pt"
    torch.save(payload, step_path)

    latest_path = checkpoint_dir / "latest.pt"
    torch.save(payload, latest_path)

    if is_best:
        best_path = checkpoint_dir / "best.pt"
        torch.save(payload, best_path)

    _rotate_checkpoints(checkpoint_dir, keep_last_n=keep_last_n)
    return step_path


def _rotate_checkpoints(checkpoint_dir: Path, keep_last_n: int) -> None:
    """Delete step_*.pt files beyond the most recent `keep_last_n` --
    frees disk (and, transitively, avoids anyone accidentally memmap'ing
    a huge pile of stale multi-hundred-MB checkpoints). `latest.pt` and
    `best.pt` are never touched by this.
    """
    step_files = sorted(
        checkpoint_dir.glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    excess = len(step_files) - keep_last_n
    for old_file in step_files[:max(0, excess)]:
        old_file.unlink()


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    map_location: str | torch.device | None = None,
) -> CheckpointMetadata:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. If this is a fresh run, that's expected -- "
            f"training will start from step 0."
        )

    payload = torch.load(path, map_location=map_location or "cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    return CheckpointMetadata(
        step=payload["step"],
        model_name=payload.get("model_name", "unknown"),
        train_loss=payload.get("train_loss"),
        val_loss=payload.get("val_loss"),
        best_val_loss=payload.get("best_val_loss"),
        extra=payload.get("extra", {}),
    )


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    latest_path = Path(checkpoint_dir) / "latest.pt"
    return latest_path if latest_path.exists() else None
