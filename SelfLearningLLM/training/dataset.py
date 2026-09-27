"""training/dataset.py

`PackedDataset` reads a `.bin` shard produced by `data/packing.py` through
`numpy.memmap` rather than loading it into RAM. Each `__getitem__` call
copies out exactly `seq_length + 1` `uint16` values (a few KB at most) and
casts them to a `torch.long` tensor -- the rest of a multi-hundred-MB or
multi-GB shard is never touched except by the OS's own page cache, which
evicts pages under memory pressure automatically (unlike a Python list
holding everything live).

The classic nanoGPT-style sampling scheme is used: any starting offset in
`[0, num_tokens - seq_length - 1]` is valid, so one epoch's worth of
"documents" is really just `num_tokens - seq_length` overlapping windows;
`__len__` reflects that. This maximizes how much of the packed data a
fixed number of training steps actually sees, at the cost of adjacent
samples overlapping -- a standard, deliberate tradeoff for LM
pretraining at this scale.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.packing import PACKED_DTYPE, PackManifest


class PackedDataset(Dataset):
    def __init__(self, bin_path: str | Path, seq_length: int):
        self.bin_path = Path(bin_path)
        manifest_path = self.bin_path.with_suffix(".manifest.json")
        if not self.bin_path.exists():
            raise FileNotFoundError(
                f"Packed shard not found: {self.bin_path}. "
                f"Run `python -m data.packing --input <corpus> --output-train-bin {self.bin_path}` first."
            )
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest for {self.bin_path}: {manifest_path}")

        self.manifest = PackManifest.load(manifest_path)
        self.seq_length = seq_length

        # `mmap_mode="r"` is the whole point: this does NOT read the file
        # into memory. It maps the file's pages into the process's
        # address space lazily; only the slices actually indexed below
        # ever get paged in.
        self._memmap: np.memmap | None = None  # opened lazily per-worker, see _ensure_open

        if self.manifest.num_tokens <= seq_length:
            raise ValueError(
                f"Shard has only {self.manifest.num_tokens} tokens, which is <= "
                f"seq_length={seq_length}; cannot form even one training example."
            )

    def _ensure_open(self) -> np.memmap:
        # Opened lazily (not in __init__) so that each DataLoader worker
        # process (num_workers > 0) opens its own memmap handle rather
        # than inheriting one across a fork, which can be unsafe/slow
        # depending on platform.
        if self._memmap is None:
            self._memmap = np.memmap(self.bin_path, dtype=PACKED_DTYPE, mode="r")
        return self._memmap

    def __len__(self) -> int:
        return max(0, self.manifest.num_tokens - self.seq_length)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        mm = self._ensure_open()
        # Copy only this window out of the memmap -- .astype() forces a
        # real (small) copy so the returned tensor doesn't keep pinning
        # the underlying memmap page in a way that fights the DataLoader
        # collate/pin_memory pipeline.
        window = np.asarray(mm[idx : idx + self.seq_length + 1], dtype=np.int64)
        input_ids = torch.from_numpy(window[:-1].copy())
        targets = torch.from_numpy(window[1:].copy())
        return input_ids, targets
