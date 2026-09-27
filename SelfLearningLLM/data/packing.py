"""data/packing.py

Tokenizes a text corpus and packs the resulting token ids into a flat
binary file of `uint16` (vocab_size=16384 fits comfortably under 65536,
so 2 bytes/token instead of the 8 bytes/token a Python int list or a
default int64 numpy array would cost -- a 4x memory/disk reduction before
we even get to the memmap trick below).

Memory efficiency is the whole point of this module:
  - The corpus is read and tokenized one DOCUMENT at a time (paragraphs,
    split on blank lines) and each document's tokens are appended to the
    output file immediately via `array.tofile`, rather than tokenizing
    the entire corpus into one giant Python list first. Peak RAM is
    O(largest single document), not O(corpus size).
  - The packed `.bin` file is later read with `numpy.memmap` (see
    training/dataset.py), so training never loads the whole token array
    into RAM either -- the OS pages in only the byte ranges a given
    training batch actually touches. This is what lets a 300M-token
    Stage B/C corpus (far bigger than 16GB RAM could hold as Python
    objects) be trained on at all.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from tokenizer.tokenizer import SLMTokenizer

PACKED_DTYPE = np.uint16  # valid as long as vocab_size <= 65536


@dataclass
class PackManifest:
    num_tokens: int
    dtype: str
    vocab_size: int
    num_documents: int
    source: str

    def save(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "PackManifest":
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))


def _iter_documents(path: Path) -> Iterator[str]:
    """Yield one document (a blank-line-separated paragraph block) at a
    time, reading the file incrementally rather than splitting an
    entire-file string all at once.
    """
    buffer_lines: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.rstrip("\n")
            if stripped.strip() == "":
                if buffer_lines:
                    yield "\n".join(buffer_lines)
                    buffer_lines = []
            else:
                buffer_lines.append(stripped)
    if buffer_lines:
        yield "\n".join(buffer_lines)


def _iter_all_documents(input_path: Path) -> Iterator[str]:
    if input_path.is_file():
        yield from _iter_documents(input_path)
    elif input_path.is_dir():
        for fp in sorted(input_path.rglob("*.txt")):
            yield from _iter_documents(fp)
    else:
        raise FileNotFoundError(f"input path does not exist: {input_path}")


def pack_corpus(
    input_path: Path,
    tokenizer: SLMTokenizer,
    output_bin: Path,
    val_fraction: float = 0.0,
    output_val_bin: Path | None = None,
    seed: int = 1337,
) -> tuple[PackManifest, PackManifest | None]:
    """Tokenize `input_path` and write packed uint16 shard(s).

    If `val_fraction > 0`, documents are deterministically shuffled (by
    `seed`) and split into train/val BEFORE packing, so validation data
    never leaks into the training shard. Each document is written as
    `encode(doc, add_bos=False, add_eos=True)` -- the trailing EOS acts as
    the document separator inside the flat token stream.

    Returns (train_manifest, val_manifest_or_None).
    """
    if tokenizer.vocab_size > np.iinfo(PACKED_DTYPE).max + 1:
        raise ValueError(
            f"tokenizer vocab_size={tokenizer.vocab_size} does not fit in {PACKED_DTYPE}; "
            f"pick a wider packed dtype."
        )

    output_bin.parent.mkdir(parents=True, exist_ok=True)
    documents = list(_iter_all_documents(input_path))
    if not documents:
        raise ValueError(f"No documents found in {input_path} (expected blank-line-separated paragraphs)")

    if val_fraction > 0.0:
        if output_val_bin is None:
            raise ValueError("output_val_bin is required when val_fraction > 0")
        rng = random.Random(seed)
        shuffled = documents[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_fraction))
        val_docs = shuffled[:n_val]
        train_docs = shuffled[n_val:]
    else:
        train_docs, val_docs = documents, []

    train_manifest = _write_shard(train_docs, tokenizer, output_bin, source=str(input_path))
    val_manifest = None
    if val_docs:
        val_manifest = _write_shard(val_docs, tokenizer, output_val_bin, source=str(input_path))

    return train_manifest, val_manifest


def _write_shard(docs: list[str], tokenizer: SLMTokenizer, output_bin: Path, source: str) -> PackManifest:
    num_tokens = 0
    with open(output_bin, "wb") as f:
        for doc in docs:
            ids = tokenizer.encode(doc, add_bos=False, add_eos=True)
            arr = np.asarray(ids, dtype=PACKED_DTYPE)
            arr.tofile(f)  # streamed write -- no accumulation across documents
            num_tokens += len(ids)

    manifest = PackManifest(
        num_tokens=num_tokens,
        dtype=str(np.dtype(PACKED_DTYPE)),
        vocab_size=tokenizer.vocab_size,
        num_documents=len(docs),
        source=source,
    )
    manifest.save(output_bin.with_suffix(".manifest.json"))
    return manifest


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer")
    parser.add_argument("--output-train-bin", type=str, default="data/train/train.bin")
    parser.add_argument("--output-val-bin", type=str, default="data/val/val.bin")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    tok = SLMTokenizer.from_pretrained(args.tokenizer_dir)
    train_manifest, val_manifest = pack_corpus(
        input_path=Path(args.input),
        tokenizer=tok,
        output_bin=Path(args.output_train_bin),
        val_fraction=args.val_fraction,
        output_val_bin=Path(args.output_val_bin),
        seed=args.seed,
    )
    print(
        f"Train shard: {args.output_train_bin} -- {train_manifest.num_tokens:,} tokens, "
        f"{train_manifest.num_documents} documents"
    )
    if val_manifest:
        print(
            f"Val shard  : {args.output_val_bin} -- {val_manifest.num_tokens:,} tokens, "
            f"{val_manifest.num_documents} documents"
        )


if __name__ == "__main__":
    main()
