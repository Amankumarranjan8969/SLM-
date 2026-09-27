"""tokenizer/train_tokenizer.py

Trains a byte-level BPE tokenizer (GPT-2 style byte alphabet, so every
possible byte sequence is representable -- no OOV at the byte level) using
the Rust-backed `tokenizers` library.

Performance notes (this is the part of Phase 2 that matters for
"optimize the code"):
  - Corpus files are streamed line-by-line through a generator
    (`_iter_corpus_lines`) and fed to `Tokenizer.train_from_iterator`,
    instead of reading entire files into a Python list/string first. This
    keeps peak RAM roughly constant regardless of corpus size -- important
    since Stage B/C corpora (WikiText / FineWeb subset) can be far larger
    than what comfortably fits in a naive "read the whole file" approach.
  - The actual BPE merge-counting and training loop runs in the
    library's Rust core (`tokenizers`, not `transformers`), which
    parallelizes pre-tokenization and pair-counting across CPU cores
    automatically -- we do not add a slower pure-Python training loop on
    top of it.
  - `show_progress=True` gives real-time feedback without holding
    anything extra in memory.

Usage:
    python -m tokenizer.train_tokenizer \
        --input data/raw/dev_sample_corpus.txt \
        --vocab-size 16384 \
        --output-dir tokenizer/

    # Multiple files / a whole directory of .txt files:
    python -m tokenizer.train_tokenizer --input data/cleaned --vocab-size 16384
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers
from tokenizers.models import BPE

# Keep this list in sync with tokenizer/tokenizer.py::SPECIAL_TOKENS.
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<USER>", "<ASSISTANT>"]

# Read files in fixed-size chunks and split on newlines ourselves, rather
# than using Python's line iteration on a giant file object held open for
# the whole run -- avoids surprises with very long lines (e.g. minified
# code, single-line JSON) blowing up memory on a naive `for line in f`.
_CHUNK_SIZE_BYTES = 8 * 1024 * 1024  # 8 MB read chunks


def _iter_file_lines(path: Path) -> Iterator[str]:
    """Yield lines from a single file, reading in bounded-size chunks."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        leftover = ""
        while True:
            chunk = f.read(_CHUNK_SIZE_BYTES)
            if not chunk:
                break
            chunk = leftover + chunk
            lines = chunk.split("\n")
            leftover = lines.pop()  # last (possibly partial) line carries over
            for line in lines:
                if line:
                    yield line
        if leftover:
            yield leftover


def _iter_corpus_lines(input_path: Path) -> Iterator[str]:
    """Yield lines across one file or every *.txt file under a directory.

    This is the iterator handed to `train_from_iterator`; nothing here
    materializes the full corpus in memory at once.
    """
    if input_path.is_file():
        yield from _iter_file_lines(input_path)
    elif input_path.is_dir():
        txt_files = sorted(input_path.rglob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(
                f"No .txt files found under directory: {input_path}. "
                f"Point --input at a file or a directory containing .txt files."
            )
        for fp in txt_files:
            yield from _iter_file_lines(fp)
    else:
        raise FileNotFoundError(f"--input path does not exist: {input_path}")


def build_tokenizer(vocab_size: int, min_frequency: int) -> tuple[Tokenizer, trainers.BpeTrainer]:
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    # ByteLevel pre-tokenizer maps every byte (0-255) into the initial
    # alphabet, so BPE learns merges over byte sequences directly -- this
    # is what makes the tokenizer byte-level (any UTF-8 text, including
    # unseen scripts/emoji/binary-ish text, is representable without
    # ever falling back to <UNK> once the byte alphabet is in the vocab).
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    return tokenizer, trainer


def train(
    input_path: Path,
    output_dir: Path,
    vocab_size: int = 16384,
    min_frequency: int = 2,
    context_length: int = 512,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, trainer = build_tokenizer(vocab_size, min_frequency)

    t0 = time.time()
    tokenizer.train_from_iterator(_iter_corpus_lines(input_path), trainer=trainer)
    elapsed = time.time() - t0

    tokenizer_json_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json_path))

    actual_vocab_size = tokenizer.get_vocab_size()
    special_token_ids = {tok: tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS}

    config = {
        "tokenizer_type": "byte_level_bpe",
        "requested_vocab_size": vocab_size,
        # NOTE: with a small/low-diversity corpus, the trainer may converge
        # to fewer merges than requested (it stops once no more frequent
        # pairs exist). This is the ACTUAL measured vocab size from this
        # run, not the requested one -- do not assume they match.
        "actual_vocab_size": actual_vocab_size,
        "context_length": context_length,
        "special_tokens": SPECIAL_TOKENS,
        "special_token_ids": special_token_ids,
        "min_frequency": min_frequency,
        "training_input": str(input_path),
        "training_seconds": round(elapsed, 3),
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, required=True, help="File or directory of .txt files")
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default="tokenizer")
    args = parser.parse_args()

    config = train(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        context_length=args.context_length,
    )

    print(f"Trained tokenizer on: {config['training_input']}")
    print(f"  requested vocab_size : {config['requested_vocab_size']}")
    print(f"  actual vocab_size    : {config['actual_vocab_size']}")
    print(f"  training time        : {config['training_seconds']}s")
    print(f"  special token ids    : {config['special_token_ids']}")
    print(f"Saved to: {args.output_dir}/tokenizer.json, {args.output_dir}/config.json")


if __name__ == "__main__":
    main()
