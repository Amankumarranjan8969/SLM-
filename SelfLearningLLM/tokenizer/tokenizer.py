"""tokenizer/tokenizer.py

Thin, opinionated wrapper around the trained `tokenizers.Tokenizer` that
gives the rest of the project (data pipeline, training loop, chat CLI) one
stable API for encode/decode/padding/special tokens, and does so through
the library's batched Rust calls rather than a Python-level per-item loop.

Performance notes:
  - `encode_batch` / `decode_batch` call `Tokenizer.encode_batch_fast` /
    `Tokenizer.decode_batch`, which run in the Rust core across the whole
    batch at once. A naive `[self.encode(t) for t in texts]` would work
    but pays Python call overhead per item; for a data-loading hot path
    (packing millions of tokens) that overhead adds up.
  - Padding is applied once, vectorized, after batch encoding -- not per
    string inside a Python loop.
  - The tokenizer object itself is loaded once and reused; this class is
    safe to hold as a long-lived singleton in the data pipeline / trainer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tokenizers import Tokenizer

# Must match tokenizer/train_tokenizer.py::SPECIAL_TOKENS exactly.
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<USER>", "<ASSISTANT>"]


@dataclass
class EncodedBatch:
    """Result of a padded batch encode: parallel lists, ready to tensorize."""

    input_ids: list[list[int]]
    attention_mask: list[list[int]]


class SLMTokenizer:
    """Wraps a trained byte-level BPE `tokenizers.Tokenizer`.

    Load with `SLMTokenizer.from_pretrained("tokenizer/")`, which expects
    `tokenizer.json` + `config.json` produced by
    `tokenizer/train_tokenizer.py` in that directory.
    """

    def __init__(self, tokenizer: Tokenizer, config: dict):
        self._tk = tokenizer
        self._config = config

        missing = [t for t in SPECIAL_TOKENS if self._tk.token_to_id(t) is None]
        if missing:
            raise ValueError(
                f"Loaded tokenizer is missing required special tokens: {missing}. "
                f"Was it trained with tokenizer/train_tokenizer.py?"
            )

        self.pad_id: int = self._tk.token_to_id("<PAD>")
        self.bos_id: int = self._tk.token_to_id("<BOS>")
        self.eos_id: int = self._tk.token_to_id("<EOS>")
        self.unk_id: int = self._tk.token_to_id("<UNK>")
        self.user_id: int = self._tk.token_to_id("<USER>")
        self.assistant_id: int = self._tk.token_to_id("<ASSISTANT>")

        self.vocab_size: int = self._tk.get_vocab_size()
        self.context_length: int = config.get("context_length", 512)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pretrained(cls, tokenizer_dir: str | Path) -> "SLMTokenizer":
        tokenizer_dir = Path(tokenizer_dir)
        tk_path = tokenizer_dir / "tokenizer.json"
        cfg_path = tokenizer_dir / "config.json"
        if not tk_path.exists():
            raise FileNotFoundError(
                f"No tokenizer.json in {tokenizer_dir}. "
                f"Train one first: python -m tokenizer.train_tokenizer --input <corpus> "
                f"--output-dir {tokenizer_dir}"
            )
        tokenizer = Tokenizer.from_file(str(tk_path))
        config = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        return cls(tokenizer, config)

    # ------------------------------------------------------------------ #
    # Single-example encode / decode
    # ------------------------------------------------------------------ #

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        truncate_to: Optional[int] = None,
    ) -> list[int]:
        ids = self._tk.encode(text, add_special_tokens=False).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        max_len = truncate_to or self.context_length
        if max_len is not None and len(ids) > max_len:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tk.decode(ids, skip_special_tokens=skip_special_tokens)

    # ------------------------------------------------------------------ #
    # Batched encode / decode (preferred hot path -- see module docstring)
    # ------------------------------------------------------------------ #

    def encode_batch(
        self,
        texts: list[str],
        add_bos: bool = True,
        add_eos: bool = True,
        padding: bool = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> EncodedBatch:
        """Encode many strings in one Rust-side batch call, then pad.

        Padding is right-padding with `self.pad_id`; `attention_mask` is 1
        for real tokens and 0 for padding, matching the convention used by
        `training/trainer.py` (Phase 4) for loss masking.
        """
        encodings = self._tk.encode_batch(texts, add_special_tokens=False)

        max_len = max_length or self.context_length
        all_ids: list[list[int]] = []
        for enc in encodings:
            ids = enc.ids
            if add_bos:
                ids = [self.bos_id] + ids
            if add_eos:
                ids = ids + [self.eos_id]
            if len(ids) > max_len:
                ids = ids[:max_len]
            all_ids.append(ids)

        if not padding:
            return EncodedBatch(input_ids=all_ids, attention_mask=[[1] * len(ids) for ids in all_ids])

        target_len = max((len(ids) for ids in all_ids), default=0)
        if pad_to_multiple_of:
            remainder = target_len % pad_to_multiple_of
            if remainder != 0:
                target_len += pad_to_multiple_of - remainder
        target_len = min(target_len, max_len) if target_len > max_len else target_len

        padded_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for ids in all_ids:
            pad_len = target_len - len(ids)
            padded_ids.append(ids + [self.pad_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        return EncodedBatch(input_ids=padded_ids, attention_mask=attention_mask)

    def decode_batch(self, ids_batch: list[list[int]], skip_special_tokens: bool = True) -> list[str]:
        return self._tk.decode_batch(ids_batch, skip_special_tokens=skip_special_tokens)

    # ------------------------------------------------------------------ #
    # Instruction-format helpers (used by training/finetune.py, Phase 8)
    # ------------------------------------------------------------------ #

    def encode_instruction_example(
        self, user_text: str, assistant_text: str
    ) -> tuple[list[int], list[int]]:
        """Encode a `<USER> ... <ASSISTANT> ...` pair and return
        `(input_ids, loss_mask)` where `loss_mask[i] == 1` iff token `i`
        belongs to the assistant response (per section 7: "Only calculate
        training loss on assistant tokens").
        """
        prompt_ids = [self.bos_id, self.user_id] + self._tk.encode(
            user_text, add_special_tokens=False
        ).ids
        response_ids = (
            [self.assistant_id]
            + self._tk.encode(assistant_text, add_special_tokens=False).ids
            + [self.eos_id]
        )
        input_ids = prompt_ids + response_ids
        loss_mask = [0] * len(prompt_ids) + [1] * len(response_ids)

        if len(input_ids) > self.context_length:
            input_ids = input_ids[: self.context_length]
            loss_mask = loss_mask[: self.context_length]

        return input_ids, loss_mask

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"SLMTokenizer(vocab_size={self.vocab_size}, "
            f"context_length={self.context_length}, "
            f"pad={self.pad_id}, bos={self.bos_id}, eos={self.eos_id}, unk={self.unk_id})"
        )
