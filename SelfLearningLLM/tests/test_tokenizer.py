"""tests/test_tokenizer.py

Phase 2 tests. These run against a tokenizer actually trained on
data/raw/dev_sample_corpus.txt (see tokenizer/train_tokenizer.py) --
a small, session-scoped fixture trains it once so every test gets a real,
working tokenizer rather than a mock.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tokenizer.train_tokenizer import SPECIAL_TOKENS, train
from tokenizer.tokenizer import SLMTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_CORPUS = REPO_ROOT / "data" / "raw" / "dev_sample_corpus.txt"


@pytest.fixture(scope="session")
def trained_tokenizer_dir(tmp_path_factory) -> Path:
    out_dir = tmp_path_factory.mktemp("tokenizer_test_artifacts")
    train(input_path=DEV_CORPUS, output_dir=out_dir, vocab_size=1024, context_length=512)
    return out_dir


@pytest.fixture(scope="session")
def tok(trained_tokenizer_dir) -> SLMTokenizer:
    return SLMTokenizer.from_pretrained(trained_tokenizer_dir)


# --------------------------------------------------------------------------- #
# Training artifacts
# --------------------------------------------------------------------------- #


def test_training_produces_expected_files(trained_tokenizer_dir):
    assert (trained_tokenizer_dir / "tokenizer.json").exists()
    assert (trained_tokenizer_dir / "config.json").exists()


def test_special_tokens_present_with_distinct_ids(tok):
    ids = {tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id, tok.user_id, tok.assistant_id}
    assert len(ids) == len(SPECIAL_TOKENS), "special token ids must be pairwise distinct"


def test_vocab_size_is_at_least_the_special_tokens_plus_byte_alphabet(tok):
    # 256 possible bytes + special tokens is the floor; real merges add more.
    assert tok.vocab_size >= 256 + len(SPECIAL_TOKENS)


# --------------------------------------------------------------------------- #
# Round trip: text -> tokens -> text
# --------------------------------------------------------------------------- #


ROUND_TRIP_CASES = [
    "Once there was a small robot who lived at the edge of a quiet forest.",
    "def factorial(n: int) -> int:\n    return 1 if n == 0 else n * factorial(n - 1)",
    "café, naïve, façade, Zürich, 日本語, Ελληνικά, Кириллица, 🙂🚀🤖",
    "The quadratic formula solves ax^2 + bx + c = 0 for x.",
    "",
    "   leading and trailing spaces   ",
    "Tabs\tand\nnewlines\nmixed together.",
    "A completely novel sentence with words never seen during training whatsoever.",
    "don't, can't, it's, \"quoted\", (parens), [brackets], {braces}",
]


@pytest.mark.parametrize("text", ROUND_TRIP_CASES)
def test_round_trip_single(tok, text):
    ids = tok.encode(text)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    decoded = tok.decode(ids)  # skip_special_tokens=True by default
    assert decoded == text


def test_round_trip_without_bos_eos(tok):
    text = "Round trip with no special tokens added."
    ids = tok.encode(text, add_bos=False, add_eos=False)
    assert tok.bos_id not in ids
    assert tok.eos_id not in ids
    assert tok.decode(ids) == text


def test_round_trip_keep_special_tokens_when_requested(tok):
    text = "hello"
    ids = tok.encode(text)
    decoded_with_special = tok.decode(ids, skip_special_tokens=False)
    assert decoded_with_special != text  # BOS/EOS text markers should show up
    assert tok.decode(ids, skip_special_tokens=True) == text


# --------------------------------------------------------------------------- #
# Byte-level coverage / UNK behavior
# --------------------------------------------------------------------------- #


def test_byte_level_coverage_means_no_unk_on_arbitrary_unicode(tok):
    """Byte-level BPE can represent any valid UTF-8 string via raw bytes,
    even words/scripts never seen in training, so <UNK> should not appear
    for ordinary text."""
    weird = "Some brand-new 語彙 that was definitely не seen в тренировке. 🧪"
    ids = tok.encode(weird)
    assert tok.unk_id not in ids
    assert tok.decode(ids) == weird


def test_unk_token_is_reserved_and_addressable(tok):
    # <UNK> must exist in the vocab even though byte-level coverage means
    # it is essentially never emitted by the encoder itself -- it's a
    # required special token per the project spec's "Unknown token
    # handling" requirement, and remains available for any future
    # non-byte-level vocabulary restriction.
    assert tok.unk_id is not None
    assert tok.decode([tok.unk_id], skip_special_tokens=False) != ""


# --------------------------------------------------------------------------- #
# Padding / batching
# --------------------------------------------------------------------------- #


def test_encode_batch_pads_to_longest_in_batch(tok):
    texts = ["short", "a somewhat longer sentence than the first one"]
    batch = tok.encode_batch(texts, padding=True)
    lengths = [len(ids) for ids in batch.input_ids]
    assert lengths[0] == lengths[1], "all sequences in a padded batch must share length"
    assert len(batch.attention_mask) == 2
    # first sequence is shorter -> should have trailing zeros in its mask
    assert batch.attention_mask[0][-1] == 0
    assert batch.attention_mask[1][-1] == 1


def test_encode_batch_attention_mask_matches_pad_positions(tok):
    texts = ["hi", "a much longer piece of text to force padding on the short one"]
    batch = tok.encode_batch(texts, padding=True)
    for ids, mask in zip(batch.input_ids, batch.attention_mask):
        for token_id, m in zip(ids, mask):
            if m == 0:
                assert token_id == tok.pad_id
            else:
                assert token_id != tok.pad_id or token_id in (tok.bos_id, tok.eos_id)


def test_encode_batch_without_padding_returns_ragged_lengths(tok):
    texts = ["short", "a somewhat longer sentence than the first one"]
    batch = tok.encode_batch(texts, padding=False)
    lengths = [len(ids) for ids in batch.input_ids]
    assert lengths[0] != lengths[1]


def test_encode_batch_matches_single_encode(tok):
    texts = ["first example", "second, different example here"]
    batch = tok.encode_batch(texts, padding=False)
    for text, ids in zip(texts, batch.input_ids):
        assert ids == tok.encode(text)


def test_decode_batch_matches_single_decode(tok):
    texts = ["alpha beta gamma", "delta epsilon"]
    batch = tok.encode_batch(texts, padding=True)
    decoded = tok.decode_batch(batch.input_ids)
    for text, d in zip(texts, decoded):
        assert d == text


def test_truncation_respects_context_length(tok):
    long_text = "word " * 2000
    ids = tok.encode(long_text, truncate_to=32)
    assert len(ids) == 32


# --------------------------------------------------------------------------- #
# Instruction-format helper (used by Phase 8 finetuning)
# --------------------------------------------------------------------------- #


def test_instruction_example_masks_loss_on_prompt_only(tok):
    input_ids, loss_mask = tok.encode_instruction_example(
        user_text="What is 2 + 2?", assistant_text="4"
    )
    assert len(input_ids) == len(loss_mask)
    assert input_ids[0] == tok.bos_id
    assert loss_mask[0] == 0  # BOS is prompt-side
    assert input_ids[1] == tok.user_id
    assert loss_mask[1] == 0
    assert tok.assistant_id in input_ids
    assistant_pos = input_ids.index(tok.assistant_id)
    assert loss_mask[assistant_pos] == 1
    # everything from <ASSISTANT> onward (inclusive) is loss-eligible
    assert all(m == 1 for m in loss_mask[assistant_pos:])
    # everything before <ASSISTANT> is masked out
    assert all(m == 0 for m in loss_mask[:assistant_pos])
    assert input_ids[-1] == tok.eos_id
    assert loss_mask[-1] == 1


def test_instruction_example_truncates_to_context_length(tok):
    long_user_text = "word " * 2000
    input_ids, loss_mask = tok.encode_instruction_example(long_user_text, "answer")
    assert len(input_ids) <= tok.context_length
    assert len(input_ids) == len(loss_mask)


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #


def test_from_pretrained_raises_helpful_error_on_missing_dir(tmp_path):
    empty_dir = tmp_path / "no_tokenizer_here"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="Train one first"):
        SLMTokenizer.from_pretrained(empty_dir)


def test_train_tokenizer_rejects_missing_input(tmp_path):
    from tokenizer.train_tokenizer import train as train_fn

    with pytest.raises(FileNotFoundError):
        train_fn(
            input_path=tmp_path / "does_not_exist.txt",
            output_dir=tmp_path / "out",
        )
