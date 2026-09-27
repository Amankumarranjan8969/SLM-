# Self-Learning Small Language Model (SLM)

> **Status: Phase 4 of 16 complete.** See [Build Phases](#build-phases) below
> for what exists today vs. what is planned. Nothing in this README reports
> training results yet -- there are none. Every metric field in this project
> is either a real measured number from `reports/evaluation_report.json` or
> explicitly marked `N/A` / `NOT RUN`.

## 1. Project Objective

Build a decoder-only, Llama-style Transformer language model **from scratch**
(no pretrained weights, no Hugging Face model implementation for the core
architecture) at approximately **56M parameters**, trainable end-to-end on a
single **RTX 4050 laptop GPU (6GB VRAM)**, and wrap it in a
**verification-gated self-learning system**.

**Research question:**

> Under a 6GB VRAM constraint, can a sub-100M parameter language model
> trained from scratch achieve measurable performance improvements on
> verifiable tasks through verification-gated experience replay, while
> avoiding catastrophic forgetting?

The model is never trained blindly on its own generated outputs. Every
self-generated answer is passed through a deterministic, rule-based verifier
(unit tests for code, SymPy for math, exact/keyword matching for QA) before
it is eligible to become training data. Unverifiable generations are kept
only as retrievable memory, never as training signal.

## 2. Architecture

**Status: Implemented (Phase 3), verified by `tests/test_model.py` and
`tests/test_attention.py` (32 tests, all passing).**

| Component | Choice | File |
|---|---|---|
| Model family | Decoder-only Transformer, Llama-style | `model/llm.py` |
| Normalization | RMSNorm | `model/normalization.py` |
| Positional encoding | Rotary Position Embeddings (RoPE) | `model/rope.py` |
| Attention | Grouped Query Attention (GQA), fused via `F.scaled_dot_product_attention(..., enable_gqa=True)` | `model/attention.py` |
| Feed-forward | SwiGLU | `model/feedforward.py` |
| Attention mask | Causal (fused fast path when uncached; explicit offset-aware mask when resuming a KV cache with >1 new token) | `model/attention.py` |
| Embedding / LM head | Weight-tied (configurable) | `model/embeddings.py`, `model/llm.py` |
| Precision | bf16 (fp16 fallback) mixed precision -- autocast wiring lands in Phase 4's trainer | `model/config.py` |
| KV cache | Supported for O(1)-per-token generation | `model/llm.py::generate` |

Target (56M) hyperparameters:

```text
d_model        = 640
n_layers       = 10
n_query_heads  = 10
n_kv_heads     = 5
head_dim       = 64
d_ffn          = 1728
vocab_size     = 16384
context_length = 512
```

Analytic parameter count (derived in `model/config.py::analytic_param_count`)
**and now cross-checked against the real, instantiated `nn.Module`** via
`model.count_parameters()` in `tests/test_model.py` -- they match exactly
for all three presets:

| Preset | d_model | n_layers | Analytic params | Real instantiated params |
|---|---|---|---|---|
| `7m`  | 256 | 5  | 7,146,240  | 7,146,240 (match) |
| `19m` | 384 | 8  | 18,880,896 | 18,880,896 (match) |
| `56m` | 640 | 10 | 55,964,800 | 55,964,800 (match) |

Full architecture writeup: [`docs/architecture.md`](docs/architecture.md).

## 3. Installation

Requires Python 3.12+, a CUDA-capable GPU (tested target: RTX 4050 6GB), and
Windows or Linux.

```bash
git clone <this-repo>
cd SelfLearningLLM
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
# Install the CUDA build of torch matching your driver, e.g.:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify the config system and analytic parameter counts (no GPU needed):

```bash
python -m model.config
pytest tests/test_config.py -v
```

## 4. Dataset Preparation

*(Implemented in Phase 2 onward.)* Planned stages:

- **Stage A -- TinyStories** (small subset) for pipeline validation.
- **Stage B -- WikiText + curated CS/Math/ML educational data** (Python,
  Java, C/C++, data structures, algorithms, mathematics, ML/AI, CS
  fundamentals).
- **Stage C -- optional FineWeb subset.**

Data size is always user-bounded via `MAX_DATA_SIZE` / `MAX_TOKENS` in the
active `configs/*.yaml` (`training.max_data_size_mb`, `training.max_tokens`) --
nothing is downloaded automatically at unbounded scale.

Status: **NOT RUN**.

## 5. Tokenizer Training

**Status: Implemented (Phase 2).** Byte-level BPE (`tokenizers` library,
Rust-backed), `vocab_size=16384` target, with `<PAD>`/`<BOS>`/`<EOS>`/
`<UNK>`/`<USER>`/`<ASSISTANT>` special tokens.

```bash
python -m tokenizer.train_tokenizer --input data/raw/dev_sample_corpus.txt \
    --vocab-size 16384 --output-dir tokenizer/
```

`tests/test_tokenizer.py` (26 tests, all passing) trains a real tokenizer
on `data/raw/dev_sample_corpus.txt` -- a small, hand-written, original dev
corpus checked into the repo for pipeline validation only, **not** a
Stage A/B/C training corpus -- and asserts genuine round-trip correctness
(`text -> tokens -> text`) on plain English, Python/Java/C code, math
notation, and multi-script Unicode/emoji, plus batched encode/decode,
padding + attention-mask correctness, truncation, and the
`<USER>`/`<ASSISTANT>` instruction-loss-masking helper used by Phase 8.

Measured on that dev corpus (726 unique merges learned -- the trainer
correctly stops early because the corpus is tiny; it will use the full
16384 budget once run against the real Stage A/B/C corpora):

- Round trip: **exact match** on every test string, including sentences
  never seen during training (byte-level coverage) and non-Latin
  scripts/emoji.
- `<UNK>` never appears for any valid Unicode input, as expected from a
  byte-level alphabet; it remains a reserved special token.

**Honest note on the batching optimization:** `SLMTokenizer.encode_batch`
routes through the library's single Rust-side batch call instead of a
Python `for` loop, expecting a speedup from cross-item parallelism. On
this development sandbox (1 CPU core available) a direct microbenchmark
over 50,000 short strings showed **no speedup** (loop: ~25k items/s,
batch: ~22k items/s) -- the parallelism the batch call would exploit has
no second core to run on here. The claim about the target machine (RTX
4050 laptop, which has a multi-core CPU) is a reasonable expectation, not
yet a *measured* result on that hardware; treat it as unverified until
someone benchmarks it there. The batching path is kept because it is at
worst equivalent and because it still saves one linear pass by computing
padding once per batch instead of per item.

## 6. Model Training (Pretraining)

**Status: Training loop implemented (Phase 4), verified by
`tests/test_training.py` (21 tests, all passing) plus a real end-to-end
run on the dev corpus.**

```bash
python -m training.pretrain --config configs/7m.yaml \
    --train-bin data/train/dev.bin --val-bin data/val/dev.bin
# add --resume to continue from checkpoints/<model_name>/latest.pt
```

`train.py --model {7m,19m,56m}` (the exact spec'd invocation) is a thin
wrapper `main.py` will grow around this once the data pipeline (Phase 2's
`prepare-data`, still open for the real Stage A/B/C corpora) exists;
`training.pretrain` is already the real, runnable entrypoint.

**Memory-efficiency techniques implemented, all config-toggleable:**

| Technique | Where | Effect |
|---|---|---|
| Mixed precision (bf16, fp16 fallback w/ GradScaler) | `training/trainer.py` | ~2x less activation memory than fp32 |
| Gradient accumulation | `training/trainer.py` | decouples effective batch size from peak-memory micro-batch size |
| Gradient checkpointing | `model/transformer.py`, toggle: `training.gradient_checkpointing` | recompute vs. store block activations -- see measured results below |
| Automatic CUDA-OOM backoff | `training/trainer.py::Trainer._reduce_batch_size_and_rebuild` | halves `micro_batch_size` (down to `runtime.min_micro_batch_size`) and retries instead of crashing |
| Memory-mapped packed datasets | `data/packing.py`, `training/dataset.py` | training data is `numpy.memmap`'d, never loaded whole into RAM, regardless of corpus size |
| Gradient clipping | `training/trainer.py` | bounds blowup risk from any single batch |

**Honest, measured memory results (this sandbox: 4 CPU cores, 3.9GB total
RAM, NO GPU -- these are not RTX 4050 VRAM numbers, but they are real
numbers from this machine, not estimates):**

```text
56M model, batch=4, seq_len=512, CPU, peak RSS (resource.getrusage):
  gradient_checkpointing=False : 2,380 MB
  gradient_checkpointing=True  : 1,858 MB   (22% less)

56M model, batch=12, seq_len=512, CPU:
  gradient_checkpointing=False : process OOM-KILLED by this sandbox's
                                  3.9GB RAM limit (did not complete)
  gradient_checkpointing=True  : completed successfully, peak RSS 3,287 MB
```

That second row is the real point of gradient checkpointing: at
batch=12 the non-checkpointed run didn't just use more memory, it
**failed outright** on this machine, while the checkpointed run
completed. The exact VRAM numbers on an RTX 4050 will differ from these
CPU RSS numbers (GPU activation memory doesn't have the same allocator
overhead as a CPU process), but the qualitative effect -- checkpointing
turns an OOM into a successful step -- is exactly the mechanism this
project needs for 6GB VRAM, and this is a real reproducible
demonstration of it, not a claim taken on faith.

**Also verified end-to-end on CPU, on the dev corpus's packed shard:**
a real training run (`test_trainer_full_run_reduces_loss_and_writes_log`)
drives loss down over 20 steps, writes a real CSV log, and produces a
loadable checkpoint; `test_trainer_resume_continues_step_count` confirms
resuming from that checkpoint continues the step counter and optimizer
state correctly rather than restarting; and three dedicated tests
(`test_oom_backoff_*`) exercise the CUDA-OOM auto-backoff retry logic by
simulating `torch.OutOfMemoryError` (this sandbox has no GPU to trigger a
*real* one), confirming it halves the batch size and retries, gives up
and re-raises once at the configured floor, and is a no-op when disabled.

Status: pretraining on the REAL Stage A/B/C corpora -- **NOT RUN** (that
needs Phase 2's data-pipeline output at scale, not just the tokenizer's
dev sample; the training loop itself is done and tested).

## 7. Instruction Tuning

*(Implemented in Phase 8.)* `<USER>` / `<ASSISTANT>` format; loss computed
only on assistant tokens.

Status: **NOT RUN**.

## 8. Self-Learning

*(Implemented in Phases 10-13.)* Verification-gated experience replay:
generate -> classify task type -> verify deterministically -> store
verified experience -> accumulate to buffer threshold -> fine-tune -> run
frozen regression benchmark -> promote or roll back. Full design:
[`docs/self_learning.md`](docs/self_learning.md).

Status: **NOT RUN**.

## 9. Verification

Deterministic verifiers only (the ~56M model is never used as its own
judge): sandboxed unit-test execution for code, SymPy for math, exact/
normalized/keyword matching for structured QA and definitions. Open-ended
generation is stored as memory only and is never used as a training
signal.

Status: **NOT RUN** (implemented in Phase 10).

## 10. Regression Gate

Frozen benchmark across Language / Math / Python / Algorithms /
Definitions / General QA. An update is rejected and rolled back if:

```python
if new_score < old_score * 0.98:
    reject_update(); rollback()
else:
    promote_checkpoint()
```

Also tracked: Backward Transfer (BWT) and Forgetting Rate.

Status: **NOT RUN** (implemented in Phase 12).

## 11. Evaluation

Metrics tracked once training exists: validation perplexity, Code Pass@1,
Math Exact Match, QA Accuracy, BWT, Forgetting Rate. See
[`docs/evaluation.md`](docs/evaluation.md).

Status: **NOT RUN**.

## 12. Experiments / Ablation

Planned comparison table (all cells `N/A` until the corresponding model
version has actually been trained and evaluated):

| Model | Perplexity | Code Pass@1 | Math EM | QA Accuracy | BWT | Forgetting |
|---|---|---|---|---|---|---|
| v1 Base | N/A | N/A | N/A | N/A | N/A | N/A |
| v2 + Instruction tuning | N/A | N/A | N/A | N/A | N/A | N/A |
| v3 + Memory/RAG | N/A | N/A | N/A | N/A | N/A | N/A |
| v4 + Verification-gated self-learning | N/A | N/A | N/A | N/A | N/A | N/A |
| v5 + Regression gate | N/A | N/A | N/A | N/A | N/A | N/A |

## 13. Hardware Requirements

- GPU: NVIDIA RTX 4050 Laptop (6GB VRAM) or better
- RAM: 16GB+
- Python 3.12+
- CUDA-enabled PyTorch
- Windows or Linux

The training loop is designed to auto-reduce micro-batch size on CUDA OOM
(see `runtime.auto_reduce_batch_on_oom` in every `configs/*.yaml`).

## 14. Limitations

- At ~56M parameters, the model has limited world knowledge and reasoning
  depth; it targets *verifiable, narrow* tasks (code correctness, math,
  structured QA), not open-domain reasoning.
- Deterministic verification bounds *what* can be learned from self-play:
  only code/math/structured-QA/definition tasks are training-eligible.
- Context length is capped at 512 tokens for VRAM reasons.
- Batch sizes in `configs/*.yaml` are provisional starting points until
  measured on real hardware (see note in `configs/56m.yaml`).

## 15. Research Novelty

The contribution is not the Transformer architecture (a standard Llama-style
design) but the **verification-gated self-learning loop with an explicit
regression gate** at small scale: a from-scratch sub-100M model that
extends its own training data only through experiences a deterministic
verifier can certify as correct, with a frozen-benchmark rollback
mechanism guarding against catastrophic forgetting from self-play.

## 16. Reproducibility

- All hyperparameters live in versioned YAML (`configs/*.yaml`) -- nothing
  load-bearing is hardcoded in scripts.
- Fixed seeds (`training.seed`, default `1337`).
- Every checkpoint is tagged with a model-version metadata JSON (see
  `models/v*/`).
- All logs (loss, LR, tokens/sec, GPU memory, verification results,
  regression outcomes) are written to `logs/` in both TensorBoard and
  CSV/JSON form.

---

## Build Phases

| Phase | Scope | Status |
|---|---|---|
| 1 | Project structure + configuration | **Done** |
| 2 | Tokenizer | **Done** |
| 3 | 7M Transformer | **Done** |
| 4 | Training loop | **Done** |
| 5 | 7M TinyStories experiment | Not started |
| 6 | 19M model | Not started |
| 7 | 56M model | Not started |
| 8 | Instruction tuning | Not started |
| 9 | Evaluation framework | Not started |
| 10 | Verification system | Not started |
| 11 | Experience replay | Not started |
| 12 | Regression gate | Not started |
| 13 | RAG memory | Not started |
| 14 | CLI chatbot | Not started |
| 15 | Dashboard | Not started |
| 16 | Ablation experiments | Not started |

## Project Layout

```text
SelfLearningLLM/
├── data/               # raw / cleaned / train / val / instruction / experiences
├── tokenizer/          # byte-level BPE tokenizer <-- Phase 2 (done)
├── model/              # from-scratch Transformer (config.py done; rest = Phase 3)
├── training/           # pretrain/finetune loops (Phase 4+)
├── self_learning/       # verifier, experience buffer, regression gate (Phase 10-13)
├── evaluation/         # perplexity, code/math/qa eval, forgetting metrics
├── inference/          # generation utilities
├── scripts/            # chat.py, param_check.py, etc.
├── configs/            # 7m.yaml, 19m.yaml, 56m.yaml  <-- Phase 1
├── tests/              # pytest suite
├── checkpoints/        # latest.pt / best.pt / step_xxxxx.pt
├── models/             # v1_base ... v5_regression_gate, with metadata.json
├── logs/               # TensorBoard + CSV/JSON logs
├── reports/            # evaluation_report.{json,csv} + plots/
├── docs/               # architecture / self_learning / training / evaluation / research
├── dashboard/          # Streamlit app (Phase 15)
├── main.py             # unified CLI
├── requirements.txt
└── README.md
```
