# Training

**Status: Phase 4 complete.** Covers `training/trainer.py`,
`training/optimizer.py`, `training/scheduler.py`, `training/checkpoint.py`,
`training/dataset.py`, `data/packing.py`, and `training/pretrain.py`.

## Data flow

```text
raw corpus (.txt, blank-line-separated documents)
    -> data/packing.py::pack_corpus
       - tokenizes one document at a time (streamed, not all at once)
       - writes token ids as flat uint16 to a .bin file (2 bytes/token)
       - writes a sibling .manifest.json (num_tokens, vocab_size, ...)
    -> training/dataset.py::PackedDataset
       - opens the .bin via numpy.memmap (lazily, per DataLoader worker)
       - __getitem__ copies out exactly one (seq_length+1)-token window
    -> torch.utils.data.DataLoader
    -> training/trainer.py::Trainer
```

No stage loads the whole corpus into RAM as Python objects. This is the
main reason a 300M-token Stage B/C corpus (which would not fit as a
Python list of ints in 16GB RAM) is trainable at all on this hardware
tier.

## Why uint16 for packed tokens

`vocab_size=16384 < 65536 = 2**16`, so every token id fits in an unsigned
16-bit integer. Using `np.uint16` instead of the numpy/Python default
`int64` cuts the packed file (and therefore the memmap'd working set) to
a quarter of the size. If a future run raises `vocab_size` above 65536,
`data/packing.py::pack_corpus` will raise immediately rather than
silently truncating token ids -- see the `PACKED_DTYPE` check at the top
of `pack_corpus`.

## Mixed precision

`model.dtype` (bf16 by default) is resolved once per run in
`training/trainer.py::resolve_autocast_dtype`:

- On CUDA with bf16 hardware support (Ampere/Ada, which includes the RTX
  4050): `torch.bfloat16`, no loss scaling needed.
- On CUDA without bf16 support: falls back to `torch.float16`, WITH
  `torch.amp.GradScaler` (loss scaling), since fp16's narrower exponent
  range needs it to avoid gradient underflow.
- On CPU (this dev sandbox): `torch.float32`. Autocast in bf16 works on
  CPU too but buys nothing without matching hardware, so it's skipped
  for simplicity and correctness.

## Gradient accumulation vs. micro-batch size

`training.micro_batch_size` is what actually gets materialized in memory
at once. `training.gradient_accumulation_steps` batches of that size are
each forward/backward'd (losses summed, scaled by
`1/gradient_accumulation_steps` before `.backward()`) before a single
optimizer step. The **effective** batch size for gradient quality is
`micro_batch_size * gradient_accumulation_steps`; peak activation memory
only ever has to hold one micro-batch.

## Gradient checkpointing

See `model/transformer.py::Transformer.forward`. When
`training.gradient_checkpointing=True` (and the model is in training
mode, and no KV cache is active), each `TransformerBlock`'s forward pass
is wrapped in `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`:
its activations are freed right after the forward pass and recomputed
during backward, instead of all `n_layers` blocks' worth of activations
staying resident simultaneously. Verified two ways:

1. **Numerical equivalence**: identical loss and byte-for-byte identical
   gradients with and without checkpointing on a fixed seed (checked
   directly during Phase 4 development).
2. **Real memory measurement** (see README.md > section 6 for the exact
   numbers): on this CPU-only, 3.9GB-RAM sandbox, checkpointing cut peak
   RSS by 22% at batch=4, and was the difference between a completed
   training step and an OOM-killed process at batch=12.

## CUDA-OOM auto-backoff

`Trainer.train()` wraps each micro-batch's forward/backward in a
`try/except torch.OutOfMemoryError`. On a caught OOM (and
`runtime.auto_reduce_batch_on_oom=True`), `Trainer._reduce_batch_size_and_rebuild`
halves `micro_batch_size` (never below `runtime.min_micro_batch_size`),
calls `torch.cuda.empty_cache()`, rebuilds the train/val `DataLoader`s
around the smaller size, and the training step is retried from scratch
rather than the whole run crashing. If backoff is disabled, or the floor
is already reached, the exception propagates normally.

This sandbox has no GPU, so a REAL `torch.OutOfMemoryError` can't be
triggered here; `tests/test_training.py::test_oom_backoff_*` verify the
retry/give-up/disabled logic by monkeypatching `Trainer._forward_backward`
to raise a simulated `torch.OutOfMemoryError` on the first call. The
control flow exercised is identical to the real-GPU case; what's
untested here is specifically PyTorch's own OOM-recovery behavior on
real CUDA allocators, which needs the actual target hardware.

## Checkpointing

`training/checkpoint.py::save_checkpoint` writes `model_state_dict`,
`optimizer_state_dict` (so AdamW's per-parameter momentum/variance
buffers survive a resume -- skipping this causes a visible loss spike
after every resume), `scheduler_state_dict`, `step`, and loss metadata to
`step_XXXXXXX.pt`, `latest.pt`, and (when the run just hit a new best
validation loss) `best.pt`. `_rotate_checkpoints` deletes old
`step_*.pt` files beyond `training.keep_last_n_checkpoints`, leaving
`latest.pt`/`best.pt` untouched. `--resume` on `training/pretrain.py`
loads `latest.pt` if present and continues the step counter exactly.

## Logging

Every `training.log_every` steps, `Trainer._log` appends one row to
`logs/<model_name>_train_log.csv` (step, train/val loss, LR, tokens/sec,
GPU memory -- `"N/A (CPU)"` when there's no GPU, ETA, elapsed time) and,
if `tensorboard` is importable, mirrors the scalar values to a
TensorBoard event file under `logs/<model_name>_tensorboard/`. If
`tensorboard` isn't installed, that half is silently skipped and a
one-line notice is printed once; CSV logging is always the source of
truth.
