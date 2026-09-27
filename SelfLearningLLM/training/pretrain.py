"""training/pretrain.py

CLI entrypoint for causal LM pretraining. This is what `main.py pretrain`
(Phase 4 onward) delegates to.

Usage:
    python -m training.pretrain --config configs/7m.yaml \
        --train-bin data/train/dev.bin --val-bin data/val/dev.bin

    # Resume a crashed/stopped run from checkpoints/latest.pt:
    python -m training.pretrain --config configs/7m.yaml \
        --train-bin data/train/dev.bin --resume
"""

from __future__ import annotations

import argparse
from pathlib import Path

from model.config import ExperimentConfig
from model.llm import SLMForCausalLM
from training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--train-bin", type=str, required=True)
    parser.add_argument("--val-bin", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None, help="Override config's training.max_steps")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    model = SLMForCausalLM.from_config(config.model)

    real_params = model.count_parameters()
    print(f"[pretrain] Model: {config.model.name} -- {real_params:,} parameters (~{real_params/1e6:.2f}M)")

    trainer = Trainer(
        model=model,
        config=config,
        train_bin=args.train_bin,
        val_bin=args.val_bin,
        checkpoint_dir=Path(args.checkpoint_dir) / config.model.name,
        log_dir=args.log_dir,
        model_name=config.model.name,
    )

    if args.resume:
        trainer.resume_if_available()

    print(
        f"[pretrain] device={trainer.device} autocast_dtype={trainer.autocast_dtype} "
        f"micro_batch_size={trainer.state.micro_batch_size} "
        f"grad_accum_steps={config.training.gradient_accumulation_steps} "
        f"gradient_checkpointing={config.training.gradient_checkpointing}"
    )

    trainer.train(max_steps=args.max_steps)
    print(f"[pretrain] Finished at step {trainer.state.step}. Checkpoints in {trainer.checkpoint_dir}")


if __name__ == "__main__":
    main()
