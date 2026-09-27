"""main.py

Unified CLI for the Self-Learning SLM project.

Phase 1 status: argument parsing + config loading works end-to-end.
Every subcommand's *implementation* is landed in a later phase (see
README.md > Build Phases); until then it prints a clear "not implemented
yet" message naming the phase, instead of silently doing nothing or
faking a result.

Usage:
    python main.py prepare-data --config configs/7m.yaml
    python main.py train-tokenizer --config configs/7m.yaml
    python main.py pretrain --config configs/7m.yaml
    python main.py finetune --config configs/7m.yaml
    python main.py evaluate --config configs/7m.yaml
    python main.py chat --config configs/56m.yaml
    python main.py self-learn --config configs/56m.yaml
    python main.py dashboard
    python main.py pipeline --config configs/7m.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from model.config import ExperimentConfig

# Which build phase implements each subcommand, so a "not implemented"
# message is actually useful instead of a bare error.
_PHASE_OF_COMMAND = {
    "prepare-data": 2,
    "train-tokenizer": 2,
    "pretrain": 4,
    "finetune": 8,
    "evaluate": 9,
    "chat": 14,
    "self-learn": 11,
    "dashboard": 15,
    "pipeline": "2-16 (runs the full sequence)",
}


def _load_config(args: argparse.Namespace) -> ExperimentConfig | None:
    if getattr(args, "config", None) is None:
        return None
    return ExperimentConfig.from_yaml(args.config)


def _not_implemented(command: str, args: argparse.Namespace) -> None:
    phase = _PHASE_OF_COMMAND.get(command, "?")
    print(f"[main.py] '{command}' is not implemented yet (planned for Phase {phase}).")
    if getattr(args, "config", None):
        cfg = _load_config(args)
        print(f"[main.py] Config loaded OK from {args.config}:")
        print(f"  model preset : {cfg.model.name}")
        print(f"  analytic params: {cfg.model.analytic_param_count()['total_params']:,}")
    print(
        "[main.py] See README.md > 'Build Phases' for current project status. "
        "Nothing was trained, generated, or evaluated by this call."
    )
    sys.exit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Self-Learning Small Language Model -- unified CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_cmd(name: str, help_text: str, needs_config: bool = True):
        p = sub.add_parser(name, help=help_text)
        if needs_config:
            p.add_argument(
                "--config",
                type=str,
                default=None,
                help="Path to a YAML config, e.g. configs/7m.yaml",
            )
        return p

    add_cmd("prepare-data", "Clean, dedup, split, tokenize, and pack the training data.")
    add_cmd("train-tokenizer", "Train the byte-level BPE tokenizer on project datasets.")
    add_cmd("pretrain", "Run causal LM pretraining.")
    add_cmd("finetune", "Run instruction tuning.")
    add_cmd("evaluate", "Run the evaluation suite (perplexity, code/math/qa, forgetting).")
    add_cmd("chat", "Launch the terminal chatbot with verification-gated self-learning.")
    add_cmd("self-learn", "Run one experience-replay + regression-gate self-learning cycle.")
    add_cmd("dashboard", "Launch the Streamlit dashboard.", needs_config=False)
    add_cmd("pipeline", "Run the complete pipeline end-to-end.")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "dashboard":
        _not_implemented("dashboard", args)
        return

    if getattr(args, "config", None) is None:
        # Default to the smallest model so `python main.py pretrain` "just
        # works" for a first pipeline debug run, per the project's own
        # rule: "The 7M model should be used to debug the complete
        # pipeline. Do not waste hours training a broken 56M model."
        default_config = Path("configs/7m.yaml")
        if default_config.exists():
            args.config = str(default_config)
            print(f"[main.py] --config not given, defaulting to {default_config}")

    _not_implemented(args.command, args)


if __name__ == "__main__":
    main()
