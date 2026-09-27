"""scripts/param_check.py

Prints the analytic (closed-form) parameter count for each model preset
(7m/19m/56m) defined in model/config.py. Useful for verifying architecture
choices BEFORE spending GPU time -- and, once model/llm.py exists (Phase 3),
for cross-checking against the real `sum(p.numel() for p in model.parameters())`.

Usage:
    python scripts/param_check.py
    python scripts/param_check.py --preset 56m
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import MODEL_PRESETS, get_model_preset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=list(MODEL_PRESETS),
        help="Show only this preset (default: show all).",
    )
    args = parser.parse_args()

    names = [args.preset] if args.preset else list(MODEL_PRESETS)
    for name in names:
        cfg = get_model_preset(name)
        print(cfg.summary())
        print()


if __name__ == "__main__":
    main()
