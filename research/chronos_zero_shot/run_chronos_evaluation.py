#!/usr/bin/env python3
"""
Evaluation and research-pipeline runner for Chronos-2 zero-shot directional forecasting.

Hard contracts enforced:
- Maximum evaluation origin <= 266 (origins >= 267 and locked evaluation 268-315 strictly forbidden)
- locked_evaluation_read=False enforced on all inputs and artifacts
- Windows: tuning [120, 179], validation [180, 219], confirmation [220, 266]
- Tuning-only selection of Platt calibration and active blend weight
- Confirmation window is descriptive only
- Active production unchanged; promotion_eligible is strictly False
- Matched monthly coverage with candidate ranked by abs(blend - 0.5)
- Contamination caveat acknowledged: requires live or post-release paper trading
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path if run directly
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Chronos-2 zero-shot directional evaluation and metrics generation."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root directory (defaults to project root)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Imported after sys.path setup above so the package resolves when run directly.
    from forecast_select.chronos_pipeline import (
        build_chronos_zero_shot,
        chronos_zero_shot_status,
    )

    args = parse_args(argv)
    target_root = args.root.resolve()

    print(f"[Chronos-2 Evaluation] Running research evaluation on {target_root}...")
    summary = build_chronos_zero_shot(root=target_root)
    print(f"[Chronos-2 Evaluation] Selected active weight: {summary['selected_active_weight']}")
    print(f"[Chronos-2 Evaluation] Non-locked gate passed: {summary['nonlocked_gate_passed']}")
    print(f"[Chronos-2 Evaluation] Promotion eligible: {summary['promotion_eligible']}")

    status = chronos_zero_shot_status(root=target_root)
    print("\n[Chronos-2 Status Summary]")
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
