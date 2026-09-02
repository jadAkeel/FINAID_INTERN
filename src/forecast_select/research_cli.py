from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

RESEARCH_COMMANDS = {
    "build-risk-gate",
    "show-risk-gate",
    "build-directional-downside",
    "show-directional-downside",
    "build-context-selector",
    "show-context-selector",
    "build-unified-controller",
    "show-unified-controller",
    "build-regime-adaptive",
    "show-regime-adaptive",
    "build-regime-robustness",
    "show-regime-robustness",
    "build-correctness-audit",
    "show-correctness-audit",
    "build-down-sensing",
    "show-down-sensing",
    "build-selection-score-v2",
    "show-selection-score-v2",
    "build-directional-ranker-v1",
    "show-directional-ranker-v1",
}


def register_research_subparsers(
    subparsers: Any,
    deprecated_prefix: bool = False,
) -> None:
    help_prefix = "[Research / Unpromoted] " if deprecated_prefix else ""

    subparsers.add_parser(
        "build-risk-gate",
        help=f"{help_prefix}Build the experimental Downside Risk Gate",
    )
    subparsers.add_parser(
        "show-risk-gate",
        help=f"{help_prefix}Show the experimental Downside Risk Gate result",
    )
    subparsers.add_parser(
        "build-directional-downside",
        help=f"{help_prefix}Build the experimental bidirectional top-15 selector",
    )
    subparsers.add_parser(
        "show-directional-downside",
        help=f"{help_prefix}Show the Directional Downside Selector result",
    )
    subparsers.add_parser(
        "build-context-selector",
        help=f"{help_prefix}Build the experimental Contextual Defensive Selector",
    )
    subparsers.add_parser(
        "show-context-selector",
        help=f"{help_prefix}Show the Contextual Defensive Selector result",
    )
    subparsers.add_parser(
        "build-unified-controller",
        help=f"{help_prefix}Build the non-promoting unified forecast controller",
    )
    subparsers.add_parser(
        "show-unified-controller",
        help=f"{help_prefix}Show the unified forecast controller result",
    )
    build_regime = subparsers.add_parser(
        "build-regime-adaptive",
        help=f"{help_prefix}Build the non-promoting regime-adaptive bidirectional selector",
    )
    build_regime.add_argument(
        "--cap",
        type=int,
        default=None,
        help="Override the monthly selection cap, for example 15 or 20",
    )
    subparsers.add_parser(
        "show-regime-adaptive",
        help=f"{help_prefix}Show the regime-adaptive bidirectional selector result",
    )
    subparsers.add_parser(
        "build-regime-robustness",
        help=f"{help_prefix}Build the regime-adaptive replacement robustness study",
    )
    subparsers.add_parser(
        "show-regime-robustness",
        help=f"{help_prefix}Show the regime-adaptive robustness study result",
    )
    subparsers.add_parser(
        "build-correctness-audit",
        help=f"{help_prefix}Audit score semantics and causal correctness calibration",
    )
    subparsers.add_parser(
        "show-correctness-audit",
        help=f"{help_prefix}Show the correctness-calibration audit decision",
    )
    subparsers.add_parser(
        "build-down-sensing",
        help=f"{help_prefix}Build the extreme-down sensing and guarded replacement study",
    )
    subparsers.add_parser(
        "show-down-sensing",
        help=f"{help_prefix}Show the down-sensing study result",
    )
    subparsers.add_parser(
        "build-selection-score-v2",
        help=f"{help_prefix}Build the bounded selection-score meta-ranker audit",
    )
    subparsers.add_parser(
        "show-selection-score-v2",
        help=f"{help_prefix}Show the selection-score v2 audit result",
    )
    subparsers.add_parser(
        "build-directional-ranker-v1",
        help=f"{help_prefix}Build the direct Up/Down directional ranker audit",
    )
    subparsers.add_parser(
        "show-directional-ranker-v1",
        help=f"{help_prefix}Show the directional ranker v1 audit result",
    )


def execute_research_command(
    command: str,
    args: argparse.Namespace,
    root: Path | None = None,
) -> int:
    target_root = Path(ROOT if root is None else root)

    if command == "build-risk-gate":
        mod = importlib.import_module("forecast_select.downside_pipeline")
        print(mod.build_downside_risk_gate(target_root))
    elif command == "show-risk-gate":
        mod = importlib.import_module("forecast_select.downside_pipeline")
        print(
            json.dumps(
                mod.downside_risk_gate_status(target_root),
                indent=2,
                sort_keys=True,
            )
        )
    elif command == "build-directional-downside":
        mod = importlib.import_module("forecast_select.directional_downside_pipeline")
        print(mod.build_directional_downside_selector(target_root))
    elif command == "show-directional-downside":
        mod = importlib.import_module("forecast_select.directional_downside_pipeline")
        print(
            json.dumps(
                mod.directional_downside_status(target_root),
                indent=2,
                sort_keys=True,
            )
        )
    elif command == "build-context-selector":
        mod = importlib.import_module("forecast_select.contextual_pipeline")
        print(mod.build_contextual_defensive_selector(target_root))
    elif command == "show-context-selector":
        mod = importlib.import_module("forecast_select.contextual_pipeline")
        print(
            json.dumps(
                mod.contextual_defensive_status(target_root),
                indent=2,
                sort_keys=True,
            )
        )
    elif command == "build-unified-controller":
        mod = importlib.import_module("forecast_select.unified_pipeline")
        print(mod.build_unified_controller(target_root))
    elif command == "show-unified-controller":
        mod = importlib.import_module("forecast_select.unified_pipeline")
        print(
            json.dumps(
                mod.unified_controller_status(target_root),
                indent=2,
                sort_keys=True,
            )
        )
    elif command == "build-regime-adaptive":
        mod = importlib.import_module("forecast_select.regime_adaptive_pipeline")
        cap = getattr(args, "cap", None)
        print(mod.build_regime_adaptive_selector(target_root, cap=cap))
    elif command == "show-regime-adaptive":
        mod = importlib.import_module("forecast_select.regime_adaptive_pipeline")
        print(
            json.dumps(
                mod.regime_adaptive_status(target_root),
                indent=2,
                sort_keys=True,
            )
        )
    elif command == "build-regime-robustness":
        mod = importlib.import_module("forecast_select.robustness_pipeline")
        print(mod.build_regime_adaptive_robustness(target_root))
    elif command == "show-regime-robustness":
        mod = importlib.import_module("forecast_select.robustness_pipeline")
        print(
            json.dumps(
                mod.regime_adaptive_robustness_status(target_root),
                indent=2,
                sort_keys=True,
            )
        )
    elif command == "build-correctness-audit":
        mod = importlib.import_module("forecast_select.calibration_audit")
        print(mod.build_correctness_calibration_audit(target_root))
    elif command == "show-correctness-audit":
        mod = importlib.import_module("forecast_select.calibration_audit")
        print(
            json.dumps(
                mod.correctness_calibration_status(target_root),
                indent=2,
                sort_keys=True,
            )
        )
    elif command == "build-down-sensing":
        mod = importlib.import_module("forecast_select.down_sensing_pipeline")
        print(mod.build_down_sensing_gate(target_root))
    elif command == "show-down-sensing":
        mod = importlib.import_module("forecast_select.down_sensing_pipeline")
        print(
            json.dumps(
                mod.down_sensing_status(target_root),
                indent=2,
                sort_keys=True,
            )
        )
    elif command == "build-selection-score-v2":
        mod = importlib.import_module("forecast_select.selection_score_v2_runner")
        print(mod.build_selection_score_v2_audit(target_root))
    elif command == "show-selection-score-v2":
        mod = importlib.import_module("forecast_select.selection_score_v2_runner")
        print(
            json.dumps(
                mod.selection_score_v2_status(target_root),
                indent=2,
                sort_keys=True,
            )
        )
    elif command == "build-directional-ranker-v1":
        mod = importlib.import_module("forecast_select.directional_ranker_v1_runner")
        print(mod.build_directional_ranker_v1_audit(target_root))
    elif command == "show-directional-ranker-v1":
        mod = importlib.import_module("forecast_select.directional_ranker_v1_runner")
        print(
            json.dumps(
                mod.directional_ranker_v1_status(target_root),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        raise ValueError(f"Unknown research command: {command}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forecast-select-research",
        description="Research, audit, and experimental commands for forecast-select",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    register_research_subparsers(sub, deprecated_prefix=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return execute_research_command(args.command, args, ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
