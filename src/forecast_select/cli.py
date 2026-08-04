from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contextual_pipeline import (
    build_contextual_defensive_selector,
    contextual_defensive_status,
)
from .downside_pipeline import (
    build_downside_risk_gate,
    downside_risk_gate_status,
)
from .directional_downside_pipeline import (
    build_directional_downside_selector,
    directional_downside_status,
)
from .future_forecast import write_next_three_forecast
from .project import audit_data, check_project
from .uptrend_pipeline import ROOT, active_model_status, build_active_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forecast-select",
        description="Readable monthly directional forecasting research pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit-data", help="Validate and profile the input workbook")
    sub.add_parser("build-model", help="Validate or build the Uptrend Selector")
    sub.add_parser("show-results", help="Show the registered model result")
    sub.add_parser(
        "build-risk-gate",
        help="Build the experimental Downside Risk Gate",
    )
    sub.add_parser(
        "show-risk-gate",
        help="Show the experimental Downside Risk Gate result",
    )
    sub.add_parser(
        "build-directional-downside",
        help="Build the experimental bidirectional top-15 selector",
    )
    sub.add_parser(
        "show-directional-downside",
        help="Show the Directional Downside Selector result",
    )
    sub.add_parser(
        "build-context-selector",
        help="Build the experimental Contextual Defensive Selector",
    )
    sub.add_parser(
        "show-context-selector",
        help="Show the Contextual Defensive Selector result",
    )
    sub.add_parser(
        "forecast-next-three",
        help="Forecast the next three monthly directions without future values",
    )
    sub.add_parser("check-project", help="Check active artifact integrity")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(ROOT)
    if args.command == "audit-data":
        print(audit_data(root))
    elif args.command == "build-model":
        print(build_active_model(root))
    elif args.command == "show-results":
        print(json.dumps(active_model_status(root), indent=2, sort_keys=True))
    elif args.command == "build-risk-gate":
        print(build_downside_risk_gate(root))
    elif args.command == "show-risk-gate":
        print(json.dumps(
            downside_risk_gate_status(root),
            indent=2,
            sort_keys=True,
        ))
    elif args.command == "build-directional-downside":
        print(build_directional_downside_selector(root))
    elif args.command == "show-directional-downside":
        print(json.dumps(
            directional_downside_status(root),
            indent=2,
            sort_keys=True,
        ))
    elif args.command == "build-context-selector":
        print(build_contextual_defensive_selector(root))
    elif args.command == "show-context-selector":
        print(json.dumps(
            contextual_defensive_status(root),
            indent=2,
            sort_keys=True,
        ))
    elif args.command == "forecast-next-three":
        print(write_next_three_forecast(root))
    elif args.command == "check-project":
        print(check_project(root))
    return 0
