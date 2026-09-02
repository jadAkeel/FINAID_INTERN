from __future__ import annotations

import argparse
import json
from pathlib import Path

from .active_model import active_model_status, build_active_model
from .future_forecast import (
    write_next_three_forecast as write_uptrend_next_three_forecast,
)
from .future_regime_forecast import write_regime_adaptive_next_three_forecast
from .project import audit_data, check_project
from .research_cli import (
    RESEARCH_COMMANDS,
    execute_research_command,
    register_research_subparsers,
)
from .uptrend_pipeline import (
    ROOT,
    active_model_status as uptrend_model_status,
    build_active_model as build_uptrend_model,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forecast-select",
        description="Readable monthly directional forecasting research pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Production Core Commands
    sub.add_parser("audit-data", help="Validate and profile the input workbook")
    sub.add_parser(
        "build-model",
        help="Validate or build the active Regime Adaptive model",
    )
    sub.add_parser("show-results", help="Show the active model result")
    sub.add_parser(
        "build-uptrend-model",
        help="Validate or build the retained Uptrend Selector baseline",
    )
    sub.add_parser(
        "show-uptrend-results",
        help="Show the retained Uptrend Selector baseline result",
    )
    sub.add_parser(
        "forecast-next-three",
        help="Forecast three horizons with the active Regime Adaptive model",
    )
    sub.add_parser(
        "forecast-uptrend-next-three",
        help="Run the retained Uptrend-only three-horizon forecast",
    )
    sub.add_parser(
        "forecast-regime-next-three",
        help="Forecast three direct horizons with the frozen regime-adaptive policy",
    )
    sub.add_parser("check-project", help="Check active artifact integrity")

    # Research / Experimental Compatibility Routes (deferred execution)
    register_research_subparsers(sub, deprecated_prefix=True)

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
    elif args.command == "build-uptrend-model":
        print(build_uptrend_model(root))
    elif args.command == "show-uptrend-results":
        print(json.dumps(uptrend_model_status(root), indent=2, sort_keys=True))
    elif args.command == "forecast-next-three":
        print(write_regime_adaptive_next_three_forecast(root))
    elif args.command == "forecast-uptrend-next-three":
        print(write_uptrend_next_three_forecast(root))
    elif args.command == "forecast-regime-next-three":
        print(write_regime_adaptive_next_three_forecast(root))
    elif args.command == "check-project":
        print(check_project(root))
    elif args.command in RESEARCH_COMMANDS:
        return execute_research_command(args.command, args, root)
    else:
        raise ValueError(f"Unknown command: {args.command}")
    return 0
