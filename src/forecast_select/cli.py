from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    elif args.command == "check-project":
        print(check_project(root))
    return 0
