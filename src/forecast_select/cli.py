from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import immutable_inventory
from .pipeline import ROOT, assemble_catboost_full_v2, evaluate_artifact, make_freeze_manifest, make_monthly_forecast, prepare, read_config, run_audit, run_backtest, run_catboost_chunk, run_catboost_full_v2, run_ensemble, run_level_c


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forecast-select")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    backtest = sub.add_parser("backtest")
    backtest.add_argument("--models", choices=["baselines", "classical", "catboost", "all"], default="classical")
    sub.add_parser("ensemble")
    sub.add_parser("level-c")
    catboost_full = sub.add_parser("catboost-full")
    catboost_full.add_argument("--chunk-size", type=int, default=8)
    catboost_full.add_argument("--chunk-index", type=int)
    catboost_full.add_argument("--assemble", action="store_true")
    sub.add_parser("freeze")
    sub.add_parser("locked-audit")
    sub.add_parser("train-final")
    sub.add_parser("predict-month")
    sub.add_parser("report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(ROOT)
    if args.command == "audit":
        inventory = immutable_inventory(root.parent.parent / "Downloads", root / "reports/source_inventory.json")
        path = run_audit(root)
        print(json.dumps({"inventory": inventory, "data_profile": str(path)}, indent=2))
    elif args.command == "backtest":
        baseline = ["majority", "persistence", "reversal", "momentum_3", "momentum_6", "momentum_12", "mean_reversion", "ar1", "ar2"]
        models = baseline if args.models == "baselines" else baseline + ["global_logistic"] if args.models == "classical" else ["catboost_global"] if args.models == "catboost" else read_config(root)["models"]
        output_name = {"baselines": "dev_oof.parquet", "classical": "dev_classical_oof.parquet", "catboost": "dev_catboost_oof.parquet", "all": "dev_all_oof.parquet"}[args.models]
        path = run_backtest(root, models=models, output_name=output_name)
        evaluate_artifact(root, path, f"dev_{args.models}", read_config(root)["reliability_floor"])
        print(path)
    elif args.command == "ensemble":
        print(run_ensemble(root))
    elif args.command == "level-c":
        print(run_level_c(root))
    elif args.command == "catboost-full":
        path = run_catboost_full_v2(root, chunk_size=args.chunk_size, chunk_index=args.chunk_index, assemble=args.assemble)
        if args.assemble or args.chunk_index is None:
            evaluate_artifact(root, path, "catboost_full_v2", read_config(root)["reliability_floor"])
        print(path)
    elif args.command == "freeze":
        print(make_freeze_manifest(root))
    elif args.command == "locked-audit":
        _, _, _, layout, config = prepare(root)
        path = run_backtest(root, models=["majority", "persistence", "global_logistic"], origins=layout.audit_origins, output_name="locked_audit_v1.parquet")
        evaluate_artifact(root, path, "locked_audit_v1", config["reliability_floor"])
        print(path)
    elif args.command == "train-final":
        print("Final training is represented by the frozen global_logistic method; no new audit evaluation is performed.")
    elif args.command == "predict-month":
        print(make_monthly_forecast(root))
    elif args.command == "report":
        from .reporting import write_final_report
        print(write_final_report(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
