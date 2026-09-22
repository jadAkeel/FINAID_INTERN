"""End-to-end runner for the Local / Interaction Logistic Up Selector study.

    python -m forecast_select.local_logistic_runner tune
    python -m forecast_select.local_logistic_runner evaluate
    python -m forecast_select.local_logistic_runner all

``tune`` searches the local feature set, local ``C``, minimum local history,
shrinkage weight and interaction ``C`` on the tuning origins 120-179 only.
``evaluate`` freezes that choice and scores every candidate on tuning,
validation and confirmation without any further search.

The locked origins 268-315 are unreachable: the workbook is read with
``nrows`` capped at position 267 and every origin list passes through
``assert_origins_unlocked``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .io import atomic_write_json, atomic_write_parquet
from .local_logistic import sample_aware_weights, shrink_probabilities
from .local_logistic_metrics import (
    calibration_table,
    paired_comparison,
    per_indicator_table,
    raw_metrics,
    selection_overlap,
    selector_metrics,
)
from .local_logistic_pipeline import (
    ROOT,
    LocalSpec,
    apply_selector,
    assert_origins_unlocked,
    effective_local,
    load_experiment_settings,
    prepare_experiment_panel,
    run_walk_forward,
)


RESEARCH_DIR = "research/local_logistic"


def _artifacts(root: Path) -> Path:
    path = root / RESEARCH_DIR / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _metrics_dir(root: Path) -> Path:
    path = root / RESEARCH_DIR / "metrics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _origins(window) -> list[int]:
    start, end = (int(value) for value in window)
    return list(range(start, end + 1))


def _local_specs(settings: dict) -> list[LocalSpec]:
    local = settings["local_model"]
    return [
        LocalSpec(feature_set=name, logistic_c=float(value))
        for name in local["feature_set_grid"]
        for value in local["logistic_c_grid"]
    ]


# ---------------------------------------------------------------------------
# Stage 1 - tuning search (tuning origins only)
# ---------------------------------------------------------------------------


def run_tuning(root: Path = ROOT, force: bool = False) -> dict:
    settings = load_experiment_settings(root)
    tuning = _origins(settings["tuning_origins"])
    assert_origins_unlocked(tuning, settings)
    artifacts = _artifacts(root)
    signal_path = artifacts / "tuning_signals.parquet"

    shared = prepare_experiment_panel(root)
    if signal_path.exists() and not force:
        signals = pd.read_parquet(signal_path)
        print(f"reused cached tuning signals ({len(signals)} rows)")
    else:
        print(f"fitting tuning walk-forward over origins {tuning[0]}-{tuning[-1]}")
        produced = run_walk_forward(
            shared,
            tuning,
            local_specs=_local_specs(settings),
            interaction_cs=settings["interaction_model"]["logistic_c_grid"],
            progress=True,
        )
        signals = produced["signals"]
        atomic_write_parquet(signals, signal_path)
        atomic_write_json(produced["model_size"], artifacts / "model_size.json")
        print(f"tuning walk-forward took {produced['runtime_seconds']:.0f}s")

    window = tuple(int(v) for v in settings["tuning_origins"])
    baseline = apply_selector(signals, "p_global", shared)
    baseline_metrics = selector_metrics(baseline, window)
    print("tuning baseline:", baseline_metrics["hits"], "/", baseline_metrics["calls"])

    records = []
    local_grid = settings["local_model"]["minimum_local_rows_grid"]
    weight_grid = settings["shrinkage"]["local_weight_grid"]
    for spec in _local_specs(settings):
        for minimum_rows in local_grid:
            p_local, fallback = effective_local(signals, spec.key, int(minimum_rows))
            for weight in [1.0, *weight_grid]:
                column = "_candidate"
                signals[column] = shrink_probabilities(
                    signals["p_global"].to_numpy(dtype=float), p_local, float(weight)
                )
                result = apply_selector(signals, column, shared)
                metrics = selector_metrics(result, window)
                records.append({
                    "family": "pure_local" if weight == 1.0 else "fixed_shrinkage",
                    "feature_set": spec.feature_set,
                    "logistic_c": spec.logistic_c,
                    "minimum_local_rows": int(minimum_rows),
                    "local_weight": float(weight),
                    "fallback_rate": float(fallback.mean()),
                    **metrics,
                    "hits_vs_baseline": metrics["hits"] - baseline_metrics["hits"],
                })
            print(f"  searched {spec.key} min_rows={minimum_rows}", flush=True)

    for value in settings["interaction_model"]["logistic_c_grid"]:
        column = f"p_interaction__c{float(value):g}"
        result = apply_selector(signals, column, shared)
        metrics = selector_metrics(result, window)
        records.append({
            "family": "interaction",
            "feature_set": "",
            "logistic_c": float(value),
            "minimum_local_rows": 0,
            "local_weight": float("nan"),
            "fallback_rate": float("nan"),
            **metrics,
            "hits_vs_baseline": metrics["hits"] - baseline_metrics["hits"],
        })
    signals.drop(columns=[c for c in signals.columns if c == "_candidate"], inplace=True)

    search = pd.DataFrame(records)
    search.to_csv(_metrics_dir(root) / "tuning_search.csv", index=False)

    def _best(family: str) -> dict:
        subset = search[search["family"].eq(family)].copy()
        subset = subset.sort_values(
            ["accuracy", "logistic_c", "minimum_local_rows", "local_weight"],
            ascending=[False, True, True, True],
        )
        return subset.iloc[0].to_dict()

    best_local = _best("pure_local")
    best_shrinkage = _best("fixed_shrinkage")
    best_interaction = _best("interaction")

    # Sample-aware shrinkage reuses the frozen fixed-shrinkage feature set,
    # local C and minimum-history rule; only w_max and n_ref are searched.
    aware_records = []
    spec_key = LocalSpec(
        str(best_shrinkage["feature_set"]), float(best_shrinkage["logistic_c"])
    ).key
    minimum_rows = int(best_shrinkage["minimum_local_rows"])
    p_local, fallback = effective_local(signals, spec_key, minimum_rows)
    rows = signals["local_training_rows"].to_numpy(dtype=float)
    for maximum_weight in weight_grid:
        for reference in settings["shrinkage"]["sample_aware_reference_rows_grid"]:
            if int(reference) <= minimum_rows:
                continue
            weights = sample_aware_weights(
                rows, fallback, float(maximum_weight), minimum_rows, int(reference)
            )
            signals["_candidate"] = shrink_probabilities(
                signals["p_global"].to_numpy(dtype=float), p_local, weights
            )
            result = apply_selector(signals, "_candidate", shared)
            metrics = selector_metrics(result, window)
            aware_records.append({
                "family": "sample_aware_shrinkage",
                "feature_set": best_shrinkage["feature_set"],
                "logistic_c": float(best_shrinkage["logistic_c"]),
                "minimum_local_rows": minimum_rows,
                "maximum_local_weight": float(maximum_weight),
                "reference_rows": int(reference),
                "mean_effective_weight": float(weights.mean()),
                **metrics,
                "hits_vs_baseline": metrics["hits"] - baseline_metrics["hits"],
            })
    signals.drop(columns=[c for c in signals.columns if c == "_candidate"], inplace=True)
    aware = pd.DataFrame(aware_records)
    aware.to_csv(_metrics_dir(root) / "tuning_search_sample_aware.csv", index=False)
    best_aware = aware.sort_values(
        ["accuracy", "maximum_local_weight", "reference_rows"],
        ascending=[False, True, True],
    ).iloc[0].to_dict()

    frozen = {
        "tuning_window": list(window),
        "baseline": baseline_metrics,
        "pure_local": {
            "feature_set": str(best_local["feature_set"]),
            "logistic_c": float(best_local["logistic_c"]),
            "minimum_local_rows": int(best_local["minimum_local_rows"]),
            "local_weight": 1.0,
            "tuning_accuracy": float(best_local["accuracy"]),
        },
        "fixed_shrinkage": {
            "feature_set": str(best_shrinkage["feature_set"]),
            "logistic_c": float(best_shrinkage["logistic_c"]),
            "minimum_local_rows": int(best_shrinkage["minimum_local_rows"]),
            "local_weight": float(best_shrinkage["local_weight"]),
            "tuning_accuracy": float(best_shrinkage["accuracy"]),
        },
        "sample_aware_shrinkage": {
            "feature_set": str(best_aware["feature_set"]),
            "logistic_c": float(best_aware["logistic_c"]),
            "minimum_local_rows": int(best_aware["minimum_local_rows"]),
            "maximum_local_weight": float(best_aware["maximum_local_weight"]),
            "reference_rows": int(best_aware["reference_rows"]),
            "tuning_accuracy": float(best_aware["accuracy"]),
        },
        "interaction": {
            "logistic_c": float(best_interaction["logistic_c"]),
            "features": list(settings["interaction_model"]["features"]),
            "tuning_accuracy": float(best_interaction["accuracy"]),
        },
    }
    atomic_write_json(frozen, _metrics_dir(root) / "frozen_configuration.json")
    print(json.dumps(frozen, indent=2, default=str))
    return frozen


# ---------------------------------------------------------------------------
# Stage 2 - frozen evaluation across all three periods
# ---------------------------------------------------------------------------


def _candidate_columns(
    signals: pd.DataFrame, frozen: dict, settings: dict
) -> dict[str, str]:
    """Attach every frozen candidate probability to ``signals``."""
    p_global = signals["p_global"].to_numpy(dtype=float)
    rows = signals["local_training_rows"].to_numpy(dtype=float)

    pure = frozen["pure_local"]
    pure_key = LocalSpec(pure["feature_set"], float(pure["logistic_c"])).key
    p_pure, fallback_pure = effective_local(
        signals, pure_key, int(pure["minimum_local_rows"])
    )
    signals["p_local"] = p_pure
    signals["local_fallback_used"] = fallback_pure

    fixed = frozen["fixed_shrinkage"]
    fixed_key = LocalSpec(fixed["feature_set"], float(fixed["logistic_c"])).key
    p_fixed_local, fallback_fixed = effective_local(
        signals, fixed_key, int(fixed["minimum_local_rows"])
    )
    signals["p_hybrid_local_component"] = p_fixed_local
    signals["hybrid_fallback_used"] = fallback_fixed
    signals["p_hybrid"] = shrink_probabilities(
        p_global, p_fixed_local, float(fixed["local_weight"])
    )

    aware = frozen["sample_aware_shrinkage"]
    aware_key = LocalSpec(aware["feature_set"], float(aware["logistic_c"])).key
    p_aware_local, fallback_aware = effective_local(
        signals, aware_key, int(aware["minimum_local_rows"])
    )
    weights = sample_aware_weights(
        rows,
        fallback_aware,
        float(aware["maximum_local_weight"]),
        int(aware["minimum_local_rows"]),
        int(aware["reference_rows"]),
    )
    signals["sample_aware_weight"] = weights
    signals["p_hybrid_sample_aware"] = shrink_probabilities(
        p_global, p_aware_local, weights
    )

    signals["p_interaction"] = signals[
        f"p_interaction__c{float(frozen['interaction']['logistic_c']):g}"
    ].to_numpy(dtype=float)

    return {
        "A_global": "p_global",
        "B_pure_local": "p_local",
        "C_fixed_shrinkage": "p_hybrid",
        "D_sample_aware_shrinkage": "p_hybrid_sample_aware",
        "E_interaction": "p_interaction",
    }


def run_evaluation(root: Path = ROOT, force: bool = False) -> dict:
    settings = load_experiment_settings(root)
    frozen_path = _metrics_dir(root) / "frozen_configuration.json"
    if not frozen_path.exists():
        raise SystemExit("Run the tune stage first")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))

    start = int(settings["tuning_origins"][0])
    end = int(settings["confirmation_origins"][1])
    origins = list(range(start, end + 1))
    assert_origins_unlocked(origins, settings)

    artifacts = _artifacts(root)
    signal_path = artifacts / "full_signals.parquet"
    shared = prepare_experiment_panel(root)

    specs = sorted({
        LocalSpec(frozen[family]["feature_set"], float(frozen[family]["logistic_c"]))
        for family in ("pure_local", "fixed_shrinkage", "sample_aware_shrinkage")
    }, key=lambda spec: spec.key)

    if signal_path.exists() and not force:
        signals = pd.read_parquet(signal_path)
        print(f"reused cached full signals ({len(signals)} rows)")
    else:
        print(f"fitting frozen walk-forward over origins {start}-{end}")
        produced = run_walk_forward(
            shared,
            origins,
            local_specs=specs,
            interaction_cs=[float(frozen["interaction"]["logistic_c"])],
            collect_coefficients=True,
            progress=True,
        )
        signals = produced["signals"]
        atomic_write_parquet(signals, signal_path)
        atomic_write_parquet(
            produced["local_coefficients"], artifacts / "local_coefficients.parquet"
        )
        atomic_write_parquet(
            produced["interaction_coefficients"],
            artifacts / "interaction_coefficients.parquet",
        )
        atomic_write_json(produced["model_size"], artifacts / "model_size_frozen.json")
        print(f"frozen walk-forward took {produced['runtime_seconds']:.0f}s")

    columns = _candidate_columns(signals, frozen, settings)
    results = {
        name: apply_selector(signals, column, shared)
        for name, column in columns.items()
    }

    # Every candidate must have been scored on the identical eligible rows.
    reference = results["A_global"][["origin_position", "indicator_id"]]
    for name, result in results.items():
        current = result[["origin_position", "indicator_id"]]
        if not current.sort_values(
            ["origin_position", "indicator_id"]
        ).reset_index(drop=True).equals(
            reference.sort_values(
                ["origin_position", "indicator_id"]
            ).reset_index(drop=True)
        ):
            raise AssertionError(f"Candidate {name} scored a different row set")

    windows = {
        "tuning": tuple(int(v) for v in settings["tuning_origins"]),
        "validation": tuple(int(v) for v in settings["validation_origins"]),
        "confirmation": tuple(int(v) for v in settings["confirmation_origins"]),
    }
    block = int(settings["bootstrap"]["block_months"])
    replicates = int(settings["bootstrap"]["replicates"])
    seed = int(read_seed(root))

    summary: dict = {
        "experiment_id": settings["experiment_id"],
        "frozen_configuration": frozen,
        "locked_origins": settings["locked_origins"],
        "locked_origins_read": False,
        "maximum_readable_position": int(settings["maximum_readable_position"]),
        "maximum_origin_evaluated": int(signals["origin_position"].max()),
        "periods": {},
    }
    for period, window in windows.items():
        block_summary: dict = {"window": list(window), "candidates": {}}
        for name, result in results.items():
            block_summary["candidates"][name] = {
                "selector": selector_metrics(result, window),
                "raw": raw_metrics(signals, columns[name], window),
            }
            if name != "A_global":
                block_summary["candidates"][name]["paired_vs_global"] = (
                    paired_comparison(
                        results["A_global"], result, window, block, replicates, seed
                    )
                )
                _, overlap = selection_overlap(results["A_global"], result, window)
                block_summary["candidates"][name]["overlap_vs_global"] = overlap
        summary["periods"][period] = block_summary

    # Fallback behaviour.
    summary["fallback"] = {
        "pure_local_rate": float(signals["local_fallback_used"].mean()),
        "hybrid_rate": float(signals["hybrid_fallback_used"].mean()),
        "mean_local_training_rows": float(signals["local_training_rows"].mean()),
        "min_local_training_rows": int(signals["local_training_rows"].min()),
        "max_local_training_rows": int(signals["local_training_rows"].max()),
        "mean_sample_aware_weight": float(signals["sample_aware_weight"].mean()),
        "by_period": {
            period: float(
                signals.loc[
                    signals["origin_position"].between(*window),
                    "hybrid_fallback_used",
                ].mean()
            )
            for period, window in windows.items()
        },
    }

    atomic_write_json(summary, _metrics_dir(root) / "summary.json")

    # ---- auditable per-row artifact -------------------------------------
    predictions = signals[[
        "origin_position", "origin_date", "target_date", "indicator_id", "y_true",
        "p_global", "p_local", "p_hybrid", "p_hybrid_sample_aware", "p_interaction",
        "local_training_rows", "local_fallback_used", "hybrid_fallback_used",
        "sample_aware_weight",
    ]].copy()
    predictions = predictions.rename(
        columns={"local_training_rows": "local_training_sample_count"}
    )
    for name, result in results.items():
        short = name.split("_", 1)[1]
        indexed = result.set_index(["origin_position", "indicator_id"])
        keys = pd.MultiIndex.from_arrays([
            predictions["origin_position"], predictions["indicator_id"]
        ])
        predictions[f"selected_{short}"] = (
            indexed["accepted"].reindex(keys).fillna(False).to_numpy(dtype=bool)
        )
        predictions[f"rank_{short}"] = (
            indexed["selection_rank"].reindex(keys).to_numpy()
        )
        predictions[f"score_{short}"] = (
            indexed["selection_score"].reindex(keys).to_numpy()
        )
        direction = indexed["predicted_direction"].reindex(keys).to_numpy()
        predictions[f"direction_{short}"] = direction
        predictions[f"correct_{short}"] = np.where(
            predictions["y_true"].notna(),
            (direction == "Up").astype(float)
            == predictions["y_true"].to_numpy(dtype=float),
            np.nan,
        )
    predictions["period"] = np.select(
        [
            predictions["origin_position"].between(*windows["tuning"]),
            predictions["origin_position"].between(*windows["validation"]),
            predictions["origin_position"].between(*windows["confirmation"]),
        ],
        ["tuning", "validation", "confirmation"],
        default="",
    )
    atomic_write_parquet(predictions, artifacts / "predictions.parquet")
    predictions.to_csv(artifacts / "predictions.csv", index=False)

    # ---- overlap detail, per-indicator, calibration ----------------------
    for name in results:
        if name == "A_global":
            continue
        frames = []
        for period, window in windows.items():
            frame, _ = selection_overlap(results["A_global"], results[name], window)
            if not frame.empty:
                frame.insert(0, "period", period)
                frames.append(frame)
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(
                _metrics_dir(root) / f"overlap_{name}.csv", index=False
            )

    per_indicator = []
    for period, window in windows.items():
        table = per_indicator_table(signals, results, window, columns)
        table.insert(0, "period", period)
        per_indicator.append(table)
    pd.concat(per_indicator, ignore_index=True).to_csv(
        _metrics_dir(root) / "per_indicator.csv", index=False
    )

    calibration = []
    for period, window in windows.items():
        for name, column in columns.items():
            table = calibration_table(signals, column, window)
            if table.empty:
                continue
            table.insert(0, "period", period)
            table.insert(1, "candidate", name)
            calibration.append(table)
    if calibration:
        pd.concat(calibration, ignore_index=True).to_csv(
            _metrics_dir(root) / "calibration.csv", index=False
        )

    coefficient_path = artifacts / "local_coefficients.parquet"
    if coefficient_path.exists():
        stability = coefficient_stability(pd.read_parquet(coefficient_path), frozen)
        stability.to_csv(_metrics_dir(root) / "coefficient_stability.csv", index=False)

    print(json.dumps(summary["periods"], indent=2, default=str)[:4000])
    return summary


def coefficient_stability(coefficients: pd.DataFrame, frozen: dict) -> pd.DataFrame:
    """Cross-indicator and across-origin dispersion of the local slopes."""
    if coefficients.empty:
        return pd.DataFrame()
    key = LocalSpec(
        frozen["fixed_shrinkage"]["feature_set"],
        float(frozen["fixed_shrinkage"]["logistic_c"]),
    ).key
    subset = coefficients[coefficients["spec"].eq(key)]
    if subset.empty:
        subset = coefficients
    columns = [c for c in subset.columns if c.startswith("coef_")]
    records = []
    for column in columns:
        per_indicator = subset.groupby("indicator_id")[column].mean()
        across_origins = subset.groupby("indicator_id")[column].std(ddof=1)
        sign_share = subset.groupby("indicator_id")[column].apply(
            lambda values: float(np.mean(np.sign(values) == np.sign(values.mean())))
        )
        records.append({
            "feature": column.removeprefix("coef_"),
            "mean_coefficient": float(per_indicator.mean()),
            "cross_indicator_std": float(per_indicator.std(ddof=1)),
            "cross_indicator_min": float(per_indicator.min()),
            "cross_indicator_max": float(per_indicator.max()),
            "mean_within_indicator_std_over_origins": float(across_origins.mean()),
            "variance_ratio_within_over_between": (
                float(across_origins.mean() ** 2 / per_indicator.std(ddof=1) ** 2)
                if per_indicator.std(ddof=1) > 0
                else float("nan")
            ),
            "mean_sign_consistency": float(sign_share.mean()),
            "indicators_positive": int((per_indicator > 0).sum()),
            "indicators_negative": int((per_indicator < 0).sum()),
        })
    return pd.DataFrame(records)


def read_seed(root: Path) -> int:
    import yaml

    with (root / "configs/config.yaml").open(encoding="utf-8") as handle:
        return int(yaml.safe_load(handle)["seed"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="local-logistic-experiment",
        description="Local / Interaction Logistic Up Selector research runner",
    )
    parser.add_argument("stage", choices=["tune", "evaluate", "all"])
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if args.stage in ("tune", "all"):
        run_tuning(root, force=args.force)
    if args.stage in ("evaluate", "all"):
        run_evaluation(root, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
