from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .expansion_experiment_runner import _reference_bundle
from .expansion_quality import paired_monthly_hit_bootstrap
from .experiment_cache import (
    replay_cache_generation,
    write_experiment_ledger_row,
)
from .features import FEATURE_FAMILY_COLUMNS
from .indicator_selection import summarize_selected_predictions
from .io import atomic_write_json, atomic_write_parquet
from .regime_adaptive_pipeline import _apply, _build_inputs
from .regime_experiment_runner import _band_metrics, _metrics, _window
from .uptrend_pipeline import (
    ROOT,
    _configuration_hash,
    build_uptrend_predictions,
)


EXPERIMENT_RELEASE = "top15_feature_family_ablation_v1"


def _experiment_key(
    root: Path,
    family: str,
    reference_key: str,
) -> str:
    payload = {
        "experiment_release": EXPERIMENT_RELEASE,
        "family": family,
        "feature_columns": FEATURE_FAMILY_COLUMNS[family],
        "reference_cache_key": reference_key,
        "uptrend_config_hash": _configuration_hash(root, (family,)),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_root(root: Path, family: str) -> Path:
    return (
        root
        / "research/regime_adaptive_selector/feature_ablation"
        / family
    )


def _add_base_up_rank(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    ready = result["level_c_ready"].fillna(False).astype(bool)
    result["base_up_rank"] = np.nan
    result.loc[ready, "base_up_rank"] = (
        result.loc[ready]
        .groupby("origin_position")["p_up_selection_score"]
        .rank(method="first", ascending=False)
    )
    return result


def _baseline_base(source: pd.DataFrame) -> pd.DataFrame:
    result = source.copy()
    result["accepted"] = result["base_accepted"].fillna(False).astype(bool)
    result["predicted_direction"] = result[
        "base_predicted_direction"
    ].astype(str)
    return result


def _window_metrics(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    return {
        "early_tuning": _metrics(_window(frame, [120, 149])),
        "late_tuning": _metrics(_window(frame, [150, 179])),
        "tuning": _metrics(_window(frame, [120, 179])),
        "validation": _metrics(_window(frame, [180, 219])),
        "confirmation_descriptive": _metrics(_window(frame, [220, 266])),
    }


def _nonnegative_windows(
    candidate: dict[str, dict[str, float | int]],
    baseline: dict[str, dict[str, float | int]],
) -> bool:
    return all(
        candidate[name]["hits"] >= baseline[name]["hits"]
        and candidate[name]["accuracy"] >= baseline[name]["accuracy"]
        for name in ["early_tuning", "late_tuning", "validation"]
    )


def _load_cached_ablation(
    root: Path,
    family: str,
    experiment_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    output_root = _artifact_root(root, family)
    summary_path = output_root / "summary.json"
    base_path = output_root / "base_predictions.parquet"
    adaptive_path = output_root / "adaptive_predictions.parquet"
    if not all(path.exists() for path in [summary_path, base_path, adaptive_path]):
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("experiment_key") != experiment_key:
        return None
    base = pd.read_parquet(base_path)
    adaptive = pd.read_parquet(adaptive_path)
    for frame in [base, adaptive]:
        if int(frame["origin_position"].max()) >= 268:
            raise ValueError("Feature ablation cache includes locked origins")
        if frame["locked_evaluation_read"].fillna(True).astype(bool).any():
            raise ValueError("Feature ablation cache reports locked evidence")
    return base, adaptive


def _build_exact_ablation(
    root: Path,
    family: str,
    settings: dict,
    selected_params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    started = perf_counter()
    base = build_uptrend_predictions(
        root,
        origin_range=(120, 266),
        feature_families=(family,),
    )
    base["locked_evaluation_read"] = False
    inputs = _build_inputs(
        root,
        settings,
        int(settings["selection"]["maximum_selection_count"]),
        base_predictions=base,
    )
    inputs = _add_base_up_rank(inputs)
    adaptive = _apply(inputs, settings, selected_params, cap=None)
    selected = adaptive[adaptive["accepted"].fillna(False).astype(bool)]
    monthly = selected.groupby("origin_position")["indicator_id"].agg(
        ["count", "nunique"]
    )
    if not monthly["count"].between(15, 20).all():
        raise AssertionError("Feature ablation violated the 15-20 call contract")
    if not monthly["count"].eq(monthly["nunique"]).all():
        raise AssertionError("Feature ablation selected duplicate indicators")
    adaptive["locked_evaluation_read"] = False
    adaptive["feature_ablation_family"] = family
    adaptive["feature_ablation_release"] = EXPERIMENT_RELEASE
    runtime_seconds = perf_counter() - started
    return base, adaptive, runtime_seconds


def _ledger_row(
    family: str,
    adaptive: pd.DataFrame,
    adaptive_windows: dict[str, dict[str, float | int]],
    bootstrap: dict[str, float | int],
    runtime_seconds: float,
    accepted: bool,
    rejection_reasons: list[str],
    reference_key,
    root: Path,
) -> dict[str, object]:
    all_metrics = _metrics(adaptive)
    core = _band_metrics(adaptive, 1, 15)
    middle = _band_metrics(adaptive, 16, 17)
    tail = _band_metrics(adaptive, 18, 20)
    selected = adaptive[adaptive["accepted"].fillna(False).astype(bool)]
    down = selected[selected["predicted_direction"].eq("Down")]
    down_hits = int(down["y_true"].eq(0.0).sum())
    caps = selected.groupby("origin_position").size().value_counts().sort_index()
    return {
        "experiment_id": f"phase2_feature_{family}",
        "hypothesis": f"The {family} family improves the Top-15 core",
        "changed_variables": json.dumps(FEATURE_FAMILY_COLUMNS[family]),
        "selected_on": "tuning_120_179; validation_gate; confirmation_descriptive",
        "data_hash": reference_key.source_data_hash,
        "config_hash": _configuration_hash(root, (family,)),
        **all_metrics,
        "early_tuning_calls": adaptive_windows["early_tuning"]["calls"],
        "early_tuning_hits": adaptive_windows["early_tuning"]["hits"],
        "early_tuning_accuracy": adaptive_windows["early_tuning"]["accuracy"],
        "late_tuning_calls": adaptive_windows["late_tuning"]["calls"],
        "late_tuning_hits": adaptive_windows["late_tuning"]["hits"],
        "late_tuning_accuracy": adaptive_windows["late_tuning"]["accuracy"],
        "validation_calls": adaptive_windows["validation"]["calls"],
        "validation_hits": adaptive_windows["validation"]["hits"],
        "validation_accuracy": adaptive_windows["validation"]["accuracy"],
        "confirmation_calls": adaptive_windows[
            "confirmation_descriptive"
        ]["calls"],
        "confirmation_hits": adaptive_windows[
            "confirmation_descriptive"
        ]["hits"],
        "confirmation_accuracy": adaptive_windows[
            "confirmation_descriptive"
        ]["accuracy"],
        "cap_distribution": json.dumps(
            {str(int(cap)): int(count) for cap, count in caps.items()},
            sort_keys=True,
        ),
        "rank_1_15_calls": core["calls"],
        "rank_1_15_hits": core["hits"],
        "rank_1_15_accuracy": core["accuracy"],
        "rank_16_17_calls": middle["calls"],
        "rank_16_17_hits": middle["hits"],
        "rank_16_17_accuracy": middle["accuracy"],
        "rank_18_20_calls": tail["calls"],
        "rank_18_20_hits": tail["hits"],
        "rank_18_20_accuracy": tail["accuracy"],
        "down_calls": int(len(down)),
        "down_hits": down_hits,
        "down_precision": float(down_hits / len(down)) if len(down) else np.nan,
        "bootstrap_p10": bootstrap["bootstrap_p10"],
        "bootstrap_p90": bootstrap["bootstrap_p90"],
        "runtime_seconds": runtime_seconds,
        "cache_status": "exact_build",
        "cache_key": reference_key.digest(),
        "cache_generation": replay_cache_generation(root, reference_key),
        "accepted": accepted,
        "rejection_reason": "|".join(rejection_reasons),
    }


def build_feature_family_ablation(
    family: str,
    root: Path = ROOT,
) -> dict[str, object]:
    if family not in FEATURE_FAMILY_COLUMNS:
        raise ValueError(f"Unknown feature family: {family}")
    (
        _,
        _,
        source,
        settings,
        project,
        reference_summary,
        reference_key,
    ) = _reference_bundle(root)
    experiment_key = _experiment_key(root, family, reference_key.digest())
    cached = _load_cached_ablation(root, family, experiment_key)
    if cached is None:
        base, adaptive, runtime_seconds = _build_exact_ablation(
            root,
            family,
            settings,
            dict(reference_summary["selected_parameters"]),
        )
        cache_status = "exact_build"
    else:
        base, adaptive = cached
        runtime_seconds = 0.0
        cache_status = "hit"

    base_baseline = _baseline_base(source)
    base_windows = _window_metrics(base)
    base_baseline_windows = _window_metrics(base_baseline)
    adaptive_windows = _window_metrics(adaptive)
    adaptive_baseline_windows = _window_metrics(source)
    base_gate = _nonnegative_windows(base_windows, base_baseline_windows)
    adaptive_gate = _nonnegative_windows(
        adaptive_windows, adaptive_baseline_windows
    )
    paired_validation = paired_monthly_hit_bootstrap(
        adaptive,
        source,
        settings["validation_origins"],
        block_months=int(project["bootstrap_blocks"]),
        replicates=int(project["bootstrap_replicates"]),
        seed=int(project["seed"]),
    )
    bootstrap = summarize_selected_predictions(
        _window(adaptive, settings["validation_origins"]),
        block_months=int(project["bootstrap_blocks"]),
        bootstrap_replicates=int(project["bootstrap_replicates"]),
        seed=int(project["seed"]),
    )
    paired_gate = bool(paired_validation["bootstrap_p10"] >= 0.0)
    accepted = bool(base_gate and adaptive_gate and paired_gate)
    rejection_reasons = []
    if not base_gate:
        rejection_reasons.append("base_top15_temporal_or_validation_gate")
    if not adaptive_gate:
        rejection_reasons.append("adaptive_temporal_or_validation_gate")
    if not paired_gate:
        rejection_reasons.append("paired_validation_bootstrap_p10")

    output_root = _artifact_root(root, family)
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(base, output_root / "base_predictions.parquet")
    atomic_write_parquet(adaptive, output_root / "adaptive_predictions.parquet")
    payload = {
        "experiment_id": f"phase2_feature_{family}",
        "experiment_key": experiment_key,
        "experiment_release": EXPERIMENT_RELEASE,
        "family": family,
        "feature_columns": FEATURE_FAMILY_COLUMNS[family],
        "cache_status": cache_status,
        "runtime_seconds": runtime_seconds,
        "active_model_changed": False,
        "locked_evaluation_read": False,
        "locked_origins": settings["locked_origins"],
        "selected_on": "tuning_and_validation_without_confirmation_or_locked",
        "confirmation_used_for_selection": False,
        "base": {
            "candidate": base_windows,
            "baseline": base_baseline_windows,
            "gate_passed": base_gate,
        },
        "adaptive": {
            "candidate": adaptive_windows,
            "baseline": adaptive_baseline_windows,
            "gate_passed": adaptive_gate,
        },
        "paired_validation": paired_validation,
        "validation_bootstrap": bootstrap,
        "accepted": accepted,
        "rejection_reasons": rejection_reasons,
    }
    atomic_write_json(payload, output_root / "summary.json")
    ledger_row = _ledger_row(
        family,
        adaptive,
        adaptive_windows,
        bootstrap,
        runtime_seconds,
        accepted,
        rejection_reasons,
        reference_key,
        root,
    )
    ledger_row["cache_status"] = cache_status
    write_experiment_ledger_row(
        root / "research/regime_adaptive_selector/metrics/experiment_ledger.csv",
        ledger_row,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=sorted(FEATURE_FAMILY_COLUMNS))
    args = parser.parse_args()
    print(
        json.dumps(
            build_feature_family_ablation(args.family),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
