from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .downside_challengers import apply_normalized_bidirectional_selector
from .expansion_experiment_runner import _reference_bundle
from .expansion_quality import paired_monthly_hit_bootstrap
from .experiment_cache import (
    replay_cache_generation,
    write_experiment_ledger_row,
)
from .indicator_selection import summarize_selected_predictions
from .io import atomic_write_json, atomic_write_parquet
from .regime_adaptive_pipeline import _apply
from .regime_experiment_runner import _band_metrics, _metrics, _window
from .uptrend_pipeline import ROOT


def _windows(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    return {
        "early_tuning": _metrics(_window(frame, [120, 149])),
        "late_tuning": _metrics(_window(frame, [150, 179])),
        "tuning": _metrics(_window(frame, [120, 179])),
        "validation": _metrics(_window(frame, [180, 219])),
        "confirmation_descriptive": _metrics(_window(frame, [220, 266])),
    }


def _down_evidence(frame: pd.DataFrame, bounds: list[int]) -> dict[str, float | int]:
    selected = _window(frame, bounds)
    selected = selected[
        selected["accepted"].fillna(False).astype(bool)
        & selected["predicted_direction"].eq("Down")
        & selected["y_true"].notna()
    ]
    hits = int(selected["y_true"].eq(0.0).sum())
    return {
        "calls": int(len(selected)),
        "hits": hits,
        "precision": float(hits / len(selected)) if len(selected) else np.nan,
    }


def _expansion_only_mask(frame: pd.DataFrame) -> pd.Series:
    """Identify selected expansion rows that were not Down replacements."""
    regime_replacement = frame.get(
        "regime_replacement", pd.Series(False, index=frame.index)
    ).fillna(False)
    normalized_replacement = frame.get(
        "normalized_replacement", pd.Series(False, index=frame.index)
    ).fillna(False)
    return (
        frame["base_up_rank"].between(16, 20)
        & ~regime_replacement
        & ~normalized_replacement
    )


def _evidence(
    predictions: pd.DataFrame,
    baseline: pd.DataFrame,
    settings: dict,
    project: dict,
) -> dict[str, object]:
    windows = _windows(predictions)
    baseline_windows = _windows(baseline)
    down = {
        "early_tuning": _down_evidence(predictions, [120, 149]),
        "late_tuning": _down_evidence(predictions, [150, 179]),
        "tuning": _down_evidence(predictions, [120, 179]),
        "validation": _down_evidence(predictions, [180, 219]),
        "confirmation_descriptive": _down_evidence(predictions, [220, 266]),
    }
    nonnegative = all(
        windows[name]["hits"] >= baseline_windows[name]["hits"]
        for name in ["early_tuning", "late_tuning", "validation"]
    )
    adequate = bool(
        down["tuning"]["calls"] >= 8
        and down["early_tuning"]["calls"] >= 3
        and down["late_tuning"]["calls"] >= 3
        and down["validation"]["calls"] >= 3
    )
    paired_validation = paired_monthly_hit_bootstrap(
        predictions,
        baseline,
        settings["validation_origins"],
        block_months=int(project["bootstrap_blocks"]),
        replicates=int(project["bootstrap_replicates"]),
        seed=int(project["seed"]),
    )
    bootstrap = summarize_selected_predictions(
        _window(predictions, settings["validation_origins"]),
        block_months=int(project["bootstrap_blocks"]),
        bootstrap_replicates=int(project["bootstrap_replicates"]),
        seed=int(project["seed"]),
    )
    accepted = bool(
        nonnegative
        and adequate
        and paired_validation["bootstrap_p10"] >= 0.0
    )
    selected = predictions[predictions["accepted"].fillna(False).astype(bool)]
    expansion_only = selected[_expansion_only_mask(selected)]
    if not expansion_only["predicted_direction"].eq("Up").all():
        raise AssertionError("Expansion-only calls were flipped Down")
    monthly = selected.groupby("origin_position")["indicator_id"].agg(
        ["count", "nunique"]
    )
    if not monthly["count"].between(15, 20).all():
        raise AssertionError("Down challenger violated the 15-20 contract")
    if not monthly["count"].eq(monthly["nunique"]).all():
        raise AssertionError("Down challenger selected duplicates")
    reasons = []
    if not nonnegative:
        reasons.append("negative_temporal_or_validation_hit_delta")
    if not adequate:
        reasons.append("inadequate_down_call_evidence")
    if paired_validation["bootstrap_p10"] < 0.0:
        reasons.append("paired_validation_bootstrap_p10")
    return {
        "accepted": accepted,
        "adequate_down_evidence": adequate,
        "bootstrap": bootstrap,
        "down": down,
        "nonnegative_hit_deltas": nonnegative,
        "paired_validation": paired_validation,
        "rejection_reasons": reasons,
        "windows": windows,
        "baseline_windows": baseline_windows,
    }


def _ledger_row(
    experiment_id: str,
    hypothesis: str,
    predictions: pd.DataFrame,
    evidence: dict[str, object],
    runtime_seconds: float,
    reference_key,
    root: Path,
) -> dict[str, object]:
    metrics = _metrics(predictions)
    core = _band_metrics(predictions, 1, 15)
    middle = _band_metrics(predictions, 16, 17)
    tail = _band_metrics(predictions, 18, 20)
    selected = predictions[predictions["accepted"].fillna(False).astype(bool)]
    caps = selected.groupby("origin_position").size().value_counts().sort_index()
    down = evidence["down"]["tuning"]
    windows = evidence["windows"]
    return {
        "experiment_id": experiment_id,
        "hypothesis": hypothesis,
        "changed_variables": experiment_id,
        "selected_on": "tuning_stability; validation_gate; confirmation_descriptive",
        "data_hash": reference_key.source_data_hash,
        "config_hash": reference_key.relevant_config_hash,
        **metrics,
        "early_tuning_calls": windows["early_tuning"]["calls"],
        "early_tuning_hits": windows["early_tuning"]["hits"],
        "early_tuning_accuracy": windows["early_tuning"]["accuracy"],
        "late_tuning_calls": windows["late_tuning"]["calls"],
        "late_tuning_hits": windows["late_tuning"]["hits"],
        "late_tuning_accuracy": windows["late_tuning"]["accuracy"],
        "validation_calls": windows["validation"]["calls"],
        "validation_hits": windows["validation"]["hits"],
        "validation_accuracy": windows["validation"]["accuracy"],
        "confirmation_calls": windows["confirmation_descriptive"]["calls"],
        "confirmation_hits": windows["confirmation_descriptive"]["hits"],
        "confirmation_accuracy": windows[
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
        "down_calls": down["calls"],
        "down_hits": down["hits"],
        "down_precision": down["precision"],
        "bootstrap_p10": evidence["bootstrap"]["bootstrap_p10"],
        "bootstrap_p90": evidence["bootstrap"]["bootstrap_p90"],
        "runtime_seconds": runtime_seconds,
        "cache_status": "hit",
        "cache_key": reference_key.digest(),
        "cache_generation": replay_cache_generation(root, reference_key),
        "accepted": evidence["accepted"],
        "rejection_reason": "|".join(evidence["rejection_reasons"]),
    }


def _write_nonlocked_roadmap_summary(
    ledger_path: Path,
) -> Path:
    """Publish a compact, ledger-derived conclusion for the research suite."""
    ledger = pd.read_csv(ledger_path)
    selected_experiment_id = "phase3_current_guarded_up_first"
    selected = ledger.loc[
        ledger["experiment_id"].eq(selected_experiment_id)
    ]
    if len(selected) != 1:
        raise AssertionError("Research ledger is missing its retained reference row")
    reference = selected.iloc[0]
    validation_accuracy = pd.to_numeric(
        ledger["validation_accuracy"], errors="coerce"
    )
    payload = {
        "selected_experiment_id": selected_experiment_id,
        "experiment_ledger_rows": int(len(ledger)),
        "active_model_changed": False,
        "confirmation_used_for_selection": False,
        "locked_evaluation_read": False,
        "reference": {
            "validation": {
                "calls": int(reference["validation_calls"]),
                "hits": int(reference["validation_hits"]),
                "accuracy": float(reference["validation_accuracy"]),
            }
        },
        "milestones": {
            "validation_65_percent_reached": bool(
                validation_accuracy.ge(0.65).fillna(False).any()
            )
        },
    }
    output = ledger_path.parent / "nonlocked_roadmap_summary.json"
    atomic_write_json(payload, output)
    return output


def build_downside_challengers(root: Path = ROOT) -> dict[str, object]:
    (
        replay,
        _,
        source,
        settings,
        project,
        summary,
        reference_key,
    ) = _reference_bundle(root)
    params = dict(summary["selected_parameters"])
    baseline_started = perf_counter()
    baseline = _apply(replay, settings, params, cap=None)
    baseline_runtime = perf_counter() - baseline_started

    replacement_params = dict(params)
    replacement_params["maximum_replacements"] = 1
    replacement_started = perf_counter()
    maximum_one_replacement = _apply(
        replay,
        settings,
        replacement_params,
        cap=None,
    )
    replacement_runtime = perf_counter() - replacement_started

    normalized_started = perf_counter()
    normalized = apply_normalized_bidirectional_selector(
        replay,
        baseline,
        down_threshold=float(params["down_threshold"]),
        stress_trigger=float(params["stress_trigger"]),
        hard_down_threshold=float(settings["stress"]["hard_down_threshold"]),
        normalized_margin=0.05,
        maximum_down_actions=1,
        minimum_cap=int(settings["selection"]["minimum_selection_count"]),
    )
    normalized_runtime = perf_counter() - normalized_started

    candidates = {
        "current_guarded_up_first": (baseline, baseline_runtime),
        "maximum_one_down_replacement": (
            maximum_one_replacement,
            replacement_runtime,
        ),
        "normalized_bidirectional_percentile": (
            normalized,
            normalized_runtime,
        ),
    }
    results = {}
    ledger_path = (
        root
        / "research/regime_adaptive_selector/metrics/experiment_ledger.csv"
    )
    artifact_root = root / "research/regime_adaptive_selector/downside_challengers"
    artifact_root.mkdir(parents=True, exist_ok=True)
    for name, (predictions, runtime) in candidates.items():
        evidence = _evidence(predictions, baseline, settings, project)
        if name == "current_guarded_up_first":
            evidence["accepted"] = True
            evidence["rejection_reasons"] = []
        results[name] = evidence
        atomic_write_parquet(predictions, artifact_root / f"{name}.parquet")
        write_experiment_ledger_row(
            ledger_path,
            _ledger_row(
                f"phase3_{name}",
                f"Evaluate {name} without forcing Down calls",
                predictions,
                evidence,
                runtime,
                reference_key,
                root,
            ),
        )
    _write_nonlocked_roadmap_summary(ledger_path)
    payload = {
        "experiment_phase": "downside_challengers",
        "active_model_changed": False,
        "confirmation_used_for_selection": False,
        "locked_evaluation_read": False,
        "locked_origins": settings["locked_origins"],
        "normalized_scale": "within_origin_cross_sectional_percentile_ranks",
        "results": results,
    }
    atomic_write_json(payload, artifact_root / "summary.json")
    return payload


if __name__ == "__main__":
    print(json.dumps(build_downside_challengers(), indent=2, sort_keys=True))
