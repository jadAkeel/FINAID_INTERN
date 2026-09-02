from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .experiment_cache import (
    ReplayCacheKey,
    attach_replay_outcomes,
    load_replay_cache,
    replay_cache_generation,
    split_replay_source,
    write_experiment_ledger_row,
    write_replay_cache,
)
from .indicator_selection import summarize_selected_predictions
from .io import sha256_file
from .regime_adaptive_pipeline import (
    _apply,
    _read_yaml,
    regime_adaptive_predictions_artifact,
    regime_adaptive_summary_path,
)
from .uptrend_pipeline import ROOT


def _correct(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["predicted_direction"].eq("Up") & frame["y_true"].eq(1.0)
    ) | (
        frame["predicted_direction"].eq("Down") & frame["y_true"].eq(0.0)
    )


def _metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    selected = frame[
        frame["accepted"].fillna(False).astype(bool)
        & frame["y_true"].notna()
    ].copy()
    correct = _correct(selected)
    return {
        "calls": int(len(selected)),
        "hits": int(correct.sum()),
        "accuracy": float(correct.mean()) if len(selected) else np.nan,
    }


def _band_metrics(
    frame: pd.DataFrame,
    lower: int,
    upper: int,
) -> dict[str, float | int]:
    selected = frame[
        frame["accepted"].fillna(False).astype(bool)
        & frame["base_up_rank"].between(lower, upper)
        & frame["y_true"].notna()
    ].copy()
    correct = _correct(selected)
    return {
        "calls": int(len(selected)),
        "hits": int(correct.sum()),
        "accuracy": float(correct.mean()) if len(selected) else np.nan,
    }


def _window(frame: pd.DataFrame, bounds: list[int]) -> pd.DataFrame:
    return frame[frame["origin_position"].between(*map(int, bounds))]


def _assert_replay_equivalent(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    expected_inputs: pd.DataFrame,
) -> None:
    keys = ["origin_position", "indicator_id"]
    columns = [
        "y_true",
        "accepted",
        "predicted_direction",
        "selection_rank",
        "regime_cap",
        "regime_replacement",
        "p_up_selection_score",
        "p_up_base",
        "p_down",
        "risk_adjusted_up_score",
        "down_candidate",
        "directional_confidence",
        "selection_score",
        "regime_selection_changed",
        "regime_direction_changed",
    ]
    left = expected[keys + columns].sort_values(keys).reset_index(drop=True)
    right = actual[keys + columns].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False)
    expected_ranks = expected_inputs[keys + ["base_up_rank"]].sort_values(
        keys
    ).reset_index(drop=True)
    actual_ranks = actual[keys + ["base_up_rank"]].sort_values(
        keys
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        expected_ranks,
        actual_ranks,
        check_dtype=False,
    )


def build_reference_replay(root: Path = ROOT) -> dict[str, object]:
    settings = _read_yaml(root / "configs/regime_adaptive_selector.yaml")
    project = _read_yaml(root / "configs/config.yaml")
    locked_start = int(settings["locked_origins"][0])
    source = pd.read_parquet(regime_adaptive_predictions_artifact(root))
    if int(source["origin_position"].max()) >= locked_start:
        raise ValueError("Reference artifact includes locked origins")
    if source["locked_evaluation_read"].fillna(True).astype(bool).any():
        raise ValueError("Reference artifact reports that locked evidence was read")
    summary = json.loads(
        regime_adaptive_summary_path(root).read_text(encoding="utf-8")
    )
    selected_params = dict(summary["selected_parameters"])
    data_hashes = source["data_hash"].dropna().astype(str).unique()
    config_hashes = source["config_hash"].dropna().astype(str).unique()
    if len(data_hashes) != 1 or len(config_hashes) != 1:
        raise ValueError("Reference artifact has ambiguous provenance")

    model_settings = {
        "selection": settings["selection"],
        "asset_group_overlay": settings["asset_group_overlay"],
        "generalized_correlation_overlay": settings[
            "generalized_correlation_overlay"
        ],
        "forward_regime": settings["forward_regime"],
        "stress": settings["stress"],
    }
    key = ReplayCacheKey(
        source_data_hash=str(data_hashes[0]),
        source_artifact_hash=sha256_file(
            regime_adaptive_predictions_artifact(root)
        ),
        relevant_config_hash=str(config_hashes[0]),
        origin_start=int(source["origin_position"].min()),
        origin_end=int(source["origin_position"].max()),
        feature_contract_version="regime_replay_inputs_with_base_up_rank_v1",
        model_settings=model_settings,
        selected_parameters=selected_params,
        release_identifier=str(settings["experiment_release"]),
    )
    source_inputs, source_outcomes = split_replay_source(
        source,
        locked_start=locked_start,
    )
    try:
        cached_inputs, cached_outcomes = load_replay_cache(
            root,
            key,
            locked_start=locked_start,
        )
        cache_status = "hit"
    except FileNotFoundError:
        write_replay_cache(
            root,
            key,
            source_inputs,
            source_outcomes,
            locked_start=locked_start,
        )
        cached_inputs, cached_outcomes = load_replay_cache(
            root,
            key,
            locked_start=locked_start,
        )
        cache_status = "miss_built"
    replay_inputs = attach_replay_outcomes(cached_inputs, cached_outcomes)
    started = perf_counter()
    replayed = _apply(replay_inputs, settings, selected_params, cap=None)
    runtime_seconds = perf_counter() - started
    _assert_replay_equivalent(source, replayed, source_inputs)

    all_metrics = _metrics(replayed)
    early = _metrics(_window(replayed, [120, 149]))
    late = _metrics(_window(replayed, [150, 179]))
    validation = _metrics(_window(replayed, settings["validation_origins"]))
    confirmation = _metrics(
        _window(replayed, settings["confirmation_origins"])
    )
    core = _band_metrics(replayed, 1, 15)
    middle = _band_metrics(replayed, 16, 17)
    tail = _band_metrics(replayed, 18, 20)
    selected = replayed[replayed["accepted"].fillna(False).astype(bool)]
    down = selected[selected["predicted_direction"].eq("Down")]
    down_correct = _correct(down)
    cap_distribution = (
        selected.groupby("origin_position").size().value_counts().sort_index()
    )
    bootstrap = summarize_selected_predictions(
        _window(replayed, settings["validation_origins"]),
        block_months=int(project["bootstrap_blocks"]),
        bootstrap_replicates=int(project["bootstrap_replicates"]),
        seed=int(project["seed"]),
    )
    ledger_row = {
        "experiment_id": "phase0_reference_replay_v1",
        "hypothesis": "A provenance-keyed replay cache reproduces the reference",
        "changed_variables": "none; exact reference replay",
        "selected_on": "existing non-locked reference",
        "data_hash": key.source_data_hash,
        "config_hash": key.relevant_config_hash,
        **all_metrics,
        "early_tuning_calls": early["calls"],
        "early_tuning_hits": early["hits"],
        "early_tuning_accuracy": early["accuracy"],
        "late_tuning_calls": late["calls"],
        "late_tuning_hits": late["hits"],
        "late_tuning_accuracy": late["accuracy"],
        "validation_calls": validation["calls"],
        "validation_hits": validation["hits"],
        "validation_accuracy": validation["accuracy"],
        "confirmation_calls": confirmation["calls"],
        "confirmation_hits": confirmation["hits"],
        "confirmation_accuracy": confirmation["accuracy"],
        "cap_distribution": json.dumps(
            {str(int(cap)): int(count) for cap, count in cap_distribution.items()},
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
        "down_hits": int(down_correct.sum()),
        "down_precision": (
            float(down_correct.mean()) if len(down) else np.nan
        ),
        "bootstrap_p10": float(bootstrap["bootstrap_p10"]),
        "bootstrap_p90": float(bootstrap["bootstrap_p90"]),
        "runtime_seconds": float(runtime_seconds),
        "cache_status": cache_status,
        "cache_key": key.digest(),
        "cache_generation": replay_cache_generation(root, key),
        "accepted": True,
        "rejection_reason": "",
    }
    ledger_path = (
        root
        / "research/regime_adaptive_selector/metrics/experiment_ledger.csv"
    )
    write_experiment_ledger_row(ledger_path, ledger_row)
    return {
        "cache_key": key.digest(),
        "cache_path": str(
            root
            / "research/regime_adaptive_selector/cache"
            / key.digest()
        ),
        "cache_status": cache_status,
        "equivalent": True,
        "ledger_path": str(ledger_path),
        "runtime_seconds": runtime_seconds,
        "validation": validation,
    }


if __name__ == "__main__":
    print(json.dumps(build_reference_replay(), indent=2, sort_keys=True))
