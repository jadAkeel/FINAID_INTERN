from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .io import atomic_write_json, atomic_write_parquet, sha256_file


CACHE_SCHEMA_VERSION = "regime_replay_cache_v1"

LEDGER_COLUMNS = [
    "experiment_id",
    "hypothesis",
    "changed_variables",
    "selected_on",
    "data_hash",
    "config_hash",
    "calls",
    "hits",
    "accuracy",
    "early_tuning_calls",
    "early_tuning_hits",
    "early_tuning_accuracy",
    "late_tuning_calls",
    "late_tuning_hits",
    "late_tuning_accuracy",
    "validation_calls",
    "validation_hits",
    "validation_accuracy",
    "confirmation_calls",
    "confirmation_hits",
    "confirmation_accuracy",
    "cap_distribution",
    "rank_1_15_calls",
    "rank_1_15_hits",
    "rank_1_15_accuracy",
    "rank_16_17_calls",
    "rank_16_17_hits",
    "rank_16_17_accuracy",
    "rank_18_20_calls",
    "rank_18_20_hits",
    "rank_18_20_accuracy",
    "down_calls",
    "down_hits",
    "down_precision",
    "bootstrap_p10",
    "bootstrap_p90",
    "runtime_seconds",
    "cache_status",
    "cache_key",
    "cache_generation",
    "accepted",
    "rejection_reason",
]

REPLAY_INPUT_ALLOWLIST = [
    "origin_position",
    "origin_date",
    "target_date",
    "indicator_id",
    "eligible",
    "data_quality_ok",
    "feature_version",
    "data_hash",
    "config_hash",
    "p_up_raw",
    "p_up",
    "p_up_calibrated",
    "indicator_prior",
    "indicator_history_rows",
    "level_c_ready",
    "calibration_fit_through_origin",
    "reliability_fit_through_origin",
    "local_model_available",
    "pattern_history_rows",
    "base_predicted_direction",
    "base_accepted",
    "p_down_global",
    "p_down_local",
    "p_down_pattern",
    "p_down_indicator_prior",
    "down_exhaustion_flag",
    "down_fit_through_origin",
    "down_return_1",
    "down_momentum_3",
    "down_market_breadth",
    "nonselected_warning_score",
    "nonselected_warning_reason",
    "adaptive_data_quality_excluded",
    "adaptive_exclusion_reason",
    "level_c_ready_before_adaptive_exclusion",
    "p_up_generalized_graph",
    "p_up_generalized_calibrated",
    "generalized_graph_fit_through_origin",
    "generalized_graph_window_months",
    "generalized_graph_minimum_pairs",
    "generalized_graph_alpha",
    "asset_group",
    "asset_group_prior",
    "asset_group_market_prior",
    "asset_group_relative_logit",
    "asset_group_prior_fit_through_origin",
    "p_up_selection_score",
    "asset_group_overlay_weight",
    "p_down_base",
    "market_mean_return",
    "market_breadth",
    "market_breadth_3",
    "market_breadth_change_3",
    "market_dispersion",
    "peer_nonselected_count",
    "peer_nonselected_available_count",
    "peer_nonselected_breadth_up",
    "peer_nonselected_mean_return",
    "peer_nonselected_median_return",
    "peer_nonselected_dispersion",
    "peer_nonselected_negative_share",
    "peer_nonselected_weak_momentum_share",
    "peer_nonselected_mean_momentum_3",
    "peer_nonselected_exhaustion_share",
    "peer_nonselected_lead_negative_share",
    "previous_shock",
    "previous_shock_share",
    "market_mean_return_stress",
    "market_breadth_stress",
    "market_breadth_3_stress",
    "market_breadth_change_3_stress",
    "market_dispersion_stress",
    "peer_nonselected_breadth_up_stress",
    "peer_nonselected_mean_return_stress",
    "peer_nonselected_negative_share_stress",
    "peer_nonselected_weak_momentum_share_stress",
    "peer_nonselected_exhaustion_share_stress",
    "peer_nonselected_lead_negative_share_stress",
    "previous_shock_share_stress",
    "market_stress",
    "peer_stress",
    "shock_stress",
    "regime_stress",
    "regime_label",
    "forecast_market_breadth",
    "forecast_market_breadth_fit_through_origin",
    "forecast_market_breadth_observation_through_origin",
]


@dataclass(frozen=True)
class ReplayCacheKey:
    source_data_hash: str
    source_artifact_hash: str
    relevant_config_hash: str
    origin_start: int
    origin_end: int
    feature_contract_version: str
    model_settings: Mapping[str, Any]
    selected_parameters: Mapping[str, Any]
    release_identifier: str
    cache_schema_version: str = CACHE_SCHEMA_VERSION

    def payload(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        encoded = json.dumps(
            self.payload(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _assert_nonlocked_origins(
    frame: pd.DataFrame,
    *,
    locked_start: int,
) -> None:
    if frame.empty:
        raise ValueError("Replay cache frames must not be empty")
    origins = pd.to_numeric(frame["origin_position"], errors="raise")
    if int(origins.max()) >= int(locked_start):
        raise ValueError("Locked origins must never enter the replay cache")
    if (
        "locked_evaluation_read" in frame
        and frame["locked_evaluation_read"].fillna(True).astype(bool).any()
    ):
        raise ValueError("Replay source reports that locked evidence was read")


def split_replay_source(
    predictions: pd.DataFrame,
    *,
    locked_start: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate outcome-free selector inputs from non-locked evaluation labels."""
    required = {
        "origin_position",
        "indicator_id",
        "y_true",
        "p_up_selection_score",
        "level_c_ready",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Replay source is missing columns: {sorted(missing)}")
    _assert_nonlocked_origins(predictions, locked_start=locked_start)
    if predictions.duplicated(["origin_position", "indicator_id"]).any():
        raise ValueError("Replay source has duplicate origin/indicator rows")

    outcomes = predictions[[
        "origin_position",
        "indicator_id",
        "y_true",
    ]].copy()
    input_columns = [
        column
        for column in REPLAY_INPUT_ALLOWLIST
        if column in predictions.columns
    ]
    inputs = predictions[input_columns].copy()
    if "base_accepted" in inputs:
        inputs["accepted"] = inputs["base_accepted"].fillna(False).astype(bool)
    if "base_predicted_direction" in inputs:
        inputs["predicted_direction"] = inputs[
            "base_predicted_direction"
        ].astype(str)
    if "p_down_base" in inputs:
        inputs["p_down"] = pd.to_numeric(
            inputs["p_down_base"], errors="coerce"
        )

    ready = inputs["level_c_ready"].fillna(False).astype(bool)
    inputs["base_up_rank"] = pd.NA
    inputs.loc[ready, "base_up_rank"] = (
        inputs.loc[ready]
        .groupby("origin_position")["p_up_selection_score"]
        .rank(method="first", ascending=False)
    )
    inputs["base_up_rank"] = pd.to_numeric(
        inputs["base_up_rank"], errors="coerce"
    )
    return inputs, outcomes


def attach_replay_outcomes(
    inputs: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    if "y_true" in inputs:
        raise ValueError("Replay inputs must be outcome-free")
    _assert_matching_keys(inputs, outcomes)
    return inputs.merge(
        outcomes,
        on=["origin_position", "indicator_id"],
        how="left",
        validate="one_to_one",
    )


def replay_cache_path(root: Path, key: ReplayCacheKey) -> Path:
    return (
        root
        / "research/regime_adaptive_selector/cache"
        / key.digest()
    )


def _key_index(frame: pd.DataFrame) -> pd.MultiIndex:
    if frame.duplicated(["origin_position", "indicator_id"]).any():
        raise ValueError("Replay cache frames have duplicate keys")
    return pd.MultiIndex.from_frame(
        frame[["origin_position", "indicator_id"]].sort_values(
            ["origin_position", "indicator_id"]
        )
    )


def _assert_matching_keys(
    inputs: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> None:
    if not _key_index(inputs).equals(_key_index(outcomes)):
        raise ValueError("Replay input and outcome keys do not match exactly")


def replay_cache_generation(root: Path, key: ReplayCacheKey) -> str:
    pointer_path = replay_cache_path(root, key) / "current.json"
    if not pointer_path.exists():
        raise FileNotFoundError("Replay cache generation pointer is missing")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    generation = str(pointer.get("generation", ""))
    if not generation or Path(generation).name != generation:
        raise ValueError("Replay cache generation pointer is invalid")
    return generation


def write_replay_cache(
    root: Path,
    key: ReplayCacheKey,
    inputs: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    locked_start: int,
) -> Path:
    if "y_true" in inputs:
        raise ValueError("Replay inputs must not contain outcomes")
    _assert_nonlocked_origins(inputs, locked_start=locked_start)
    _assert_nonlocked_origins(outcomes, locked_start=locked_start)
    _assert_matching_keys(inputs, outcomes)
    if int(inputs["origin_position"].min()) != key.origin_start:
        raise ValueError("Replay input origin start does not match the cache key")
    if int(inputs["origin_position"].max()) != key.origin_end:
        raise ValueError("Replay input origin end does not match the cache key")
    if int(outcomes["origin_position"].min()) != key.origin_start:
        raise ValueError("Replay outcome origin start does not match the cache key")
    if int(outcomes["origin_position"].max()) != key.origin_end:
        raise ValueError("Replay outcome origin end does not match the cache key")

    target = replay_cache_path(root, key)
    generations = target / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    generation = uuid.uuid4().hex
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=generations))
    published = generations / generation
    input_path = staging / "inputs.parquet"
    outcome_path = staging / "outcomes.parquet"
    atomic_write_parquet(inputs, input_path)
    atomic_write_parquet(outcomes, outcome_path)
    manifest = {
        "cache_key": key.payload(),
        "cache_key_digest": key.digest(),
        "generation": generation,
        "input_columns": inputs.columns.tolist(),
        "input_hash": sha256_file(input_path),
        "input_rows": int(len(inputs)),
        "locked_evaluation_read": False,
        "outcome_columns": outcomes.columns.tolist(),
        "outcome_hash": sha256_file(outcome_path),
        "outcome_rows": int(len(outcomes)),
    }
    atomic_write_json(manifest, staging / "manifest.json")
    os.replace(staging, published)
    atomic_write_json({"generation": generation}, target / "current.json")
    return published


def load_replay_cache(
    root: Path,
    key: ReplayCacheKey,
    *,
    locked_start: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = replay_cache_path(root, key)
    generation = replay_cache_generation(root, key)
    published = target / "generations" / generation
    manifest_path = published / "manifest.json"
    input_path = published / "inputs.parquet"
    outcome_path = published / "outcomes.parquet"
    if not all(path.exists() for path in [manifest_path, input_path, outcome_path]):
        raise FileNotFoundError("Replay cache is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("cache_key") != key.payload():
        raise ValueError("Replay cache key does not match its manifest")
    if manifest.get("cache_key_digest") != key.digest():
        raise ValueError("Replay cache digest does not match its manifest")
    if manifest.get("generation") != generation:
        raise ValueError("Replay cache generation does not match its manifest")
    if manifest.get("input_hash") != sha256_file(input_path):
        raise ValueError("Replay input cache hash does not match its manifest")
    if manifest.get("outcome_hash") != sha256_file(outcome_path):
        raise ValueError("Replay outcome cache hash does not match its manifest")

    inputs = pd.read_parquet(input_path)
    outcomes = pd.read_parquet(outcome_path)
    if "y_true" in inputs:
        raise ValueError("Replay input cache contains outcomes")
    _assert_nonlocked_origins(inputs, locked_start=locked_start)
    _assert_nonlocked_origins(outcomes, locked_start=locked_start)
    _assert_matching_keys(inputs, outcomes)
    if len(inputs) != int(manifest["input_rows"]):
        raise ValueError("Replay input row count does not match its manifest")
    if len(outcomes) != int(manifest["outcome_rows"]):
        raise ValueError("Replay outcome row count does not match its manifest")
    if inputs.columns.tolist() != manifest["input_columns"]:
        raise ValueError("Replay input schema does not match its manifest")
    if outcomes.columns.tolist() != manifest["outcome_columns"]:
        raise ValueError("Replay outcome schema does not match its manifest")
    if int(inputs["origin_position"].min()) != key.origin_start:
        raise ValueError("Replay input origin start does not match the cache key")
    if int(inputs["origin_position"].max()) != key.origin_end:
        raise ValueError("Replay input origin end does not match the cache key")
    if int(outcomes["origin_position"].min()) != key.origin_start:
        raise ValueError("Replay outcome origin start does not match the cache key")
    if int(outcomes["origin_position"].max()) != key.origin_end:
        raise ValueError("Replay outcome origin end does not match the cache key")
    return inputs, outcomes


def write_experiment_ledger_row(
    path: Path,
    row: Mapping[str, Any],
) -> None:
    unknown = set(row).difference(LEDGER_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown experiment ledger fields: {sorted(unknown)}")
    normalized = {column: row.get(column) for column in LEDGER_COLUMNS}
    if path.exists():
        ledger = pd.read_csv(path)
        missing = set(LEDGER_COLUMNS).difference(ledger.columns)
        for column in missing:
            ledger[column] = pd.NA
        extra = set(ledger.columns).difference(LEDGER_COLUMNS)
        if extra:
            raise ValueError(f"Experiment ledger has unknown fields: {sorted(extra)}")
        ledger = ledger[LEDGER_COLUMNS]
        ledger = ledger[
            ledger["experiment_id"].ne(normalized["experiment_id"])
        ].reset_index(drop=True)
        records = ledger.to_dict(orient="records")
        records.append(normalized)
        ledger = pd.DataFrame.from_records(records, columns=LEDGER_COLUMNS)
    else:
        ledger = pd.DataFrame([normalized], columns=LEDGER_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(fd)
    try:
        ledger.to_csv(temporary_name, index=False)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
