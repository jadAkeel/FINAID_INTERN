from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .uptrend_model import fit_uptrend_model, predict_uptrend_probability
from .features import build_feature_panel
from .io import atomic_write_json, atomic_write_parquet, load_workbook, sha256_file
from .schemas import validate_oof_columns
from .indicator_selection import (
    propagate_correlation_graph,
    select_top_indicators,
    summarize_selected_predictions,
)
from .targets import build_targets
from .validation import (
    assert_target_alignment,
    assert_target_history_available,
    causal_training_rows,
)


ROOT = Path(__file__).resolve().parents[2]


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _configuration_hash(root: Path) -> str:
    payload = {
        "project": _read_yaml(root / "configs/config.yaml"),
        "model": _read_yaml(root / "configs/uptrend_model.yaml"),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prepare(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    config = _read_yaml(root / "configs/config.yaml")
    model_settings = _read_yaml(root / "configs/uptrend_model.yaml")
    frame = load_workbook(root / config["data_path"])
    targets = build_targets(frame)
    assert_target_alignment(targets, frame)
    features = build_feature_panel(
        frame,
        availability_lag=int(model_settings["availability_lag_months"]),
        include_structured=True,
    )
    panel = features.merge(
        targets[[
            "origin_position", "indicator_id", "target_date", "y_true",
            "zero_change", "target_available", "value_t", "value_t1",
        ]],
        on=["origin_position", "indicator_id"],
        how="left",
        validate="one_to_one",
    )
    panel["eligible"] = (
        panel["target_available"]
        & panel["observed"].eq(1)
        & panel["origin_position"].gt(int(config["minimum_history_months"]))
    )
    panel["data_quality_ok"] = panel["eligible"]
    return frame, targets, panel, config, model_settings


def _prediction_rows(
    origin: int,
    test: pd.DataFrame,
    probabilities: np.ndarray,
    data_hash: str,
    config_hash: str,
    feature_version: str,
    model_release: str,
    seed: int,
    runtime_seconds: float,
) -> pd.DataFrame:
    rows = test[[
        "origin_position", "origin_date", "target_date", "indicator_id",
        "y_true", "eligible", "data_quality_ok",
    ]].copy()
    rows["run_id"] = "uptrend_selector_research"
    rows["model_id"] = "uptrend_logistic"
    rows["model_version"] = model_release
    rows["p_up_raw"] = probabilities
    rows["p_up"] = probabilities
    rows["predicted_direction"] = np.where(probabilities >= 0.5, "Up", "Down")
    rows["fit_window"] = f"<=position_{origin - 2}"
    rows["feature_version"] = feature_version
    rows["data_hash"] = data_hash
    rows["config_hash"] = config_hash
    rows["seed"] = seed
    rows["runtime_seconds"] = runtime_seconds
    rows["error_flag"] = False
    rows["error_message"] = None
    rows["ineligibility_reason"] = ""
    return rows


def build_uptrend_predictions(
    root: Path = ROOT,
    origin_range: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Build causal walk-forward predictions for the configured or requested origins."""
    frame, targets, panel, config, model_settings = _prepare(root)
    start, end = (
        origin_range
        if origin_range is not None
        else model_settings["selection_origins"]
    )
    lag = int(model_settings["availability_lag_months"])
    data_hash = sha256_file(root / config["data_path"])
    config_hash = _configuration_hash(root)
    model_config = model_settings["model"]
    eligible = panel[panel["eligible"]].copy()
    logistic_parts = []
    for origin in range(int(start), int(end) + 1):
        train = causal_training_rows(eligible, origin, availability_lag=lag)
        test = eligible[eligible["origin_position"].eq(origin)].copy()
        assert_target_history_available(train, origin, availability_lag=lag)
        started = time.perf_counter()
        model = fit_uptrend_model(
            train,
            seed=int(config["seed"]),
            logistic_c=float(model_config["logistic_c"]),
            max_iter=int(model_config["logistic_max_iter"]),
        )
        probability = predict_uptrend_probability(model, test)
        logistic_parts.append(_prediction_rows(
            origin,
            test,
            probability,
            data_hash,
            config_hash,
            str(config["feature_version"]),
            str(model_settings["model_release"]),
            int(config["seed"]),
            time.perf_counter() - started,
        ))
    logistic = pd.concat(logistic_parts, ignore_index=True)

    graph_config = model_settings["graph"]
    indicators = [column for column in frame.columns if column.startswith("X")]
    graph = frame[indicators].diff().iloc[
        :int(graph_config["estimation_end"])
    ].corr(min_periods=int(graph_config["minimum_pairs"]))
    np.fill_diagonal(graph.values, 0.0)
    graph_parts = []
    for origin, group in logistic.groupby("origin_position", sort=True):
        current = group.copy()
        correlation = graph.reindex(
            index=current["indicator_id"],
            columns=current["indicator_id"],
        ).fillna(0.0).to_numpy(dtype=float)
        current["p_up_raw"] = current["p_up"].to_numpy(dtype=float)
        current["p_up"] = propagate_correlation_graph(
            current["p_up"].to_numpy(dtype=float),
            correlation,
            alpha=float(graph_config["alpha"]),
        )
        current["predicted_direction"] = np.where(current["p_up"] >= 0.5, "Up", "Down")
        current["model_id"] = model_settings["model_id"]
        current["graph_alpha"] = float(graph_config["alpha"])
        current["graph_estimation_end"] = int(graph_config["estimation_end"])
        current["graph_origin"] = int(origin)
        graph_parts.append(current)
    graph_predictions = pd.concat(graph_parts, ignore_index=True)

    selection = model_settings["selection"]
    target_history = targets.loc[
        targets["target_available"] & targets["y_true"].notna(),
        ["origin_position", "indicator_id", "y_true"],
    ]
    result = select_top_indicators(
        target_history,
        graph_predictions,
        cap=int(selection["monthly_selection_count"]),
        prior_window=int(selection["trailing_target_window"]),
        prior_weight=float(selection["indicator_prior_weight"]),
        minimum_history_months=int(selection["minimum_history_months"]),
        minimum_indicator_history=int(selection["minimum_indicator_history"]),
        availability_lag=lag,
    )
    result["model_id"] = model_settings["model_id"]
    result["model_version"] = model_settings["model_release"]
    validate_oof_columns(result.columns.tolist())
    counts = result[result["accepted"]].groupby("origin_position")["indicator_id"].agg(
        ["count", "nunique"]
    )
    expected = int(selection["monthly_selection_count"])
    if not counts["count"].eq(expected).all() or not counts["nunique"].eq(expected).all():
        raise AssertionError("The model must select the configured number of unique indicators per month")
    if (result["calibration_fit_through_origin"] > result["origin_position"] - 2).any():
        raise AssertionError("Model selection used an unavailable target")
    return result


def active_model_artifact(root: Path = ROOT) -> Path:
    return root / "artifacts/active/uptrend_predictions.parquet"


def _assert_model_invariants(
    predictions: pd.DataFrame,
    model_settings: dict,
) -> None:
    selection = model_settings["selection"]
    start, end = (int(value) for value in model_settings["selection_origins"])
    expected_origins = set(range(start, end + 1))
    actual_origins = set(
        predictions["origin_position"].dropna().astype(int).unique().tolist()
    )
    if actual_origins != expected_origins:
        raise AssertionError("Model origins do not match the registered Selection window")
    if set(predictions["run_id"].dropna().astype(str).unique()) != {
        "uptrend_selector_research"
    }:
        raise AssertionError("Model run_id is invalid")
    if set(predictions["model_id"].dropna().astype(str).unique()) != {
        model_settings["model_id"]
    }:
        raise AssertionError("Model id is invalid")
    if set(predictions["model_version"].dropna().astype(str).unique()) != {
        model_settings["model_release"]
    }:
        raise AssertionError("Model release label is invalid")

    selected = predictions[predictions["accepted"].fillna(False).astype(bool)]
    counts = selected.groupby("origin_position")["indicator_id"].agg(
        ["count", "nunique"]
    )
    expected_count = int(selection["monthly_selection_count"])
    if (
        set(counts.index.astype(int)) != expected_origins
        or not counts["count"].eq(expected_count).all()
        or not counts["nunique"].eq(expected_count).all()
    ):
        raise AssertionError(
            f"The model must select {expected_count} unique indicators per month"
        )

    ready = predictions[
        predictions["calibration_fit_through_origin"].notna()
    ]
    if (
        ready["calibration_fit_through_origin"]
        > ready["origin_position"] - 2
    ).any():
        raise AssertionError("Model selection used an unavailable target")
    if (
        ready["reliability_fit_through_origin"]
        > ready["origin_position"] - 2
    ).any():
        raise AssertionError("Model reliability used an unavailable target")


def _assert_artifact_provenance(
    predictions: pd.DataFrame,
    root: Path,
) -> None:
    config = _read_yaml(root / "configs/config.yaml")
    expected = {
        "config_hash": _configuration_hash(root),
        "data_hash": sha256_file(root / config["data_path"]),
        "feature_version": config["feature_version"],
    }
    for column, value in expected.items():
        actual = set(predictions[column].dropna().astype(str).unique())
        if actual != {str(value)}:
            raise AssertionError(
                f"Active model {column} does not match the current project"
            )


def write_model_report(predictions: pd.DataFrame, root: Path = ROOT) -> dict:
    config = _read_yaml(root / "configs/config.yaml")
    model_settings = _read_yaml(root / "configs/uptrend_model.yaml")
    summary = summarize_selected_predictions(
        predictions,
        block_months=int(config["bootstrap_blocks"]),
        bootstrap_replicates=int(config["bootstrap_replicates"]),
        seed=int(config["seed"]),
    )
    registered = model_settings["registered_result"]
    summary.update({
        "model_id": model_settings["model_id"],
        "model_name": model_settings["model_name"],
        "model_release": model_settings["model_release"],
        "registered_result_matches": bool(
            int(summary["months"]) == int(registered["months"])
            and int(summary["calls"]) == int(registered["calls"])
            and int(summary["hits"]) == int(registered["hits"])
            and np.isclose(float(summary["accuracy"]), float(registered["accuracy"]))
            and int(summary["up_calls"]) == int(registered["up_calls"])
            and int(summary["down_calls"]) == int(registered["down_calls"])
        ),
        "confirmation_read": False,
        "locked_evaluation_read": False,
        "artifact": active_model_artifact(root).relative_to(root).as_posix(),
    })
    atomic_write_json(summary, root / "reports/model_performance.json")
    lines = [
        "# Uptrend Selector performance",
        "",
        "This is the single active model pipeline.",
        "",
        f"- Selection hits / calls: `{summary['hits']} / {summary['calls']}`",
        f"- Top-15 accuracy: `{summary['accuracy']:.4%}`",
        f"- Up / Down calls: `{summary['up_calls']} / {summary['down_calls']}`",
        f"- Registered result reproduced: `{summary['registered_result_matches']}`",
        "- Confirmation read: `False`",
        "- Locked evaluation read: `False`",
        "",
        "The pipeline is Structured Logistic with corrected cross-sectional rank, followed by a frozen signed correlation graph and a causal top-indicator selector.",
    ]
    (root / "reports/model_performance.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return summary


def build_active_model(root: Path = ROOT) -> Path:
    output = active_model_artifact(root)
    model_settings = _read_yaml(root / "configs/uptrend_model.yaml")
    if output.exists():
        try:
            predictions = pd.read_parquet(output)
            validate_oof_columns(predictions.columns.tolist())
            _assert_model_invariants(predictions, model_settings)
            _assert_artifact_provenance(predictions, root)
            summary = write_model_report(predictions, root)
            if not summary["registered_result_matches"]:
                raise AssertionError(
                    "Active model artifact does not match the registered result"
                )
            return output
        except (AssertionError, KeyError, OSError, ValueError):
            pass
    predictions = build_uptrend_predictions(root)
    _assert_model_invariants(predictions, model_settings)
    _assert_artifact_provenance(predictions, root)
    atomic_write_parquet(predictions, output)
    summary = write_model_report(predictions, root)
    if not summary["registered_result_matches"]:
        raise AssertionError("Rebuilt model does not match the registered result")
    return output


def active_model_status(root: Path = ROOT) -> dict:
    output = active_model_artifact(root)
    if not output.exists():
        return {"ready": False, "artifact": str(output)}
    predictions = pd.read_parquet(output)
    model_settings = _read_yaml(root / "configs/uptrend_model.yaml")
    validate_oof_columns(predictions.columns.tolist())
    _assert_model_invariants(predictions, model_settings)
    _assert_artifact_provenance(predictions, root)
    summary = write_model_report(predictions, root)
    return {"ready": True, **summary}
