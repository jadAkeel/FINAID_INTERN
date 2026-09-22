from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from forecast_select.chronos_pipeline import (
    build_chronos_zero_shot,
    chronos_zero_shot_status,
    read_active_predictions,
)
from forecast_select.io import atomic_write_parquet


def _make_test_data(
    start_origin: int = 120,
    end_origin: int = 266,
    num_indicators: int = 20,
    seed: int = 777,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    origins = list(range(start_origin, end_origin + 1))
    indicators = [f"X{i}" for i in range(1, num_indicators + 1)]

    in_rows = []
    out_rows = []
    act_rows = []

    for o in origins:
        k = int(rng.integers(15, min(num_indicators + 1, 21)))
        accepted_set = set(rng.choice(indicators, size=k, replace=False))

        for ind in indicators:
            p_raw = float(rng.uniform(0.15, 0.85))
            p_cal = float(rng.uniform(0.15, 0.85))
            y = float(rng.choice([0.0, 1.0]))

            in_rows.append({
                "origin_position": o,
                "indicator_id": ind,
                "p_up": p_raw,
                "p_up_raw": p_raw,
                "eligible": True,
                "data_quality_ok": True,
                "error_flag": False,
                "locked_evaluation_read": False,
            })
            out_rows.append({
                "origin_position": o,
                "indicator_id": ind,
                "y_true": y,
                "change_t_to_t1": 1.0 if y == 1.0 else -1.0,
                "locked_evaluation_read": False,
            })
            act_rows.append({
                "origin_position": o,
                "indicator_id": ind,
                "p_up_calibrated": p_cal,
                "p_up": p_cal,
                "accepted": ind in accepted_set,
                "predicted_direction": "Up" if p_cal >= 0.5 else "Down",
                "eligible": True,
                "locked_evaluation_read": False,
                "y_true": y,
            })

    return pd.DataFrame(in_rows), pd.DataFrame(out_rows), pd.DataFrame(act_rows)


def test_research_cli_registration_and_no_chronos_install_needed():
    # Verify CLI can be imported and commands are registered without requiring chronos or torch
    import forecast_select.research_cli as cli_mod

    assert "build-chronos-zero-shot" in cli_mod.RESEARCH_COMMANDS
    assert "show-chronos-zero-shot" in cli_mod.RESEARCH_COMMANDS

    parser = cli_mod._parser()
    subparser_actions = [
        action for action in parser._actions if isinstance(action, cli_mod.argparse._SubParsersAction)
    ]
    assert len(subparser_actions) == 1
    choices = subparser_actions[0].choices
    assert "build-chronos-zero-shot" in choices
    assert "show-chronos-zero-shot" in choices

    # Ensure chronos package is not imported merely by importing research_cli
    assert "chronos" not in sys.modules or not hasattr(sys.modules["chronos"], "Chronos2Pipeline")


def test_read_active_predictions_filters_and_asserts_max_origin(tmp_path):
    active_path = tmp_path / "active.parquet"

    # Create active dataframe with origins up to 280 (exceeding 266)
    rows = []
    for orig in [120, 200, 266, 267, 268, 280]:
        rows.append({
            "origin_position": orig,
            "indicator_id": "X1",
            "p_up_calibrated": 0.6,
            "p_up": 0.6,
            "accepted": True,
            "predicted_direction": "Up",
            "eligible": True,
            "locked_evaluation_read": False,
            "y_true": 1.0,
        })
    df_raw = pd.DataFrame(rows)
    atomic_write_parquet(df_raw, active_path)

    # Reading with active_path must filter origin_position <= 266
    filtered_df = read_active_predictions(root=tmp_path, active_path=active_path)
    assert int(filtered_df["origin_position"].max()) <= 266
    assert set(filtered_df["origin_position"].unique()) == {120, 200, 266}

    # If active dataframe has locked_evaluation_read=True, reading must assert and fail
    locked_rows = [
        {
            "origin_position": 120,
            "indicator_id": "X1",
            "p_up_calibrated": 0.6,
            "p_up": 0.6,
            "accepted": True,
            "predicted_direction": "Up",
            "eligible": True,
            "locked_evaluation_read": True,
            "y_true": 1.0,
        }
    ]
    locked_path = tmp_path / "locked_active.parquet"
    atomic_write_parquet(pd.DataFrame(locked_rows), locked_path)
    with pytest.raises(AssertionError, match="locked_evaluation_read=True"):
        read_active_predictions(root=tmp_path, active_path=locked_path)


def test_end_to_end_pipeline_evaluation_and_atomic_writes_tmp_path(tmp_path):
    inputs_df, outcomes_df, active_df = _make_test_data(120, 266, num_indicators=20, seed=100)

    # Active predictions path in tmp_path
    active_dir = tmp_path / "artifacts/active"
    active_dir.mkdir(parents=True, exist_ok=True)
    active_path = active_dir / "regime_adaptive_predictions.parquet"
    atomic_write_parquet(active_df, active_path)

    # Config dict
    test_config = {
        "experiment_id": "chronos_zero_shot",
        "research_only": True,
        "allow_promotion": False,
        "locked_evaluation_read_allowed": False,
        "package_pin": "chronos-forecasting==2.3.2",
        "model_id": "amazon/chronos-2",
        "model_revision": "29ec3766d36d6f73f0696f85560a422f50e8498c",
        "prediction_length": 2,
        "availability_lag": 1,
        "tuning_origins": [120, 179],
        "validation_origins": [180, 219],
        "confirmation_origins": [220, 266],
        "locked_origins": [268, 315],
        "maximum_workbook_position": 267,
        "seed": 20260727,
        "bootstrap_blocks": 6,
        "bootstrap_block_size": 6,
        "bootstrap_replicates": 50,  # Fast for integration test
        "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "minimum_history_origins": 12,
    }

    # Run build_chronos_zero_shot pointing to tmp_path
    summary = build_chronos_zero_shot(
        root=tmp_path,
        inputs=inputs_df,
        outcomes=outcomes_df,
        active_predictions=active_df,
        config=test_config,
    )

    # Check returned summary
    assert summary["experiment_id"] == "chronos_zero_shot"
    assert summary["selected_active_weight"] in [0.0, 0.25, 0.5, 0.75, 1.0]
    assert summary["promotion_eligible"] is False
    assert summary["promotion_requires_locked_evaluation"] is True
    assert summary["allow_promotion"] is False
    assert summary["active_model_changed"] is False
    assert summary["locked_evaluation_read"] is False
    assert summary["maximum_origin_position"] <= 266
    assert "contamination_caveat" in summary and len(summary["contamination_caveat"]) > 20

    # Check atomic files are written under tmp_path / research / chronos_zero_shot
    artifacts_dir = tmp_path / "research/chronos_zero_shot/artifacts"
    metrics_dir = tmp_path / "research/chronos_zero_shot/metrics"

    pred_parquet = artifacts_dir / "predictions.parquet"
    matched_parquet = artifacts_dir / "matched_coverage.parquet"
    summary_json = metrics_dir / "summary.json"
    win_csv = metrics_dir / "window_metrics.csv"
    div_csv = metrics_dir / "diversity_metrics.csv"
    matched_csv = metrics_dir / "matched_coverage.csv"

    assert pred_parquet.exists()
    assert matched_parquet.exists()
    assert summary_json.exists()
    assert win_csv.exists()
    assert div_csv.exists()
    assert matched_csv.exists()

    # Predictions check
    pred_df = pd.read_parquet(pred_parquet)
    assert not pred_df.empty
    assert int(pred_df["origin_position"].max()) <= 266
    assert "p_up_blend" in pred_df.columns
    assert "p_up_chronos_calibrated" in pred_df.columns
    assert "p_up_active" in pred_df.columns

    # Test status function
    status = chronos_zero_shot_status(root=tmp_path)
    assert status["ready"] is True
    assert status["locked_evaluation_read"] is False
    assert status["maximum_origin_position"] <= 266
    assert status["promotion_eligible"] is False
    assert status["allow_promotion"] is False


def test_status_rejections_on_violations(tmp_path):
    metrics_dir = tmp_path / "research/chronos_zero_shot/metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = metrics_dir / "summary.json"

    # 1. Reject if locked_evaluation_read is True
    bad_summary = {
        "locked_evaluation_read": True,
        "maximum_origin_position": 266,
        "allow_promotion": False,
        "promotion_eligible": False,
    }
    summary_path.write_text(json.dumps(bad_summary), encoding="utf-8")
    with pytest.raises(AssertionError, match="locked_evaluation_read is not False"):
        chronos_zero_shot_status(root=tmp_path)

    # 2. Reject if maximum_origin_position > 266
    bad_summary2 = {
        "locked_evaluation_read": False,
        "maximum_origin_position": 268,
        "allow_promotion": False,
        "promotion_eligible": False,
    }
    summary_path.write_text(json.dumps(bad_summary2), encoding="utf-8")
    with pytest.raises(AssertionError, match="exceeds 266"):
        chronos_zero_shot_status(root=tmp_path)

    # 3. Reject if promotion_eligible is True
    bad_summary3 = {
        "locked_evaluation_read": False,
        "maximum_origin_position": 266,
        "allow_promotion": False,
        "promotion_eligible": True,
    }
    summary_path.write_text(json.dumps(bad_summary3), encoding="utf-8")
    with pytest.raises(AssertionError, match="promotion_eligible is not False"):
        chronos_zero_shot_status(root=tmp_path)


def test_runner_script_invocation(tmp_path, monkeypatch):
    inputs_df, outcomes_df, active_df = _make_test_data(120, 266, num_indicators=20, seed=101)

    # Setup files in tmp_path
    active_dir = tmp_path / "artifacts/active"
    active_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(active_df, active_dir / "regime_adaptive_predictions.parquet")

    # Monkeypatch load_chronos_cache to return synthetic data
    import forecast_select.chronos_pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "load_chronos_cache", lambda root, key: (inputs_df, outcomes_df))

    # Import runner
    import research.chronos_zero_shot.run_chronos_evaluation as runner_mod

    code = runner_mod.main(["--root", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "research/chronos_zero_shot/metrics/summary.json").exists()
