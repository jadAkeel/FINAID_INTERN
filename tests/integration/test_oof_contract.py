import json
from pathlib import Path

import pandas as pd

from forecast_select.project import check_project
from forecast_select.schemas import validate_oof_columns


def test_uptrend_selector_matches_contract_and_registered_result():
    path = Path("artifacts/active/uptrend_predictions.parquet")
    assert path.exists()
    frame = pd.read_parquet(path)
    validate_oof_columns(frame.columns.tolist())
    selected = frame[frame["accepted"]].copy()
    monthly = selected.groupby("origin_position")["indicator_id"].agg(
        ["count", "nunique"]
    )
    assert monthly.eq(15).all().all()
    assert len(selected) == 1500
    predicted = selected["predicted_direction"].eq("Up").astype(int)
    assert int(predicted.eq(selected["y_true"].astype(int)).sum()) == 926
    ready = frame[frame["calibration_fit_through_origin"].notna()]
    assert (
        ready["calibration_fit_through_origin"]
        <= ready["origin_position"] - 2
    ).all()


def test_locked_evaluation_is_preserved_separately():
    assert Path("artifacts/audit/locked_evaluation.parquet").exists()


def test_project_check_verifies_locked_evaluation_hash():
    report = check_project()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["locked_evaluation_preserved"] is True
    assert payload["locked_evaluation_sha256"] == (
        "04ebedf9455051b189486f61deba949299a499915aa33f11b7126efa5a035b39"
    )


def test_candidate_portfolio_has_one_artifact_per_model_family():
    canonical = Path("research/reference_models/artifacts")
    assert {path.name for path in canonical.glob("*.parquet")} == {
        "baseline_models.parquet",
        "lead_lag_logistic.parquet",
        "market_regime_selector.parquet",
        "parliament_vote.parquet",
        "structured_catboost.parquet",
        "weighted_ensemble.parquet",
    }
    assert not Path("archive").exists()
