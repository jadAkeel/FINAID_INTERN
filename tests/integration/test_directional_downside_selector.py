import json
from pathlib import Path

import pandas as pd


def test_directional_downside_artifacts_are_causal_and_stop_before_locked():
    probability_path = Path(
        "research/directional_downside_selector/artifacts/"
        "downside_probabilities.parquet"
    )
    predictions_path = Path(
        "research/directional_downside_selector/artifacts/predictions.parquet"
    )
    assert probability_path.exists()
    assert predictions_path.exists()
    probabilities = pd.read_parquet(probability_path)
    predictions = pd.read_parquet(predictions_path)
    assert int(probabilities["origin_position"].min()) == 120
    assert int(probabilities["origin_position"].max()) == 266
    assert (
        probabilities["down_fit_through_origin"]
        <= probabilities["origin_position"] - 2
    ).all()
    assert probabilities["data_hash"].nunique() == 1
    assert probabilities["config_hash"].nunique() == 1
    assert not probabilities["locked_evaluation_read"].any()
    assert not predictions["locked_evaluation_read"].any()
    selected = predictions[predictions["accepted"]]
    monthly = selected.groupby("origin_position")["indicator_id"].agg(
        ["count", "nunique"]
    )
    assert monthly.eq(15).all().all()


def test_directional_downside_result_never_auto_promotes():
    path = Path(
        "research/directional_downside_selector/metrics/summary.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["active_model_changed"] is False
    assert payload["locked_evaluation_read"] is False
