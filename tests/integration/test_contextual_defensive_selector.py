import json
from pathlib import Path

import pandas as pd


def test_contextual_selector_artifact_is_causal_and_stops_before_locked_data():
    path = Path(
        "research/contextual_defensive_selector/artifacts/predictions.parquet"
    )
    assert path.exists()
    predictions = pd.read_parquet(path)
    assert int(predictions["origin_position"].min()) == 120
    assert int(predictions["origin_position"].max()) == 266
    assert (
        predictions["regime_observation_through_origin"]
        <= predictions["origin_position"] - 1
    ).all()
    assert not predictions["locked_evaluation_read"].any()

    selected = predictions[predictions["accepted"]]
    monthly = selected.groupby("origin_position")["indicator_id"].agg(
        ["count", "nunique"]
    )
    assert monthly["count"].eq(15).all()
    assert monthly["nunique"].eq(15).all()


def test_contextual_selector_records_the_frozen_unpromoted_result():
    path = Path(
        "research/contextual_defensive_selector/metrics/summary.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["selected_stress_threshold"] == 0.45
    assert payload["selected_role_indicators"] == ["X44", "X49"]
    assert payload["discovery_contextual"]["hits"] == 934
    assert payload["confirmation_contextual"]["hits"] == 436
    assert payload["confirmation_base"]["hits"] == 436
    assert payload["promotion_eligible"] is False
    assert payload["active_model_changed"] is False
    assert payload["locked_evaluation_read"] is False
