import json
from pathlib import Path

import pandas as pd


def test_downside_risk_gate_artifacts_are_causal_and_stop_before_locked_data():
    risk_path = Path(
        "research/downside_risk_gate/artifacts/risk_predictions.parquet"
    )
    gated_path = Path(
        "research/downside_risk_gate/artifacts/gated_predictions.parquet"
    )
    assert risk_path.exists()
    assert gated_path.exists()

    risk = pd.read_parquet(risk_path)
    gated = pd.read_parquet(gated_path)
    assert int(risk["origin_position"].min()) == 120
    assert int(risk["origin_position"].max()) == 266
    assert int(gated["origin_position"].max()) == 266
    assert (
        risk["risk_fit_through_origin"]
        <= risk["origin_position"] - 2
    ).all()
    assert "X16" not in set(risk["indicator_id"])
    assert not gated["locked_evaluation_read"].any()

    selected = gated[gated["accepted"]]
    monthly = selected.groupby("origin_position")["indicator_id"].agg(
        ["count", "nunique"]
    )
    assert monthly.eq(15).all().all()


def test_downside_risk_gate_result_is_recorded_as_unpromoted_experiment():
    path = Path("research/downside_risk_gate/metrics/summary.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["selected_penalty"] == 0.0
    assert payload["active_model_changed"] is False
    assert payload["locked_evaluation_read"] is False
    assert payload["confirmation_read"] is True
