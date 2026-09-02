import json
from pathlib import Path

import pandas as pd


def test_down_sensing_artifacts_are_causal_and_stop_before_locked_data():
    scores_path = Path(
        "research/down_sensing_gate/artifacts/extreme_scores.parquet"
    )
    assert scores_path.exists()
    scores = pd.read_parquet(scores_path)
    assert int(scores["origin_position"].min()) >= 120
    assert int(scores["origin_position"].max()) <= 267
    assert (
        scores["extreme_fit_through_origin"] <= scores["origin_position"] - 2
    ).all()
    finite = scores["p_extreme_down"].between(0.0, 1.0)
    assert finite.all()


def test_down_sensing_result_is_recorded_as_unpromoted_experiment():
    path = Path("research/down_sensing_gate/metrics/summary.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["active_model_changed"] is False
    assert payload["locked_evaluation_read"] is False
    assert payload["locked_origins"] == [268, 315]
    assert payload["confirmation_read"] is True
    assert payload.get("promotion_eligible") in (False, True)
    diagnostics = payload["score_diagnostics"]
    assert len(diagnostics) == 3
