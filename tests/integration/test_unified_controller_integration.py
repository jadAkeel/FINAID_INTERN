import json
from pathlib import Path

import pandas as pd


def test_unified_controller_artifact_is_causal_and_keeps_monthly_cap():
    path = Path(
        "research/unified_forecast_controller/artifacts/predictions.parquet"
    )
    assert path.exists()
    predictions = pd.read_parquet(path)
    assert int(predictions["origin_position"].min()) == 120
    assert int(predictions["origin_position"].max()) == 266
    assert not predictions["locked_evaluation_read"].any()
    selected = predictions[predictions["accepted"]]
    monthly = selected.groupby("origin_position")["indicator_id"].agg(
        ["count", "nunique"]
    )
    assert monthly["count"].eq(15).all()
    assert monthly["nunique"].eq(15).all()


def test_unified_controller_is_non_promoting():
    path = Path("research/unified_forecast_controller/metrics/summary.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["active_model_changed"] is False
    assert payload["locked_evaluation_read"] is False

