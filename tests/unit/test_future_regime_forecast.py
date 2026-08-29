import json

import pandas as pd

from forecast_select import future_regime_forecast


def _forecasts() -> pd.DataFrame:
    rows = []
    for horizon, month in [(1, "2026-06"), (2, "2026-07"), (3, "2026-08")]:
        for rank in range(1, 16):
            rows.append({
                "forecast_horizon_months": horizon,
                "forecast_month": month,
                "model_scope": (
                    "frozen_regime_adaptive_one_step"
                    if horizon == 1
                    else "experimental_direct_horizon_regime_adaptive"
                ),
                "training_fit_through_origin": 315 - horizon,
                "generalized_graph_fit_through_origin": 315,
                "regime_label": "mixed",
                "regime_stress": 0.5,
                "forecast_market_breadth": 0.7,
                "forecast_market_breadth_fit_through_origin": 314,
                "regime_cap": 15,
                "selection_rank": rank,
                "indicator_id": f"X{rank}",
                "predicted_direction": "Up",
                "selection_score": 0.75 - rank / 100,
                "p_up_base": 0.7,
                "p_down": 0.3,
                "asset_group": "test",
            })
    return pd.DataFrame(rows)


def test_regime_adaptive_writer_serializes_three_direct_horizons(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/config.yaml").write_text(
        "data_path: data/monthly_indicators.xlsx\n",
        encoding="utf-8",
    )
    (tmp_path / "configs/regime_adaptive_selector.yaml").write_text(
        "availability_lag_months: 1\n",
        encoding="utf-8",
    )
    summary_path = tmp_path / "research/regime_adaptive_selector/metrics/summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps({
        "selection_mode": "conservative_fallback_no_stable_candidate",
        "selected_parameters": {"allow_down_predictions": False},
        "generalized_correlation_overlay": {
            "enabled": True,
            "window_months": 48,
        },
    }), encoding="utf-8")

    frame = pd.DataFrame({
        "position": [315, 316],
        "Dates": pd.to_datetime(["2026-04-30", "2026-05-29"]),
    })
    monkeypatch.setattr(future_regime_forecast, "load_workbook", lambda _: frame)
    monkeypatch.setattr(
        future_regime_forecast,
        "build_regime_adaptive_monthly_forecasts",
        lambda *args, **kwargs: _forecasts(),
    )
    monkeypatch.setattr(future_regime_forecast, "sha256_file", lambda _: "hash")
    monkeypatch.setattr(
        future_regime_forecast.pd,
        "read_parquet",
        lambda _: pd.DataFrame({"origin_position": [120, 266]}),
    )

    output = future_regime_forecast.write_regime_adaptive_next_three_forecast(
        tmp_path
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["generated_from_data_through"] == "2026-05-29"
    assert payload["feature_information_through"] == "2026-04-30"
    assert [row["forecast_month"] for row in payload["forecasts"]] == [
        "2026-06",
        "2026-07",
        "2026-08",
    ]
    assert [row["horizon_months"] for row in payload["forecasts"]] == [1, 2, 3]
    assert all(len(row["selections"]) == 15 for row in payload["forecasts"])
    assert payload["generalized_correlation_overlay"]["enabled"] is True
    assert all(
        row["generalized_graph_fit_through_origin"] == 315
        for row in payload["forecasts"]
    )
    assert all(
        row["forecast_market_breadth"] == 0.7
        and row["forward_regime_fit_through_origin"] == 314
        for row in payload["forecasts"]
    )
    assert all(
        "actual" not in selection and "correct" not in selection
        for row in payload["forecasts"]
        for selection in row["selections"]
    )
