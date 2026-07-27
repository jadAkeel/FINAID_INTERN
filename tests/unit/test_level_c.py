import numpy as np
import pandas as pd

from forecast_select.calibration import score_level_c


def _history() -> pd.DataFrame:
    rows = []
    for origin in range(1, 15):
        for indicator in ["X1", "X2", "X3"]:
            p = 0.8 if (origin + int(indicator[1:])) % 2 else 0.2
            y = int(p >= 0.5)
            rows.append({"origin_position": origin, "indicator_id": indicator, "p_up": p, "y_true": y})
    return pd.DataFrame(rows)


def test_level_c_uses_only_earlier_level_b_rows_and_caps_acceptance():
    history = _history()
    current = pd.DataFrame({
        "origin_position": [15] * 25,
        "indicator_id": [f"X{i}" for i in range(1, 26)],
        "p_up": np.linspace(0.51, 0.99, 25),
        "y_true": [np.nan] * 25,
    })
    result = score_level_c(history, current, floor=0.50, cap=20, min_history_months=12, bootstrap_replicates=50)
    assert result["level_c_ready"].all()
    assert int(result["accepted"].sum()) <= 20
    assert (result["calibration_fit_through_origin"] < result["origin_position"]).all()
    assert (result["reliability_fit_through_origin"] < result["origin_position"]).all()


def test_level_c_rejects_without_earlier_history():
    history = _history().query("origin_position < 5")
    current = pd.DataFrame({"origin_position": [15], "indicator_id": ["X1"], "p_up": [0.9], "y_true": [np.nan]})
    result = score_level_c(history, current, min_history_months=12)
    assert not bool(result.loc[0, "level_c_ready"])
    assert not bool(result.loc[0, "accepted"])
    assert result.loc[0, "rejection_reason"] == "insufficient_earlier_level_b_history"

