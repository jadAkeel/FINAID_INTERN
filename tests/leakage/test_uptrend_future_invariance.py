import numpy as np
import pandas as pd

from forecast_select.features import build_feature_panel


def test_future_values_do_not_change_earlier_uptrend_features():
    rows = 90
    frame = pd.DataFrame({
        "Dates": pd.date_range("2010-01-31", periods=rows, freq="ME"),
        "position": range(1, rows + 1),
        "X1": np.arange(rows, dtype=float) + 10,
        "X2": np.arange(rows, dtype=float) * 1.5 + 20,
        "X3": np.arange(rows, dtype=float) * -0.25 + 100,
    })
    changed = frame.copy()
    changed.loc[70:, ["X1", "X2", "X3"]] *= 100
    baseline = build_feature_panel(frame, availability_lag=1, include_structured=True)
    perturbed = build_feature_panel(changed, availability_lag=1, include_structured=True)
    columns = ["pca_factor_1", "pca_loading_1", "peer_corr_abs_topk_mean", "regime_breadth_3"]
    earlier = baseline[baseline["origin_position"] <= 70].sort_values(["origin_position", "indicator_id"])
    changed_earlier = perturbed[perturbed["origin_position"] <= 70].sort_values(["origin_position", "indicator_id"])
    np.testing.assert_allclose(earlier[columns].to_numpy(dtype=float), changed_earlier[columns].to_numpy(dtype=float), equal_nan=True)
