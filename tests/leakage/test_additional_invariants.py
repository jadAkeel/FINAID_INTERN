import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_select.features import build_feature_panel
from forecast_select.targets import build_targets


def test_final_row_has_no_available_label():
    frame = pd.DataFrame({"Dates": pd.date_range("2020-01-31", periods=4, freq="ME"), "position": [1, 2, 3, 4], "X1": [1.0, 2.0, 3.0, 4.0]})
    targets = build_targets(frame)
    assert not bool(targets.loc[targets["origin_position"].eq(4), "target_available"].iloc[0])
    assert pd.isna(targets.loc[targets["origin_position"].eq(4), "y_true"].iloc[0])


def test_asof_lag_does_not_use_current_row_value():
    frame = pd.DataFrame({"Dates": pd.date_range("2020-01-31", periods=20, freq="ME"), "position": range(1, 21), "X1": np.arange(20, dtype=float)})
    lagged = build_feature_panel(frame, availability_lag=1)
    row = lagged[(lagged["origin_position"] == 10) & (lagged["indicator_id"] == "X1")].iloc[0]
    assert row["level"] == frame.loc[frame["position"].eq(9), "X1"].iloc[0]


def test_locked_evaluation_hash_is_stable_and_present():
    path = Path("artifacts/audit/locked_evaluation.parquet")
    assert path.exists()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "04ebedf9455051b189486f61deba949299a499915aa33f11b7126efa5a035b39"
