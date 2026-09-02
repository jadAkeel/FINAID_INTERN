import pandas as pd
import pytest

from forecast_select.io import load_workbook


def test_load_workbook_can_stop_before_locked_rows(tmp_path):
    path = tmp_path / "monthly.xlsx"
    frame = pd.DataFrame({
        "Dates": pd.date_range("2020-01-31", periods=8, freq="ME"),
        "X1": range(8),
        "X2": range(10, 18),
    })
    frame.to_excel(path, sheet_name="Sheet1", index=False)
    loaded = load_workbook(path, maximum_position=5)
    assert loaded["position"].tolist() == [1, 2, 3, 4, 5]
    assert loaded["Dates"].max() == frame.loc[4, "Dates"]


def test_load_workbook_rejects_nonpositive_maximum_position(tmp_path):
    with pytest.raises(ValueError, match="must be positive"):
        load_workbook(tmp_path / "unused.xlsx", maximum_position=0)
