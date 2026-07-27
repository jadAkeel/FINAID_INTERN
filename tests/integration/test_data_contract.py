from pathlib import Path

from forecast_select.io import load_workbook


def test_supplied_workbook_contract():
    path = Path("data/raw/FinalList_Extended.xlsx")
    frame = load_workbook(path)
    assert len(frame) == 316
    assert len([c for c in frame.columns if c.startswith("X")]) == 50
    assert frame["Dates"].iloc[0].strftime("%Y-%m-%d") == "2000-02-29"
    assert frame["Dates"].iloc[-1].strftime("%Y-%m-%d") == "2026-05-29"

