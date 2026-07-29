from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(payload: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_parquet(frame: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, target)


def load_workbook(path: str | Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="Sheet1", engine="openpyxl")
    if frame.empty or frame.columns[0] != "Dates":
        raise ValueError("Expected a non-empty Sheet1 whose first column is Dates")
    frame["Dates"] = pd.to_datetime(frame["Dates"], errors="raise")
    indicators = [str(c) for c in frame.columns[1:]]
    if indicators != [f"X{i}" for i in range(1, len(indicators) + 1)]:
        raise ValueError("Indicator columns must be contiguous X1..Xn")
    if frame["Dates"].duplicated().any() or not frame["Dates"].is_monotonic_increasing:
        raise ValueError("Dates must be sorted and unique")
    periods = frame["Dates"].dt.to_period("M")
    if periods.duplicated().any():
        raise ValueError("Dates must contain at most one row per calendar month")
    expected = pd.period_range(periods.iloc[0], periods.iloc[-1], freq="M")
    if periods.tolist() != expected.tolist():
        raise ValueError("Dates contain missing calendar months")
    for column in indicators:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        nonnumeric = frame[column].notna() & numeric.isna()
        if nonnumeric.any():
            raise ValueError(f"Non-numeric values found in {column}")
        frame[column] = numeric.astype(float)
    frame["period"] = periods.astype(str)
    frame["position"] = range(1, len(frame) + 1)
    return frame
