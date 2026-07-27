from __future__ import annotations

import pandas as pd


def build_targets(frame: pd.DataFrame) -> pd.DataFrame:
    indicators = [c for c in frame.columns if c.startswith("X")]
    rows = []
    for indicator in indicators:
        current = frame[indicator]
        future = current.shift(-1)
        change = future - current
        rows.append(pd.DataFrame({
            "origin_position": frame["position"],
            "origin_date": frame["Dates"],
            "target_date": frame["Dates"].shift(-1),
            "indicator_id": indicator,
            "value_t": current,
            "value_t1": future,
            "change_t_to_t1": change,
            "y_true": (change > 0).astype("float64").where(change.notna()),
            "zero_change": (change == 0).astype("boolean").where(change.notna()),
            "target_available": future.notna() & current.notna(),
        }))
    return pd.concat(rows, ignore_index=True)


def target_for_origin(frame: pd.DataFrame, origin_position: int, indicator: str) -> int | None:
    row = frame.loc[frame["position"].eq(origin_position), indicator]
    future = frame.loc[frame["position"].eq(origin_position + 1), indicator]
    if row.empty or future.empty or pd.isna(row.iloc[0]) or pd.isna(future.iloc[0]):
        return None
    return int(future.iloc[0] > row.iloc[0])

