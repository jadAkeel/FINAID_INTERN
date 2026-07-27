from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ValidationLayout:
    n_rows: int
    feature_warmup_end: int
    initial_estimation_end: int
    development_origins: tuple[int, ...]
    audit_origins: tuple[int, ...]
    production_origin: int


def make_layout(n_rows: int, audit_size: int = 48) -> ValidationLayout:
    if n_rows < audit_size + 20:
        raise ValueError("Not enough rows for a locked audit")
    # Positions are 1-based: the last raw row is production origin n_rows,
    # so the last evaluable origin is n_rows - 1.
    audit_start = n_rows - audit_size
    audit_end = n_rows - 1
    initial_end = min(119, audit_start - 1)
    dev_start = 120
    if dev_start >= audit_start:
        raise ValueError("Not enough rows for the development origins")
    return ValidationLayout(
        n_rows=n_rows,
        feature_warmup_end=12,
        initial_estimation_end=initial_end,
        development_origins=tuple(range(dev_start, audit_start)),
        audit_origins=tuple(range(audit_start, audit_end + 1)),
        production_origin=n_rows,
    )


def origin_rows(frame: pd.DataFrame, origins: tuple[int, ...]) -> pd.DataFrame:
    return frame[frame["origin_position"].isin(origins)].copy()


def assert_no_same_month_training(train: pd.DataFrame, origin: int) -> None:
    if not train.empty and int(train["origin_position"].max()) >= origin:
        raise AssertionError("Training rows include the forecast origin or future")


def assert_target_alignment(targets: pd.DataFrame, frame: pd.DataFrame) -> None:
    check = targets.dropna(subset=["y_true"]).sample(min(200, len(targets.dropna(subset=["y_true"]))), random_state=7)
    for row in check.itertuples(index=False):
        current = frame.loc[frame["position"].eq(row.origin_position), row.indicator_id].iloc[0]
        future = frame.loc[frame["position"].eq(row.origin_position + 1), row.indicator_id].iloc[0]
        expected = int(future > current)
        if int(row.y_true) != expected:
            raise AssertionError("Target alignment mismatch")
