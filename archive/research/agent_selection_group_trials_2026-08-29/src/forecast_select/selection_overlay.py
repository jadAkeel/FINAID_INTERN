"""Recent-miss + group-stability selection overlay (Family F).

Builds a refined ``p_up_selection_score`` by penalising indicators whose
recent causal history shows a high miss rate and rewarding groups whose
recent up-rate is stable. All statistics are fit through ``t-2`` so no
future label leaks.

The module exposes two public functions:

* :func:`build_recent_misses` and :func:`build_group_stability` build the
  per-origin causal statistics.
* :func:`apply_selection_overlay` updates the input frame in place with
  the new ``p_up_selection_score`` (named ``p_up_selection_score`` by
  default) while preserving all other columns.

The selected parameters are documented in
``research/regime_adaptive_selection_group_v2/``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _logit(p):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_recent_misses(
    frame: pd.DataFrame,
    *,
    window_months: int = 6,
    label_lag: int = 2,
) -> pd.DataFrame:
    """Return a frame with ``fit_through_origin``, ``indicator_id``,
    ``recent_miss_rate`` and ``recent_call_count``.

    At ``fit_through_origin = t - label_lag`` the rolling miss is the mean
    of ``1 - y_true`` for the previous ``window_months`` months.
    """
    work = frame[["origin_position", "indicator_id", "y_true"]].copy()
    work["y_true"] = pd.to_numeric(work["y_true"], errors="coerce")
    work = work.dropna(subset=["y_true"])
    work = work.sort_values(["indicator_id", "origin_position"]).reset_index(drop=True)
    work["miss"] = (1.0 - work["y_true"]).astype(float)
    work["rolling_miss"] = (
        work.groupby("indicator_id")["miss"]
        .transform(
            lambda s: s.shift(1)
            .rolling(window_months, min_periods=max(2, window_months // 3))
            .mean()
        )
    )
    work["rolling_calls"] = (
        work.groupby("indicator_id")["miss"]
        .transform(
            lambda s: s.shift(1)
            .rolling(window_months, min_periods=max(2, window_months // 3))
            .count()
        )
    )
    work["fit_through_origin"] = work["origin_position"] - int(label_lag)
    return work[
        ["fit_through_origin", "indicator_id", "rolling_miss", "rolling_calls"]
    ].rename(
        columns={
            "rolling_miss": "recent_miss_rate",
            "rolling_calls": "recent_call_count",
        }
    )


def build_group_stability(
    frame: pd.DataFrame,
    *,
    window_months: int = 6,
    label_lag: int = 2,
) -> pd.DataFrame:
    """Return a frame with ``fit_through_origin``, ``asset_group``,
    ``group_rolling_up_rate`` and ``group_rolling_std``.
    """
    work = frame[["origin_position", "asset_group", "y_true"]].copy()
    work["y_true"] = pd.to_numeric(work["y_true"], errors="coerce")
    work = work.dropna(subset=["y_true", "asset_group"])
    grouped = (
        work.groupby(["asset_group", "origin_position"])["y_true"]
        .agg(up_rate="mean", n_indicators="count")
        .reset_index()
    )
    grouped = grouped.sort_values(["asset_group", "origin_position"]).reset_index(drop=True)
    grouped["rolling_up_rate"] = (
        grouped.groupby("asset_group")["up_rate"]
        .transform(
            lambda s: s.shift(1)
            .rolling(window_months, min_periods=max(2, window_months // 3))
            .mean()
        )
    )
    grouped["rolling_std"] = (
        grouped.groupby("asset_group")["up_rate"]
        .transform(
            lambda s: s.shift(1)
            .rolling(window_months, min_periods=max(2, window_months // 3))
            .std()
        )
    )
    grouped["fit_through_origin"] = grouped["origin_position"] - int(label_lag)
    return grouped[
        ["fit_through_origin", "asset_group", "rolling_up_rate", "rolling_std"]
    ].rename(
        columns={
            "rolling_up_rate": "group_rolling_up_rate",
            "rolling_std": "group_rolling_std",
        }
    )


def apply_selection_overlay(
    frame: pd.DataFrame,
    misses: pd.DataFrame,
    stability: pd.DataFrame,
    *,
    base_threshold: float = 0.45,
    threshold_relax: float = 0.10,
    history_full: int = 4,
    history_zero: int = 1,
    stability_bonus: float = 0.30,
    miss_penalty_strength: float = 0.40,
    stability_penalty_std: float = 0.5,
    output_column: str = "p_up_selection_score",
) -> pd.DataFrame:
    """Apply the recent-miss + group-stability overlay.

    Returns a copy of ``frame`` with an additional column ``output_column``
    containing the refined probability. The original ``p_up_selection_score``
    is preserved as ``p_up_selection_score_baseline`` so the caller can
    fall back to the un-overlaid score if needed.
    """
    out = frame.copy()
    out["p_up_selection_score_baseline"] = pd.to_numeric(
        out.get("p_up_selection_score", out.get("p_up")), errors="coerce"
    ).fillna(pd.to_numeric(out["p_up"], errors="coerce"))

    out = out.merge(
        misses,
        left_on=["origin_position", "indicator_id"],
        right_on=["fit_through_origin", "indicator_id"],
        how="left",
        validate="many_to_one",
    )
    out = out.merge(
        stability,
        left_on=["origin_position", "asset_group"],
        right_on=["fit_through_origin", "asset_group"],
        how="left",
        validate="many_to_one",
    )

    base = pd.to_numeric(out["p_up_selection_score_baseline"], errors="coerce").fillna(
        pd.to_numeric(out["p_up"], errors="coerce")
    ).clip(1e-6, 1.0 - 1e-6)
    base_logit = _logit(base)

    rolling_miss = pd.to_numeric(out["recent_miss_rate"], errors="coerce")
    rolling_calls = pd.to_numeric(out["recent_call_count"], errors="coerce")
    calls = rolling_calls.fillna(0.0)
    history_weight = ((calls - history_zero) / max(1, history_full - history_zero)).clip(0.0, 1.0)
    dynamic_threshold = base_threshold - threshold_relax * history_weight
    miss_penalty = (
        (rolling_miss - dynamic_threshold).clip(lower=0.0)
        / (1.0 - dynamic_threshold).clip(lower=1e-3)
    ).where(calls.ge(history_zero), 0.0)
    miss_penalty = miss_penalty.fillna(0.0).clip(0.0, 1.0) * history_weight

    group_up = pd.to_numeric(out["group_rolling_up_rate"], errors="coerce").fillna(0.5)
    group_std = pd.to_numeric(out["group_rolling_std"], errors="coerce").fillna(0.2)
    stability_value = (
        (group_up - 0.5) - stability_penalty_std * group_std
    ).clip(-0.5, 0.5)
    adjusted = (
        base_logit
        - miss_penalty_strength * miss_penalty
        + stability_bonus * stability_value.fillna(0.0)
    )
    out[output_column] = np.clip(_sigmoid(adjusted), 1e-6, 1.0 - 1e-6)
    out["selection_overlay_recent_miss"] = rolling_miss
    out["selection_overlay_recent_calls"] = calls
    out["selection_overlay_miss_penalty"] = miss_penalty
    out["selection_overlay_history_weight"] = history_weight
    out["selection_overlay_stability_bonus"] = stability_bonus * stability_value.fillna(0.0)
    out["selection_overlay_base_threshold"] = float(base_threshold)
    out["selection_overlay_stability_bonus_strength"] = float(stability_bonus)
    return out
