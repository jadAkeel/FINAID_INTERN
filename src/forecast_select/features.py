from __future__ import annotations

import numpy as np
import pandas as pd


def _stale_run(series: pd.Series) -> pd.Series:
    changed = series.diff().ne(0) & series.notna() & series.shift(1).notna()
    groups = changed.cumsum()
    return series.notna().groupby(groups).cumcount().astype(float).where(series.notna(), np.nan)


def build_feature_panel(frame: pd.DataFrame, availability_lag: int = 1) -> pd.DataFrame:
    """Build causal features. At origin t, feature values use observations through t-lag."""
    indicators = [c for c in frame.columns if c.startswith("X")]
    source = frame[indicators].shift(availability_lag)
    feature_parts = []
    changes = source.diff()
    for indicator in indicators:
        s = source[indicator]
        d = changes[indicator]
        rolling_mean = s.rolling(12, min_periods=6).mean()
        rolling_std = s.rolling(12, min_periods=6).std()
        mad = s.rolling(12, min_periods=6).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
        out = pd.DataFrame({
            "origin_position": frame["position"],
            "origin_date": frame["Dates"],
            "indicator_id": indicator,
            "level": s,
            "diff_1": d,
            "pct_change_1": d / s.shift(1).replace(0, np.nan),
            "direction_1": (d > 0).astype(float),
            "direction_lag_1": (d.shift(1) > 0).astype(float),
            "direction_lag_2": (d.shift(2) > 0).astype(float),
            "direction_lag_3": (d.shift(3) > 0).astype(float),
            "direction_lag_6": (d.shift(6) > 0).astype(float),
            "direction_lag_12": (d.shift(12) > 0).astype(float),
            "change_lag_1": d.shift(1),
            "change_lag_2": d.shift(2),
            "change_lag_3": d.shift(3),
            "change_lag_6": d.shift(6),
            "change_lag_12": d.shift(12),
            "momentum_3": s / s.shift(3).replace(0, np.nan) - 1,
            "momentum_6": s / s.shift(6).replace(0, np.nan) - 1,
            "momentum_12": s / s.shift(12).replace(0, np.nan) - 1,
            "rolling_mean_12": rolling_mean,
            "rolling_std_12": rolling_std,
            "rolling_mad_12": mad,
            "robust_z_12": (s - rolling_mean) / (1.4826 * mad.replace(0, np.nan)),
            "distance_mean_6": s / s.rolling(6, min_periods=3).mean().replace(0, np.nan) - 1,
            "distance_mean_12": s / rolling_mean.replace(0, np.nan) - 1,
            "stale_run": _stale_run(s),
            "observed": s.notna().astype(float),
            "time_since_observation": (~s.notna()).groupby(s.notna().cumsum()).cumcount().astype(float),
        })
        feature_parts.append(out)
    panel = pd.concat(feature_parts, ignore_index=True)
    changes_long = changes.stack(dropna=False).rename("current_change").reset_index()
    changes_long.columns = ["row_index", "indicator_id", "current_change"]
    stats = changes_long.groupby("row_index")["current_change"].agg(
        cross_section_median="median", cross_section_dispersion="std"
    )
    stats["cross_section_breadth"] = changes_long.assign(up=changes_long["current_change"] > 0).groupby("row_index")["up"].mean()
    stats["cross_section_rank"] = changes_long.groupby("row_index")["current_change"].rank(pct=True).groupby(changes_long["row_index"]).mean()
    panel["row_index"] = panel["origin_position"] - 1
    panel = panel.join(stats, on="row_index")
    panel = panel.drop(columns="row_index")
    return panel

