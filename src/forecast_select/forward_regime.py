from __future__ import annotations

from collections.abc import Iterable
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .features import build_feature_panel


MARKET_FEATURES = [
    "direction_1",
    "direction_lag_1",
    "direction_lag_2",
    "direction_lag_3",
    "direction_lag_6",
    "direction_lag_12",
    "pct_change_1",
    "momentum_3",
    "momentum_6",
    "momentum_9",
    "momentum_12",
    "robust_z_12",
    "distance_mean_6",
    "distance_mean_12",
    "cross_section_breadth",
    "cross_section_dispersion",
    "pca_factor_1",
    "pca_factor_2",
    "pca_explained_variance_1",
    "pca_explained_variance_2",
    "peer_direction_consensus",
    "regime_breadth_3",
    "regime_dispersion_12",
    "regime_volatility_12",
    "relative_volatility_12",
    "diff_z_12",
    "momentum_acceleration_3_12",
]

GROUP_FEATURES = [
    "direction_1",
    "direction_lag_1",
    "momentum_3",
    "momentum_6",
    "momentum_12",
    "robust_z_12",
]


def _origin_feature_panel(
    frame: pd.DataFrame,
    targets: pd.DataFrame,
    indicator_groups: dict[str, list[str]],
    excluded_indicators: set[str],
    availability_lag: int,
) -> tuple[pd.DataFrame, list[str]]:
    features = build_feature_panel(
        frame,
        availability_lag=availability_lag,
        include_structured=True,
    )
    panel = features.merge(
        targets[["origin_position", "indicator_id", "y_true"]],
        on=["origin_position", "indicator_id"],
        how="left",
        validate="one_to_one",
    )
    panel = panel[
        ~panel["indicator_id"].astype(str).isin(excluded_indicators)
    ].copy()
    group_map = {
        str(indicator): str(group)
        for group, indicators in indicator_groups.items()
        for indicator in indicators
    }
    panel["asset_group"] = panel["indicator_id"].astype(str).map(
        group_map
    ).fillna("ungrouped")

    scale = pd.to_numeric(
        panel["rolling_std_12"], errors="coerce"
    ).replace(0, np.nan)
    level_scale = pd.to_numeric(
        panel["rolling_mean_12"], errors="coerce"
    ).abs().replace(0, np.nan)
    panel["relative_volatility_12"] = scale / level_scale
    panel["diff_z_12"] = (
        pd.to_numeric(panel["diff_1"], errors="coerce") / scale
    )
    panel["momentum_acceleration_3_12"] = (
        pd.to_numeric(panel["momentum_3"], errors="coerce")
        - pd.to_numeric(panel["momentum_12"], errors="coerce")
    )

    # Some structured features legitimately have no history at the earliest
    # origins. Pandas delegates their median to NumPy, which warns for an
    # all-NaN slice even though the resulting NaN is expected and imputed.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Mean of empty slice",
            category=RuntimeWarning,
        )
        aggregate = panel.groupby("origin_position")[MARKET_FEATURES].agg(
            ["mean", "median", "std"]
        )
    aggregate.columns = [
        f"market_{column}_{stat}" for column, stat in aggregate.columns
    ]
    origin = aggregate.reset_index()

    group = panel.pivot_table(
        index="origin_position",
        columns="asset_group",
        values=GROUP_FEATURES,
        aggfunc="mean",
    )
    group.columns = [
        f"group_{asset_group}_{column}"
        for column, asset_group in group.columns
    ]
    origin = origin.merge(
        group.reset_index(),
        on="origin_position",
        how="left",
    )

    target = panel.groupby("origin_position")["y_true"].mean().rename(
        "market_breadth_target"
    )
    history = pd.DataFrame({
        "origin_position": sorted(panel["origin_position"].unique())
    }).merge(
        target.reset_index(),
        on="origin_position",
        how="left",
    )
    for lag in range(2, 25):
        history[f"breadth_lag_{lag}"] = history[
            "market_breadth_target"
        ].shift(lag)
    known = history["market_breadth_target"].shift(2)
    for window in [3, 6, 9, 12, 18, 24, 36, 60]:
        minimum = max(2, window // 2)
        history[f"breadth_mean_{window}"] = known.rolling(
            window,
            min_periods=minimum,
        ).mean()
        history[f"breadth_std_{window}"] = known.rolling(
            window,
            min_periods=minimum,
        ).std()
    history["breadth_change_2_3"] = (
        history["breadth_lag_2"] - history["breadth_lag_3"]
    )
    history["breadth_acceleration"] = (
        history["breadth_lag_2"]
        - 2.0 * history["breadth_lag_3"]
        + history["breadth_lag_4"]
    )
    origin = origin.merge(history, on="origin_position", how="left")

    dates = panel[["origin_position", "origin_date"]].drop_duplicates(
        "origin_position"
    )
    dates["origin_date"] = pd.to_datetime(dates["origin_date"])
    dates["month_sin"] = np.sin(
        2.0 * np.pi * dates["origin_date"].dt.month / 12.0
    )
    dates["month_cos"] = np.cos(
        2.0 * np.pi * dates["origin_date"].dt.month / 12.0
    )
    dates["time_index"] = dates["origin_position"] / 100.0
    origin = origin.merge(
        dates.drop(columns="origin_date"),
        on="origin_position",
        how="left",
    )
    feature_columns = [
        column
        for column in origin.columns
        if column not in {"origin_position", "market_breadth_target"}
    ]
    return origin.sort_values("origin_position"), feature_columns


def build_forward_market_breadth_forecast(
    frame: pd.DataFrame,
    targets: pd.DataFrame,
    indicator_groups: dict[str, list[str]],
    excluded_indicators: set[str],
    forecast_origins: Iterable[int],
    availability_lag: int,
    model_settings: dict,
) -> pd.DataFrame:
    """Forecast market breadth with targets available through origin - 2."""
    origins = sorted({int(origin) for origin in forecast_origins})
    if not origins:
        raise ValueError("Forward regime forecast requires at least one origin")
    maximum_origin = max(origins)
    safe_frame = frame[frame["position"].le(maximum_origin)].copy()
    safe_targets = targets[
        targets["origin_position"].le(maximum_origin - 2)
    ].copy()
    origin_panel, feature_columns = _origin_feature_panel(
        safe_frame,
        safe_targets,
        indicator_groups,
        excluded_indicators,
        availability_lag,
    )
    rows = []
    for origin in origins:
        train = origin_panel[
            origin_panel["market_breadth_target"].notna()
            & origin_panel["origin_position"].le(origin - 2)
        ]
        test = origin_panel[origin_panel["origin_position"].eq(origin)]
        if train.empty or test.empty:
            raise ValueError(
                f"Forward regime data is unavailable for origin {origin}"
            )
        available_features = [
            column
            for column in feature_columns
            if train[column].notna().any()
        ]
        model = Pipeline([
            (
                "impute",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=int(model_settings["max_iter"]),
                    learning_rate=float(model_settings["learning_rate"]),
                    max_depth=int(model_settings["max_depth"]),
                    min_samples_leaf=int(
                        model_settings["minimum_samples_leaf"]
                    ),
                    l2_regularization=float(
                        model_settings["l2_regularization"]
                    ),
                    random_state=int(model_settings["seed"]),
                ),
            ),
        ])
        model.fit(
            train[available_features],
            train["market_breadth_target"].astype(float),
        )
        probability = float(
            np.clip(
                model.predict(test[available_features])[0],
                0.02,
                0.98,
            )
        )
        rows.append({
            "origin_position": origin,
            "forecast_market_breadth": probability,
            "forecast_market_breadth_fit_through_origin": int(
                train["origin_position"].max()
            ),
            "forecast_market_breadth_observation_through_origin": origin - 1,
        })
    return pd.DataFrame(rows)
