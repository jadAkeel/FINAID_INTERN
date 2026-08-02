from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DOWNSIDE_NUMERIC_FEATURES = [
    "risk_return_1",
    "risk_return_lag_1",
    "risk_return_lag_2",
    "risk_momentum_3",
    "risk_momentum_6",
    "risk_momentum_12",
    "risk_volatility_3",
    "risk_volatility_6",
    "risk_volatility_12",
    "risk_drawdown_12",
    "risk_distance_mean_12",
    "risk_negative_share_3",
    "risk_previous_shock",
    "risk_previous_shock_share",
    "risk_previous_group_shock_share",
    "risk_market_mean_return",
    "risk_market_median_return",
    "risk_breadth_up",
    "risk_dispersion",
    "risk_market_volatility_6",
    "risk_breadth_mean_3",
    "risk_average_correlation_12",
    "risk_thematic_equity_return",
    "risk_commodity_return",
    "risk_us_sector_return",
    "risk_global_equity_return",
    "risk_fixed_income_return",
    "risk_currency_return",
    "risk_X18_return",
    "risk_X20_return",
    "risk_X32_return",
    "risk_X37_return",
    "risk_X39_return",
    "risk_X42_return",
    "risk_X44_return",
]


def indicator_group_map(settings: dict) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for group, indicators in settings["indicator_groups"].items():
        for indicator in indicators:
            if indicator in mapping:
                raise ValueError(f"Indicator {indicator} belongs to multiple groups")
            mapping[str(indicator)] = str(group)
    return mapping


def _rolling_mad(
    series: pd.Series,
    window: int,
    minimum_history: int,
) -> pd.Series:
    history = series.shift(1)
    return history.rolling(window, min_periods=minimum_history).apply(
        lambda values: float(
            np.median(np.abs(values - np.median(values)))
        ),
        raw=True,
    )


def build_sudden_drop_labels(
    targets: pd.DataFrame,
    trailing_window: int,
    minimum_history: int,
    lower_quantile: float,
    robust_z: float,
) -> pd.DataFrame:
    """Label unusually large negative returns using only earlier target returns."""
    if trailing_window < minimum_history or minimum_history < 12:
        raise ValueError("Shock history windows are invalid")
    if not 0.0 < lower_quantile < 0.5:
        raise ValueError("lower_quantile must be between 0 and 0.5")
    if robust_z <= 0:
        raise ValueError("robust_z must be positive")

    parts = []
    ordered = targets.sort_values(
        ["indicator_id", "origin_position"]
    )
    for indicator, group in ordered.groupby("indicator_id", sort=False):
        current = pd.to_numeric(group["value_t"], errors="coerce")
        future = pd.to_numeric(group["value_t1"], errors="coerce")
        target_return = future.div(current.replace(0, np.nan)).sub(1.0)
        history = target_return.shift(1)
        lower_tail = history.rolling(
            trailing_window,
            min_periods=minimum_history,
        ).quantile(lower_quantile)
        historical_median = history.rolling(
            trailing_window,
            min_periods=minimum_history,
        ).median()
        historical_mad = _rolling_mad(
            target_return,
            trailing_window,
            minimum_history,
        )
        robust_scale = 1.4826 * historical_mad
        robust_threshold = historical_median - robust_z * robust_scale
        valid = (
            target_return.notna()
            & lower_tail.notna()
            & robust_threshold.notna()
            & robust_scale.gt(0)
        )
        sudden_drop = (
            target_return.lt(0)
            & target_return.le(lower_tail)
            & target_return.le(robust_threshold)
            & valid
        )
        part = group[[
            "origin_position",
            "origin_date",
            "target_date",
            "indicator_id",
        ]].copy()
        part["target_return"] = target_return.to_numpy(dtype=float)
        part["shock_lower_tail"] = lower_tail.to_numpy(dtype=float)
        part["shock_robust_threshold"] = robust_threshold.to_numpy(dtype=float)
        part["shock_label_valid"] = valid.to_numpy(dtype=bool)
        part["sudden_drop"] = sudden_drop.to_numpy(dtype=bool)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _average_pairwise_correlation(window: pd.DataFrame) -> float:
    eligible = window.loc[:, window.notna().sum().ge(8)]
    if eligible.shape[1] < 3:
        return np.nan
    correlation = eligible.corr(min_periods=8).to_numpy(dtype=float)
    upper = correlation[np.triu_indices_from(correlation, k=1)]
    upper = upper[np.isfinite(upper)]
    return float(upper.mean()) if len(upper) else np.nan


def build_downside_feature_panel(
    frame: pd.DataFrame,
    settings: dict,
) -> pd.DataFrame:
    """Build past-only downside features at each forecast origin."""
    lag = int(settings["availability_lag_months"])
    indicators = [
        column for column in frame.columns if column.startswith("X")
    ]
    source = frame[indicators].shift(lag)
    returns = source.pct_change(fill_method=None)
    mapping = indicator_group_map(settings)

    cross = pd.DataFrame({
        "origin_position": frame["position"],
        "risk_market_mean_return": returns.mean(axis=1),
        "risk_market_median_return": returns.median(axis=1),
        "risk_breadth_up": (
            returns.gt(0).sum(axis=1)
            / returns.notna().sum(axis=1).replace(0, np.nan)
        ),
        "risk_dispersion": returns.std(axis=1),
    })
    cross["risk_market_volatility_6"] = cross[
        "risk_market_mean_return"
    ].rolling(6, min_periods=3).std()
    cross["risk_breadth_mean_3"] = cross["risk_breadth_up"].rolling(
        3,
        min_periods=2,
    ).mean()
    cross["risk_average_correlation_12"] = [
        _average_pairwise_correlation(
            returns.iloc[max(0, index - 11):index + 1]
        )
        for index in range(len(returns))
    ]
    for group in settings["indicator_groups"]:
        members = [
            indicator
            for indicator, indicator_group in mapping.items()
            if indicator_group == group and indicator in returns
        ]
        cross[f"risk_{group}_return"] = returns[members].mean(axis=1)
    for indicator in ["X18", "X20", "X32", "X37", "X39", "X42", "X44"]:
        cross[f"risk_{indicator}_return"] = returns[indicator]

    parts = []
    for indicator in indicators:
        level = source[indicator]
        indicator_return = returns[indicator]
        part = pd.DataFrame({
            "origin_position": frame["position"],
            "indicator_id": indicator,
            "indicator_group": mapping.get(indicator, "unmapped"),
            "risk_return_1": indicator_return,
            "risk_return_lag_1": indicator_return.shift(1),
            "risk_return_lag_2": indicator_return.shift(2),
            "risk_momentum_3": level.div(level.shift(3).replace(0, np.nan)).sub(1),
            "risk_momentum_6": level.div(level.shift(6).replace(0, np.nan)).sub(1),
            "risk_momentum_12": level.div(level.shift(12).replace(0, np.nan)).sub(1),
            "risk_volatility_3": indicator_return.rolling(3, min_periods=2).std(),
            "risk_volatility_6": indicator_return.rolling(6, min_periods=3).std(),
            "risk_volatility_12": indicator_return.rolling(12, min_periods=6).std(),
            "risk_drawdown_12": level.div(
                level.rolling(12, min_periods=6).max().replace(0, np.nan)
            ).sub(1),
            "risk_distance_mean_12": level.div(
                level.rolling(12, min_periods=6).mean().replace(0, np.nan)
            ).sub(1),
            "risk_negative_share_3": indicator_return.lt(0).where(
                indicator_return.notna()
            ).rolling(3, min_periods=2).mean(),
        })
        parts.append(part.merge(
            cross,
            on="origin_position",
            how="left",
            validate="one_to_one",
        ))
    return pd.concat(parts, ignore_index=True)


def add_known_shock_features(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    settings: dict,
) -> pd.DataFrame:
    """Attach the newest shock outcome whose target is available at the origin."""
    lag = int(settings["availability_lag_months"])
    known = labels[[
        "origin_position",
        "indicator_id",
        "shock_label_valid",
        "sudden_drop",
    ]].copy()
    known["origin_position"] = known["origin_position"] + lag + 1
    known["risk_previous_shock"] = known["sudden_drop"].astype(float).where(
        known["shock_label_valid"].astype(bool)
    )
    mapping = indicator_group_map(settings)
    known["indicator_group"] = known["indicator_id"].map(mapping)
    valid = known[known["shock_label_valid"].astype(bool)].copy()
    market_share = valid.groupby("origin_position")[
        "sudden_drop"
    ].mean().rename("risk_previous_shock_share")
    group_share = valid.groupby([
        "origin_position",
        "indicator_group",
    ])["sudden_drop"].mean().rename("risk_previous_group_shock_share")
    known = known.merge(
        market_share,
        on="origin_position",
        how="left",
        validate="many_to_one",
    ).merge(
        group_share,
        on=["origin_position", "indicator_group"],
        how="left",
        validate="many_to_one",
    )
    return panel.merge(
        known[[
            "origin_position",
            "indicator_id",
            "risk_previous_shock",
            "risk_previous_shock_share",
            "risk_previous_group_shock_share",
        ]],
        on=["origin_position", "indicator_id"],
        how="left",
        validate="one_to_one",
    )


@dataclass
class DownsideRiskModel:
    model: Any
    feature_columns: list[str]


def fit_downside_risk_model(
    train: pd.DataFrame,
    seed: int,
    logistic_c: float,
    max_iter: int,
) -> DownsideRiskModel:
    numeric = [
        column for column in DOWNSIDE_NUMERIC_FEATURES
        if column in train.columns
    ]
    categorical = ["indicator_id", "indicator_group"]
    preprocessor = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                (
                    "impute",
                    SimpleImputer(strategy="median", add_indicator=True),
                ),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical,
        ),
    ], remainder="drop")
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        (
            "classifier",
            LogisticRegression(
                C=logistic_c,
                penalty="l2",
                solver="lbfgs",
                class_weight="balanced",
                max_iter=max_iter,
                random_state=seed,
            ),
        ),
    ])
    if train["sudden_drop"].nunique() < 2:
        raise ValueError("Downside model requires both shock classes")
    feature_columns = [*numeric, *categorical]
    pipeline.fit(
        train[feature_columns],
        train["sudden_drop"].astype(int),
    )
    return DownsideRiskModel(pipeline, feature_columns)


def predict_downside_probability(
    model: DownsideRiskModel,
    test: pd.DataFrame,
) -> np.ndarray:
    prepared = test[model.feature_columns].copy()
    for column in ["indicator_id", "indicator_group"]:
        prepared[column] = prepared[column].fillna("missing").astype(str)
    return np.clip(
        model.model.predict_proba(prepared)[:, 1],
        1e-6,
        1.0 - 1e-6,
    )


def apply_downside_risk_gate(
    base_predictions: pd.DataFrame,
    risk_predictions: pd.DataFrame,
    penalty: float,
    cap: int,
) -> pd.DataFrame:
    """Rerank Up candidates by subtracting relative downside risk."""
    if penalty < 0:
        raise ValueError("penalty must be non-negative")
    if cap < 1:
        raise ValueError("cap must be positive")
    risk_columns = [
        "origin_position",
        "indicator_id",
        "p_sudden_drop",
        "sudden_drop",
        "shock_label_valid",
        "risk_fit_through_origin",
    ]
    merged = base_predictions.merge(
        risk_predictions[risk_columns],
        on=["origin_position", "indicator_id"],
        how="left",
        validate="one_to_one",
    )
    merged["base_accepted"] = merged["accepted"].fillna(False).astype(bool)
    merged["risk_percentile"] = merged.groupby(
        "origin_position"
    )["p_sudden_drop"].rank(method="average", pct=True)
    merged["risk_gate_penalty"] = float(penalty)
    merged["risk_adjusted_score"] = (
        pd.to_numeric(merged["selection_score"], errors="coerce")
        - penalty * merged["risk_percentile"]
    )
    merged["accepted"] = False
    merged["selection_rank"] = np.nan
    merged["rejection_reason"] = "risk_gate_monthly_cap"

    for origin, positions in merged.groupby(
        "origin_position",
        sort=True,
    ).groups.items():
        current = merged.loc[positions]
        candidates = current[
            current["level_c_ready"].fillna(False).astype(bool)
            & current["p_sudden_drop"].notna()
            & current["predicted_direction"].eq("Up")
        ].sort_values("risk_adjusted_score", ascending=False)
        if len(candidates) < cap:
            raise AssertionError(
                f"Origin {origin} has fewer than {cap} eligible Up candidates"
            )
        accepted_index = candidates.head(cap).index
        merged.loc[accepted_index, "accepted"] = True
        merged.loc[accepted_index, "selection_rank"] = np.arange(
            1,
            cap + 1,
        )
        merged.loc[accepted_index, "rejection_reason"] = ""

    merged["risk_gate_changed"] = (
        merged["base_accepted"]
        != merged["accepted"].fillna(False).astype(bool)
    )
    return merged


def summarize_gate_predictions(predictions: pd.DataFrame) -> dict[str, float | int]:
    selected = predictions[
        predictions["accepted"].fillna(False).astype(bool)
        & predictions["y_true"].notna()
    ].copy()
    selected["correct"] = selected["predicted_direction"].eq("Up").astype(
        int
    ).eq(selected["y_true"].astype(int))
    return {
        "months": int(selected["origin_position"].nunique()),
        "calls": int(len(selected)),
        "hits": int(selected["correct"].sum()),
        "accuracy": float(selected["correct"].mean()),
        "up_calls": int(selected["predicted_direction"].eq("Up").sum()),
        "down_calls": int(selected["predicted_direction"].eq("Down").sum()),
        "changed_calls": int(
            predictions[
                predictions["risk_gate_changed"].fillna(False).astype(bool)
            ].shape[0]
            // 2
        ),
    }
