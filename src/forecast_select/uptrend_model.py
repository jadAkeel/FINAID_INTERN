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

from .features import FEATURE_FAMILY_COLUMNS


FEATURE_COLUMNS = [
    "level", "diff_1", "pct_change_1", "direction_1",
    "direction_lag_1", "direction_lag_2", "direction_lag_3",
    "direction_lag_6", "direction_lag_12",
    "change_lag_1", "change_lag_2", "change_lag_3",
    "change_lag_6", "change_lag_12",
    "momentum_3", "momentum_6", "momentum_9", "momentum_12",
    "rolling_mean_12", "rolling_std_12", "rolling_mad_12",
    "robust_z_12", "distance_mean_6", "distance_mean_12",
    "stale_run", "observed", "time_since_observation",
    "cross_section_median", "cross_section_dispersion",
    "cross_section_breadth", "cross_section_rank",
    "pca_factor_1", "pca_factor_2",
    "pca_loading_1", "pca_loading_2",
    "pca_explained_variance_1", "pca_explained_variance_2",
    "peer_corr_abs_topk_mean", "peer_corr_signed_top1",
    "peer_direction_consensus", "peer_available_count",
    "regime_breadth_3", "regime_dispersion_12", "regime_volatility_12",
]


@dataclass
class UptrendModel:
    model: Any
    feature_columns: list[str]


def fit_uptrend_model(
    train: pd.DataFrame,
    seed: int,
    logistic_c: float,
    max_iter: int,
    feature_families: tuple[str, ...] = (),
) -> UptrendModel:
    requested = [
        column
        for family in feature_families
        for column in FEATURE_FAMILY_COLUMNS[family]
    ]
    numeric = [
        column
        for column in [*FEATURE_COLUMNS, *requested]
        if column in train.columns
    ]
    preprocessor = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ),
        (
            "indicator",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ["indicator_id"],
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
                max_iter=max_iter,
                random_state=seed,
            ),
        ),
    ])
    if train["y_true"].nunique() < 2:
        raise ValueError("Uptrend model requires both target classes")
    feature_columns = [*numeric, "indicator_id"]
    pipeline.fit(train[feature_columns], train["y_true"].astype(int))
    return UptrendModel(pipeline, feature_columns)


def predict_uptrend_probability(
    model: UptrendModel,
    test: pd.DataFrame,
) -> np.ndarray:
    prepared = test[model.feature_columns].copy()
    prepared["indicator_id"] = prepared["indicator_id"].astype(str)
    return np.clip(model.model.predict_proba(prepared)[:, 1], 1e-6, 1.0 - 1e-6)
