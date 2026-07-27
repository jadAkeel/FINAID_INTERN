from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - optional runtime dependency
    CatBoostClassifier = None  # type: ignore[assignment,misc]


FEATURE_COLUMNS = [
    "level", "diff_1", "pct_change_1", "direction_1", "direction_lag_1", "direction_lag_2", "direction_lag_3", "direction_lag_6", "direction_lag_12",
    "change_lag_1", "change_lag_2", "change_lag_3", "change_lag_6", "change_lag_12", "momentum_3", "momentum_6", "momentum_12",
    "rolling_mean_12", "rolling_std_12", "rolling_mad_12", "robust_z_12", "distance_mean_6", "distance_mean_12", "stale_run", "observed", "time_since_observation",
    "cross_section_median", "cross_section_dispersion", "cross_section_breadth", "cross_section_rank",
]


@dataclass
class FittedModel:
    model_id: str
    model: Any
    feature_columns: list[str]


def _clip_probability(value: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(np.asarray(value, dtype=float), 1e-6, 1 - 1e-6)


def baseline_probability(name: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    if name == "majority":
        return np.full(len(test), float(train["y_true"].mean()) if len(train) else 0.5)
    if name == "persistence":
        return _clip_probability(0.2 + 0.6 * test["direction_1"].fillna(0.5).to_numpy())
    if name == "reversal":
        return _clip_probability(0.8 - 0.6 * test["direction_1"].fillna(0.5).to_numpy())
    if name in {"momentum_3", "momentum_6", "momentum_12"}:
        signal = test[name].fillna(0).to_numpy()
        return _clip_probability(expit(2.0 * signal))
    if name == "mean_reversion":
        signal = -test["robust_z_12"].fillna(0).to_numpy()
        return _clip_probability(expit(0.8 * signal))
    if name in {"ar1", "ar2"}:
        lag = 1 if name == "ar1" else 2
        signal = test[f"change_lag_{lag}"].fillna(0).to_numpy()
        train_signal = train[f"change_lag_{lag}"].dropna()
        scale = float(train_signal.std()) if len(train_signal) and train_signal.std() > 0 else 1.0
        return _clip_probability(expit(0.75 * signal / scale))
    raise ValueError(f"Unknown baseline: {name}")


def fit_global_logistic(train: pd.DataFrame, seed: int = 7) -> FittedModel:
    numeric = [c for c in FEATURE_COLUMNS if c in train.columns]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), numeric),
        ("indicator", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["indicator_id"]),
    ], remainder="drop")
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(C=0.25, penalty="l2", solver="lbfgs", max_iter=500, random_state=seed)),
    ])
    if train["y_true"].nunique() < 2:
        raise ValueError("Global logistic requires both target classes in its training window")
    pipeline.fit(train[numeric + ["indicator_id"]], train["y_true"].astype(int))
    return FittedModel("global_logistic", pipeline, numeric + ["indicator_id"])


def fit_catboost(train: pd.DataFrame, seed: int = 7) -> FittedModel:
    if CatBoostClassifier is None:
        raise RuntimeError("catboost is not installed")
    numeric = [c for c in FEATURE_COLUMNS if c in train.columns]
    features = numeric + ["indicator_id"]
    model = CatBoostClassifier(iterations=120, depth=4, learning_rate=0.04, l2_leaf_reg=8.0, loss_function="Logloss", verbose=False, random_seed=seed, allow_writing_files=False, thread_count=2)
    prepared = train[features].copy()
    prepared["indicator_id"] = prepared["indicator_id"].astype(str)
    model.fit(prepared, train["y_true"].astype(int), cat_features=[len(features) - 1])
    return FittedModel("catboost_global", model, features)


def predict_fitted(fitted: FittedModel, test: pd.DataFrame) -> np.ndarray:
    prepared = test[fitted.feature_columns].copy()
    if "indicator_id" in prepared:
        prepared["indicator_id"] = prepared["indicator_id"].astype(str)
    return _clip_probability(fitted.model.predict_proba(prepared)[:, 1])

