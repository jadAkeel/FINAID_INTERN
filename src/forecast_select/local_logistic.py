"""Local (per-indicator) and indicator-interaction logistic models.

This module is research-only. It never replaces the production Uptrend
Selector; it only produces alternative ``p_up`` signals that are fed through
the unchanged graph -> prior -> top-15 selector stages.

Temporal contract (identical to the production Uptrend Selector):

* Features at forecast origin ``t`` use observations through ``t - lag``.
* Training labels stop at ``t - lag - 1`` (``t - 2`` for ``lag = 1``).
* Every estimator, imputer and scaler is fitted on that causal training slice
  only, separately at every forecast origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .uptrend_model import FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Compact local feature sets
# ---------------------------------------------------------------------------
# A local model sees roughly one row per month for a single indicator, so it
# cannot support the 44 global predictors. Each candidate set below is a
# strict subset of the production feature panel (no new feature engineering),
# drawn from the families named in the research brief: short-term direction
# lags, change lags, momentum, volatility, robust z-score, distance from
# rolling means, cross-sectional breadth / rank.

LOCAL_FEATURE_SETS: dict[str, list[str]] = {
    "lean5": [
        "direction_lag_1",
        "momentum_3",
        "momentum_6",
        "distance_mean_12",
        "cross_section_rank",
    ],
    "core8": [
        "direction_lag_1",
        "change_lag_1",
        "momentum_3",
        "momentum_6",
        "rolling_std_12",
        "robust_z_12",
        "distance_mean_12",
        "cross_section_rank",
    ],
    "wide12": [
        "direction_lag_1",
        "direction_lag_2",
        "direction_lag_3",
        "change_lag_1",
        "momentum_3",
        "momentum_6",
        "momentum_12",
        "rolling_std_12",
        "robust_z_12",
        "distance_mean_12",
        "cross_section_rank",
        "cross_section_breadth",
    ],
}


def local_feature_columns(name: str) -> list[str]:
    if name not in LOCAL_FEATURE_SETS:
        raise ValueError(f"Unknown local feature set: {name}")
    return list(LOCAL_FEATURE_SETS[name])


# ---------------------------------------------------------------------------
# Local logistic
# ---------------------------------------------------------------------------


@dataclass
class LocalLogisticModel:
    """Per-indicator logistic models plus the history that produced them."""

    models: dict[str, Any] = field(default_factory=dict)
    training_rows: dict[str, int] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)


def _local_pipeline(
    logistic_c: float,
    max_iter: int,
    seed: int,
    solver: str,
) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=False)),
        ("scale", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                C=logistic_c,
                penalty="l2",
                solver=solver,
                max_iter=max_iter,
                random_state=seed,
            ),
        ),
    ])


def fit_local_logistic_models(
    train: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
    logistic_c: float,
    max_iter: int,
    minimum_local_rows: int,
    minimum_local_class_rows: int,
    solver: str = "liblinear",
) -> LocalLogisticModel:
    """Fit one logistic regression per indicator on same-indicator history only.

    ``train`` must already be restricted to causally available rows. An
    indicator is skipped (and later falls back to the global model) when it
    has fewer than ``minimum_local_rows`` labelled rows or fewer than
    ``minimum_local_class_rows`` rows in its minority class.
    """
    missing = [c for c in feature_columns if c not in train.columns]
    if missing:
        raise ValueError(f"Local feature columns missing from panel: {missing}")
    models: dict[str, Any] = {}
    rows: dict[str, int] = {}
    for indicator, group in train.groupby("indicator_id", sort=False):
        key = str(indicator)
        labels = group["y_true"].dropna()
        rows[key] = int(len(labels))
        counts = labels.astype(int).value_counts()
        if (
            len(labels) < minimum_local_rows
            or len(counts) < 2
            or int(counts.min()) < minimum_local_class_rows
        ):
            continue
        usable = group.loc[labels.index, feature_columns]
        # An all-missing column cannot be median-imputed, so skip such
        # indicators rather than silently injecting zeros.
        if bool(usable.notna().sum().eq(0).any()):
            continue
        model = _local_pipeline(logistic_c, max_iter, seed, solver)
        model.fit(usable, labels.astype(int))
        models[key] = model
    return LocalLogisticModel(
        models=models,
        training_rows=rows,
        feature_columns=list(feature_columns),
    )


def predict_local_probability(
    model: LocalLogisticModel,
    test: pd.DataFrame,
    fallback: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(p_local, local_rows, fallback_used)`` aligned with ``test``.

    Rows for indicators without a fitted local model receive the supplied
    ``fallback`` probability (the global model) and are flagged.
    """
    fallback = np.asarray(fallback, dtype=float)
    if len(fallback) != len(test):
        raise ValueError("fallback length must match the test frame")
    probability = fallback.copy()
    counts = np.zeros(len(test), dtype=int)
    used_fallback = np.ones(len(test), dtype=bool)
    positions = {key: index for index, key in enumerate(test.index)}
    for indicator, index in test.groupby("indicator_id", sort=False).groups.items():
        key = str(indicator)
        slots = np.array([positions[value] for value in index], dtype=int)
        counts[slots] = int(model.training_rows.get(key, 0))
        local = model.models.get(key)
        if local is None:
            continue
        probability[slots] = np.clip(
            local.predict_proba(test.loc[index, model.feature_columns])[:, 1],
            1e-6,
            1.0 - 1e-6,
        )
        used_fallback[slots] = False
    return probability, counts, used_fallback


def local_coefficients(model: LocalLogisticModel) -> pd.DataFrame:
    """Return the fitted local slopes, one row per indicator."""
    records = []
    for indicator, pipeline in sorted(model.models.items()):
        classifier = pipeline.named_steps["classifier"]
        record: dict[str, Any] = {
            "indicator_id": indicator,
            "training_rows": int(model.training_rows.get(indicator, 0)),
            "intercept": float(classifier.intercept_[0]),
        }
        for name, value in zip(model.feature_columns, classifier.coef_[0]):
            record[f"coef_{name}"] = float(value)
        records.append(record)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Shrinkage / partial pooling
# ---------------------------------------------------------------------------


def shrink_probabilities(
    p_global: np.ndarray,
    p_local: np.ndarray,
    local_weight: np.ndarray | float,
) -> np.ndarray:
    """``p = w * p_local + (1 - w) * p_global`` with ``w`` in ``[0, 1]``."""
    weight = np.asarray(local_weight, dtype=float)
    if bool(np.any(weight < 0.0)) or bool(np.any(weight > 1.0)):
        raise ValueError("local_weight must lie in [0, 1]")
    blended = weight * np.asarray(p_local, dtype=float) + (
        1.0 - weight
    ) * np.asarray(p_global, dtype=float)
    return np.clip(blended, 1e-6, 1.0 - 1e-6)


def sample_aware_weights(
    local_rows: np.ndarray,
    fallback_used: np.ndarray,
    maximum_weight: float,
    minimum_rows: int,
    reference_rows: int,
) -> np.ndarray:
    """Ramp the local weight linearly with available same-indicator history.

    ``w(n) = w_max * clip((n - n_min) / (n_ref - n_min), 0, 1)`` and ``w = 0``
    wherever the global fallback was used. The rule is fixed in advance; only
    ``w_max`` and ``n_ref`` are selected on tuning.
    """
    if reference_rows <= minimum_rows:
        raise ValueError("reference_rows must exceed minimum_rows")
    rows = np.asarray(local_rows, dtype=float)
    ramp = (rows - float(minimum_rows)) / float(reference_rows - minimum_rows)
    weight = float(maximum_weight) * np.clip(ramp, 0.0, 1.0)
    return np.where(np.asarray(fallback_used, dtype=bool), 0.0, weight)


# ---------------------------------------------------------------------------
# Indicator-interaction logistic
# ---------------------------------------------------------------------------


@dataclass
class InteractionLogisticModel:
    numeric_columns: list[str]
    interaction_columns: list[str]
    numeric_pipeline: Any
    encoder: Any
    classifier: Any
    coefficient_names: list[str]

    @property
    def n_coefficients(self) -> int:
        return int(len(self.coefficient_names))


def _interaction_design(
    model: InteractionLogisticModel,
    frame: pd.DataFrame,
    fit: bool,
) -> np.ndarray:
    numeric = frame[model.numeric_columns]
    indicator = frame[["indicator_id"]].astype(str)
    if fit:
        scaled = model.numeric_pipeline.fit_transform(numeric)
        encoded = model.encoder.fit_transform(indicator)
    else:
        scaled = model.numeric_pipeline.transform(numeric)
        encoded = model.encoder.transform(indicator)
    slots = [model.numeric_columns.index(name) for name in model.interaction_columns]
    selected = scaled[:, slots]
    # (n, n_indicators, n_interaction_features) -> flattened interaction block.
    interactions = (
        encoded[:, :, None] * selected[:, None, :]
    ).reshape(len(frame), -1)
    return np.hstack([scaled, encoded, interactions])


def fit_interaction_logistic_model(
    train: pd.DataFrame,
    seed: int,
    logistic_c: float,
    max_iter: int,
    interaction_features: list[str],
) -> InteractionLogisticModel:
    """One pooled logistic regression whose selected slopes vary by indicator.

    The base design matches the production global model (44 scaled numeric
    predictors plus missingness indicators, plus a one-hot indicator block).
    On top of that it adds ``indicator_id x feature`` products for a small,
    predefined feature list, so a handful of slopes are indicator-specific
    while everything else stays pooled.
    """
    numeric_columns = [c for c in FEATURE_COLUMNS if c in train.columns]
    missing = [c for c in interaction_features if c not in numeric_columns]
    if missing:
        raise ValueError(f"Interaction features missing from panel: {missing}")
    if train["y_true"].nunique() < 2:
        raise ValueError("Interaction model requires both target classes")
    model = InteractionLogisticModel(
        numeric_columns=numeric_columns,
        interaction_columns=list(interaction_features),
        numeric_pipeline=Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]),
        encoder=OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        classifier=LogisticRegression(
            C=logistic_c,
            penalty="l2",
            solver="lbfgs",
            max_iter=max_iter,
            random_state=seed,
        ),
        coefficient_names=[],
    )
    design = _interaction_design(model, train, fit=True)
    model.classifier.fit(design, train["y_true"].astype(int))
    indicators = [str(value) for value in model.encoder.categories_[0]]
    interaction_width = len(indicators) * len(model.interaction_columns)
    missing_width = (
        design.shape[1] - len(numeric_columns) - len(indicators) - interaction_width
    )
    names = list(numeric_columns)
    names += [f"missing_indicator_{index}" for index in range(missing_width)]
    names += [f"indicator[{value}]" for value in indicators]
    names += [
        f"indicator[{value}]x{feature}"
        for value in indicators
        for feature in model.interaction_columns
    ]
    model.coefficient_names = names
    return model


def predict_interaction_probability(
    model: InteractionLogisticModel,
    test: pd.DataFrame,
) -> np.ndarray:
    design = _interaction_design(model, test, fit=False)
    return np.clip(
        model.classifier.predict_proba(design)[:, 1],
        1e-6,
        1.0 - 1e-6,
    )


def interaction_coefficients(model: InteractionLogisticModel) -> pd.DataFrame:
    return pd.DataFrame({
        "term": model.coefficient_names,
        "coefficient": model.classifier.coef_[0].astype(float),
    })
