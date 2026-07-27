from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class _ProbModel:
    model: object | None
    constant: float | None = None

    def predict(self, values: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.full(len(values), float(self.constant if self.constant is not None else 0.5))
        return np.clip(self.model.predict_proba(values)[:, 1], 1e-6, 1 - 1e-6)


def _fit_binary_model(features: np.ndarray, target: np.ndarray, seed: int) -> _ProbModel:
    values = np.asarray(target, dtype=int)
    if len(values) == 0:
        return _ProbModel(None, 0.5)
    if len(np.unique(values)) < 2:
        return _ProbModel(None, float(values.mean()))
    model = Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(C=0.5, max_iter=300, random_state=seed)),
    ])
    model.fit(features, values)
    return _ProbModel(model)


def _logit(probability: pd.Series | np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(values / (1.0 - values))


def fit_platt(history: pd.DataFrame, seed: int = 20260727) -> _ProbModel:
    raw = _logit(history["p_up"].to_numpy())[:, None]
    return _fit_binary_model(raw, history["y_true"].astype(int).to_numpy(), seed)


def apply_platt(model: _ProbModel, frame: pd.DataFrame) -> np.ndarray:
    return model.predict(_logit(frame["p_up"].to_numpy())[:, None])


def _correctness_features(frame: pd.DataFrame) -> np.ndarray:
    raw = np.clip(frame["p_up"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    calibrated = np.clip(frame["p_up_calibrated"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    return np.column_stack([
        calibrated,
        np.abs(raw - 0.5) * 2.0,
        np.abs(calibrated - 0.5) * 2.0,
    ])


def _block_residual_quantile(history: pd.DataFrame, block_months: int, replicates: int, seed: int) -> float:
    if history.empty:
        return 0.0
    residual = history.assign(_residual=history["correct"].astype(float) - history["correctness_probability"])
    monthly = residual.groupby("origin_position", sort=True)["_residual"].mean().to_numpy()
    if len(monthly) == 0:
        return 0.0
    if len(monthly) < 2:
        return float(monthly[0])
    rng = np.random.default_rng(seed)
    block = max(1, min(int(block_months), len(monthly)))
    samples = []
    for _ in range(max(50, int(replicates))):
        starts = rng.integers(0, len(monthly), size=max(1, int(np.ceil(len(monthly) / block))))
        sample = np.concatenate([np.take(monthly, np.arange(start, start + block) % len(monthly)) for start in starts])[:len(monthly)]
        samples.append(float(sample.mean()))
    return float(np.quantile(samples, 0.10))


def score_level_c(
    history: pd.DataFrame,
    current: pd.DataFrame,
    floor: float = 0.55,
    cap: int = 20,
    min_history_months: int = 12,
    block_months: int = 6,
    bootstrap_replicates: int = 200,
    seed: int = 20260727,
) -> pd.DataFrame:
    """Score current Level-B rows using only earlier Level-B rows.

    Calibration, correctness fitting, and the block-bootstrap bias correction are
    refit independently for each origin from strictly earlier origins.
    """
    result = current.copy()
    for column, default in {
        "p_up_calibrated": np.nan, "directional_confidence": np.nan,
        "correctness_probability": np.nan, "correctness_lcb": np.nan,
        "level_c_ready": False, "accepted": False, "selection_rank": np.nan,
        "rejection_reason": "insufficient_earlier_level_b_history",
        "calibration_fit_through_origin": np.nan,
        "reliability_fit_through_origin": np.nan,
    }.items():
        result[column] = default
    if result.empty:
        return result
    for origin, positions in result.groupby("origin_position", sort=True).groups.items():
        current_rows = result.loc[positions].copy()
        history_rows = history[(history["origin_position"] < origin) & history["p_up"].notna() & history["y_true"].notna()].copy()
        if history_rows["origin_position"].nunique() < min_history_months:
            continue
        calibrator = fit_platt(history_rows, seed)
        history_rows["p_up_calibrated"] = apply_platt(calibrator, history_rows)
        current_rows["p_up_calibrated"] = apply_platt(calibrator, current_rows)
        history_rows["directional_confidence"] = np.maximum(history_rows["p_up"], 1.0 - history_rows["p_up"])
        current_rows["directional_confidence"] = np.maximum(current_rows["p_up"], 1.0 - current_rows["p_up"])
        history_rows["correct"] = (history_rows["y_true"].astype(int) == (history_rows["p_up"] >= 0.5).astype(int)).astype(int)
        correctness_model = _fit_binary_model(_correctness_features(history_rows), history_rows["correct"].to_numpy(), seed)
        history_rows["correctness_probability"] = correctness_model.predict(_correctness_features(history_rows))
        current_rows["correctness_probability"] = correctness_model.predict(_correctness_features(current_rows))
        bias_lcb = _block_residual_quantile(history_rows, block_months, bootstrap_replicates, seed + int(origin))
        current_rows["correctness_lcb"] = np.clip(current_rows["correctness_probability"] + bias_lcb, 0.0, 1.0)
        current_rows["level_c_ready"] = True
        current_rows["calibration_fit_through_origin"] = origin - 1
        current_rows["reliability_fit_through_origin"] = origin - 1
        current_rows["rejection_reason"] = np.where(current_rows["correctness_lcb"] >= floor, "", "below_reliability_floor")
        passing = current_rows[current_rows["correctness_lcb"] >= floor].sort_values("correctness_lcb", ascending=False)
        accepted_index = passing.head(cap).index
        current_rows.loc[accepted_index, "accepted"] = True
        current_rows.loc[passing.index.difference(accepted_index), "rejection_reason"] = "monthly_cap_20"
        current_rows.loc[accepted_index, "selection_rank"] = np.arange(1, len(accepted_index) + 1)
        result.loc[current_rows.index, current_rows.columns] = current_rows
    return result


def evaluate_level_c(predictions: pd.DataFrame, floor: float, block_months: int = 6, bootstrap_replicates: int = 500, seed: int = 20260727) -> dict[str, float | int | None]:
    ready = predictions[predictions["level_c_ready"] & predictions["y_true"].notna()].copy()
    if ready.empty:
        return {"ready_rows": 0, "ready_months": 0, "full_accuracy": None, "calibrated_brier": None, "correctness_brier": None, "accepted": 0, "coverage": 0.0, "accepted_accuracy": None, "lcb_p10": None, "lcb_p90": None}
    direction = (ready["p_up_calibrated"] >= 0.5).astype(int)
    full_accuracy = float((direction == ready["y_true"].astype(int)).mean())
    correct = (ready["y_true"].astype(int) == (ready["p_up"] >= 0.5).astype(int)).astype(int)
    accepted = ready[ready["accepted"]]
    monthly_values = accepted.groupby("origin_position").apply(lambda g: float((g["y_true"].astype(int) == (g["p_up"] >= 0.5).astype(int)).mean()), include_groups=False).to_numpy()
    if len(monthly_values):
        rng = np.random.default_rng(seed)
        samples = []
        block = max(1, min(block_months, len(monthly_values)))
        for _ in range(max(50, bootstrap_replicates)):
            starts = rng.integers(0, len(monthly_values), size=max(1, int(np.ceil(len(monthly_values) / block))))
            sample = np.concatenate([np.take(monthly_values, np.arange(s, s + block) % len(monthly_values)) for s in starts])[:len(monthly_values)]
            samples.append(float(sample.mean()))
        lcb_p10, lcb_p90 = float(np.quantile(samples, 0.10)), float(np.quantile(samples, 0.90))
    else:
        lcb_p10 = lcb_p90 = None
    return {"ready_rows": int(len(ready)), "ready_months": int(ready["origin_position"].nunique()), "full_accuracy": full_accuracy, "calibrated_brier": float(np.mean((ready["y_true"] - ready["p_up_calibrated"]) ** 2)), "correctness_brier": float(np.mean((correct - ready["correctness_probability"]) ** 2)), "accepted": int(len(accepted)), "coverage": float(len(accepted) / len(ready)), "accepted_accuracy": float((accepted["y_true"].astype(int) == (accepted["p_up"] >= 0.5).astype(int)).mean()) if len(accepted) else None, "lcb_p10": lcb_p10, "lcb_p90": lcb_p90}

