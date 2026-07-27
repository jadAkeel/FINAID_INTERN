from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, log_loss, matthews_corrcoef, roc_auc_score


def classification_metrics(predictions: pd.DataFrame) -> dict[str, float | None]:
    p = predictions.dropna(subset=["y_true", "p_up"])
    if p.empty:
        return {k: None for k in ["n", "accuracy", "balanced_accuracy", "f1", "mcc", "brier", "log_loss", "roc_auc"]}
    y = p["y_true"].astype(int).to_numpy()
    prob = p["p_up"].clip(1e-6, 1 - 1e-6).to_numpy()
    pred = (prob >= 0.5).astype(int)
    result: dict[str, float | None] = {"n": float(len(p)), "accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "f1": float(f1_score(y, pred, zero_division=0)), "mcc": float(matthews_corrcoef(y, pred)), "brier": float(brier_score_loss(y, prob)), "log_loss": float(log_loss(y, prob, labels=[0, 1]))}
    result["roc_auc"] = float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else None
    return result


def selective_metrics(predictions: pd.DataFrame, floor: float = 0.55, cap: int = 20) -> dict[str, float | int | None]:
    p = predictions.dropna(subset=["y_true", "p_up"]).copy()
    if p.empty:
        return {"accepted": 0, "coverage": 0.0, "accepted_accuracy": None, "months": 0}
    p["confidence"] = (p["p_up"] - 0.5).abs() * 2
    p["accept"] = p["confidence"] >= max(0.0, 2 * floor - 1)
    selected = p[p["accept"]].sort_values(["origin_position", "confidence"], ascending=[True, False]).groupby("origin_position", sort=False).head(cap)
    return {"accepted": int(len(selected)), "coverage": float(len(selected) / len(p)), "accepted_accuracy": float((selected["y_true"].astype(int) == (selected["p_up"] >= 0.5).astype(int)).mean()) if len(selected) else None, "months": int(p["origin_position"].nunique())}


def monthly_block_bootstrap(predictions: pd.DataFrame, metric: str = "accuracy", block: int = 6, reps: int = 500, seed: int = 7) -> dict[str, float | int]:
    p = predictions.dropna(subset=["y_true", "p_up"]).copy()
    months = sorted(p["origin_position"].unique())
    if not months:
        return {"replicates": reps, "block": block, "lower_90": float("nan"), "upper_90": float("nan")}
    monthly = []
    for origin, group in p.groupby("origin_position"):
        monthly.append(float(((group["p_up"] >= 0.5).astype(int) == group["y_true"].astype(int)).mean()))
    rng = np.random.default_rng(seed)
    values = []
    n = len(monthly)
    for _ in range(reps):
        starts = rng.integers(0, n, size=max(1, int(np.ceil(n / block))))
        sample = np.concatenate([np.take(monthly, np.arange(start, start + block) % n) for start in starts])[:n]
        values.append(float(np.mean(sample)))
    return {"replicates": reps, "block": block, "lower_90": float(np.quantile(values, 0.05)), "upper_90": float(np.quantile(values, 0.95)), "lower_95": float(np.quantile(values, 0.025)), "upper_95": float(np.quantile(values, 0.975))}

