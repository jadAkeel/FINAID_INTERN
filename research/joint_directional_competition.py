"""Fixed-rule joint Up/Down research check on non-locked OOF predictions.

Run from the repository root: python -m research.joint_directional_competition
No policy threshold or weight is selected using Validation or Confirmation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "artifacts/active/regime_adaptive_predictions.parquet"
DOWN = ROOT / "research/down_v3/artifacts/v3_predictions.parquet"
OUTPUT = ROOT / "research/joint_directional_competition"
WINDOWS = {"validation": (180, 219), "confirmation": (220, 266)}


def _logit(values: pd.Series) -> np.ndarray:
    clipped = np.clip(values.to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def fit_calibrators(panel: pd.DataFrame, through_origin: int = 148) -> dict:
    history = panel.loc[panel["origin_position"].le(through_origin)]
    if history["origin_position"].nunique() < 24:
        raise ValueError("At least 24 earlier months are needed for calibration")
    models = {}
    for name, score, target in (
        ("up", "p_up_selection_score", history["y_true"]),
        ("down", "logistic", 1 - history["y_true"]),
    ):
        model = LogisticRegression(C=1.0, max_iter=1000)
        model.fit(_logit(history[score]), target.astype(int))
        models[name] = model
    return models


def score_directions(panel: pd.DataFrame, calibrators: dict) -> pd.DataFrame:
    """Return complementary directional scores on one common scale."""
    scored = panel.copy()
    up = calibrators["up"].predict_proba(_logit(scored["p_up_selection_score"]))[:, 1]
    down = calibrators["down"].predict_proba(_logit(scored["logistic"]))[:, 1]
    scored["joint_p_up"] = 0.5 * (up + 1 - down)
    scored["joint_p_down"] = 1 - scored["joint_p_up"]
    return scored


def select_month(month: pd.DataFrame) -> pd.DataFrame:
    """Let every eligible indicator's stronger direction compete for all slots."""
    ready = month.loc[
        month["level_c_ready"].fillna(False).astype(bool)
        & month["joint_p_up"].notna()
        & month["joint_p_down"].notna()
    ].copy()
    if ready.empty or ready["indicator_id"].duplicated().any():
        raise ValueError("A month needs unique eligible indicators")
    caps = month["regime_cap"].dropna().unique()
    if len(caps) != 1 or not 15 <= int(caps[0]) <= 20:
        raise ValueError("A month must have one cap between 15 and 20")
    cap = int(caps[0])
    if len(ready) < cap:
        raise ValueError("Fewer eligible indicators than the monthly cap")
    ready["direction"] = np.where(
        ready["joint_p_down"].gt(ready["joint_p_up"]), "Down", "Up"
    )
    ready["call_score"] = np.maximum(ready["joint_p_up"], ready["joint_p_down"])
    selected = ready.sort_values(
        ["call_score", "indicator_id"], ascending=[False, True]
    ).head(cap).copy()
    selected["selection_rank"] = np.arange(1, cap + 1)
    if "y_true" in selected:
        selected["call_correct"] = np.where(
            selected["direction"].eq("Up"),
            selected["y_true"],
            1 - selected["y_true"],
        ).astype(int)
    return selected


def select_panel(panel: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [select_month(month) for _, month in panel.groupby("origin_position", sort=True)],
        ignore_index=True,
    )


def _summary(rows: pd.DataFrame) -> dict:
    down = rows.loc[rows["direction"].eq("Down")]
    return {
        "calls": int(len(rows)),
        "hits": int(rows["call_correct"].sum()),
        "accuracy": float(rows["call_correct"].mean()),
        "down_calls": int(len(down)),
        "down_hits": int(down["call_correct"].sum()),
    }


def _confidence_audit(rows: pd.DataFrame) -> dict:
    bins = rows.copy()
    bins["score_quintile"] = pd.qcut(
        bins["call_score"], 5, labels=False, duplicates="drop"
    )
    quintiles = [
        {
            "quintile_low_to_high": int(key) + 1,
            "calls": int(len(group)),
            "mean_score": float(group["call_score"].mean()),
            "observed_accuracy": float(group["call_correct"].mean()),
        }
        for key, group in bins.groupby("score_quintile", sort=True)
    ]
    return {
        "correctness_auc": float(roc_auc_score(rows["call_correct"], rows["call_score"])),
        "correctness_brier": float(brier_score_loss(rows["call_correct"], rows["call_score"])),
        "score_quintiles": quintiles,
        "higher_score_more_accurate_by_quintile": all(
            left["observed_accuracy"] <= right["observed_accuracy"]
            for left, right in zip(quintiles, quintiles[1:])
        ),
    }


def main() -> None:
    active = pd.read_parquet(ACTIVE)
    down = pd.read_parquet(DOWN)[["origin_position", "indicator_id", "logistic"]]
    panel = active.merge(down, on=["origin_position", "indicator_id"], validate="one_to_one")
    panel = panel.loc[
        panel["origin_position"].between(120, 266)
        & panel["level_c_ready"].fillna(False).astype(bool)
        & panel["y_true"].notna()
        & panel["p_up_selection_score"].notna()
        & panel["logistic"].notna()
    ].copy()
    if panel.empty or panel["origin_position"].max() > 266:
        raise ValueError("Only non-locked labeled origins 120–266 may be used")
    calibrators = fit_calibrators(panel)
    evaluated = score_directions(
        panel.loc[panel["origin_position"].ge(180)], calibrators
    )
    selected = select_panel(evaluated)
    baseline = evaluated.loc[evaluated["accepted"].fillna(False).astype(bool)].copy()
    baseline["direction"] = baseline["predicted_direction"]
    baseline["call_correct"] = np.where(
        baseline["direction"].eq("Up"), baseline["y_true"], 1 - baseline["y_true"]
    ).astype(int)
    if not selected.groupby("origin_position").size().eq(
        baseline.groupby("origin_position").size()
    ).all():
        raise AssertionError("Candidate and baseline must have equal monthly coverage")
    report = {
        "method": "fixed_equal_weight_complementary_directional_calibration",
        "calibration_labels_through_origin": 148,
        "selection_rule": "one direction per indicator, top original monthly cap, no Down quota",
        "windows": {},
        "evidence_limit": (
            "Validation and Confirmation have been viewed in prior research; "
            "upstream scores were also developed on earlier viewed windows. "
            "This is an exploratory replay, not a fresh blind test."
        ),
        "production_changed": False,
    }
    for name, (first, last) in WINDOWS.items():
        candidate_rows = selected.loc[selected["origin_position"].between(first, last)]
        baseline_rows = baseline.loc[baseline["origin_position"].between(first, last)]
        candidate = _summary(candidate_rows)
        reference = _summary(baseline_rows)
        report["windows"][name] = {
            "candidate": candidate,
            "baseline": reference,
            "hits_delta": candidate["hits"] - reference["hits"],
            "confidence": _confidence_audit(candidate_rows),
        }
    validation = report["windows"]["validation"]
    confirmation = report["windows"]["confirmation"]
    report["nonlocked_gate_passed"] = (
        validation["hits_delta"] > 0
        and confirmation["hits_delta"] >= 0
        and validation["confidence"]["correctness_auc"] > 0.5
        and validation["confidence"]["higher_score_more_accurate_by_quintile"]
    )
    OUTPUT.mkdir(exist_ok=True)
    selected.to_parquet(OUTPUT / "predictions.parquet", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
