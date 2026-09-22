"""Research-only correctness-trained Up/Down mix with a 15–20 total-call cap.

Run from the repository root: python -m research.trained_directional_mix
The training and policy screen use only origins available before Validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research.flexible_directional_selection import select_panel, summarize

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "artifacts/active/regime_adaptive_predictions.parquet"
DOWN = ROOT / "research/down_v3/artifacts/v3_predictions.parquet"
OUTPUT = ROOT / "research/trained_directional_mix"


def features(panel: pd.DataFrame, direction_down: int) -> np.ndarray:
    up = panel["p_up_selection_score"].to_numpy(dtype=float)
    down = panel["logistic"].to_numpy(dtype=float)
    stress = panel["regime_stress"].to_numpy(dtype=float)
    flag = np.full(len(panel), direction_down, dtype=float)
    return np.column_stack((up, down, stress, flag, flag * up, flag * down, flag * stress))


def fit_correctness(panel: pd.DataFrame, *, through_origin: int, regularization: float):
    training = panel[panel["origin_position"].le(through_origin)]
    if training["origin_position"].nunique() < 24:
        raise ValueError("At least 24 prior months are required")
    x = np.vstack((features(training, 0), features(training, 1)))
    y_up = training["y_true"].to_numpy(dtype=int)
    y = np.concatenate((y_up, 1 - y_up))
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=regularization, max_iter=1000)
    )
    model.fit(x, y)
    return model


def score(panel: pd.DataFrame, model) -> pd.DataFrame:
    scored = panel.copy()
    scored["up_cal"] = model.predict_proba(features(scored, 0))[:, 1]
    scored["down_cal"] = model.predict_proba(features(scored, 1))[:, 1]
    return scored


def baseline_rows(panel: pd.DataFrame) -> pd.DataFrame:
    baseline = panel[panel["accepted"].fillna(False)].copy()
    baseline["direction"] = baseline["predicted_direction"]
    baseline["call_correct"] = np.where(
        baseline["direction"].eq("Down"), 1 - baseline["y_true"], baseline["y_true"]
    ).astype(int)
    baseline["selection_source"] = np.where(
        baseline["direction"].eq("Down"), "down_retained", "up"
    )
    return baseline


def main() -> None:
    active = pd.read_parquet(ACTIVE)
    down = pd.read_parquet(DOWN)[["origin_position", "indicator_id", "logistic"]]
    panel = active.merge(down, on=["origin_position", "indicator_id"], validate="one_to_one")
    panel = panel[
        panel["origin_position"].between(120, 266)
        & panel["level_c_ready"].fillna(False)
        & panel["y_true"].notna()
        & panel["p_up_selection_score"].notna()
        & panel["logistic"].notna()
        & panel["regime_stress"].notna()
    ].copy()
    if panel.empty or panel["origin_position"].max() > 266:
        raise ValueError("Only non-locked labeled origins 120–266 may be used")
    baseline = baseline_rows(panel)

    # At origin 150, labels through 148 are available. Freeze this first model
    # while screening policy choices on 150–179, then freeze the final model
    # through 178 for Validation starting at origin 180.
    tune = panel[panel["origin_position"].between(150, 179)]
    best = None
    for regularization in (0.1, 1.0, 10.0):
        screen_model = fit_correctness(panel, through_origin=148, regularization=regularization)
        scored_tune = score(tune, screen_model)
        for down_floor in (0.50, 0.55, 0.60):
            for down_margin in (0.00, 0.03, 0.05, 0.08):
                policy = {
                    "down_floor": down_floor,
                    "down_margin": down_margin,
                    "extra_floor": 0.0,
                }
                selected = select_panel(scored_tune, policy, optional_extra=False)
                down_count = int(selected["direction"].eq("Down").sum())
                # The monthly cap is fixed for both arms. Prefer fewer Down
                # calls on ties because they require an additional model.
                key = (int(selected["call_correct"].sum()), -down_count, down_margin, down_floor)
                if best is None or key > best[0]:
                    best = (key, regularization, policy, selected)

    _, regularization, policy, screened = best
    final_model = fit_correctness(panel, through_origin=178, regularization=regularization)
    scored = score(panel[panel["origin_position"].ge(180)], final_model)
    selected = select_panel(scored, policy, optional_extra=False)
    assert selected.groupby("origin_position").size().between(15, 20).all()
    assert selected.groupby("origin_position")["direction"].apply(
        lambda directions: directions.eq("Down").sum()
    ).le(5).all()
    assert not selected.duplicated(["origin_position", "indicator_id"]).any()

    validation = scored[scored["origin_position"].between(180, 219)]
    y_up = validation["y_true"].to_numpy(dtype=int)
    probabilities = np.concatenate(
        (validation["up_cal"].to_numpy(), validation["down_cal"].to_numpy())
    )
    correctness = np.concatenate((y_up, 1 - y_up))
    candidate_all = summarize(selected)
    reference_all = summarize(baseline)
    candidate = {window: candidate_all[window] for window in ("validation", "confirmation")}
    reference = {window: reference_all[window] for window in ("validation", "confirmation")}
    report = {
        "training_labels_through_origin": 178,
        "screen_training_labels_through_origin": 148,
        "policy_selected_on": "origins_150_179",
        "provenance_warning": (
            "Raw V3 Down scores are walk-forward by origin, but the V3 feature "
            "family was chosen in earlier research using Tuning 120-179. "
            "The policy screen is exploratory, not an untouched holdout."
        ),
        "screen": {
            "candidate_hits": int(screened["call_correct"].sum()),
            "candidate_calls": int(len(screened)),
            "candidate_down_calls": int(screened["direction"].eq("Down").sum()),
            "baseline_hits": int(baseline.loc[
                baseline["origin_position"].between(150, 179), "call_correct"
            ].sum()),
        },
        "regularization": regularization,
        "policy": policy,
        "baseline": reference,
        "candidate": candidate,
        "validation_correctness_auc_all_options": float(roc_auc_score(correctness, probabilities)),
        "validation_correctness_brier_all_options": float(brier_score_loss(correctness, probabilities)),
        "validation_hits_delta": candidate["validation"]["hits"] - reference["validation"]["hits"],
        "confirmation_hits_delta": candidate["confirmation"]["hits"] - reference["confirmation"]["hits"],
        "production_changed": False,
    }
    OUTPUT.mkdir(exist_ok=True)
    selected.to_parquet(OUTPUT / "predictions.parquet", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
