"""Research-only bounded search for a confidence-weighted Up/Down mix.

Run: python -m research.weighted_directional_mix
Weights are selected on origins 150–179. Monthly total calls stay at 15–20.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from research.flexible_directional_selection import select_panel, summarize

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "artifacts/active/regime_adaptive_predictions.parquet"
DOWN = ROOT / "research/down_v3/artifacts/v3_predictions.parquet"
OUTPUT = ROOT / "research/weighted_directional_mix"
WEIGHTS = (0.00, 0.25, 0.50, 0.75, 1.00)
POLICIES = (
    {"down_floor": 0.50, "down_margin": 0.00, "extra_floor": 0.0},
    {"down_floor": 0.55, "down_margin": 0.05, "extra_floor": 0.0},
    {"down_floor": 0.60, "down_margin": 0.08, "extra_floor": 0.0},
)


def _logit(values: pd.Series) -> np.ndarray:
    p = np.clip(values.to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_arm_calibration(panel: pd.DataFrame, through_origin: int) -> dict:
    history = panel[panel["origin_position"].le(through_origin)]
    if history["origin_position"].nunique() < 24:
        raise ValueError("At least 24 prior origins are needed for calibration")
    sources = {
        "up": ("p_up_selection_score", history["y_true"]),
        "down": ("logistic", 1 - history["y_true"]),
        "up_prior": ("indicator_prior", history["y_true"]),
        "down_prior": ("p_down_indicator_prior", 1 - history["y_true"]),
    }
    calibrators = {}
    for name, (column, target) in sources.items():
        model = LogisticRegression(C=1e6, max_iter=1000)
        model.fit(_logit(history[column].fillna(0.5)), target.astype(int))
        calibrators[name] = model
    return calibrators


def score_arms(panel: pd.DataFrame, calibrators: dict) -> pd.DataFrame:
    result = panel.copy()
    for name, column in (
        ("up", "p_up_selection_score"),
        ("down", "logistic"),
        ("up_prior", "indicator_prior"),
        ("down_prior", "p_down_indicator_prior"),
    ):
        result[f"p_{name}_arm"] = calibrators[name].predict_proba(
            _logit(result[column].fillna(0.5))
        )[:, 1]
    return result


def blend_arms(
    panel: pd.DataFrame,
    up_weight: float,
    down_weight: float,
    source: str = "opposite",
) -> pd.DataFrame:
    result = panel.copy()
    if source == "opposite":
        up_other = 1 - result["p_down_arm"]
        down_other = 1 - result["p_up_arm"]
    elif source == "prior":
        up_other = result["p_up_prior_arm"]
        down_other = result["p_down_prior_arm"]
    else:
        raise ValueError(f"Unknown weight source: {source}")
    result["up_cal"] = (
        up_weight * result["p_up_arm"]
        + (1 - up_weight) * up_other
    )
    result["down_cal"] = (
        down_weight * result["p_down_arm"]
        + (1 - down_weight) * down_other
    )
    return result


def _baseline(panel: pd.DataFrame) -> pd.DataFrame:
    selected = panel[panel["accepted"].fillna(False)].copy()
    selected["direction"] = selected["predicted_direction"]
    selected["call_correct"] = np.where(
        selected["direction"].eq("Down"), 1 - selected["y_true"], selected["y_true"]
    ).astype(int)
    selected["selection_source"] = np.where(
        selected["direction"].eq("Down"), "down_retained", "up"
    )
    return selected


def accuracy_by_score_quintile(rows: pd.DataFrame) -> list[dict]:
    ranked = rows.copy()
    ranked["score_quintile"] = pd.qcut(
        ranked["call_score"], 5, labels=False, duplicates="drop"
    )
    return [
        {
            "quintile_low_to_high": int(quintile) + 1,
            "calls": int(len(group)),
            "accuracy": float(group["call_correct"].mean()),
            "mean_score": float(group["call_score"].mean()),
        }
        for quintile, group in ranked.groupby("score_quintile", sort=True)
    ]


def main() -> None:
    active = pd.read_parquet(ACTIVE)
    raw_down = pd.read_parquet(DOWN)[["origin_position", "indicator_id", "logistic"]]
    panel = active.merge(raw_down, on=["origin_position", "indicator_id"], validate="one_to_one")
    panel = panel[
        panel["origin_position"].between(120, 266)
        & panel["level_c_ready"].fillna(False)
        & panel["y_true"].notna()
        & panel["p_up_selection_score"].notna()
        & panel["logistic"].notna()
    ].copy()
    if panel.empty or panel["origin_position"].max() > 266:
        raise ValueError("The trial must stay inside non-locked labeled origins")
    baseline = _baseline(panel)

    # At origin 150, labels through 148 are available. Keep this calibration
    # frozen after selecting weights, so score-scale drift from a refit cannot
    # silently change the Down-call count.
    calibrators = fit_arm_calibration(panel, through_origin=148)
    scored = score_arms(panel[panel["origin_position"].ge(150)], calibrators)
    screen = scored[scored["origin_position"].le(179)]
    best = None
    grid_rows = []
    for source in ("opposite", "prior"):
        for up_weight in WEIGHTS:
            for down_weight in WEIGHTS:
                blended = blend_arms(screen, up_weight, down_weight, source)
                for policy in POLICIES:
                    selected = select_panel(blended, policy, optional_extra=False)
                    down_count = int(selected["direction"].eq("Down").sum())
                    grid_rows.append({
                        "source": source,
                        "up_weight": up_weight,
                        "down_weight": down_weight,
                        "down_floor": policy["down_floor"],
                        "down_margin": policy["down_margin"],
                        "screen_hits": int(selected["call_correct"].sum()),
                        "screen_calls": int(len(selected)),
                        "screen_down_calls": down_count,
                    })
                    key = (int(selected["call_correct"].sum()), -down_count, up_weight, down_weight)
                    if best is None or key > best[0]:
                        best = (key, source, up_weight, down_weight, policy, selected)

    _, source, up_weight, down_weight, policy, screen_selected = best
    evaluated = blend_arms(
        scored[scored["origin_position"].ge(180)], up_weight, down_weight, source
    )
    selected = select_panel(evaluated, policy, optional_extra=False)
    assert selected.groupby("origin_position").size().between(15, 20).all()
    assert selected.groupby("origin_position")["direction"].apply(
        lambda directions: directions.eq("Down").sum()
    ).le(5).all()
    assert not selected.duplicated(["origin_position", "indicator_id"]).any()

    candidate_full = summarize(selected)
    baseline_full = summarize(baseline)
    windows = ("validation", "confirmation")
    candidate = {window: candidate_full[window] for window in windows}
    reference = {window: baseline_full[window] for window in windows}
    validation = selected[selected["origin_position"].between(180, 219)]
    report = {
        "calibration_labels_through_origin": 148,
        "weights_selected_on": "origins_150_179",
        "weight_source": source,
        "up_weight": up_weight,
        "down_weight": down_weight,
        "policy": policy,
        "screen_calls": int(len(screen_selected)),
        "screen_hits": int(screen_selected["call_correct"].sum()),
        "screen_down_calls": int(screen_selected["direction"].eq("Down").sum()),
        "screen_selected_score_auc": float(
            roc_auc_score(screen_selected["call_correct"], screen_selected["call_score"])
        ),
        "screen_score_quintiles": accuracy_by_score_quintile(screen_selected),
        "baseline": reference,
        "candidate": candidate,
        "validation_selected_score_auc": float(
            roc_auc_score(validation["call_correct"], validation["call_score"])
        ),
        "validation_selected_score_brier": float(
            brier_score_loss(validation["call_correct"], validation["call_score"])
        ),
        "validation_score_quintiles": accuracy_by_score_quintile(validation),
        "validation_hits_delta": candidate["validation"]["hits"] - reference["validation"]["hits"],
        "confirmation_hits_delta": candidate["confirmation"]["hits"] - reference["confirmation"]["hits"],
        "provenance_warning": (
            "The V3 feature family and active Up overlay were developed using "
            "previously viewed windows; this is exploratory evidence only."
        ),
        "production_changed": False,
    }
    OUTPUT.mkdir(exist_ok=True)
    selected.to_parquet(OUTPUT / "predictions.parquet", index=False)
    pd.DataFrame(grid_rows).to_csv(OUTPUT / "weight_screen.csv", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
