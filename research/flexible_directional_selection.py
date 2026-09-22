"""Research-only joint Up/Down selection on the saved non-locked OOF panel.

Run from the repository root: python research/flexible_directional_selection.py
Tuning chooses policy thresholds; Validation is the decision gate. The active
model and its artifacts are never modified.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "artifacts/active/regime_adaptive_predictions.parquet"
DOWN = ROOT / "research/down_v3/artifacts/v3_predictions.parquet"
OUTPUT = ROOT / "research/flexible_directional_selection"
WINDOWS = {"tuning": (120, 179), "validation": (180, 219), "confirmation": (220, 266)}


def calibrate(tuning: pd.DataFrame, score: str, target: pd.Series, values: pd.Series) -> np.ndarray:
    """Put both arms on a Tuning-fitted correctness scale."""
    def logit(series: pd.Series) -> np.ndarray:
        p = np.clip(series.to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1.0 - p)).reshape(-1, 1)

    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(logit(tuning[score]), target.astype(int))
    return model.predict_proba(logit(values))[:, 1]


def select_month(
    month: pd.DataFrame,
    *,
    down_floor: float,
    down_margin: float,
    extra_floor: float,
    optional_extra: bool,
) -> pd.DataFrame:
    """Choose each indicator's stronger eligible direction, then 15 to cap calls."""
    ready = month.loc[month["level_c_ready"].fillna(False).astype(bool)].copy()
    ready = ready[ready["up_cal"].notna() & ready["down_cal"].notna()]
    cap = min(20, int(ready["regime_cap"].iloc[0]))
    if cap < 15 or len(ready) < 15:
        raise ValueError("At least 15 eligible indicators and a cap >= 15 are required")
    ready["direction"] = np.where(
        (ready["down_cal"] >= down_floor)
        & (ready["down_cal"] >= ready["up_cal"] + down_margin),
        "Down",
        "Up",
    )
    ready["call_score"] = np.where(
        ready["direction"].eq("Down"), ready["down_cal"], ready["up_cal"]
    )
    ready = ready.sort_values(
        ["call_score", "indicator_id"], ascending=[False, True]
    )
    chosen = []
    down_count = 0
    for _, row in ready.iterrows():
        if row["direction"] == "Down" and down_count >= 5:
            continue
        if len(chosen) >= 15 and optional_extra and row["call_score"] < extra_floor:
            break
        chosen.append(row)
        down_count += row["direction"] == "Down"
        if len(chosen) == cap:
            break
    if len(chosen) < 15:
        raise ValueError("The direction and Down-cap rules left fewer than 15 calls")
    result = pd.DataFrame(chosen).copy()
    result["call_correct"] = np.where(
        result["direction"].eq("Down"), 1 - result["y_true"], result["y_true"]
    ).astype(int)
    result["selection_rank"] = np.arange(1, len(result) + 1)
    result["selection_source"] = "up"
    down_rows = result["direction"].eq("Down")
    result.loc[down_rows, "selection_source"] = "down_external"
    result.loc[down_rows & result["accepted"].fillna(False), "selection_source"] = "down_overlay"
    if "predicted_direction" in result:
        retained = down_rows & result["accepted"].fillna(False) & result["predicted_direction"].eq("Down")
        result.loc[retained, "selection_source"] = "down_retained"
    return result


def select_panel(panel: pd.DataFrame, policy: dict, *, optional_extra: bool) -> pd.DataFrame:
    return pd.concat(
        [
            select_month(month, **policy, optional_extra=optional_extra)
            for _, month in panel.groupby("origin_position", sort=True)
        ],
        ignore_index=True,
    )


def summarize(rows: pd.DataFrame) -> dict:
    result = {}
    for name, (first, last) in WINDOWS.items():
        sample = rows[rows["origin_position"].between(first, last)]
        down = sample[sample["direction"].eq("Down")]
        result[name] = {
            "calls": int(len(sample)),
            "hits": int(sample["call_correct"].sum()),
            "accuracy": float(sample["call_correct"].mean()),
            "down_calls": int(len(down)),
            "down_hits": int(down["call_correct"].sum()),
            "down_overlay": int(down["selection_source"].eq("down_overlay").sum()),
            "down_retained": int(down["selection_source"].eq("down_retained").sum()),
            "down_external": int(down["selection_source"].eq("down_external").sum()),
            "calls_per_month": {
                int(k): int(v)
                for k, v in sample.groupby("origin_position").size().value_counts().sort_index().items()
            },
        }
    return result


def main() -> None:
    active = pd.read_parquet(ACTIVE)
    down = pd.read_parquet(DOWN)[["origin_position", "indicator_id", "logistic_platt"]]
    panel = active.merge(down, on=["origin_position", "indicator_id"], validate="one_to_one")
    panel = panel[panel["origin_position"].between(120, 266)].copy()
    assert panel["origin_position"].max() <= 266
    tune = panel[panel["origin_position"].between(120, 179)]
    panel["up_cal"] = calibrate(tune, "p_up_selection_score", tune["y_true"], panel["p_up_selection_score"])
    panel["down_cal"] = calibrate(tune, "logistic_platt", 1 - tune["y_true"], panel["logistic_platt"])
    baseline = panel[panel["accepted"].fillna(False)].copy()
    baseline["direction"] = baseline["predicted_direction"]
    baseline["call_correct"] = np.where(
        baseline["direction"].eq("Down"), 1 - baseline["y_true"], baseline["y_true"]
    ).astype(int)
    baseline["selection_source"] = np.where(baseline["direction"].eq("Down"), "down_retained", "up")

    # All policy choices are made on Tuning. Validation is never used to select.
    best = None
    tune_panel = panel[panel["origin_position"].between(120, 179)]
    for floor in (0.50, 0.55, 0.60):
        for margin in (0.00, 0.03, 0.05, 0.08):
            for extra in (0.55, 0.60, 0.65):
                policy = {"down_floor": floor, "down_margin": margin, "extra_floor": extra}
                selected = select_panel(tune_panel, policy, optional_extra=True)
                # Accuracy is the product objective; ties prefer more coverage,
                # then stricter Down evidence.
                key = (float(selected["call_correct"].mean()), len(selected), margin, floor, extra)
                if best is None or key > best[0]:
                    best = (key, policy)
    policy = best[1]
    flexible = select_panel(panel, policy, optional_extra=True)
    same_cap = select_panel(panel, policy, optional_extra=False)
    monthly_counts = flexible.groupby("origin_position").size().rename("matched_cap")
    matched_baseline = baseline.merge(
        monthly_counts, left_on="origin_position", right_index=True, validate="many_to_one"
    )
    matched_baseline = matched_baseline[
        matched_baseline["selection_rank"] <= matched_baseline["matched_cap"]
    ]
    assert flexible.groupby("origin_position").size().between(15, 20).all()
    assert flexible.groupby("origin_position")["direction"].apply(lambda s: s.eq("Down").sum()).le(5).all()
    assert not flexible.duplicated(["origin_position", "indicator_id"]).any()
    baseline_result = summarize(baseline)
    flexible_result = summarize(flexible)
    same_cap_result = summarize(same_cap)
    matched_result = summarize(matched_baseline)
    report = {
        "selected_on": "tuning_120_179",
        "calibration_note": "Tuning-fitted Platt scales; Tuning metrics are in-sample for policy selection.",
        "policy": policy,
        "baseline": baseline_result,
        "matched_coverage_baseline": matched_result,
        "flexible": flexible_result,
        "same_cap": same_cap_result,
        "validation_matched_hits_delta": (
            flexible_result["validation"]["hits"] - matched_result["validation"]["hits"]
        ),
        "confirmation_matched_hits_delta": (
            flexible_result["confirmation"]["hits"] - matched_result["confirmation"]["hits"]
        ),
        "non_locked_screen_passed": (
            flexible_result["validation"]["hits"] > matched_result["validation"]["hits"]
            and flexible_result["confirmation"]["hits"] >= matched_result["confirmation"]["hits"]
            and same_cap_result["validation"]["hits"] >= baseline_result["validation"]["hits"]
        ),
        "promotion_eligible": False,
        "production_changed": False,
    }
    OUTPUT.mkdir(exist_ok=True)
    flexible.to_parquet(OUTPUT / "predictions.parquet", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
