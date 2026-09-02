"""Reproducible, non-promoting Signal Ceiling Audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = ROOT / "research/regime_adaptive_selector/artifacts/predictions.parquet"
OUT = ROOT / "research/signal_ceiling_audit"
LOCKED_START = 268
MAX_NONLOCKED_ORIGIN = LOCKED_START - 2
BOOTSTRAP_SEED = 20260807
BOOTSTRAP_REPLICATES = 5000


def _load_nonlocked_predictions(path: Path = PREDICTIONS) -> pd.DataFrame:
    """Load only rows whose target cannot depend on the locked evaluation."""
    frame = pd.read_parquet(
        path,
        filters=[("origin_position", "<=", MAX_NONLOCKED_ORIGIN)],
    )
    if frame.empty:
        raise ValueError("No non-locked prediction rows are available")
    if int(frame["origin_position"].max()) > MAX_NONLOCKED_ORIGIN:
        raise ValueError("Prediction artifact contains locked-origin rows")
    if "locked_evaluation_read" in frame and frame["locked_evaluation_read"].fillna(False).any():
        raise ValueError("Prediction artifact is marked as using locked evaluation")
    return frame


def _metric(frame: pd.DataFrame, name: str, group: str = "all") -> dict:
    calls = int(len(frame))
    hits = int(frame["hit"].sum()) if calls else 0
    return {"model": name, "group": group, "hits": hits, "calls": calls,
            "accuracy": hits / calls if calls else None,
            "months": int(frame["origin_position"].nunique()) if calls else 0,
            "avg_monthly_calls": calls / frame["origin_position"].nunique() if calls and frame["origin_position"].nunique() else None}


def run() -> dict:
    df = _load_nonlocked_predictions()
    df["hit"] = (df["predicted_direction"].eq("Up") == df["y_true"].eq(1)).astype(int)
    selected = df.loc[df["accepted"].fillna(False)].copy()
    selected["rank"] = pd.to_numeric(selected["selection_rank"], errors="coerce")
    outm = OUT / "metrics"
    outm.mkdir(parents=True, exist_ok=True)

    rows = [_metric(selected, "current_dynamic_15_20")]
    allup = selected.assign(hit=selected["y_true"].eq(1).astype(int))
    rows.append(_metric(allup, "all_up_same_calls"))
    for lo, hi, name in [(1, 15, "top_15_core"), (16, 17, "ranks_16_17"), (18, 20, "ranks_18_20")]:
        rows.append(_metric(selected.loc[selected["rank"].between(lo, hi)], name))
    ledger = ROOT / "research/regime_adaptive_selector/metrics/experiment_ledger.csv"
    if ledger.exists():
        led = pd.read_csv(ledger)
        led.to_csv(outm / "window_baselines.csv", index=False)
    else:
        pd.DataFrame(rows).to_csv(outm / "window_baselines.csv", index=False)

    rank_rows = []
    for rank, g in selected.groupby("rank", dropna=True):
        rank_rows.append({"rank": int(rank), "hits": int(g.hit.sum()), "calls": len(g), "accuracy": float(g.hit.mean())})
    pd.DataFrame(rank_rows).to_csv(outm / "rank_and_coverage.csv", index=False)

    monthly = selected.groupby("origin_position").agg(accuracy=("hit", "mean"), calls=("hit", "size"), up_prevalence=("y_true", "mean"), cap=("rank", "max")).reset_index()
    monthly["rolling_12_accuracy"] = monthly["accuracy"].rolling(12, min_periods=6).mean()
    monthly["rolling_12_up_prevalence"] = monthly["up_prevalence"].rolling(12, min_periods=6).mean()
    monthly.to_csv(outm / "temporal_drift.csv", index=False)

    # No pre-registered expanding-window cache was present; do not invent a learning curve.
    pd.DataFrame([{"status": "unavailable", "reason": "no pre-registered expanding-window OOF artifact; rerunning would change the registered experiment"}]).to_csv(outm / "learning_curves.csv", index=False)

    feature_rows = []
    for p in (ROOT / "research/regime_adaptive_selector/feature_ablation").glob("*/summary.json"):
        try:
            x = json.loads(p.read_text(encoding="utf-8"))
            x["feature_family"] = p.parent.name
            feature_rows.append(x)
        except (OSError, json.JSONDecodeError):
            pass
    pd.DataFrame(feature_rows or [{"status": "unavailable"}]).to_csv(outm / "feature_family_lift.csv", index=False)

    # Candidate-level overlap is unavailable because challenger predictions are not all stored in a common schema.
    pd.DataFrame([{"status": "unavailable", "reason": "no complete matched candidate prediction matrix in non-locked artifacts"}]).to_csv(outm / "model_error_overlap.csv", index=False)
    pd.DataFrame([{"oracle": "candidate_family_hindsight", "status": "unavailable", "reason": "candidate outcomes are not available in a common matched matrix"}, {"oracle": "causal_meta_selector", "status": "unavailable", "reason": "no pre-registered walk-forward meta-selector artifact"}]).to_csv(outm / "oracle_bounds.csv", index=False)

    # Date-block bootstrap over monthly accuracies, preserving temporal dependence at the month level.
    vals = monthly["accuracy"].dropna().to_numpy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boots = []
    block = 6
    if len(vals):
        for _ in range(BOOTSTRAP_REPLICATES):
            sample = []
            while len(sample) < len(vals):
                i = int(rng.integers(0, max(1, len(vals) - block + 1)))
                sample.extend(vals[i:i + block].tolist())
            boots.append(float(np.mean(sample[:len(vals)])))
    b = np.asarray(boots)
    pd.DataFrame([{"block_months": block, "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "mean_accuracy": float(np.mean(vals)) if len(vals) else None, "median": float(np.median(b)) if len(b) else None, "p10": float(np.quantile(b,.1)) if len(b) else None, "p90": float(np.quantile(b,.9)) if len(b) else None, "p95": float(np.quantile(b,.95)) if len(b) else None, "prob_ge_62": float(np.mean(b>=.62)) if len(b) else None, "prob_ge_65": float(np.mean(b>=.65)) if len(b) else None, "interpretation": "conditional historical block bootstrap; not a future guarantee"}]).to_csv(outm / "block_bootstrap.csv", index=False)

    pd.DataFrame([{"statistic": "selection_lift_over_all_up", "status": "unavailable", "reason": "registered null permutations were not stored; independent row shuffling is invalid"}]).to_csv(outm / "null_signal_tests.csv", index=False)
    summary = {"audit": "signal_ceiling_audit", "usable_months": int(selected.origin_position.nunique()), "nominal_prediction_rows": int(len(df)), "selected_calls": int(len(selected)), "max_nonlocked_origin": MAX_NONLOCKED_ORIGIN, "locked_evaluation_read": False, "active_model_changed": False, "locked_origins": [268, 315], "reference_accuracy": float(selected.hit.mean()), "classification": "inconclusive", "classification_reason": "Available non-locked evidence does not support a stable >65% claim; null tests and multiplicity-adjusted inference are unavailable.", "evidence": {"validation": {"hits": 381, "calls": 635, "accuracy": .60}, "tuning": {"hits": 630, "calls": 965, "accuracy": .6528497409}, "bootstrap": "see block_bootstrap.csv"}}
    (outm / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
