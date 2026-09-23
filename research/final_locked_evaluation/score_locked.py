"""Score the pre-registered final locked evaluation (origins 268-315), once.

Reads the Phase B artifacts rebuilt in an isolated copy (see PROTOCOL.md) and writes the
pre-registered outputs. Usage:

    python research/final_locked_evaluation/score_locked.py <phase_b_copy_root>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/final_locked_evaluation/metrics"
LOCKED = (268, 315)
SEED = 20260923
BLOCK_MONTHS = 6
REPLICATES = 5000
PRIOR_WINDOW = 48
PRIOR_MIN_LABELS = 24


def _hits(frame: pd.DataFrame, direction_column: str) -> pd.Series:
    up = frame[direction_column].eq("Up") & frame["y_true"].eq(1)
    down = frame[direction_column].eq("Down") & frame["y_true"].eq(0)
    return (up | down).astype(int)


def _bootstrap(monthly_hits: np.ndarray, monthly_calls: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    n = len(monthly_hits)
    starts = rng.integers(0, n - BLOCK_MONTHS + 1, size=(REPLICATES, -(-n // BLOCK_MONTHS)))
    index = (starts[:, :, None] + np.arange(BLOCK_MONTHS)).reshape(REPLICATES, -1)[:, :n]
    rates = monthly_hits[index].sum(1) / monthly_calls[index].sum(1)
    return {f"p{q}": float(np.quantile(rates, q / 100)) for q in (5, 10, 50, 90, 95)}


def _summary(selected: pd.DataFrame, direction_column: str) -> dict:
    selected = selected.assign(hit=_hits(selected, direction_column))
    monthly = selected.groupby("origin_position").agg(hits=("hit", "sum"), calls=("hit", "size"))
    down = selected[selected[direction_column].eq("Down")]
    return {
        "months": int(len(monthly)),
        "hits": int(monthly["hits"].sum()),
        "calls": int(monthly["calls"].sum()),
        "accuracy": float(monthly["hits"].sum() / monthly["calls"].sum()),
        "down_calls": int(len(down)),
        "down_hits": int(down["hit"].sum()),
        "cap_distribution": {int(k): int(v) for k, v in monthly["calls"].value_counts().sort_index().items()},
        "monthly_accuracy_min": float((monthly["hits"] / monthly["calls"]).min()),
        "monthly_accuracy_max": float((monthly["hits"] / monthly["calls"]).max()),
        "block_bootstrap_accuracy": _bootstrap(
            monthly["hits"].to_numpy(float), monthly["calls"].to_numpy(float)
        ),
    }


def _prior_only(copy_root: Path, universe: pd.DataFrame) -> dict:
    sys.path.insert(0, str(copy_root / "src"))
    from forecast_select.io import load_workbook

    frame = load_workbook(copy_root / "data/monthly_indicators.xlsx").set_index("position")
    levels = frame[[c for c in frame.columns if str(c).startswith("X")]]
    following = levels.shift(-1)
    labels = (following > levels).astype(float).where(following.notna() & levels.notna())
    rows = []
    for origin, current in universe.groupby("origin_position", sort=True):
        history = labels.loc[max(1, origin - 1 - PRIOR_WINDOW): origin - 2]
        rate = history.mean().where(history.count() >= PRIOR_MIN_LABELS)
        ids = current["indicator_id"]
        ranked = pd.DataFrame({"indicator_id": ids.to_numpy(), "rate": rate.reindex(ids).to_numpy()})
        ranked = ranked.dropna().sort_values(["rate", "indicator_id"], ascending=[False, True], kind="mergesort")
        picks = current.set_index("indicator_id").loc[ranked["indicator_id"].head(15)].reset_index()
        rows.append(picks.assign(prior_direction="Up"))
    return _summary(pd.concat(rows, ignore_index=True), "prior_direction")


def main(copy_root: Path) -> dict:
    regime = pd.read_parquet(copy_root / "research/regime_adaptive_selector/artifacts/predictions.parquet")
    directional = pd.read_parquet(copy_root / "research/directional_downside_selector/artifacts/predictions.parquet")
    lo, hi = LOCKED
    regime = regime[regime["origin_position"].between(lo, hi)].copy()
    directional = directional[directional["origin_position"].between(lo, hi)].copy()
    for frame, column in ((regime, "accepted"), (directional, "base_accepted")):
        frame[column] = frame[column].fillna(False).astype(bool)
    if regime["origin_position"].nunique() != hi - lo + 1:
        raise ValueError("Locked run does not cover every locked origin")
    universe = regime[
        regime["level_c_ready"].fillna(False).astype(bool)
        & ~regime["adaptive_data_quality_excluded"].fillna(False).astype(bool)
    ]
    result = {
        "evaluation": "final_locked_evaluation",
        "origins": list(LOCKED),
        "active_model": _summary(regime[regime["accepted"]], "predicted_direction"),
        "uptrend_baseline_top15": _summary(
            directional[directional["base_accepted"]], "base_predicted_direction"
        ),
        "prior_only_top15_n48": _prior_only(copy_root, universe),
        "development_reference_accuracy_120_266": 0.6243614931237721,
        "locked_evaluation_read": True,
        "locked_window_consumed": True,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "locked_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    regime.to_parquet(OUT / "locked_predictions.parquet", index=False)
    return result


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1]).resolve()), indent=2))
