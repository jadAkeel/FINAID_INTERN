"""Causal prior-only selector diagnostic.

Selects the top-k indicators each month by their trailing Up-rate alone and
reports every window length side by side. Nothing is selected or tuned here:
the point is to measure how much of the active model's accuracy a zero-feature
base-rate rule already explains.

Causal contract at origin t: the direction label d_s = 1[X_{s+1} > X_s] is
usable only for s <= t-2, and the scored target is d_t. No origin above 266 is
read, so the locked evaluation (268-315) is untouched.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/monthly_indicators.xlsx"
OUT = ROOT / "research/prior_only_selector/metrics"
MAX_ORIGIN = 266
MIN_LABELS = 24
WINDOWS = {"tuning": (120, 179), "validation": (180, 219), "confirmation": (220, 266)}
PRODUCTION = {"tuning": (686, 1071), "validation": (395, 675), "confirmation": (508, 799)}


def load_directions() -> pd.DataFrame:
    frame = pd.read_excel(DATA).set_index("Dates")
    levels = frame[[c for c in frame.columns if c.startswith("X")]]
    # Position 267 is the last level read: origin 266's target needs X_267.
    levels = levels.iloc[: MAX_ORIGIN + 2]
    nxt = levels.shift(-1)
    return (nxt > levels).astype(float).where(nxt.notna() & levels.notna())


def evaluate(directions: pd.DataFrame, window: int, k: int, mode: str = "up_only") -> dict:
    hits = {w: [0, 0] for w in WINDOWS}
    for origin in range(120, MAX_ORIGIN + 1):
        start = max(0, origin - 1 - window) if window else 0
        history = directions.iloc[start : origin - 1]  # rows s <= origin-2
        rate = history.mean()
        usable = (history.count() >= MIN_LABELS) & directions.iloc[origin].notna()
        if mode == "up_only":
            picks = rate.where(usable).nlargest(k).index
            predicted = pd.Series(1.0, index=picks)
        else:
            picks = (rate - 0.5).abs().where(usable).nlargest(k).index
            predicted = (rate[picks] >= 0.5).astype(float)
        target = directions.iloc[origin][picks]
        month_hits, month_calls = int((target == predicted).sum()), int(len(picks))
        for name, (lo, hi) in WINDOWS.items():
            if lo <= origin <= hi:
                hits[name][0] += month_hits
                hits[name][1] += month_calls
    row = {"window_months": window or "all", "k": k, "mode": mode}
    for name, (h, c) in hits.items():
        row[f"{name}_hits"], row[f"{name}_calls"] = h, c
        row[f"{name}_accuracy"] = h / c if c else np.nan
    return row


def main() -> pd.DataFrame:
    directions = load_directions()
    assert len(directions) == MAX_ORIGIN + 2, "locked rows must not be loaded"
    rows = []
    for window in (24, 36, 48, 60, 96, 0):
        rows.append(evaluate(directions, window, 15))
    for window in (36, 48, 0):
        rows.append(evaluate(directions, window, 20))
    for window in (36, 48, 0):
        rows.append(evaluate(directions, window, 15, mode="majority"))
    table = pd.DataFrame(rows)
    prod = {"window_months": "n/a", "k": "15-20", "mode": "production_regime_adaptive"}
    for name, (h, c) in PRODUCTION.items():
        prod[f"{name}_hits"], prod[f"{name}_calls"], prod[f"{name}_accuracy"] = h, c, h / c
    table = pd.concat([pd.DataFrame([prod]), table], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "prior_only_windows.csv", index=False)
    return table


if __name__ == "__main__":
    result = main()
    cols = ["mode", "window_months", "k"] + [f"{w}_accuracy" for w in WINDOWS]
    print(result[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
