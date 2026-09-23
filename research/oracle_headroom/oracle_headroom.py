"""Hindsight headroom analysis: how far is production from bounds no causal model can beat?

Every "oracle" here uses outcomes it could not know at the time. None is a candidate. They
bound what any drift-based or timing-based improvement could possibly deliver, at
production's own per-origin call count and universe.

- long-run drift oracle: ranks by each indicator's Up-rate over all non-locked labels,
  i.e. it knows the true long-run drift.
- window drift oracle: ranks by each indicator's Up-rate inside the evaluation window,
  i.e. it also knows how drift shifted in that window.
- breadth timing oracle: keeps production's picks but calls every pick Down in months when
  most of the universe fell, i.e. perfect market-direction timing.
- perfect Up selection: picks only indicators that rose, as many as exist, up to the cap.
  The shortfall is what the 15-20 coverage contract costs an Up-only selector.

Reads the workbook with maximum_position=267 via the seasonal-prior loader, so the locked
evaluation (origins 268-315) is untouched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/seasonal_prior"))
from seasonal_prior import WINDOWS, load_directions, load_universe  # noqa: E402

OUT = ROOT / "research/oracle_headroom/metrics"
LAST_LABEL = 266


def _window_of(origin: int) -> str:
    return next(name for name, (lo, hi) in WINDOWS.items() if lo <= origin <= hi)


def run() -> pd.DataFrame:
    directions, _, _ = load_directions()
    universe = load_universe(directions)
    longrun = directions.loc[:LAST_LABEL].mean()
    window_rate = {
        name: directions.loc[lo:hi].mean() for name, (lo, hi) in WINDOWS.items()
    }
    rows = []
    for origin, current in universe.groupby("origin_position", sort=True):
        origin = int(origin)
        cap = int(current["accepted"].sum())
        truth = current.set_index("indicator_id")["y_true"]
        production = current[current["accepted"]]
        prod_hits = int(
            (production["y_true"].eq(1) & ~production["prod_down"]).sum()
            + (production["y_true"].eq(0) & production["prod_down"]).sum()
        )
        broad_down = bool(truth.mean() < 0.5)

        def top_by(rate: pd.Series) -> pd.Index:
            ranked = rate.reindex(truth.index).sort_values(ascending=False, kind="mergesort")
            return ranked.head(cap).index

        longrun_picks = top_by(longrun)
        window_picks = top_by(window_rate[_window_of(origin)])
        prod_picks = pd.Index(production["indicator_id"])
        rows.append({
            "origin_position": origin,
            "window": _window_of(origin),
            "cap": cap,
            "universe_breadth": float(truth.mean()),
            "broad_down_month": broad_down,
            "production": prod_hits,
            "longrun_drift_oracle": int(truth[longrun_picks].eq(1).sum()),
            "window_drift_oracle": int(truth[window_picks].eq(1).sum()),
            "breadth_timing_oracle": int(
                truth[prod_picks].eq(0).sum() if broad_down else truth[prod_picks].eq(1).sum()
            ),
            "longrun_plus_breadth_timing": int(
                truth[longrun_picks].eq(0).sum() if broad_down
                else truth[longrun_picks].eq(1).sum()
            ),
            "perfect_up_selection": int(min(cap, truth.eq(1).sum())),
        })
    return pd.DataFrame(rows)


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "production", "longrun_drift_oracle", "window_drift_oracle",
        "breadth_timing_oracle", "longrun_plus_breadth_timing", "perfect_up_selection",
    ]
    table = []
    for window in [*WINDOWS, "all_nonlocked"]:
        part = records if window == "all_nonlocked" else records[records["window"].eq(window)]
        calls = int(part["cap"].sum())
        row = {"window": window, "months": len(part), "calls": calls,
               "broad_down_months": int(part["broad_down_month"].sum())}
        for column in columns:
            row[f"{column}_hits"] = int(part[column].sum())
            row[f"{column}_accuracy"] = float(part[column].sum() / calls)
        table.append(row)
    return pd.DataFrame(table)


def main() -> pd.DataFrame:
    records = run()
    table = summarize(records)
    OUT.mkdir(parents=True, exist_ok=True)
    records.to_csv(OUT / "monthly_oracles.csv", index=False)
    table.to_csv(OUT / "headroom.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({
        "analysis": "oracle_headroom",
        "hindsight": True,
        "candidate": False,
        "max_level_position_read": 267,
        "locked_evaluation_read": False,
        "windows": table.to_dict(orient="records"),
    }, indent=2), encoding="utf-8")
    return table


if __name__ == "__main__":
    result = main()
    show = ["window", "months", "broad_down_months"] + [
        c for c in result.columns if c.endswith("_accuracy")
    ]
    pd.set_option("display.width", 250)
    print(result[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
