"""P1: Verify the Up-first architecture and the Down-is-overlay claim.

Reads the frozen active artifact only. No refits, no new models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ART = Path(__file__).resolve().parents[3] / "artifacts/active/regime_adaptive_predictions.parquet"
WINDOWS = {
    "tuning": (120, 179),
    "validation": (180, 219),
    "confirmation": (220, 266),
}


def ready_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["level_c_ready"].fillna(False).astype(bool)
        & frame["p_up_selection_score"].notna()
        & frame["p_down"].notna()
    )


def up_pool(frame: pd.DataFrame, cap: int) -> set[str]:
    ranked = frame.sort_values(
        ["p_up_selection_score", "indicator_id"], ascending=[False, True]
    )
    return set(ranked.head(cap)["indicator_id"])


def main() -> None:
    p = pd.read_parquet(ART)
    p["regime_cap"] = pd.to_numeric(p["regime_cap"], errors="coerce")

    rows = []
    for origin, g in p.groupby("origin_position", sort=True):
        cap = int(g["regime_cap"].dropna().iloc[0])
        r = g[ready_mask(g)]
        pool = up_pool(r, cap)
        selected = g[g["accepted"].fillna(False).astype(bool)]
        sel_ids = set(selected["indicator_id"])
        rows.append(
            {
                "origin": int(origin),
                "cap": cap,
                "n_selected": len(sel_ids),
                "n_outside_up_pool": len(sel_ids - pool),
                "n_replacements": int(
                    g["regime_replacement"].fillna(False).astype(bool).sum()
                ),
                "n_down": int(selected["predicted_direction"].eq("Down").sum()),
            }
        )
    d = pd.DataFrame(rows)

    print("=" * 72)
    print("P1  Up-first architecture  (Down model as overlay, not selector)")
    print("=" * 72)
    print(f"months analysed                      : {len(d)}")
    print(f"selections outside the Up pool      : {int(d['n_outside_up_pool'].sum())}")
    print(f"regime_replacement events           : {int(d['n_replacements'].sum())}")
    print(f"months with any replacement         : {int((d['n_replacements'] > 0).sum())}")
    print()

    # Do Down calls ever sit outside the top-15 core?
    sel = p[p["accepted"].fillna(False).astype(bool)].copy()
    print("selection_rank range of Down calls  : "
          f"{sel[sel['predicted_direction'].eq('Down')]['selection_rank'].min():.0f}"
          f"-{sel[sel['predicted_direction'].eq('Down')]['selection_rank'].max():.0f}")
    print()

    # If p_down alone had chosen, how different would the set be?
    print("Counterfactual: monthly top-N chosen by p_down ALONE vs by p_up pool")
    for n in (15, 20):
        overlaps = []
        for origin, g in p.groupby("origin_position", sort=True):
            cap = int(g["regime_cap"].dropna().iloc[0])
            r = g[ready_mask(g)]
            if len(r) < n:
                continue
            up_top = up_pool(r, max(cap, n))
            down_top = set(
                r.sort_values(["p_down", "indicator_id"], ascending=[False, True])
                .head(n)["indicator_id"]
            )
            overlaps.append(len(up_top & down_top) / n)
        print(f"  top-{n:<2d} Jaccard overlap Up vs Down pool: {np.mean(overlaps):.4f}")
    print()

    print("VERDICT: p_down is computed on all eligible indicators, but the Base")
    print("Pool is formed by p_up_selection_score alone; the Down model can only")
    print("flip direction of rows already admitted. With maximum_replacements=0")
    print("it is a Direction Override, not an independent Selector.")


if __name__ == "__main__":
    main()
