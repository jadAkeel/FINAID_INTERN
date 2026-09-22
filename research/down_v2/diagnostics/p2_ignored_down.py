"""P2: Quantify historically ignored strong-Down candidates.

An "X40 case" = a ready indicator NOT in the Up base pool whose p_down is high.
Question: if we had called them Down, would they have been right?
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ART = Path(__file__).resolve().parents[3] / "artifacts/active/regime_adaptive_predictions.parquet"
WINDOWS = {
    "tuning": (120, 179),
    "validation": (180, 219),
    "confirmation": (220, 266),
}


def ready_frame(p: pd.DataFrame) -> pd.DataFrame:
    return p[
        p["level_c_ready"].fillna(False).astype(bool)
        & p["p_up_selection_score"].notna()
        & p["p_down"].notna()
        & p["y_true"].notna()
    ].copy()


def main() -> None:
    p = pd.read_parquet(ART)
    p["regime_cap"] = pd.to_numeric(p["regime_cap"], errors="coerce")
    r = ready_frame(p)
    r["down_correct"] = r["y_true"].eq(0.0)

    # flag membership of the Up base pool
    in_pool = []
    for origin, g in p.groupby("origin_position", sort=True):
        cap = int(g["regime_cap"].dropna().iloc[0])
        rg = g[
            g["level_c_ready"].fillna(False).astype(bool)
            & g["p_up_selection_score"].notna()
            & g["p_down"].notna()
        ]
        ranked = rg.sort_values(
            ["p_up_selection_score", "indicator_id"], ascending=[False, True]
        )
        pool = set(ranked.head(cap)["indicator_id"])
        in_pool.append(
            pd.Series(
                g["indicator_id"].isin(pool).values,
                index=g.index,
            )
        )
    p["in_up_pool"] = pd.concat(in_pool).reindex(p.index).fillna(False)
    r = ready_frame(p)
    r["down_correct"] = r["y_true"].eq(0.0)
    r["in_up_pool"] = p.loc[r.index, "in_up_pool"].astype(bool)
    x40 = r[~r["in_up_pool"]].copy()

    print("=" * 76)
    print("P2  Strong-Down candidates ignored by the Up-first pool")
    print("=" * 76)


    print(f"ready rows outside the Up pool        : {len(x40)} of {len(r)}")
    print(f"  ... with p_down >= 0.65             : {int((x40['p_down'] >= 0.65).sum())}")
    print(f"  ... with p_down >= 0.70             : {int((x40['p_down'] >= 0.70).sum())}")
    print(f"  ... with p_down >= 0.80 (hard gate) : {int((x40['p_down'] >= 0.80).sum())}")
    print()

    print("--- Reliability: predicted p_down vs observed Down rate, ignored rows ---")
    bins = [0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.01]
    x40["bucket"] = pd.cut(x40["p_down"], bins, right=False)
    agg = x40.groupby("bucket", observed=True).agg(
        n=("down_correct", "size"),
        mean_p_down=("p_down", "mean"),
        observed_down_rate=("down_correct", "mean"),
    )
    agg["overconfidence"] = agg["mean_p_down"] - agg["observed_down_rate"]
    print(agg.round(4).to_string())
    print()
    base_rate = 1 - r["y_true"].mean()
    print(f"overall Down base rate among all ready rows : {base_rate:.4f}")
    print("  -> a calibrated p_down would have observed_down_rate ~= mean_p_down.")
    print("     Even at p_down>=0.70 the observed rate barely exceeds the base rate.")
    print()

    # what did the Up pool actually achieve, for scale
    inp = r[r["in_up_pool"].astype(bool)]
    print(f"reference: rows IN the Up pool, actual Down rate: {1 - inp['y_true'].mean():.4f}")
    print()

    print("--- Per-window: ignored strong-Down (p_down>=0.65) accuracy as Down ---")
    for name, (a, b) in WINDOWS.items():
        w = x40[(x40["origin_position"].between(a, b)) & (x40["p_down"] >= 0.65)]
        if len(w):
            print(
                f"  {name:12s} n={len(w):4d} down_acc={w['down_correct'].mean():.4f} "
                f"(base rate {1 - w['y_true'].mean():.4f})"
            )
    print()

    # strongest ignored per month: would swapping it for the weakest Up call help?
    print("--- Swap counterfactual: strongest ignored Down vs weakest selected Up ---")
    rows = []
    for origin, g in p.groupby("origin_position", sort=True):
        cap = int(g["regime_cap"].dropna().iloc[0])
        gg = g[g["y_true"].notna()]
        x = gg[~gg["in_up_pool"].astype(bool) & gg["p_down"].notna()]
        s = gg[gg["accepted"].fillna(False).astype(bool)]
        if x.empty or s.empty:
            continue
        best = x.loc[x["p_down"].idxmax()]
        victim = s.loc[s["p_up_selection_score"].idxmin()]
        rows.append(
            {
                "origin": int(origin),
                "x_pdown": best["p_down"],
                "x_pup": best["p_up_base"],
                "x_down_correct": bool(best["y_true"] == 0.0),
                "victim_correct": bool(
                    (victim["predicted_direction"] == "Up" and victim["y_true"] == 1.0)
                    or (victim["predicted_direction"] == "Down" and victim["y_true"] == 0.0)
                ),
            }
        )
    d = pd.DataFrame(rows)
    d["x_down_correct"] = d["x_down_correct"].astype(int)
    d["victim_correct"] = d["victim_correct"].astype(int)
    print(f"months with a swappable pair: {len(d)}")
    print(f"  ignored X correct as Down : {int(d['x_down_correct'].sum())} ({d['x_down_correct'].mean():.4f})")
    print(f"  victim already correct    : {int(d['victim_correct'].sum())} ({d['victim_correct'].mean():.4f})")
    print(f"  NET hit delta             : {int((d['x_down_correct'] - d['victim_correct']).sum()):+d}")
    print()
    print("--- Same swap but with a strict margin filter (p_down - p_up_base >= M) ---")
    for margin in (0.05, 0.10, 0.20, 0.30):
        m = d[(d["x_pdown"] - d["x_pup"]) >= margin]
        if len(m) == 0:
            continue
        net = int((m["x_down_correct"].astype(int) - m["victim_correct"].astype(int)).sum())
        print(
            f"  margin>={margin:.2f}: months={len(m):3d} "
            f"X_acc={m['x_down_correct'].mean():.4f} victim_acc={m['victim_correct'].mean():.4f} "
            f"net={net:+d}"
        )
    print()
    print("--- Same swap with a p_down floor ---")
    for floor in (0.60, 0.65, 0.70, 0.75):
        m = d[d["x_pdown"] >= floor]
        if len(m) == 0:
            continue
        net = int((m["x_down_correct"].astype(int) - m["victim_correct"].astype(int)).sum())
        print(
            f"  p_down>={floor:.2f}: months={len(m):3d} "
            f"X_acc={m['x_down_correct'].mean():.4f} victim_acc={m['victim_correct'].mean():.4f} "
            f"net={net:+d}"
        )


if __name__ == "__main__":
    main()
