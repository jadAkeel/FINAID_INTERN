"""P11: is there ANY regime in which Down V2 is strong enough to select on?

The user's section 7 allows a regime-conditioned Down selector if the signal is
only good in some regimes -- but not concluded from Tuning alone. This checks
out-of-pool Down V2 candidates by regime on Validation and Confirmation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

ACTIVE = ROOT / "artifacts/active/regime_adaptive_predictions.parquet"
V2 = ROOT / "research/down_v2/artifacts/down_v2_predictions.parquet"
OUT = ROOT / "research/down_v2/metrics/p11_regime_conditional.json"

WINDOWS = {
    "tuning": (120, 179),
    "validation": (180, 219),
    "confirmation": (220, 266),
}


def pool_members(g: pd.DataFrame) -> set:
    cap = int(g["regime_cap"].iloc[0])
    ready = g[g["level_c_ready"].fillna(False).astype(bool)]
    ranked = ready.sort_values(
        ["p_up_selection_score", "indicator_id"], ascending=[False, True]
    )
    return set(ranked.head(cap)["indicator_id"])


def main() -> None:
    art = pd.read_parquet(ACTIVE)
    v2 = pd.read_parquet(V2)[
        ["origin_position", "indicator_id", "p_platt"]
    ].rename(columns={"p_platt": "p_down_v2"})
    j = art.merge(v2, on=["origin_position", "indicator_id"], how="left")
    j["p_down_v2"] = j["p_down_v2"].fillna(j["p_down"].fillna(0.0))
    j["down_correct"] = 1.0 - j["y_true"]

    outs = []
    for origin, g in j.groupby("origin_position", sort=True):
        pool = pool_members(g)
        out = g[
            g["level_c_ready"].fillna(False).astype(bool)
            & ~g["indicator_id"].isin(pool)
        ].copy()
        out["in_pool"] = False
        ins = g[g["accepted"].fillna(False).astype(bool)].copy()
        ins["in_pool"] = True
        outs.append(pd.concat([out, ins], ignore_index=True))
    x = pd.concat(outs, ignore_index=True)

    # baseline: the Up pool's own down rate (what a Down call gives up)
    results = {}
    print("=" * 78)
    print("P11  Regime-conditional Down V2 (out-of-pool candidates only)")
    print("=" * 78)
    for wname, (a, b) in WINDOWS.items():
        w = x[x["origin_position"].between(a, b)]
        r = {}
        for label, m in {
            "all": pd.Series(True, index=w.index),
            "stressed": w["regime_label"].eq("stressed"),
            "mixed": w["regime_label"].eq("mixed"),
            "calm": w["regime_label"].eq("calm"),
            "high_shock": w["shock_stress"].ge(0.65),
            "low_dispersion": w["market_dispersion"]
            .le(w["market_dispersion"].quantile(0.33)),
            "falling_breadth": w["market_breadth_3"].lt(0.45),
            "v2_top5": pd.Series(False, index=w.index),
        }.items():
            if label == "v2_top5":
                sel = (
                    w.sort_values(["origin_position", "p_down_v2"], ascending=[True, False])
                    .groupby("origin_position")
                    .head(5)
                )
            else:
                sel = w[m & w["p_down_v2"].ge(0.40)]
            if len(sel) == 0:
                r[label] = {"n": 0}
                continue
            r[label] = {
                "n": int(len(sel)),
                "down_precision": float(sel["down_correct"].mean()),
                "months": int(sel["origin_position"].nunique()),
            }
        results[wname] = r
        print(f"\n--- {wname} ---")
        for label, d in r.items():
            if d["n"]:
                print(
                    f"  {label:15s} n={d['n']:4d} months={d['months']:3d} "
                    f"down_precision={d['down_precision']:.4f}"
                )
            else:
                print(f"  {label:15s} n=0")

    # the bar a Down call must clear: the Up pool's realized hit rate
    print("\n--- Bar to clear: realized Up-pool hit rate (foregone per Down call) ---")
    for wname, (a, b) in WINDOWS.items():
        w = x[x["origin_position"].between(a, b) & x["in_pool"]]
        print(f"  {wname:12s} up_pool hit_rate={w['y_true'].mean():.4f} (n={len(w)})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        pd.json_normalize(results, sep="_").to_json(indent=2, orient="records"),
        encoding="utf-8",
    )
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
