"""Post-hoc, exploratory: can Chronos-2's better calibration be used anywhere else?

Written after the locked window was consumed, on non-locked origins 120-266 that earlier research
had already viewed. It is a descriptive check, not a candidate. It tests three uses:

1. market timing: the monthly mean Chronos P(Up) as a broad-down signal;
2. per-call reliability: whether Chronos agreement separates production's hits from its misses;
3. Down calls: replacing production's lowest-ranked pick with Chronos's most bearish indicator.
"""
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
CHRONOS = ROOT / "research/chronos_zero_shot/artifacts/predictions.parquet"
ACTIVE = ROOT / "artifacts/active/regime_adaptive_predictions.parquet"
MONTHS = ROOT / "research/breadth_timing/metrics/monthly_scores.csv"
OUT = ROOT / "research/chronos_zero_shot/metrics/post_hoc_uses.csv"
WINDOWS = {"tuning": (120, 179), "validation": (180, 219), "confirmation": (220, 266)}


def _window(origin: int) -> str:
    return next(name for name, (lo, hi) in WINDOWS.items() if lo <= origin <= hi)


def main() -> pd.DataFrame:
    chronos = pd.read_parquet(CHRONOS)
    chronos = chronos[chronos["common_eligible"].fillna(False).astype(bool)]
    if int(chronos["origin_position"].max()) > 266:
        raise ValueError("Locked origins present")
    active = pd.read_parquet(ACTIVE, columns=[
        "origin_position", "indicator_id", "accepted", "predicted_direction",
        "selection_rank", "y_true", "level_c_ready", "adaptive_data_quality_excluded",
    ])
    active["accepted"] = active["accepted"].fillna(False).astype(bool)
    universe = active[
        active["level_c_ready"].fillna(False).astype(bool)
        & ~active["adaptive_data_quality_excluded"].fillna(False).astype(bool)
    ]
    months = pd.read_csv(MONTHS, index_col=0)
    accepted = chronos[chronos["active_accepted"].fillna(False).astype(bool)]
    months = months.join(accepted.groupby("origin_position")["p_up_chronos_raw"].mean().rename("mean_p"))

    swaps = []
    for origin, current in universe.groupby("origin_position"):
        picks = current[current["accepted"]].sort_values("selection_rank")
        pool = chronos[
            chronos["origin_position"].eq(origin)
            & ~chronos["indicator_id"].isin(picks["indicator_id"])
            & chronos["indicator_id"].isin(current["indicator_id"])
        ].sort_values("p_up_chronos_raw")
        worst = picks.iloc[-1]
        worst_hit = int(
            (worst["predicted_direction"] == "Up") == (worst["y_true"] == 1)
        )
        truth = current.set_index("indicator_id")["y_true"]
        for threshold in (0.5, 0.4, 0.3):
            best = pool[pool["p_up_chronos_raw"] < threshold].head(1)
            gain = 0 if best.empty else int(truth[best["indicator_id"].iloc[0]] == 0) - worst_hit
            swaps.append({"origin_position": origin, "threshold": threshold,
                          "swapped": int(not best.empty), "gain": gain})
    swaps = pd.DataFrame(swaps)

    rows = []
    for name, (lo, hi) in WINDOWS.items():
        part = months.loc[lo:hi]
        up_calls = accepted[
            accepted["origin_position"].between(lo, hi)
            & accepted["active_predicted_direction"].eq("Up")
        ]
        agree = up_calls["p_up_chronos_raw"] >= 0.5
        row = {
            "window": name,
            "timing_auc_low_mean_p_vs_flip_target": roc_auc_score(part["flip_target"], -part["mean_p"]),
            "reliability_auc_on_production_up_calls": roc_auc_score(
                up_calls["y_true"], up_calls["p_up_chronos_raw"]
            ),
            "chronos_agrees_share": float(agree.mean()),
        }
        window_swaps = swaps[swaps["origin_position"].between(lo, hi)]
        for threshold, group in window_swaps.groupby("threshold"):
            row[f"swap_p<{threshold}_count"] = int(group["swapped"].sum())
            row[f"swap_p<{threshold}_hit_gain"] = int(group["gain"].sum())
        rows.append(row)
    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    return table


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    print(main().round(3).T.to_string())
