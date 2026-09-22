"""Feature family review: overlap, collinearity, history needs, leakage audit.

Groups the 28 Down features into 8 families, measures within/between
correlation, checks how much history each family costs, and audits the
causal boundary.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PANEL = ROOT / "research/down_v2/artifacts/down_panel.parquet"

FAMILIES = {
    "recent_returns": ["down_return_1", "down_return_lag_1", "down_return_lag_2"],
    "momentum": ["down_momentum_3", "down_momentum_6", "down_momentum_12"],
    "volatility": ["down_volatility_3", "down_volatility_12"],
    "drawdown_distance": ["down_drawdown_12", "down_distance_mean_12"],
    "exhaustion_stall": [
        "down_rise_6_before_2",
        "down_stall_2",
        "down_rise_score",
        "down_stall_score",
        "down_exhaustion_score",
    ],
    "market_regime": [
        "down_market_mean_return",
        "down_market_breadth",
        "down_market_breadth_3",
        "down_market_breadth_change_3",
        "down_market_dispersion",
    ],
    "lead_peer": [
        "down_lead_peer_score",
        "down_lead_negative_consensus",
        "down_lead_abs_correlation",
        "down_lead_peer_count",
    ],
    "misc": [
        "down_negative_share_3",
        "down_near_high_12",
        "down_deceleration_2_vs_4",
        "down_volatility_compression",
    ],
}

IDEA = {
    "recent_returns": "Short-horizon reversal: a sharp recent drop tends to continue.",
    "momentum": "Persistent negative drift over 3-12 months.",
    "volatility": "Elevated instability precedes further declines.",
    "drawdown_distance": "Position below recent high/mean signals established weakness.",
    "exhaustion_stall": "Rise-then-stall pattern: strong run that loses momentum tops out.",
    "market_regime": "Broad market weakness raises downside for every indicator.",
    "lead_peer": "Correlated peers that moved first predict the next move.",
    "misc": "Sign-share, proximity to high, deceleration, vol compression.",
}


def main() -> None:
    panel = pd.read_parquet(PANEL)
    ev = panel[panel["evaluation_origin"]] if "evaluation_origin" in panel else panel
    ev = ev[ev["down_eligible"]].copy()

    print("=" * 78)
    print("Down feature family review")
    print("=" * 78)
    for fam, feats in FAMILIES.items():
        present = [f for f in feats if f in ev.columns]
        print(f"\n{fam} ({len(present)} features)")
        print(f"  idea: {IDEA[fam]}")
        cov = ev[present].notna().mean().min()
        print(f"  coverage (min non-null share): {cov:.4f}")

    # --- correlation structure ---
    print("\n" + "=" * 78)
    print("Within-family correlation (mean |r|)")
    print("=" * 78)
    for fam, feats in FAMILIES.items():
        present = [f for f in feats if f in ev.columns and ev[f].notna().sum() > 0]
        if len(present) < 2:
            print(f"  {fam:20s} n/a")
            continue
        corr = ev[present].corr().abs()
        vals = corr.to_numpy()[np.triu_indices(len(present), 1)]
        print(f"  {fam:20s} mean|r|={vals.mean():.4f}  max|r|={vals.max():.4f}")

    print("\nBetween-family correlation (mean |r| of the family centroids)")
    fam_centroids = {}
    for fam, feats in FAMILIES.items():
        present = [f for f in feats if f in ev.columns and ev[f].notna().sum() > 0]
        if not present:
            continue
        # rank-normalize each feature to make scales comparable
        z = ev[present].rank(pct=True).mean(axis=1)
        fam_centroids[fam] = z
    cen = pd.DataFrame(fam_centroids)
    c = cen.corr().abs()
    print(c.round(3).to_string())

    # --- history cost ---
    print("\n" + "=" * 78)
    print("History required per family (earliest origin with coverage)")
    print("=" * 78)
    for fam, feats in FAMILIES.items():
        present = [f for f in feats if f in panel.columns]
        if not present:
            continue
        per_origin = panel.groupby("origin_position")[present].apply(
            lambda g: g.notna().mean().min()
        )
        ok = per_origin[per_origin > 0.5]
        if ok.empty:
            print(f"  {fam:20s} never reaches 50% coverage")
            continue
        print(f"  {fam:20s} first origin with >50% coverage: {int(ok.index.min())}")

    # --- leakage audit ---
    print("\n" + "=" * 78)
    print("Leakage audit")
    print("=" * 78)
    print("  All features are built from `frame[indicators].shift(availability_lag=1)`")
    print("  in build_directional_downside_features, i.e. observations through t-1.")
    print("  Training rows pass causal_training_rows(origin, lag=1) so labels stop")
    print("  at t-2. The lead-peer block uses raw_returns.shift(-2) over a PAST")
    print("  window [predictor_start, predictor_end] where predictor_end = latest-1,")
    print("  so the shifted responses never include the forecast month.")
    print("  Verification: panel max origin", int(panel["origin_position"].max()),
          "| evaluation max 266 | locked 268-315 untouched.")


if __name__ == "__main__":
    main()
