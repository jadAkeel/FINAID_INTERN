"""Build the Down V3 feature panel: a NEW feature space, not V2 + extras.

The V2 report established that the existing absolute-price Down features are
redundant (momentum <-> drawdown_distance |r|=0.905) and that adding
complexity to them only overfits. V3 therefore changes the *kind* of
information: instead of asking "is this indicator down?", it asks
"is this indicator weaker than everything else, and is it breaking down?"

Families (all causal: `source = frame[indicators].shift(availability_lag=1)`,
so observations through t-1 only, identical to the production Down model):

  v3_rel   - cross-sectional relative weakness / rank deterioration
  v3_group - weakness relative to the indicator's own asset group
  v3_break - trend-breakdown structure (distance from highs, failed rallies)
  v3_mkt   - market context interactions

Writes: research/down_v3/artifacts/v3_panel.parquet
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from forecast_select.io import load_workbook  # noqa: E402
from forecast_select.targets import build_targets  # noqa: E402
from forecast_select.validation import assert_target_alignment  # noqa: E402

OUT = ROOT / "research/down_v3/artifacts/v3_panel.parquet"
GROUPS = ROOT / "configs/downside_risk_gate.yaml"


def safe_div(a, b):
    b = b.replace(0, np.nan)
    return a.div(b)


def build_v3_features(frame: pd.DataFrame, lag: int) -> pd.DataFrame:
    indicators = [c for c in frame.columns if c.startswith("X")]
    source = frame[indicators].shift(lag)
    returns = source.pct_change(fill_method=None)
    mkt_mean = returns.mean(axis=1)
    mkt_level = source.mean(axis=1)

    # ---------------- market context (shared across indicators) ----------------
    breadth = returns.gt(0).sum(axis=1).div(
        returns.notna().sum(axis=1).replace(0, np.nan)
    )
    mkt = pd.DataFrame(
        {
            "v3_mkt_mean_return": mkt_mean,
            "v3_mkt_breadth": breadth,
            "v3_mkt_breadth_3": breadth.rolling(3, min_periods=2).mean(),
            "v3_mkt_breadth_chg_3": breadth.diff(3),
            "v3_mkt_dispersion": returns.std(axis=1),
        }
    )

    # ---------------- per-indicator wide frames, later melted ----------------
    wide: dict[str, pd.DataFrame] = {}

    # relative weakness vs the whole universe
    for k in (1, 3, 6):
        r = returns.rolling(k, min_periods=max(1, k // 2)).mean()
        wide[f"v3_rel_return_{k}"] = r.sub(mkt_mean, axis=0)
    for k in (3, 6, 12):
        mom = safe_div(source, source.shift(k))
        mkt_mom = safe_div(mkt_level, mkt_level.shift(k))
        wide[f"v3_rel_momentum_{k}"] = mom.sub(mkt_mom, axis=0)

    rank = returns.rank(axis=1, pct=True)
    wide["v3_rel_rank"] = rank
    for k in (1, 2, 3):
        wide[f"v3_rel_rank_change_{k}"] = rank - rank.shift(k)
    wide["v3_rel_rank_slope"] = rank - rank.rolling(3, min_periods=2).mean()
    wide["v3_rel_rank_acceleration"] = rank.diff(1) - rank.diff(2)
    wide["v3_rel_pct_deterioration"] = rank.shift(3) - rank

    # group-relative weakness (X16 has no group -> its group features stay NaN)
    groups = yaml.safe_load(GROUPS.read_text(encoding="utf-8"))["indicator_groups"]
    group_of = {str(i): str(g) for g, members in groups.items() for i in members}

    def group_mean(wide_df: pd.DataFrame) -> pd.DataFrame:
        """Per-column mean over that column's own asset group (X16 -> NaN)."""
        out = pd.DataFrame(
            np.nan, index=wide_df.index, columns=wide_df.columns, dtype=float
        )
        for gname in sorted(set(group_of.values())):
            cols = [c for c in wide_df.columns if group_of.get(c) == gname]
            if cols:
                out[cols] = np.repeat(
                    wide_df[cols].mean(axis=1).to_numpy()[:, None], len(cols), axis=1
                )
        return out

    for k in (1, 3, 6):
        wide[f"v3_grp_vs_group_return_{k}"] = returns.sub(group_mean(returns))
    for k in (3, 6, 12):
        mom = safe_div(source, source.shift(k))
        wide[f"v3_grp_vs_group_momentum_{k}"] = mom.sub(group_mean(mom))

    # rank within the indicator's own group
    grp_rank = pd.DataFrame(index=returns.index, columns=indicators, dtype=float)
    for gname in sorted(set(group_of.values())):
        cols = [c for c in indicators if group_of.get(c) == gname]
        if cols:
            grp_rank[cols] = returns[cols].rank(axis=1, pct=True)
    wide["v3_grp_rank_in_group"] = grp_rank
    wide["v3_grp_rank_change_3"] = grp_rank - grp_rank.shift(3)

    # breakdown structure, vectorised across indicators
    h3 = source.rolling(3, min_periods=2).max()
    h6 = source.rolling(6, min_periods=3).max()
    h12 = source.rolling(12, min_periods=6).max()
    m3 = source.rolling(3, min_periods=2).mean()
    m6 = source.rolling(6, min_periods=3).mean()
    mom3 = safe_div(source, source.shift(3))
    mom6 = safe_div(source, source.shift(6))
    mom12 = safe_div(source, source.shift(12))
    vol12 = returns.rolling(12, min_periods=6).std()
    z = returns.div(vol12.replace(0, np.nan))

    wide["v3_brk_dist_high_3"] = safe_div(source, h3)
    wide["v3_brk_dist_high_6"] = safe_div(source, h6)
    wide["v3_brk_dist_high_12"] = safe_div(source, h12)
    wide["v3_brk_below_mean_3"] = (source < m3).astype(float)
    wide["v3_brk_below_mean_6"] = (source < m6).astype(float)
    wide["v3_brk_return_after_high_6"] = safe_div(source, h6).sub(1.0)
    wide["v3_brk_lower_high"] = (h3 < h6).astype(float)
    wide["v3_brk_momentum_turn_neg"] = ((mom3 < 0) & (mom6 > 0)).astype(float)
    wide["v3_brk_negative_acceleration"] = mom3 - mom6
    wide["v3_brk_reversal_after_strength"] = mom3 * (mom12 > 0).astype(float)
    wide["v3_brk_failed_recovery"] = ((returns > 0) & (safe_div(source, h6) < 1.0)).astype(float)
    wide["v3_brk_z_1"] = z
    wide["v3_brk_z_3"] = z.rolling(3, min_periods=2).mean()
    wide["v3_brk_neg_z_share_3"] = z.lt(0).where(z.notna()).rolling(3, min_periods=2).mean()

    # ---------------- melt to long panel ----------------
    pieces = []
    for name, w in wide.items():
        long = w.stack(dropna=False).rename(name).reset_index()
        long.columns = ["_row", "indicator_id", name]
        long["origin_position"] = frame["position"].reindex(long["_row"]).to_numpy()
        pieces.append(long.drop(columns="_row"))
    panel = pieces[0]
    for p in pieces[1:]:
        panel = panel.merge(p, on=["origin_position", "indicator_id"], how="outer")

    # market context broadcasts to every indicator
    for c in mkt.columns:
        panel[c] = panel["origin_position"].map(mkt[c])

    # interactions: weakness in a strong market is a stronger breakdown signal
    panel["v3_int_weak_in_strong"] = panel["v3_rel_return_1"].mul(
        (panel["v3_mkt_breadth"] > 0.6).astype(float)
    )
    panel["v3_int_weak_in_weak"] = panel["v3_rel_return_1"].mul(
        (panel["v3_mkt_breadth"] < 0.4).astype(float)
    )
    return panel.sort_values(
        ["origin_position", "indicator_id"]
    ).reset_index(drop=True)


def main() -> None:
    started = time.perf_counter()
    config = yaml.safe_load((ROOT / "configs/config.yaml").read_text(encoding="utf-8"))
    settings = yaml.safe_load(
        (ROOT / "configs/directional_downside_model.yaml").read_text(encoding="utf-8")
    )
    lag = int(settings["availability_lag_months"])
    end = int(settings["confirmation_origins"][1])
    # read one extra row so origin `end` has its t-1 observation available
    frame = load_workbook(ROOT / config["data_path"], maximum_position=end + 1)
    targets = build_targets(frame)
    assert_target_alignment(targets, frame)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        feats = build_v3_features(frame, lag)

    base = targets[
        [
            "origin_position",
            "indicator_id",
            "origin_date",
            "target_date",
            "y_true",
            "target_available",
            "value_t",
        ]
    ].copy()
    base["eligible"] = (
        base["target_available"].fillna(False).astype(bool)
        & base["value_t"].notna()
        & base["origin_position"].gt(int(config["minimum_history_months"]))
    )
    panel = base.merge(feats, on=["origin_position", "indicator_id"], how="left")
    panel["down_target"] = 1.0 - panel["y_true"]
    panel["down_eligible"] = (
        panel["eligible"].fillna(False).astype(bool) & panel["down_target"].notna()
    )
    panel["evaluation_origin"] = panel["origin_position"].between(120, end)
    panel = panel[panel["origin_position"] <= end].copy()
    panel["locked_evaluation_read"] = False

    v3_cols = [c for c in panel.columns if c.startswith("v3_")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT, index=False)

    print(f"rows={len(panel)} origins={panel['origin_position'].nunique()}")
    print(f"down_eligible rows={int(panel['down_eligible'].sum())}")
    print(f"down base rate={panel.loc[panel['down_eligible'], 'down_target'].mean():.4f}")
    print(f"n v3 features={len(v3_cols)}")
    cov = panel.loc[panel["down_eligible"], v3_cols].notna().mean().sort_values()
    print(f"\nlowest coverage:\n{cov.head(8).round(3).to_string()}")
    print(f"highest coverage:\n{cov.tail(5).round(3).to_string()}")
    print(f"\nruntime={time.perf_counter() - started:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
