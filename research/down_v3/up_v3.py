"""Up V3: a stronger Up ranking, focused on ranks 11-20 (the marginal slots).

Per the brief, Up V3's purpose is NOT more Up calls. It is better ranking of
the marginal calls - ranks 11-15 and 16-20 - because those are exactly the
slots a Down candidate would displace. If the weak Up slots are stronger,
fewer Down insertions will clear the bar; if they are weaker, the case for
Down insertion improves. Either way the comparison must use the same Up pool.

Feature ideas tested (all causal, built from the same shifted source frame):
  persistence - relative rank persistence / acceleration
  quality     - trend consistency, positive-month share, vol-adjusted momentum
  breadth     - confirmation from market breadth participation
  peer        - peer positive consensus and rank improvement
  group       - strength vs the indicator's own asset group

Screened on Tuning only. Validation is a gate.
Writes: research/down_v3/artifacts/up_v3_predictions.parquet
        research/down_v3/metrics/up_v3.json
"""
from __future__ import annotations

from json_output import strict_json_dumps
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from forecast_select.io import load_workbook  # noqa: E402
from forecast_select.targets import build_targets  # noqa: E402
from forecast_select.validation import causal_training_rows  # noqa: E402

OUT_PRED = ROOT / "research/down_v3/artifacts/up_v3_predictions.parquet"
OUT_MET = ROOT / "research/down_v3/metrics/up_v3.json"
GROUPS = ROOT / "configs/downside_risk_gate.yaml"

WINDOWS = {
    "tuning": (120, 179),
    "validation": (180, 219),
    "confirmation": (220, 266),
}


def build_up_v3_features(frame: pd.DataFrame, lag: int) -> pd.DataFrame:
    indicators = [c for c in frame.columns if c.startswith("X")]
    source = frame[indicators].shift(lag)
    returns = source.pct_change(fill_method=None)

    rank = returns.rank(axis=1, pct=True)
    mkt_mean = returns.mean(axis=1)
    breadth = returns.gt(0).sum(axis=1).div(
        returns.notna().sum(axis=1).replace(0, np.nan)
    )

    groups = yaml.safe_load(GROUPS.read_text(encoding="utf-8"))["indicator_groups"]
    group_of = {str(i): str(g) for g, m in groups.items() for i in m}

    def group_mean(wide: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(np.nan, index=wide.index, columns=wide.columns, dtype=float)
        for g in sorted(set(group_of.values())):
            cols = [c for c in wide.columns if group_of.get(c) == g]
            if cols:
                out[cols] = np.repeat(
                    wide[cols].mean(axis=1).to_numpy()[:, None], len(cols), axis=1
                )
        return out

    wide: dict[str, pd.DataFrame] = {}

    # persistence: does strength last?
    wide["up_rank"] = rank
    wide["up_rank_persistence_3"] = rank.rolling(3, min_periods=2).mean()
    wide["up_rank_change_1"] = rank.diff(1)
    wide["up_rank_acceleration"] = rank.diff(1) - rank.diff(2)

    # trend quality
    pos = returns.gt(0).where(returns.notna())
    for k in (3, 6):
        wide[f"up_pos_share_{k}"] = pos.rolling(k, min_periods=max(1, k // 2)).mean()
    mom3 = source.div(source.shift(3).replace(0, np.nan)).sub(1.0)
    mom6 = source.div(source.shift(6).replace(0, np.nan)).sub(1.0)
    wide["up_momentum_3"] = mom3
    wide["up_momentum_6"] = mom6
    wide["up_vol_adj_momentum_3"] = mom3.div(
        returns.rolling(12, min_periods=6).std().replace(0, np.nan)
    )
    wide["up_trend_slope_3"] = returns.rolling(3, min_periods=2).mean()
    wide["up_trend_consistency_6"] = mom6.mul(mom3.gt(0).astype(float))

    # relative strength vs universe and vs group
    wide["up_rel_return_3"] = returns.rolling(3, min_periods=2).mean().sub(mkt_mean, axis=0)
    wide["up_vs_group_3"] = returns.sub(group_mean(returns))
    wide["up_vs_group_mom_3"] = mom3.sub(group_mean(mom3))
    grp_rank = pd.DataFrame(np.nan, index=returns.index, columns=indicators, dtype=float)
    for g in sorted(set(group_of.values())):
        cols = [c for c in indicators if group_of.get(c) == g]
        if cols:
            grp_rank[cols] = returns[cols].rank(axis=1, pct=True)
    wide["up_group_rank"] = grp_rank

    # breadth confirmation
    wide["up_breadth_confirm"] = (returns.gt(0).astype(float)).mul(
        (breadth > 0.5).astype(float), axis=0
    )

    # peer consensus (contemporaneous cross-sectional, causal at t-1)
    pos_share = returns.gt(0).sum(axis=1).div(returns.notna().sum(axis=1).replace(0, np.nan))
    wide["up_peer_pos_consensus"] = pd.DataFrame(
        np.repeat(pos_share.to_numpy()[:, None], len(indicators), axis=1),
        index=returns.index,
        columns=indicators,
    )

    pieces = []
    for name, w in wide.items():
        long = w.stack(dropna=False).rename(name).reset_index()
        long.columns = ["_row", "indicator_id", name]
        long["origin_position"] = frame["position"].reindex(long["_row"]).to_numpy()
        pieces.append(long.drop(columns="_row"))
    panel = pieces[0]
    for p in pieces[1:]:
        panel = panel.merge(p, on=["origin_position", "indicator_id"], how="outer")
    return panel.sort_values(["origin_position", "indicator_id"]).reset_index(drop=True)


FAMILIES = {
    "persistence": ["up_rank", "up_rank_persistence_3", "up_rank_change_1", "up_rank_acceleration"],
    "quality": [
        "up_pos_share_3", "up_pos_share_6", "up_momentum_3", "up_momentum_6",
        "up_vol_adj_momentum_3", "up_trend_slope_3", "up_trend_consistency_6",
    ],
    "relative": ["up_rel_return_3", "up_vs_group_3", "up_vs_group_mom_3", "up_group_rank"],
    "breadth": ["up_breadth_confirm", "up_peer_pos_consensus"],
}


def build_pipeline(features: list[str], c: float) -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [("imp", SimpleImputer(strategy="median", add_indicator=True)),
                     ("sc", StandardScaler())]
                ),
                features,
            ),
            (
                "indicator",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["indicator_id"],
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", pre),
            ("clf", LogisticRegression(C=c, solver="liblinear", max_iter=1000, random_state=7)),
        ]
    )


def walk_forward(panel: pd.DataFrame, features: list[str], c: float, lag: int):
    pieces = []
    for origin in sorted(panel["origin_position"].unique()):
        if origin < 120:
            continue
        train = causal_training_rows(panel, origin, availability_lag=lag)
        if train.empty or train["y_true"].nunique() < 2:
            continue
        test = panel[panel["origin_position"].eq(origin)]
        m = build_pipeline(features, c)
        m.fit(train[[*features, "indicator_id"]], train["y_true"].astype(int))
        p = np.clip(m.predict_proba(test[[*features, "indicator_id"]])[:, 1], 1e-6, 1 - 1e-6)
        piece = test[["origin_position", "indicator_id", "y_true"]].copy()
        piece["p_up_v3"] = p
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def evaluate(pred: pd.DataFrame, score: str) -> dict:
    out = {}
    for wname, (a, b) in WINDOWS.items():
        w = pred[pred["origin_position"].between(a, b)]
        out[wname] = {"auc": float(roc_auc_score(w["y_true"], w[score]))}
        for n in (5, 10, 15, 20):
            top = (
                w.sort_values(["origin_position", score], ascending=[True, False])
                .groupby("origin_position")
                .head(n)
            )
            out[wname][f"top{n}"] = float(top["y_true"].mean())
        # the marginal slots that a Down candidate would displace
        tail = (
            w.sort_values(["origin_position", score], ascending=[True, False])
            .groupby("origin_position")
            .head(20)
        )
        tail = tail.sort_values(["origin_position", score], ascending=[True, False])
        ranks = tail.groupby("origin_position").cumcount()
        for lo, hi, lbl in ((10, 15, "ranks_11_15"), (15, 20, "ranks_16_20")):
            sub = tail[(ranks >= lo) & (ranks < hi)]
            out[wname][lbl] = float(sub["y_true"].mean()) if len(sub) else float("nan")
    return out


def main() -> None:
    started = time.perf_counter()
    config = yaml.safe_load((ROOT / "configs/config.yaml").read_text(encoding="utf-8"))
    settings = yaml.safe_load(
        (ROOT / "configs/directional_downside_model.yaml").read_text(encoding="utf-8")
    )
    lag = int(settings["availability_lag_months"])
    end = int(settings["confirmation_origins"][1])
    frame = load_workbook(ROOT / config["data_path"], maximum_position=end + 1)
    targets = build_targets(frame)
    feats = build_up_v3_features(frame, lag)
    panel = targets[
        ["origin_position", "indicator_id", "y_true", "target_available", "value_t"]
    ].merge(feats, on=["origin_position", "indicator_id"], how="left")
    panel["eligible"] = (
        panel["target_available"].fillna(False).astype(bool)
        & panel["value_t"].notna()
        & panel["origin_position"].gt(int(config["minimum_history_months"]))
    )
    panel = panel[panel["down_eligible" if "down_eligible" in panel.columns else "eligible"]]
    panel = panel[panel["origin_position"].le(end)]

    # reference: the frozen production Up score on the same rows
    art = pd.read_parquet(ROOT / "artifacts/active/regime_adaptive_predictions.parquet")
    art_ref = art[["origin_position", "indicator_id", "p_up_selection_score"]].rename(
        columns={"p_up_selection_score": "p_up_ref"}
    )
    pred = panel.merge(art_ref, on=["origin_position", "indicator_id"], how="left")
    pred = pred[pred["p_up_ref"].notna()].copy()
    pred["p_up_selection_score"] = pred["p_up_ref"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_c, best_auc = 0.1, -1.0
        allf = [c for f in FAMILIES.values() for c in f]
        for c in (0.01, 0.03, 0.1, 0.3, 1.0):
            p = walk_forward(panel, allf, c, lag)
            t = p[p["origin_position"].between(120, 179)]
            auc = roc_auc_score(t["y_true"], t["p_up_v3"])
            print(f"  C={c}: tuning AUC={auc:.4f}")
            if auc > best_auc:
                best_c, best_auc = c, auc
    print(f"selected C={best_c}")

    metrics: dict[str, object] = {"c": best_c}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, feats in FAMILIES.items():
            p = walk_forward(panel, feats, best_c, lag)
            metrics[f"only_{name}"] = {
                "n_features": len(feats),
                **evaluate(p, "p_up_v3"),
            }
            print(
                f"only_{name:12s} T={metrics[f'only_{name}']['tuning']['auc']:.4f} "
                f"V={metrics[f'only_{name}']['validation']['auc']:.4f}"
            )
        p = walk_forward(panel, allf, best_c, lag)
        metrics["full"] = {"n_features": len(allf), **evaluate(p, "p_up_v3")}
        print(
            f"{'full':12s} T={metrics['full']['tuning']['auc']:.4f} "
            f"V={metrics['full']['validation']['auc']:.4f}"
        )
        p.to_parquet(OUT_PRED, index=False)

    metrics["current_up"] = evaluate(pred, "p_up_selection_score")
    print(
        f"{'current_up':12s} T={metrics['current_up']['tuning']['auc']:.4f} "
        f"V={metrics['current_up']['validation']['auc']:.4f}"
    )

    OUT_MET.parent.mkdir(parents=True, exist_ok=True)
    OUT_MET.write_text(strict_json_dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nsaved {OUT_MET} ({time.perf_counter() - started:.0f}s)")


if __name__ == "__main__":
    main()
