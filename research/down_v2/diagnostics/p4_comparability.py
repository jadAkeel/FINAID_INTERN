"""P4: Are p_up and p_down comparable, and what unifies them best?

Loads the frozen active artifact (p_up family) and the P3 walk-forward
down refit (p_down family), joins them, and evaluates five candidate
unified scores on direction-matched correctness.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
ACTIVE = ROOT / "artifacts/active/regime_adaptive_predictions.parquet"
DOWN_REFIT = ROOT / "research/down_v2/artifacts/down_refit_predictions.parquet"

WINDOWS = {
    "tuning": (120, 179),
    "validation": (180, 219),
    "confirmation": (220, 266),
}


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    if len(np.unique(edges)) < bins + 1:
        edges = np.linspace(p.min() - 1e-9, p.max() + 1e-9, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        total += m.sum() / len(y) * abs(y[m].mean() - p[m].mean())
    return float(total)


def fit_calibrator(p: np.ndarray, y: np.ndarray, kind: str):
    """Fit a calibrator on the given (score, outcome) pairs only."""
    x = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4))).reshape(-1, 1)
    if kind == "platt":
        model = LogisticRegression(C=1e6, max_iter=1000)
        model.fit(x, y)
        return lambda s: model.predict_proba(
            np.log(np.clip(s, 1e-4, 1 - 1e-4) / (1 - np.clip(s, 1e-4, 1 - 1e-4))).reshape(-1, 1)
        )[:, 1]
    if kind == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p, y)
        return lambda s: np.clip(iso.predict(s), 1e-4, 1 - 1e-4)
    raise ValueError(kind)


def main() -> None:
    active = pd.read_parquet(ACTIVE)
    down = pd.read_parquet(DOWN_REFIT)

    keep = [
        "origin_position",
        "indicator_id",
        "y_true",
        "p_up_base",
        "p_up_selection_score",
        "level_c_ready",
        "regime_stress",
    ]
    up = active[keep].copy()
    merged = up.merge(
        down[
            [
                "origin_position",
                "indicator_id",
                "down_target",
                "p_down_global",
                "p_down_local",
                "p_down_pattern",
                "p_down_indicator_prior",
            ]
        ],
        on=["origin_position", "indicator_id"],
        how="inner",
        validate="one_to_one",
    )
    merged["p_down_blend"] = merged[
        ["p_down_global", "p_down_local", "p_down_pattern", "p_down_indicator_prior"]
    ].mean(axis=1)
    print(f"joined rows: {len(merged)}  origins: {merged['origin_position'].nunique()}")

    r = merged[
        merged["level_c_ready"].fillna(False).astype(bool)
        & merged["y_true"].notna()
    ].copy()
    r["up_correct"] = r["y_true"].eq(1.0)
    r["down_correct"] = r["y_true"].eq(0.0)

    print("\n" + "=" * 78)
    print("P4  Comparability of p_up and p_down")
    print("=" * 78)

    print("\n--- (1) RAW probabilities: scale and discrimination ---")
    print(f"{'score':22s} {'mean':>7s} {'std':>7s} {'min':>7s} {'max':>7s} {'AUC_vs_own':>11s} {'Brier':>7s}")
    for col, tgt in [
        ("p_up_base", "up_correct"),
        ("p_up_selection_score", "up_correct"),
        ("p_down_blend", "down_correct"),
        ("p_down_global", "down_correct"),
    ]:
        s, y = r[col].to_numpy(), r[tgt].to_numpy()
        print(
            f"{col:22s} {s.mean():7.4f} {s.std():7.4f} {s.min():7.4f} {s.max():7.4f} "
            f"{roc_auc_score(y, s):11.4f} {brier_score_loss(y, s):7.4f}"
        )
    print("\n  -> the two families occupy different ranges and have different")
    print("     discrimination; raw max() is therefore not meaningful.")

    print("\n--- (2) Cross-target check: does each score predict the OTHER direction? ---")
    for col in ["p_up_base", "p_down_blend"]:
        auc_up = roc_auc_score(r["up_correct"], r[col])
        auc_dn = roc_auc_score(r["down_correct"], r[col])
        print(f"  {col:18s} AUC vs Up={auc_up:.4f}  AUC vs Down={auc_dn:.4f}")

    # --- calibrate each direction on TUNING ONLY, evaluate elsewhere ---
    tune = r[r["origin_position"].between(120, 179)]
    val = r[r["origin_position"].between(180, 219)]
    conf = r[r["origin_position"].between(220, 266)]

    print("\n--- (3) Leakage-safe calibration (fit on Tuning 120-179 only) ---")
    calibrators = {}
    for name, col, tgt in [
        ("up", "p_up_base", "up_correct"),
        ("down", "p_down_blend", "down_correct"),
        ("down_global", "p_down_global", "down_correct"),
    ]:
        for kind in ("platt", "isotonic"):
            calibrators[f"{name}_{kind}"] = fit_calibrator(
                tune[col].to_numpy(), tune[tgt].to_numpy(), kind
            )

    print(f"\n{'score':24s} {'Brier':>7s} {'ECE':>7s} {'AUC':>7s} {'mean':>7s} {'p95':>7s} {'window':>12s}")
    for wname, w in [("tuning", tune), ("validation", val), ("confirmation", conf)]:
        for col, tgt in [
            ("p_up_base", "up_correct"),
            ("p_down_blend", "down_correct"),
        ]:
            s, y = w[col].to_numpy(), w[tgt].to_numpy()
            print(
                f"{col:24s} {brier_score_loss(y, s):7.4f} {ece(y, s):7.4f} "
                f"{roc_auc_score(y, s):7.4f} {s.mean():7.4f} {np.quantile(s, .95):7.4f} {wname:>12s}"
            )
        for key in ["up_platt", "up_isotonic", "down_platt", "down_isotonic"]:
            col, tgt = (
                ("p_up_base", "up_correct")
                if key.startswith("up")
                else ("p_down_blend", "down_correct")
            )
            s = calibrators[key](w[col].to_numpy())
            y = w[tgt].to_numpy()
            print(
                f"{key:24s} {brier_score_loss(y, s):7.4f} {ece(y, s):7.4f} "
                f"{roc_auc_score(y, s):7.4f} {s.mean():7.4f} {np.quantile(s, .95):7.4f} {wname:>12s}"
            )
        print()

    # --- unified candidate scores ---
    print("--- (4) UNIFIED score candidates (direction-matched correctness) ---")
    r["correct"] = np.where(r["p_down_blend"] >= r["p_up_base"], r["down_correct"], r["up_correct"])
    r["direction"] = np.where(r["p_down_blend"] >= r["p_up_base"], "Down", "Up")

    r["c_raw_max"] = np.maximum(r["p_up_base"], r["p_down_blend"])
    r["c_cal_max"] = np.maximum(
        calibrators["up_platt"](r["p_up_base"].to_numpy()),
        calibrators["down_platt"](r["p_down_blend"].to_numpy()),
    )
    for origin, g in r.groupby("origin_position"):
        r.loc[g.index, "up_pct"] = g["p_up_base"].rank(pct=True)
        r.loc[g.index, "dn_pct"] = g["p_down_blend"].rank(pct=True)
    r["c_pct_max"] = np.maximum(r["up_pct"], r["dn_pct"])
    r["c_margin"] = (r["p_down_blend"] - r["p_up_base"])
    r["c_cal_margin"] = (
        calibrators["down_platt"](r["p_down_blend"].to_numpy())
        - calibrators["up_platt"](r["p_up_base"].to_numpy())
    )
    r["c_expected"] = r["p_up_base"] * r["up_correct"].astype(float) + r["p_down_blend"] * (
        r["down_correct"].astype(float)
    )

    print(f"{'unified score':18s} {'AUC':>7s} {'AP':>7s} {'Brier':>7s}")
    for col in [
        "c_raw_max",
        "c_cal_max",
        "c_pct_max",
        "c_margin",
        "c_cal_margin",
        "c_expected",
    ]:
        y = r["correct"].to_numpy()
        s = r[col].to_numpy()
        if col in ("c_margin", "c_cal_margin"):
            # margin is not a probability; AUC/AP only
            print(f"{col:18s} {roc_auc_score(y, s):7.4f} {average_precision_score(y, s):7.4f} {'--':>7s}")
        else:
            print(f"{col:18s} {roc_auc_score(y, s):7.4f} {average_precision_score(y, s):7.4f} {brier_score_loss(y, s):7.4f}")

    print("\n--- (5) Per-window AUC of the unified candidates ---")
    print(f"{'score':18s} {'tuning':>8s} {'validation':>11s} {'confirmation':>13s}")
    for col in ["c_raw_max", "c_cal_max", "c_pct_max", "c_margin", "c_cal_margin"]:
        row = f"{col:18s}"
        for _, (a, b) in WINDOWS.items():
            w = r[r["origin_position"].between(a, b)]
            row += f" {roc_auc_score(w['correct'], w[col]):8.4f}"
        print(row)

    print("\n--- (6) What the Down side costs on a common scale ---")
    # percentile rank of p_down within its own origin vs p_up within its own origin
    for origin, g in r.groupby("origin_position"):
        r.loc[g.index, "up_pct_rank"] = g["p_up_base"].rank(pct=True, method="first")
        r.loc[g.index, "dn_pct_rank"] = g["p_down_blend"].rank(pct=True, method="first")
    top_up = r.sort_values(["origin_position", "up_pct_rank"], ascending=[True, False]).groupby("origin_position").head(15)
    top_dn = r.sort_values(["origin_position", "dn_pct_rank"], ascending=[True, False]).groupby("origin_position").head(15)
    print(f"  monthly top-15 by p_up : hit rate {top_up['up_correct'].mean():.4f}")
    print(f"  monthly top-15 by p_down: hit rate {top_dn['down_correct'].mean():.4f}")
    print("  -> this is the honest apples-to-apples comparison: at equal depth,")
    print("     how often is each model's chosen direction correct?")

    r.to_parquet(ROOT / "research/down_v2/artifacts/p4_unified_scores.parquet", index=False)
    print("\nsaved research/down_v2/artifacts/p4_unified_scores.parquet")


if __name__ == "__main__":
    main()
