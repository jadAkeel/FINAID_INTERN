"""P9: Regime dependence of the Down model.

Does the Down signal become usable in specific conditions only? Splits by
regime label, stress tercile, market breadth, dispersion, and shock state.
Uses the P3 walk-forward refit plus regime columns from the active artifact.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from json_output import strict_json_dumps  # noqa: E402
REFIT = ROOT / "research/down_v2/artifacts/down_refit_predictions.parquet"
ACTIVE = ROOT / "artifacts/active/regime_adaptive_predictions.parquet"
OUT = ROOT / "research/down_v2/metrics/p9_regime.json"


def regime_split(frame: pd.DataFrame, col: str, labels: dict) -> dict[str, pd.DataFrame]:
    out = {}
    for key, mask in labels.items():
        out[key] = frame[mask]
    return out


def summarize(frame: pd.DataFrame, score: str) -> dict[str, float | int]:
    if len(frame) < 30:
        return {"n": int(len(frame)), "auc": float("nan"), "precision_top5": float("nan")}
    auc = float(roc_auc_score(frame["down_target"], frame[score]))
    top = frame.sort_values(
        ["origin_position", score], ascending=[True, False]
    ).groupby("origin_position").head(5)
    return {
        "n": int(len(frame)),
        "months": int(frame["origin_position"].nunique()),
        "auc": auc,
        "base_rate": float(frame["down_target"].mean()),
        "precision_top5": float(top["down_target"].mean()),
    }


def main() -> None:
    pred = pd.read_parquet(REFIT)
    pred["down_target"] = pred["down_target"].astype(float)
    pred["p_down_blend"] = pred[
        ["p_down_global", "p_down_local", "p_down_pattern", "p_down_indicator_prior"]
    ].mean(axis=1)

    act = pd.read_parquet(ACTIVE)
    regime = act[
        [
            "origin_position",
            "indicator_id",
            "regime_stress",
            "regime_label",
            "market_breadth",
            "market_dispersion",
            "previous_shock",
            "previous_shock_share",
            "down_market_breadth",
        ]
    ].drop_duplicates(subset=["origin_position", "indicator_id"])
    m = pred.merge(regime, on=["origin_position", "indicator_id"], how="left")
    print(f"merged rows: {len(m)}")

    print("=" * 78)
    print("P9  Down model regime dependence")
    print("=" * 78)

    results = {}
    score = "p_down_blend"

    print("\n--- By regime_label ---")
    for label, g in m.groupby("regime_label"):
        s = summarize(g, score)
        results[f"label_{label}"] = s
        print(f"  {label:10s} n={s['n']:5d} months={s['months']:3d} "
              f"auc={s['auc']:.4f} base={s['base_rate']:.4f} top5={s['precision_top5']:.4f}")

    print("\n--- By stress tercile ---")
    m["stress_tercile"] = pd.qcut(
        m["regime_stress"], 3, labels=["low_stress", "mid_stress", "high_stress"]
    )
    for label, g in m.groupby("stress_tercile", observed=True):
        s = summarize(g, score)
        results[f"stress_{label}"] = s
        print(f"  {str(label):12s} n={s['n']:5d} auc={s['auc']:.4f} "
              f"base={s['base_rate']:.4f} top5={s['precision_top5']:.4f}")

    print("\n--- By lagged market breadth (down_market_breadth) ---")
    m["breadth_bucket"] = pd.qcut(
        m["down_market_breadth"], 4, labels=["q1_low", "q2", "q3", "q4_high"]
    )
    for label, g in m.groupby("breadth_bucket", observed=True):
        s = summarize(g, score)
        results[f"breadth_{label}"] = s
        print(f"  {str(label):10s} n={s['n']:5d} auc={s['auc']:.4f} "
              f"base={s['base_rate']:.4f} top5={s['precision_top5']:.4f}")

    print("\n--- By dispersion ---")
    m["disp_bucket"] = pd.qcut(
        m["market_dispersion"], 4, labels=["q1_low", "q2", "q3", "q4_high"]
    )
    for label, g in m.groupby("disp_bucket", observed=True):
        s = summarize(g, score)
        results[f"dispersion_{label}"] = s
        print(f"  {str(label):10s} n={s['n']:5d} auc={s['auc']:.4f} "
              f"base={s['base_rate']:.4f} top5={s['precision_top5']:.4f}")

    print("\n--- By previous shock exposure ---")
    m["shock_bucket"] = pd.cut(
        m["previous_shock"], [-0.01, 0.001, 0.1, 1.01], labels=["no_shock", "low", "high"]
    )
    for label, g in m.groupby("shock_bucket", observed=True):
        s = summarize(g, score)
        results[f"shock_{label}"] = s
        print(f"  {str(label):10s} n={s['n']:5d} auc={s['auc']:.4f} "
              f"base={s['base_rate']:.4f} top5={s['precision_top5']:.4f}")

    print("\n--- Best regime for Down, per window (validation is the gate) ---")
    for wname, (a, b) in {
        "tuning": (120, 179),
        "validation": (180, 219),
        "confirmation": (220, 266),
    }.items():
        w = m[m["origin_position"].between(a, b)]
        best = None
        for label, g in w.groupby("regime_label"):
            if len(g) < 30:
                continue
            auc = float(roc_auc_score(g["down_target"], g[score]))
            if best is None or auc > best[1]:
                best = (label, auc, len(g))
        print(f"  {wname:12s} best regime={best[0]:10s} auc={best[1]:.4f} n={best[2]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(strict_json_dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
