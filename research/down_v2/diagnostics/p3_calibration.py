"""P3: Down model calibration audit + component reliability.

Walk-forward refit identical to the frozen experiment (labels <= t-2),
then reliability / Brier / ECE on each p_down component and the blend.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from forecast_select.directional_downside import (  # noqa: E402
    fit_directional_downside_model,
    predict_directional_downside,
)
from forecast_select.validation import (  # noqa: E402
    assert_target_history_available,
    causal_training_rows,
)

PANEL = ROOT / "research/down_v2/artifacts/down_panel.parquet"
OUT = ROOT / "research/down_v2/artifacts/down_refit_predictions.parquet"
SETTINGS = yaml.safe_load(
    (ROOT / "configs/directional_downside_model.yaml").read_text(encoding="utf-8")
)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        total += m.sum() / len(y) * abs(y[m].mean() - p[m].mean())
    return float(total)


def reliability_table(y: np.ndarray, p: np.ndarray, label: str) -> pd.DataFrame:
    edges = [0.0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.01]
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append(
            {
                "score": label,
                "bucket": f"[{edges[b]:.2f},{edges[b + 1]:.2f})",
                "n": int(m.sum()),
                "mean_pred": float(p[m].mean()),
                "observed": float(y[m].mean()),
                "gap": float(p[m].mean() - y[m].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    started = time.perf_counter()
    panel = pd.read_parquet(PANEL)
    model_settings = SETTINGS["model"]
    lag = int(SETTINGS["availability_lag_months"])
    end = int(SETTINGS["confirmation_origins"][1])
    eligible = panel[panel["down_eligible"] & panel["origin_position"].le(end)].copy()
    tuning_start = int(SETTINGS["tuning_origins"][0])
    confirmation_end = int(SETTINGS["confirmation_origins"][1])

    pieces = []
    for origin in range(tuning_start, confirmation_end + 1):
        train = causal_training_rows(eligible, origin, availability_lag=lag)
        test = eligible[eligible["origin_position"].eq(origin)].copy()
        assert_target_history_available(train, origin, availability_lag=lag)
        model = fit_directional_downside_model(
            train,
            seed=7,
            global_logistic_c=float(model_settings["global_logistic_c"]),
            local_logistic_c=float(model_settings["local_logistic_c"]),
            max_iter=int(model_settings["logistic_max_iter"]),
            minimum_local_rows=int(model_settings["minimum_local_rows"]),
            minimum_local_class_rows=int(model_settings["minimum_local_class_rows"]),
        )
        probs = predict_directional_downside(
            model,
            train,
            test,
            trailing_prior_window=int(model_settings["trailing_prior_window"]),
            minimum_pattern_rows=int(model_settings["minimum_pattern_rows"]),
        )
        piece = test[
            ["origin_position", "indicator_id", "y_true", "down_target"]
        ].merge(probs, on=["origin_position", "indicator_id"], how="left")
        piece["local_model_available"] = piece["local_model_available"].fillna(False)
        pieces.append(piece)
        if origin % 20 == 0:
            print(f"  origin {origin} done ({time.perf_counter() - started:.0f}s)")

    pred = pd.concat(pieces, ignore_index=True)
    pred.to_parquet(OUT, index=False)
    print(f"\nsaved {OUT}  rows={len(pred)}")

    print("\n" + "=" * 78)
    print("P3  Down model calibration  (walk-forward refit, labels <= t-2)")
    print("=" * 78)

    components = {
        "p_down_global": pred["p_down_global"].to_numpy(),
        "p_down_local": pred["p_down_local"].to_numpy(),
        "p_down_pattern": pred["p_down_pattern"].to_numpy(),
        "p_down_indicator_prior": pred["p_down_indicator_prior"].to_numpy(),
    }
    y = pred["down_target"].to_numpy()

    print(f"{'component':24s} {'AUC':>7s} {'Brier':>7s} {'ECE':>7s} {'mean':>7s} {'p95':>7s}")
    for name, p in components.items():
        print(
            f"{name:24s} {roc_auc_score(y, p):7.4f} {brier_score_loss(y, p):7.4f} "
            f"{ece(y, p):7.4f} {p.mean():7.4f} {np.quantile(p, 0.95):7.4f}"
        )
    print()
    print(f"observed Down base rate: {y.mean():.4f}")
    print()

    tables = []
    for name, p in components.items():
        tables.append(reliability_table(y, p, name))
    rel = pd.concat(tables, ignore_index=True)
    print("--- Reliability by bucket (all components) ---")
    print(rel.to_string(index=False))
    print()
    print("KEY QUESTION: at p_down_global in [0.7,0.8), what fraction actually fell?")
    g = pred.copy()
    g["bucket"] = pd.cut(g["p_down_global"], [0.0, 0.5, 0.6, 0.65, 0.7, 0.8, 1.01], right=False)
    print(
        g.groupby("bucket", observed=True)
        .agg(n=("down_target", "size"), mean_pred=("p_down_global", "mean"),
             observed=("down_target", "mean"))
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
