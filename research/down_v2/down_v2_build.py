"""Down V2: a leaner Down model built from the ablation evidence.

Evidence driving this design (all from research/down_v2):
  P5b: every single feature family BEATS the full 28-feature model on
       validation AUC. The full model is V=0.5015; only_volatility is 0.5370.
       Complexity is actively harmful.
  P5a: the local per-indicator model is noise (AUC 0.5015-0.5055, below the
       base rate's information). The indicator prior alone is the best
       component. Blend weights with local/pattern hurt out of sample.
  P6:  class_weight='balanced' does not change AUC but breaks calibration
       (mean 0.4923 vs observed 0.4302). Removing it fixes the mean without
       costing AUC.
  P3:  raw p_down is badly overconfident (p=0.90 -> observed 0.52).
  P8:  strict thresholds tune well but collapse on validation, so the
       threshold must be chosen on Tuning and treated as fragile.
  P9:  the Down signal is strongest under stress and low dispersion.

V2 therefore: small feature set, no local model, no class weighting,
leakage-safe calibration, and a strict threshold screened on Tuning only.
"""
from __future__ import annotations

from json_output import strict_json_dumps
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from forecast_select.validation import (  # noqa: E402
    assert_target_history_available,
    causal_training_rows,
)

PANEL = ROOT / "research/down_v2/artifacts/down_panel.parquet"
OUT_PRED = ROOT / "research/down_v2/artifacts/down_v2_predictions.parquet"
OUT_METRICS = ROOT / "research/down_v2/metrics/down_v2_standalone.json"

# Lean feature set: the two families with the best validation AUC plus a
# mild drawdown/distance term (also strong standalone). No lead-peer,
# no market-regime block, no exhaustion block, no local model.
V2_FEATURES = [
    "down_volatility_3",
    "down_volatility_12",
    "down_volatility_compression",
    "down_return_1",
    "down_return_lag_1",
    "down_return_lag_2",
    "down_drawdown_12",
    "down_distance_mean_12",
    "down_momentum_3",
    "down_negative_share_3",
]

WINDOWS = {
    "tuning": (120, 179),
    "validation": (180, 219),
    "confirmation": (220, 266),
}


def build_pipeline(c: float, seed: int) -> Pipeline:
    numeric = [f for f in V2_FEATURES]
    pre = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "indicator",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["indicator_id"],
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", pre),
            (
                "classifier",
                LogisticRegression(
                    C=c,
                    penalty="l2",
                    solver="liblinear",
                    class_weight=None,
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )


def build_prior(train: pd.DataFrame, window: int) -> pd.Series:
    """Trailing per-indicator Down base rate (labels <= t-2 only)."""
    ordered = train.sort_values(["indicator_id", "origin_position"])
    recent = ordered.groupby("indicator_id", sort=False).tail(window)
    return recent.groupby("indicator_id")["down_target"].mean()


def run_walk_forward(panel: pd.DataFrame, lag: int, c: float) -> pd.DataFrame:
    pieces = []
    for origin in sorted(panel["origin_position"].unique()):
        train = causal_training_rows(panel, origin, availability_lag=lag)
        if train.empty or train["down_target"].nunique() < 2:
            continue
        test = panel[panel["origin_position"].eq(origin)].copy()
        assert_target_history_available(train, origin, availability_lag=lag)
        model = build_pipeline(c, seed=7)
        model.fit(train[[*V2_FEATURES, "indicator_id"]], train["down_target"].astype(int))
        raw = np.clip(
            model.predict_proba(test[[*V2_FEATURES, "indicator_id"]])[:, 1], 1e-6, 1 - 1e-6
        )
        piece = test[["origin_position", "indicator_id", "down_target"]].copy()
        piece["p_raw"] = raw
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def topn(frame: pd.DataFrame, score: str, n: int) -> dict[str, float | int]:
    top = frame.sort_values(
        ["origin_position", score], ascending=[True, False]
    ).groupby("origin_position").head(n)
    y = top["down_target"].to_numpy()
    return {
        "calls": int(len(top)),
        "months": int(top["origin_position"].nunique()),
        "hits": int(y.sum()),
        "precision": float(y.mean()),
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


def evaluate(frame: pd.DataFrame, score: str) -> dict[str, float | int | dict]:
    out: dict[str, float | int | dict] = {}
    for wname, (a, b) in WINDOWS.items():
        w = frame[frame["origin_position"].between(a, b)]
        y, p = w["down_target"].to_numpy(), w[score].to_numpy()
        out[wname] = {
            "n": int(len(w)),
            "auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "ece": ece(y, p),
            "mean_pred": float(p.mean()),
            "observed": float(y.mean()),
            **{f"top{n}": topn(w, score, n) for n in (1, 3, 5, 10)},
        }
    return out


def main() -> None:
    started = time.perf_counter()
    settings = yaml.safe_load(
        (ROOT / "configs/directional_downside_model.yaml").read_text(encoding="utf-8")
    )
    lag = int(settings["availability_lag_months"])
    panel = pd.read_parquet(PANEL)
    panel = panel[panel["down_eligible"]].copy()
    end = int(settings["confirmation_origins"][1])
    panel = panel[panel["origin_position"].le(end)]

    # --- C grid screened on Tuning only ---
    print("Screening regularization C on Tuning 120-179 (validation is a gate)...")
    best_c, best_auc = None, -1.0
    for c in (0.01, 0.03, 0.05, 0.1, 0.3, 1.0):
        pred = run_walk_forward(panel, lag, c)
        tune = pred[pred["origin_position"].between(120, 179)]
        auc = float(roc_auc_score(tune["down_target"], tune["p_raw"]))
        print(f"  C={c:<5} tuning AUC={auc:.4f}")
        if auc > best_auc:
            best_c, best_auc = c, auc
    print(f"selected C={best_c} (tuning AUC {best_auc:.4f})")

    pred = run_walk_forward(panel, lag, best_c)

    # --- calibration: fit on Tuning only, apply forward ---
    tune = pred[pred["origin_position"].between(120, 179)]
    yt = tune["down_target"].to_numpy()
    pr = tune["p_raw"].to_numpy()
    xt = np.log(
        np.clip(pr, 1e-6, 1 - 1e-6)
        / (1 - np.clip(pr, 1e-6, 1 - 1e-6))
    ).reshape(-1, 1)
    platt = LogisticRegression(C=1e6, max_iter=1000)
    platt.fit(xt, yt.astype(int))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(pr, yt)

    def apply_platt(s: np.ndarray) -> np.ndarray:
        x = np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6))).reshape(-1, 1)
        return np.clip(platt.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)

    pred["p_platt"] = apply_platt(pred["p_raw"].to_numpy())
    pred["p_isotonic"] = np.clip(iso.predict(pred["p_raw"].to_numpy()), 1e-6, 1 - 1e-6)

    # --- threshold screen on Tuning only ---
    print("\nThreshold screen on Tuning (p_platt):")
    best_thr, best_score = None, None
    for thr in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        w = tune.copy()
        w["s"] = pred.loc[w.index, "p_platt"]
        sel = w[w["s"].ge(thr)]
        prec = float(sel["down_target"].mean()) if len(sel) else float("nan")
        print(f"  thr={thr:.2f}: tuning calls={len(sel):4d} precision={prec:.4f}")
        if len(sel) >= 20 and (best_score is None or prec > best_score):
            best_thr, best_score = thr, prec
    print(f"selected threshold={best_thr} (tuning precision {best_score:.4f})")
    pred["threshold"] = best_thr
    pred["down_call"] = pred["p_platt"].ge(best_thr)

    pred.to_parquet(OUT_PRED, index=False)

    metrics = {
        "features": V2_FEATURES,
        "regularization_c": best_c,
        "threshold": best_thr,
        "n_features": len(V2_FEATURES),
        "raw": evaluate(pred, "p_raw"),
        "platt": evaluate(pred, "p_platt"),
        "isotonic": evaluate(pred, "p_isotonic"),
        "calls_at_threshold": {
            wname: {
                "calls": int(w["down_call"].sum()),
                "hits": int(w.loc[w["down_call"], "down_target"].sum()),
                "precision": float(w.loc[w["down_call"], "down_target"].mean())
                if w["down_call"].any()
                else float("nan"),
            }
            for wname, w in {
                n: pred[pred["origin_position"].between(a, b)]
                for n, (a, b) in WINDOWS.items()
            }.items()
        },
    }
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    OUT_METRICS.write_text(strict_json_dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    print("\n" + "=" * 78)
    print("Down V2 standalone results")
    print("=" * 78)
    for variant in ("raw", "platt", "isotonic"):
        print(f"\n--- {variant} ---")
        for wname, wm in metrics[variant].items():
            print(
                f"  {wname:12s} AUC={wm['auc']:.4f} Brier={wm['brier']:.4f} "
                f"ECE={wm['ece']:.4f} mean={wm['mean_pred']:.4f} obs={wm['observed']:.4f}"
            )
            for n in (1, 3, 5, 10):
                d = wm[f"top{n}"]
                print(f"      top{n:<2d} calls={d['calls']:4d} prec={d['precision']:.4f}")
    print("\n--- Down calls at the selected threshold ---")
    for wname, wm in metrics["calls_at_threshold"].items():
        print(f"  {wname:12s} calls={wm['calls']:4d} hits={wm['hits']:4d} precision={wm['precision']:.4f}")
    print(f"\nsaved {OUT_PRED}\nsaved {OUT_METRICS} ({time.perf_counter() - started:.0f}s)")


if __name__ == "__main__":
    main()
