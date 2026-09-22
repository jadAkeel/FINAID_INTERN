"""P6: Down training audit + the class_weight experiment.

Audit: rows per origin, class balance, rows per indicator, local-model
availability and sample sizes, coefficient stability.

Experiment: A) class_weight='balanced' (production) vs B) no class weighting
vs C) balanced + post-hoc Platt calibration. All three walk-forward, all
evaluated on AUC / Brier / calibration / top-N Down precision.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from json_output import strict_json_dumps  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))

from forecast_select.directional_downside import DIRECTIONAL_DOWNSIDE_FEATURES  # noqa: E402
from forecast_select.validation import (  # noqa: E402
    assert_target_history_available,
    causal_training_rows,
)

PANEL = ROOT / "research/down_v2/artifacts/down_panel.parquet"


def pipeline(c: float, class_weight, max_iter: int, seed: int):
    numeric = [c for c in DIRECTIONAL_DOWNSIDE_FEATURES]
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
                    class_weight=class_weight,
                    max_iter=max_iter,
                    random_state=seed,
                ),
            ),
        ]
    )


def platt(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(x, y)
    return lr.predict_proba(x)[:, 1]


def topn(frame: pd.DataFrame, score: str, n: int) -> dict[str, float]:
    top = frame.sort_values(
        ["origin_position", score], ascending=[True, False]
    ).groupby("origin_position").head(n)
    y = top["down_target"].to_numpy()
    return {
        "n": len(top),
        "precision": float(y.mean()),
        "months": int(top["origin_position"].nunique()),
    }


def run(panel: pd.DataFrame, lag: int, c: float, class_weight):
    pieces = []
    for origin in sorted(panel["origin_position"].unique()):
        train = causal_training_rows(panel, origin, availability_lag=lag)
        if train.empty or train["down_target"].nunique() < 2:
            continue
        test = panel[panel["origin_position"].eq(origin)].copy()
        assert_target_history_available(train, origin, availability_lag=lag)
        model = pipeline(c, class_weight, 500, 7)
        model.fit(
            train[[*DIRECTIONAL_DOWNSIDE_FEATURES, "indicator_id"]],
            train["down_target"].astype(int),
        )
        p = np.clip(
            model.predict_proba(
                test[[*DIRECTIONAL_DOWNSIDE_FEATURES, "indicator_id"]]
            )[:, 1],
            1e-6,
            1 - 1e-6,
        )
        piece = test[["origin_position", "indicator_id", "down_target"]].copy()
        piece["p"] = p
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def main() -> None:
    started = time.perf_counter()
    settings = yaml.safe_load(
        (ROOT / "configs/directional_downside_model.yaml").read_text(encoding="utf-8")
    )
    lag = int(settings["availability_lag_months"])
    c = float(settings["model"]["global_logistic_c"])
    panel = pd.read_parquet(PANEL)
    panel = panel[panel["down_eligible"]].copy()
    end = int(settings["confirmation_origins"][1])
    panel = panel[panel["origin_position"].le(end)]

    print("=" * 78)
    print("P6a  Training audit")
    print("=" * 78)
    ev = panel[panel["origin_position"].between(120, end)]
    print(f"evaluation rows: {len(ev)}, origins: {ev['origin_position'].nunique()}")
    rows_per_origin = ev.groupby("origin_position").size()
    print(f"rows/origin: min={rows_per_origin.min()} mean={rows_per_origin.mean():.1f} max={rows_per_origin.max()}")
    print(f"down base rate (evaluation): {ev['down_target'].mean():.4f}")
    print()
    print("training-set size at each origin (labels <= t-2):")
    sizes = []
    for origin in (120, 150, 180, 220, 266):
        train = causal_training_rows(ev, origin, availability_lag=lag)
        sizes.append((origin, len(train), float(train["down_target"].mean())))
        print(f"  origin {origin}: rows={len(train):5d} down_rate={train['down_target'].mean():.4f}")
    print()
    print("rows per indicator within the evaluation window:")
    rpi = ev.groupby("indicator_id").size()
    print(f"  min={rpi.min()} median={int(rpi.median())} max={rpi.max()}")
    print(f"  indicators with <60 rows: {int((rpi < 60).sum())}")
    print()
    print("local-model eligibility (minimum_local_rows=48, minimum_local_class_rows=8):")
    for origin in (120, 150, 180, 220, 266):
        train = causal_training_rows(ev, origin, availability_lag=lag)
        counts = train.groupby("indicator_id")["down_target"].agg(["size", lambda s: s.eq(1).sum()])
        counts.columns = ["rows", "down_rows"]
        ok = counts[(counts["rows"] >= 48) & (counts["down_rows"] >= 8)]
        print(f"  origin {origin}: {len(ok):2d} of {len(counts):2d} indicators qualify for a local model")
    print()

    print("=" * 78)
    print("P6b  class_weight experiment")
    print("=" * 78)
    results = {}
    for label, cw in [("A_balanced", "balanced"), ("B_none", None)]:
        pred = run(panel, lag, c, cw)
        pred.to_parquet(
            ROOT / f"research/down_v2/artifacts/p6_{label}.parquet", index=False
        )
        metrics = {}
        for wname, (a, b) in {
            "tuning": (120, 179),
            "validation": (180, 219),
            "confirmation": (220, 266),
        }.items():
            w = pred[pred["origin_position"].between(a, b)]
            y, p = w["down_target"].to_numpy(), w["p"].to_numpy()
            metrics[wname] = {
                "auc": float(roc_auc_score(y, p)),
                "brier": float(brier_score_loss(y, p)),
                "mean_pred": float(p.mean()),
                "obs_rate": float(y.mean()),
                **{f"top{n}": topn(w, "p", n) for n in (3, 5, 10)},
            }
        results[label] = metrics
        print(f"{label}: done ({time.perf_counter() - started:.0f}s)")

    # C: balanced + Platt calibration fitted on tuning only
    a_bal = pd.read_parquet(ROOT / "research/down_v2/artifacts/p6_A_balanced.parquet")
    tune = a_bal[a_bal["origin_position"].between(120, 179)]
    x = np.log(np.clip(tune["p"].to_numpy(), 1e-6, 1 - 1e-6) / (1 - np.clip(tune["p"].to_numpy(), 1e-6, 1 - 1e-6))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(x, tune["down_target"].astype(int))

    def apply_platt(s: pd.Series) -> np.ndarray:
        v = np.asarray(s, dtype=float)
        xx = np.log(np.clip(v, 1e-6, 1 - 1e-6) / (1 - np.clip(v, 1e-6, 1 - 1e-6))).reshape(-1, 1)
        return np.clip(lr.predict_proba(xx)[:, 1], 1e-6, 1 - 1e-6)

    a_bal["p_cal"] = apply_platt(a_bal["p"])
    a_bal.to_parquet(ROOT / "research/down_v2/artifacts/p6_C_balanced_calibrated.parquet", index=False)
    metrics = {}
    for wname, (a, b) in {
        "tuning": (120, 179),
        "validation": (180, 219),
        "confirmation": (220, 266),
    }.items():
        w = a_bal[a_bal["origin_position"].between(a, b)]
        y, p = w["down_target"].to_numpy(), w["p_cal"].to_numpy()
        metrics[wname] = {
            "auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "mean_pred": float(p.mean()),
            "obs_rate": float(y.mean()),
            **{f"top{n}": topn(w, "p_cal", n) for n in (3, 5, 10)},
        }
    results["C_balanced_calibrated"] = metrics

    print("\n--- class_weight comparison ---")
    for label, m in results.items():
        print(f"\n{label}")
        for wname, wm in m.items():
            print(
                f"  {wname:12s} AUC={wm['auc']:.4f} Brier={wm['brier']:.4f} "
                f"mean_pred={wm['mean_pred']:.4f} obs={wm['obs_rate']:.4f} "
                f"T3={wm['top3']['precision']:.4f} T5={wm['top5']['precision']:.4f} "
                f"T10={wm['top10']['precision']:.4f}"
            )

        (ROOT / "research/down_v2/metrics/p6_class_weight.json").write_text(
        strict_json_dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"\nsaved research/down_v2/metrics/p6_class_weight.json ({time.perf_counter() - started:.0f}s)")


if __name__ == "__main__":
    main()
