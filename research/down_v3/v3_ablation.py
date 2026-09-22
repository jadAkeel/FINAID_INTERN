"""V3 family-by-family ablation with full walk-forward refit.

Mirrors the P5b protocol that produced the V2 evidence: every config is a
complete 147-origin walk-forward, parameters screened on Tuning only,
Validation is a gate, Confirmation descriptive.

Configs:
  only_<family>          - that family alone (+ indicator identity)
  drop_<family>          - everything except that family
  best2 / best3          - greedily combined from the only_ results
  full                   - all 42 V3 features
  v2_reference           - the frozen V2 10-feature model, same protocol
"""
from __future__ import annotations

from json_output import strict_json_dumps
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from forecast_select.validation import causal_training_rows  # noqa: E402

PANEL = ROOT / "research/down_v3/artifacts/v3_panel.parquet"
V2_PANEL = ROOT / "research/down_v2/artifacts/down_panel.parquet"
OUT = ROOT / "research/down_v3/metrics/v3_ablation.json"
PRED_DIR = ROOT / "research/down_v3/artifacts"

WINDOWS = {
    "tuning": (120, 179),
    "validation": (180, 219),
    "confirmation": (220, 266),
}

FAMILIES = {
    "rel": [c for c in ()],
    "group": [],
    "break": [],
    "mkt": [],
    "int": [],
}

V2_FEATURES = [
    "down_volatility_3", "down_volatility_12", "down_volatility_compression",
    "down_return_1", "down_return_lag_1", "down_return_lag_2",
    "down_drawdown_12", "down_distance_mean_12",
    "down_momentum_3", "down_negative_share_3",
]


def family_columns(panel: pd.DataFrame) -> dict[str, list[str]]:
    cols = [c for c in panel.columns if c.startswith("v3_")]
    fam = {
        "rel": [c for c in cols if "_rel_" in c],
        "group": [c for c in cols if "_grp_" in c],
        "break": [c for c in cols if "_brk_" in c],
        "mkt": [c for c in cols if "_mkt_" in c],
        "int": [c for c in cols if "_int_" in c],
    }
    return fam


def build_pipeline(features: list[str], c: float) -> Pipeline:
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
                features,
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
                    random_state=7,
                ),
            ),
        ]
    )


def run_walk_forward(panel: pd.DataFrame, features: list[str], c: float, lag: int):
    pieces = []
    for origin in sorted(panel["origin_position"].unique()):
        if origin < 120:
            continue
        train = causal_training_rows(panel, origin, availability_lag=lag)
        if train.empty or train["down_target"].nunique() < 2:
            continue
        test = panel[panel["origin_position"].eq(origin)]
        model = build_pipeline(features, c)
        model.fit(train[[*features, "indicator_id"]], train["down_target"].astype(int))
        raw = np.clip(
            model.predict_proba(test[[*features, "indicator_id"]])[:, 1], 1e-6, 1 - 1e-6
        )
        piece = test[["origin_position", "indicator_id", "down_target"]].copy()
        piece["p_raw"] = raw
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def topn(frame: pd.DataFrame, score: str, n: int) -> float:
    top = (
        frame.sort_values(["origin_position", score], ascending=[True, False])
        .groupby("origin_position")
        .head(n)
    )
    return float(top["down_target"].mean())


def evaluate(pred: pd.DataFrame) -> dict[str, float]:
    out = {}
    for wname, (a, b) in WINDOWS.items():
        w = pred[pred["origin_position"].between(a, b)]
        y, p = w["down_target"].to_numpy(), w["p_raw"].to_numpy()
        out[f"{wname}_auc"] = float(roc_auc_score(y, p))
        out[f"{wname}_brier"] = float(brier_score_loss(y, p))
        for n in (1, 3, 5, 10):
            out[f"{wname}_top{n}"] = topn(w, "p_raw", n)
    return out


def main() -> None:
    started = time.perf_counter()
    panel = pd.read_parquet(PANEL)
    panel = panel[panel["down_eligible"]].copy()
    v2p = pd.read_parquet(V2_PANEL)
    v2p = v2p[v2p["down_eligible"]].copy()
    lag = 1

    fam = family_columns(panel)
    for k, v in fam.items():
        print(f"family {k}: {len(v)} features")
    all_v3 = [c for v in fam.values() for c in v]

    configs: dict[str, list[str]] = {}
    for k in fam:
        configs[f"only_{k}"] = fam[k]
        configs[f"drop_{k}"] = [c for c in all_v3 if c not in fam[k]]
    configs["full"] = all_v3

    # screen C on Tuning for the full model, reuse for all configs (fairness)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_c, best_auc = 0.1, -1.0
        for c in (0.01, 0.03, 0.1, 0.3, 1.0):
            p = run_walk_forward(panel, all_v3, c, lag)
            t = p[p["origin_position"].between(120, 179)]
            auc = roc_auc_score(t["down_target"], t["p_raw"])
            print(f"  C={c}: tuning AUC={auc:.4f}")
            if auc > best_auc:
                best_c, best_auc = c, auc
    print(f"selected C={best_c}")

    results: dict[str, dict] = {}
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for name, feats in configs.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = run_walk_forward(panel, feats, best_c, lag)
        pred.to_parquet(PRED_DIR / f"abl_{name}.parquet", index=False)
        results[name] = {"n_features": len(feats), **evaluate(pred)}
        print(
            f"{name:14s} nf={len(feats):3d} T={results[name]['tuning_auc']:.4f} "
            f"V={results[name]['validation_auc']:.4f} "
            f"C={results[name]['confirmation_auc']:.4f} "
            f"Vtop5={results[name]['validation_top5']:.4f}"
        )

    # greedy best2 / best3 by Tuning AUC among only_* configs
    solo = sorted(
        (k for k in results if k.startswith("only_")),
        key=lambda k: results[k]["tuning_auc"],
        reverse=True,
    )
    fam_rank = [k[5:] for k in solo]
    for n in (2, 3):
        combo = [c for f in fam_rank[:n] for c in fam[f]]
        name = f"best{n}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = run_walk_forward(panel, combo, best_c, lag)
        pred.to_parquet(PRED_DIR / f"abl_{name}.parquet", index=False)
        results[name] = {"n_features": len(combo), **evaluate(pred)}
        print(
            f"{name:14s} nf={len(combo):3d} T={results[name]['tuning_auc']:.4f} "
            f"V={results[name]['validation_auc']:.4f} "
            f"C={results[name]['confirmation_auc']:.4f}"
        )

    # V2 reference under the identical protocol
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = run_walk_forward(v2p, V2_FEATURES, best_c, lag)
    pred.to_parquet(PRED_DIR / "abl_v2_reference.parquet", index=False)
    results["v2_reference"] = {"n_features": len(V2_FEATURES), **evaluate(pred)}
    print(
        f"{'v2_reference':14s} nf={len(V2_FEATURES):3d} "
        f"T={results['v2_reference']['tuning_auc']:.4f} "
        f"V={results['v2_reference']['validation_auc']:.4f} "
        f"C={results['v2_reference']['confirmation_auc']:.4f}"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(strict_json_dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nsaved {OUT} ({time.perf_counter() - started:.0f}s)")


if __name__ == "__main__":
    main()
