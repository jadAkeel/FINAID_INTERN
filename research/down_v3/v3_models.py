"""Down V3 model comparison + calibration + standalone Top-N evaluation.

Stage 12 of the brief: compare on the same data and same walk-forward
  A - regularized Logistic Regression
  B - HistGradientBoostingClassifier
  C - LightGBM (dependency already present: lightgbm 4.6.0)
  D - ensemble of A+B only if they prove complementary

Stage 13: Platt / Isotonic calibration fit on Tuning only, frozen before
Validation. Stage 14: standalone Top-1..10 Down selector evaluation.

Writes:
  research/down_v3/artifacts/v3_predictions.parquet
  research/down_v3/metrics/v3_models.json
"""
from __future__ import annotations

import json
from json_output import strict_json_dumps
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from forecast_select.validation import causal_training_rows  # noqa: E402

PANEL = ROOT / "research/down_v3/artifacts/v3_panel.parquet"
OUT_PRED = ROOT / "research/down_v3/artifacts/v3_predictions.parquet"
OUT_MET = ROOT / "research/down_v3/metrics/v3_models.json"

WINDOWS = {
    "tuning": (120, 179),
    "validation": (180, 219),
    "confirmation": (220, 266),
}
RNG = np.random.default_rng(20260921)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def make_logistic(c: float) -> Pipeline:
    return Pipeline(
        [
            ("pre", ColumnTransformer([("num", "passthrough", ["numeric"])])),
            ("clf", LogisticRegression(C=c, solver="liblinear", max_iter=1000, random_state=7)),
        ]
    )


def topn_prec(frame: pd.DataFrame, score: str, n: int) -> dict[str, float]:
    top = (
        frame.sort_values(["origin_position", score], ascending=[True, False])
        .groupby("origin_position")
        .head(n)
    )
    return {
        "calls": int(len(top)),
        "hits": int(top["down_target"].sum()),
        "precision": float(top["down_target"].mean()) if len(top) else float("nan"),
    }


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    if len(np.unique(edges)) < bins + 1:
        edges = np.linspace(p.min() - 1e-9, p.max() + 1e-9, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(
        sum(
            (idx == b).sum() / len(y) * abs(y[idx == b].mean() - p[idx == b].mean())
            for b in range(bins)
            if (idx == b).any()
        )
    )


def bootstrap_se(y: np.ndarray, p: np.ndarray, n_boot: int = 400) -> float:
    """Origin-block bootstrap SE of AUC, computed on Tuning rows only."""
    rng = np.random.default_rng(20260921)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return float(np.std(aucs))


def evaluate(pred: pd.DataFrame, score: str) -> dict:
    out = {}
    for wname, (a, b) in WINDOWS.items():
        w = pred[pred["origin_position"].between(a, b)]
        y, p = w["down_target"].to_numpy(), w[score].to_numpy()
        out[wname] = {
            "n": int(len(w)),
            "auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "ece": ece(y, p),
            "mean_pred": float(p.mean()),
            "observed": float(y.mean()),
            **{f"top{n}": topn_prec(w, score, n) for n in (1, 2, 3, 4, 5, 10)},
        }
    return out


def fit_models(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    """Return {model_name: p_down_raw} for the test origin."""
    xt, yt = train[features], train["down_target"].astype(int)
    out: dict[str, np.ndarray] = {}

    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("imp", SimpleImputer(strategy="median", add_indicator=True)),
                     ("sc", StandardScaler())]
                ),
                features,
            ),
            (
                "ind",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["indicator_id"],
            ),
        ]
    )
    log = Pipeline(
        [
            ("pre", pre),
            ("clf", LogisticRegression(C=0.1, solver="liblinear", max_iter=1000, random_state=7)),
        ]
    )
    log.fit(train[[*features, "indicator_id"]], yt)
    out["logistic"] = log.predict_proba(test[[*features, "indicator_id"]])[:, 1]

    hgb = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=7,
    )
    hgb.fit(xt, yt)
    out["hgb"] = hgb.predict_proba(test[features])[:, 1]

    import lightgbm as lgb

    gbm = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=7,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=7,
        verbose=-1,
    )
    gbm.fit(xt, yt)
    out["lightgbm"] = gbm.predict_proba(test[features])[:, 1]
    return out


def main() -> None:
    started = time.perf_counter()
    panel = pd.read_parquet(PANEL)
    panel = panel[panel["down_eligible"]].copy()
    lag = 1
    # Feature set screened on TUNING ONLY. Validation is a gate, never a
    # screen: picking the family by validation AUC would be leakage.
    # Rule: among configs within one bootstrap SE of the best Tuning AUC,
    # take the one with the fewest features (parsimony tie-break).
    abl_path = ROOT / "research/down_v3/metrics/v3_ablation.json"
    if abl_path.exists():
        abl = json.loads(abl_path.read_text(encoding="utf-8"))
        solo = {
            k: v
            for k, v in abl.items()
            if k.startswith("only_")
        }
        se = {}
        for k in solo:
            p = pd.read_parquet(
                ROOT / "research/down_v3/artifacts" / f"abl_{k}.parquet"
            )
            t = p[p["origin_position"].between(120, 179)]
            se[k] = bootstrap_se(t["down_target"].to_numpy(), t["p_raw"].to_numpy())
        best_auc = max(solo[k]["tuning_auc"] for k in solo)
        thresh = best_auc - max(se.values())
        within = [k for k in solo if solo[k]["tuning_auc"] >= thresh]
        pick = min(within, key=lambda k: solo[k]["n_features"])
        feats = [c for c in panel.columns if c.startswith("v3_") and (
            f"_{pick[5:]}_" in c or (pick == "only_mkt" and "_mkt_" in c)
            or (pick == "only_int" and "_int_" in c)
        )]
        print(
            f"tuning screen: best={best_auc:.4f} se<= {max(se.values()):.4f}; "
            f"within-SE configs={within}"
        )
        print(f"selected feature set: {pick} ({len(feats)} features)")
    else:
        feats = [c for c in panel.columns if c.startswith("v3_")]
        print(f"ablation missing -> using all {len(feats)} v3 features")

    pieces = []
    for origin in sorted(panel["origin_position"].unique()):
        if origin < 120:
            continue
        train = causal_training_rows(panel, origin, availability_lag=lag)
        if train.empty or train["down_target"].nunique() < 2:
            continue
        test = panel[panel["origin_position"].eq(origin)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = fit_models(train, test, feats)
        piece = test[["origin_position", "indicator_id", "down_target"]].copy()
        for name, p in preds.items():
            piece[name] = np.clip(p, 1e-6, 1 - 1e-6)
        pieces.append(piece)
    pred = pd.concat(pieces, ignore_index=True)

    # ---- calibration on Tuning only, applied forward ----
    tune = pred[pred["origin_position"].between(120, 179)]
    yt = tune["down_target"].to_numpy()
    calibrators = {}
    for name in ("logistic", "hgb", "lightgbm"):
        pr = tune[name].to_numpy()
        platt = LogisticRegression(C=1e6, max_iter=1000)
        platt.fit(logit(pr).reshape(-1, 1), yt.astype(int))
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(pr, yt.astype(int))
        calibrators[name] = (platt, iso)
        pred[f"{name}_platt"] = np.clip(
            platt.predict_proba(logit(pred[name].to_numpy()).reshape(-1, 1))[:, 1],
            1e-6, 1 - 1e-6,
        )
        pred[f"{name}_iso"] = np.clip(iso.predict(pred[name].to_numpy()), 1e-6, 1 - 1e-6)
    # ensemble of the two most complementary models, by Tuning correlation
    corr = {
        (a, b): float(np.corrcoef(tune[a], tune[b])[0, 1])
        for a, b in (("logistic", "hgb"), ("logistic", "lightgbm"), ("hgb", "lightgbm"))
    }
    print("model correlation on Tuning:", {k: round(v, 4) for k, v in corr.items()})
    scores = {m: roc_auc_score(yt, tune[m]) for m in ("logistic", "hgb", "lightgbm")}
    lo = min(scores, key=scores.get)
    others = [m for m in scores if m != lo]
    pair = min(
        ((a, b) for a in others for b in others if a != b),
        key=lambda pr: corr.get(pr, corr.get((pr[1], pr[0]), 1.0)),
    )
    pred["ensemble"] = np.clip(
        0.5 * pred[pair[0]].to_numpy() + 0.5 * pred[pair[1]].to_numpy(), 1e-6, 1 - 1e-6
    )
    # `tune` was sliced before `ensemble` existed; re-slice so the calibrator
    # sees it. Still fit on Tuning only, frozen before Validation.
    tune = pred[pred["origin_position"].between(120, 179)]
    yt = tune["down_target"].to_numpy()
    platt = LogisticRegression(C=1e6, max_iter=1000)
    platt.fit(logit(tune["ensemble"].to_numpy()).reshape(-1, 1), yt.astype(int))
    pred["ensemble_platt"] = np.clip(
        platt.predict_proba(logit(pred["ensemble"].to_numpy()).reshape(-1, 1))[:, 1],
        1e-6, 1 - 1e-6,
    )
    pred["ensemble_pair"] = f"{pair[0]}+{pair[1]}"

    pred.to_parquet(OUT_PRED, index=False)

    metrics = {
        "features": feats,
        "ensemble_pair": pred["ensemble_pair"].iloc[0],
        "tuning_model_correlation": {f"{a}+{b}": v for (a, b), v in corr.items()},
    }
    for name in ("logistic", "hgb", "lightgbm", "logistic_platt", "hgb_platt",
                 "lightgbm_platt", "logistic_iso", "hgb_iso", "lightgbm_iso",
                 "ensemble", "ensemble_platt"):
        metrics[name] = evaluate(pred, name)

    OUT_MET.parent.mkdir(parents=True, exist_ok=True)
    OUT_MET.write_text(strict_json_dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    print("\n" + "=" * 78)
    print("Down V3 model comparison")
    print("=" * 78)
    for name in metrics:
        if not isinstance(metrics[name], dict) or "tuning" not in metrics[name]:
            continue
        print(f"\n--- {name} ---")
        for wname, d in metrics[name].items():
            if not isinstance(d, dict) or "auc" not in d:
                continue
            print(
                f"  {wname:12s} AUC={d['auc']:.4f} Brier={d['brier']:.4f} "
                f"ECE={d['ece']:.4f} mean={d['mean_pred']:.4f} obs={d['observed']:.4f}"
            )
            line = "      "
            for n in (1, 2, 3, 5, 10):
                line += f"top{n}={d[f'top{n}']['precision']:.4f} "
            print(line)
    print(f"\nsaved {OUT_PRED}\nsaved {OUT_MET} ({time.perf_counter() - started:.0f}s)")


if __name__ == "__main__":
    main()
