"""P5b: Feature-family ablation for the Down global model.

Walk-forward refit of the GLOBAL down logistic only (the part that actually
carries signal, per P5a), with one feature family removed at a time and also
each family alone. Uses the same causal boundary as production:
features <= t-1, training labels <= t-2.
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from json_output import strict_json_dumps  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))

from forecast_select.directional_downside import _model_pipeline  # noqa: E402
from forecast_select.validation import (  # noqa: E402
    assert_target_history_available,
    causal_training_rows,
)

PANEL = ROOT / "research/down_v2/artifacts/down_panel.parquet"
OUT = ROOT / "research/down_v2/artifacts/p5b_family_ablation.parquet"
METRICS = ROOT / "research/down_v2/metrics/p5b_family_ablation.json"

FAMILIES: dict[str, list[str]] = {
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
ALL_FEATURES = [f for fam in FAMILIES.values() for f in fam]


def topn_hit(frame: pd.DataFrame, score: str, n: int) -> float:
    top = frame.sort_values(
        ["origin_position", score], ascending=[True, False]
    ).groupby("origin_position").head(n)
    return float(top["down_target"].mean())


def run_config(panel: pd.DataFrame, features: list[str], lag: int, c: float) -> pd.DataFrame:
    pieces = []
    for origin in sorted(panel["origin_position"].unique()):
        train = causal_training_rows(panel, origin, availability_lag=lag)
        if train.empty or train["down_target"].nunique() < 2:
            continue
        test = panel[panel["origin_position"].eq(origin)].copy()
        assert_target_history_available(train, origin, availability_lag=lag)
        model = _model_pipeline(features, logistic_c=c, max_iter=500, seed=7,
                                include_indicator=True)
        model.fit(train[[*features, "indicator_id"]], train["down_target"].astype(int))
        p = np.clip(model.predict_proba(test[[*features, "indicator_id"]])[:, 1], 1e-6, 1 - 1e-6)
        piece = test[["origin_position", "indicator_id", "down_target"]].copy()
        piece["p"] = p
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def evaluate(frame: pd.DataFrame) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for name, (a, b) in {
        "tuning": (120, 179),
        "validation": (180, 219),
        "confirmation": (220, 266),
    }.items():
        w = frame[frame["origin_position"].between(a, b)]
        y, p = w["down_target"].to_numpy(), w["p"].to_numpy()
        out[f"{name}_auc"] = float(roc_auc_score(y, p))
        out[f"{name}_brier"] = float(brier_score_loss(y, p))
        for n in (3, 5, 10):
            out[f"{name}_top{n}"] = topn_hit(w, "p", n)
    return out


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
    present = [f for f in ALL_FEATURES if f in panel.columns]
    print(f"features present: {len(present)}/{len(ALL_FEATURES)}")

    configs: dict[str, list[str]] = {"full": present}
    for fam, feats in FAMILIES.items():
        configs[f"drop_{fam}"] = [f for f in present if f not in feats]
        configs[f"only_{fam}"] = [f for f in feats if f in present]

    results = {}
    for name, feats in configs.items():
        if not feats:
            continue
        pred = run_config(panel, feats, lag, c)
        pred.to_parquet(
            ROOT / f"research/down_v2/artifacts/p5b_{name}.parquet", index=False
        )
        metrics = evaluate(pred)
        metrics["n_features"] = len(feats)
        metrics["n_rows"] = len(pred)
        results[name] = metrics
        print(
            f"{name:26s} nf={len(feats):2d} T_AUC={metrics['tuning_auc']:.4f} "
            f"V_AUC={metrics['validation_auc']:.4f} C_AUC={metrics['confirmation_auc']:.4f} "
            f"({time.perf_counter() - started:.0f}s)"
        )

    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(strict_json_dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nsaved {METRICS}")

    print("\n--- Ranked by validation AUC ---")
    rank = sorted(results.items(), key=lambda kv: -kv[1]["validation_auc"])
    for name, m in rank:
        print(f"  {name:26s} V={m['validation_auc']:.4f} T={m['tuning_auc']:.4f} "
              f"C={m['confirmation_auc']:.4f} V_top5={m['validation_top5']:.4f}")


if __name__ == "__main__":
    main()
