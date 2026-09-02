import json
from pathlib import Path

import pandas as pd


FAMILIES = [
    "trend_persistence",
    "cross_sectional_dynamics",
    "risk_normalized",
]


def _correct(selected: pd.DataFrame) -> pd.Series:
    return (
        selected["predicted_direction"].eq("Up")
        & selected["y_true"].eq(1.0)
    ) | (
        selected["predicted_direction"].eq("Down")
        & selected["y_true"].eq(0.0)
    )


def test_feature_ablation_artifacts_match_summaries_and_ledger():
    ledger = pd.read_csv(
        "research/regime_adaptive_selector/metrics/experiment_ledger.csv"
    ).set_index("experiment_id")
    for family in FAMILIES:
        root = Path(
            "research/regime_adaptive_selector/feature_ablation"
        ) / family
        summary = json.loads(
            (root / "summary.json").read_text(encoding="utf-8")
        )
        predictions = pd.read_parquet(root / "adaptive_predictions.parquet")
        assert int(predictions["origin_position"].max()) == 266
        assert not predictions["locked_evaluation_read"].any()
        selected = predictions[predictions["accepted"]]
        monthly = selected.groupby("origin_position")["indicator_id"].agg(
            ["count", "nunique"]
        )
        assert monthly["count"].between(15, 20).all()
        assert monthly["count"].eq(monthly["nunique"]).all()
        validation = selected[selected["origin_position"].between(180, 219)]
        hits = int(_correct(validation).sum())
        assert hits == summary["adaptive"]["candidate"]["validation"]["hits"]
        experiment_id = f"phase2_feature_{family}"
        assert not bool(summary["accepted"])
        assert not bool(ledger.loc[experiment_id, "accepted"])
        assert str(ledger.loc[experiment_id, "rejection_reason"])
