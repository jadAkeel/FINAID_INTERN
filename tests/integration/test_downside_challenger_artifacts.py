import json
from pathlib import Path

import pandas as pd
import pytest


def test_nonlocked_roadmap_summary_matches_ledger():
    summary = json.loads(
        Path(
            "research/regime_adaptive_selector/metrics/"
            "nonlocked_roadmap_summary.json"
        ).read_text(encoding="utf-8")
    )
    ledger = pd.read_csv(
        "research/regime_adaptive_selector/metrics/experiment_ledger.csv"
    ).set_index("experiment_id")
    reference = ledger.loc[summary["selected_experiment_id"]]

    assert len(ledger) == summary["experiment_ledger_rows"]
    assert int(reference["validation_calls"]) == summary["reference"][
        "validation"
    ]["calls"]
    assert int(reference["validation_hits"]) == summary["reference"][
        "validation"
    ]["hits"]
    assert float(reference["validation_accuracy"]) == pytest.approx(
        summary["reference"]["validation"]["accuracy"]
    )
    assert summary["active_model_changed"] is False
    assert summary["confirmation_used_for_selection"] is False
    assert summary["locked_evaluation_read"] is False
    assert summary["milestones"]["validation_65_percent_reached"] is False


def test_downside_challenger_artifacts_match_summary_and_ledger():
    root = Path("research/regime_adaptive_selector/downside_challengers")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    ledger = pd.read_csv(
        "research/regime_adaptive_selector/metrics/experiment_ledger.csv"
    ).set_index("experiment_id")

    assert summary["active_model_changed"] is False
    assert summary["confirmation_used_for_selection"] is False
    assert summary["locked_evaluation_read"] is False

    for name, evidence in summary["results"].items():
        predictions = pd.read_parquet(root / f"{name}.parquet")
        selected = predictions[
            predictions["accepted"].fillna(False).astype(bool)
        ]
        monthly = selected.groupby("origin_position")["indicator_id"].agg(
            ["count", "nunique"]
        )
        down_per_origin = (
            selected[selected["predicted_direction"].eq("Down")]
            .groupby("origin_position")
            .size()
        )
        normalized_replacement = selected.get(
            "normalized_replacement",
            pd.Series(False, index=selected.index),
        ).fillna(False)
        regime_replacement = selected.get(
            "regime_replacement",
            pd.Series(False, index=selected.index),
        ).fillna(False)
        expansion = selected[
            selected["base_up_rank"].between(16, 20)
            & ~normalized_replacement
            & ~regime_replacement
        ]

        assert int(predictions["origin_position"].max()) == 266
        assert monthly["count"].between(15, 20).all()
        assert monthly["count"].eq(monthly["nunique"]).all()
        if name == "normalized_bidirectional_percentile":
            assert down_per_origin.le(1).all()
        if name == "maximum_one_down_replacement":
            replacements_per_origin = (
                selected[selected["regime_replacement"].fillna(False)]
                .groupby("origin_position")
                .size()
            )
            assert replacements_per_origin.le(1).all()
        assert expansion["predicted_direction"].eq("Up").all()

        experiment_id = f"phase3_{name}"
        row = ledger.loc[experiment_id]
        windows = evidence["windows"]
        assert int(row["validation_calls"]) == windows["validation"]["calls"]
        assert int(row["validation_hits"]) == windows["validation"]["hits"]
        assert float(row["validation_accuracy"]) == pytest.approx(
            windows["validation"]["accuracy"]
        )
        assert bool(row["accepted"]) is bool(evidence["accepted"])

    assert summary["results"]["current_guarded_up_first"]["accepted"] is True
    assert (
        summary["results"]["maximum_one_down_replacement"]["accepted"]
        is False
    )
    assert (
        summary["results"]["normalized_bidirectional_percentile"]["accepted"]
        is False
    )
