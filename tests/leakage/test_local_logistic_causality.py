"""Temporal-protocol guarantees for the Local / Interaction Logistic study."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast_select.local_logistic_pipeline import (
    apply_selector,
    load_experiment_settings,
    prepare_experiment_panel,
    run_walk_forward,
)
from forecast_select.uptrend_pipeline import build_uptrend_predictions
from forecast_select.validation import causal_training_rows


ROOT = Path(__file__).resolve().parents[2]
WINDOW = (120, 125)


@pytest.fixture(scope="module")
def shared():
    return prepare_experiment_panel(ROOT)


def test_experiment_panel_never_reads_the_locked_rows(shared):
    settings = load_experiment_settings(ROOT)
    locked_start = int(settings["locked_origins"][0])
    assert int(shared.frame["position"].max()) == int(
        settings["maximum_readable_position"]
    )
    assert int(shared.frame["position"].max()) < locked_start
    assert int(shared.panel["origin_position"].max()) < locked_start
    assert int(shared.targets["origin_position"].max()) < locked_start


def test_training_labels_stop_at_t_minus_two(shared):
    for origin in (120, 200, 266):
        train = causal_training_rows(shared.eligible, origin, availability_lag=1)
        assert int(train["origin_position"].max()) == origin - 2


def test_walk_forward_refuses_a_locked_origin(shared):
    with pytest.raises(AssertionError, match="Locked origins"):
        run_walk_forward(shared, [270], local_specs=[])


def test_research_selector_reproduces_the_production_uptrend_selector(shared):
    """The only thing that may differ between candidates is the p_up signal."""
    production = build_uptrend_predictions(ROOT, origin_range=WINDOW)
    produced = run_walk_forward(
        shared, range(WINDOW[0], WINDOW[1] + 1), local_specs=[]
    )
    research = apply_selector(produced["signals"], "p_global", shared)

    key = ["origin_position", "indicator_id"]
    left = production.sort_values(key).reset_index(drop=True)
    right = research.sort_values(key).reset_index(drop=True)
    assert left[key].equals(right[key])
    for column in ("p_up_raw", "p_up", "selection_score", "indicator_prior"):
        assert np.allclose(
            left[column].astype(float),
            right[column].astype(float),
            equal_nan=True,
        ), column
    assert left["accepted"].astype(bool).equals(right["accepted"].astype(bool))
    assert left["predicted_direction"].equals(right["predicted_direction"])


def test_selector_returns_fifteen_unique_indicators_per_origin(shared):
    produced = run_walk_forward(
        shared, range(WINDOW[0], WINDOW[1] + 1), local_specs=[]
    )
    result = apply_selector(produced["signals"], "p_global", shared)
    accepted = result[result["accepted"].fillna(False).astype(bool)]
    counts = accepted.groupby("origin_position")["indicator_id"].agg(
        ["count", "nunique"]
    )
    assert counts["count"].eq(15).all()
    assert counts["nunique"].eq(15).all()


def test_candidates_are_scored_on_identical_eligible_rows(shared):
    from forecast_select.local_logistic_pipeline import LocalSpec

    produced = run_walk_forward(
        shared,
        range(WINDOW[0], WINDOW[1] + 1),
        local_specs=[LocalSpec("lean5", 0.1)],
        interaction_cs=[0.1],
    )
    signals = produced["signals"]
    key = ["origin_position", "indicator_id"]
    frames = [
        apply_selector(signals, column, shared)[key]
        for column in ("p_global", "p_local__lean5_c0.1", "p_interaction__c0.1")
    ]
    for frame in frames[1:]:
        assert frame.sort_values(key).reset_index(drop=True).equals(
            frames[0].sort_values(key).reset_index(drop=True)
        )


def test_future_rows_do_not_change_an_earlier_origin_signal(shared):
    """Truncating everything after the origin leaves the prediction unchanged."""
    from forecast_select.local_logistic_pipeline import LocalSpec

    origin = 130
    full = run_walk_forward(
        shared, [origin], local_specs=[LocalSpec("core8", 0.1)]
    )["signals"]

    truncated = prepare_experiment_panel(ROOT)
    keep = truncated.eligible["origin_position"].le(origin)
    truncated.eligible.drop(
        index=truncated.eligible.index[~keep], inplace=True
    )
    partial = run_walk_forward(
        truncated, [origin], local_specs=[LocalSpec("core8", 0.1)]
    )["signals"]

    key = ["origin_position", "indicator_id"]
    left = full.sort_values(key).reset_index(drop=True)
    right = partial.sort_values(key).reset_index(drop=True)
    assert left[key].equals(right[key])
    for column in ("p_global", "p_local__core8_c0.1"):
        assert np.allclose(left[column], right[column]), column


def test_prediction_artifact_carries_the_audit_columns():
    path = ROOT / "research/local_logistic/artifacts/predictions.parquet"
    if not path.exists():
        pytest.skip("run `python -m forecast_select.local_logistic_runner all` first")
    predictions = pd.read_parquet(path)
    required = {
        "origin_position", "origin_date", "indicator_id", "y_true",
        "p_global", "p_local", "p_hybrid", "p_interaction",
        "rank_global", "rank_pure_local", "rank_fixed_shrinkage",
        "rank_interaction",
        "selected_global", "selected_pure_local", "selected_fixed_shrinkage",
        "selected_interaction",
        "correct_global", "correct_pure_local", "correct_fixed_shrinkage",
        "correct_interaction",
        "local_training_sample_count", "local_fallback_used",
    }
    assert required.issubset(set(predictions.columns))
    settings = load_experiment_settings(ROOT)
    assert int(predictions["origin_position"].max()) < int(
        settings["locked_origins"][0]
    )
