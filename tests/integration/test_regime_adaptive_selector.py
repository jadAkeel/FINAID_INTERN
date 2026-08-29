import json
from pathlib import Path

import pandas as pd


def test_regime_adaptive_artifact_is_causal_and_supports_variable_cap():
    path = Path(
        "research/regime_adaptive_selector/artifacts/predictions.parquet"
    )
    assert path.exists()
    predictions = pd.read_parquet(path)
    assert int(predictions["origin_position"].min()) == 120
    assert int(predictions["origin_position"].max()) == 266
    assert not predictions["locked_evaluation_read"].any()
    assert (
        predictions["down_fit_through_origin"]
        <= predictions["origin_position"] - 2
    ).all()
    assert (
        predictions["generalized_graph_fit_through_origin"]
        <= predictions["origin_position"] - 1
    ).all()
    assert (
        predictions["forecast_market_breadth_fit_through_origin"]
        <= predictions["origin_position"] - 2
    ).all()
    assert (
        predictions[
            "forecast_market_breadth_observation_through_origin"
        ]
        <= predictions["origin_position"] - 1
    ).all()
    selected = predictions[predictions["accepted"]]
    monthly = selected.groupby("origin_position")["indicator_id"].agg(
        ["count", "nunique"]
    )
    assert monthly["count"].between(15, 20).all()
    assert monthly["nunique"].between(15, 20).all()
    # Graduated 15..20: allow any subset of 15-20, but must be within contract and at least two distinct caps
    assert set(monthly["count"].unique()).issubset({15, 16, 17, 18, 19, 20})
    assert monthly["count"].nunique() >= 2
    assert (
        selected.groupby("origin_position")["regime_cap"].first()
        .eq(monthly["count"])
    ).all()
    assert set(selected["predicted_direction"]) <= {"Up", "Down"}
    expanded = selected[selected["regime_cap"].gt(15)].copy()
    expanded["base_score_rank"] = expanded.groupby("origin_position")[
        "p_up_selection_score"
    ].rank(method="first", ascending=False)
    assert expanded.loc[
        expanded["base_score_rank"].gt(15), "predicted_direction"
    ].eq("Up").all()
    down = selected[selected["predicted_direction"].eq("Down")]
    assert not down.empty
    assert (
        down["regime_stress"].ge(0.50)
        | down["p_down"].ge(0.80)
    ).all()
    assert "adaptive_data_quality_excluded" in predictions
    x16 = predictions[predictions["indicator_id"].eq("X16")]
    assert not x16.empty
    assert x16["adaptive_data_quality_excluded"].all()
    assert not x16["accepted"].any()
    assert "regime_replacement" in predictions
    replacements = predictions[predictions["regime_replacement"]]
    assert replacements["base_accepted"].eq(False).all()
    assert replacements["accepted"].all()
    assert replacements["predicted_direction"].eq("Down").all()


def test_regime_adaptive_result_does_not_promote_automatically():
    path = Path(
        "research/regime_adaptive_selector/metrics/summary.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["active_model_changed"] is False
    assert payload["locked_evaluation_read"] is False
    assert payload.get("promotion_eligible", False) is False
    assert 0 <= payload["selected_parameters"]["maximum_replacements"] <= 5
    assert payload["selection_mode"] in {
        "temporally_stable_development_candidate",
        "conservative_fallback_no_stable_candidate",
        "guarded_bidirectional_fallback_no_stable_candidate",
    }
    assert payload["internal_tuning_windows"] == [[120, 149], [150, 179]]
    assert "X16" in payload["excluded_indicators"]
    if payload["selection_mode"] == "guarded_bidirectional_fallback_no_stable_candidate":
        assert payload["selected_candidate_development_qualifying"] is False
        assert payload["selected_parameters"]["allow_down_predictions"] is True
        assert payload["selected_parameters"]["maximum_replacements"] == 0
        assert sum(
            payload["candidate"][window]["down_calls"]
            for window in ["tuning", "validation", "confirmation"]
        ) > 0
    elif payload["selection_mode"] == "conservative_fallback_no_stable_candidate":
        assert payload["selected_parameters"]["allow_down_predictions"] is False
    assert payload["effective_cap"] == "dynamic"
    assert payload["dynamic_cap"] is True
    assert payload["forward_regime"]["enabled"] is True
    assert payload["forward_regime"]["expansion_threshold"] == 0.65
    # Graduated mode: cap_mode should be graduated_15_to_20 with low/high thresholds
    assert payload["forward_regime"]["cap_mode"] in {"binary_15_or_20", "graduated_15_to_20"}
    if payload["forward_regime"]["cap_mode"] == "graduated_15_to_20":
        assert 0.0 <= payload["forward_regime"]["graduated_low"] < payload["forward_regime"]["graduated_high"] <= 1.0
    assert payload["asset_group_overlay"]["enabled"] is True
    assert payload["generalized_correlation_overlay"]["enabled"] is True
    assert payload["generalized_correlation_overlay"]["window_months"] == 48
    accuracy_first = payload["accuracy_first_selective"]
    assert accuracy_first["selected_on"] == "development_120_219_only"
    assert accuracy_first["confirmation_used_for_selection"] is False
    assert accuracy_first["selected_cap"] == 15
    assert accuracy_first["selected_group_weight"] == 0.25
    assert accuracy_first["development"]["calls"] == 1500
    assert accuracy_first["confirmation"]["calls"] == 705
    assert accuracy_first["coverage_vs_fixed_top15"] == 1.0
    assert accuracy_first["all_internal_windows_meet_target"] is False
    assert accuracy_first["promotion_eligible"] is False


def test_accuracy_first_artifact_is_causal_and_explicitly_selective():
    path = Path(
        "research/regime_adaptive_selector/artifacts/"
        "accuracy_first_predictions.parquet"
    )
    assert path.exists()
    predictions = pd.read_parquet(path)
    assert not predictions["locked_evaluation_read"].any()
    assert predictions["accuracy_first_selected_on"].eq(
        "development_120_219_without_confirmation_or_locked_evaluation"
    ).all()
    selected = predictions[predictions["accepted"]]
    monthly = selected.groupby("origin_position")["indicator_id"].agg(
        ["count", "nunique"]
    )
    assert monthly["count"].eq(15).all()
    assert monthly["nunique"].eq(15).all()
    assert selected["predicted_direction"].eq("Up").all()
    assert selected["selection_mode"].eq(
        "accuracy_first_fixed_coverage_up_only"
    ).all()
    pd.testing.assert_series_equal(
        predictions["selection_score"],
        predictions["p_up_selection_score"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        predictions["selection_score"],
        predictions["p_up_accuracy_first_score"],
        check_names=False,
    )
    assert predictions["generalized_graph_fit_through_origin"].le(
        predictions["origin_position"] - 1
    ).all()


def test_phase1_ledger_preserves_every_bounded_threshold_candidate():
    ledger = pd.read_csv(
        "research/regime_adaptive_selector/metrics/experiment_ledger.csv"
    )
    threshold_rows = ledger[
        ledger["experiment_id"].str.startswith("phase1_", na=False)
        & ledger["experiment_id"].str.contains("_threshold_", na=False)
    ]
    assert len(threshold_rows) == 6
    assert threshold_rows["rejection_reason"].fillna("").str.contains(
        "not_selected_on_tuning"
    ).sum() == 4

    phase1 = json.loads(
        Path(
            "research/regime_adaptive_selector/metrics/phase1_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert set(phase1["results"]["graduated"]["cap_distribution"]) == {
        "15",
        "17",
        "20",
    }
