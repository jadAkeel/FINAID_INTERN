import pandas as pd
import pytest

from forecast_select.experiment_cache import (
    LEDGER_COLUMNS,
    ReplayCacheKey,
    attach_replay_outcomes,
    load_replay_cache,
    split_replay_source,
    write_experiment_ledger_row,
    write_replay_cache,
)


def _key() -> ReplayCacheKey:
    return ReplayCacheKey(
        source_data_hash="data",
        source_artifact_hash="artifact",
        relevant_config_hash="config",
        origin_start=120,
        origin_end=121,
        feature_contract_version="features-v1",
        model_settings={"depth": 2, "nested": {"b": 2, "a": 1}},
        selected_parameters={"threshold": 0.65},
        release_identifier="release-v1",
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame({
        "origin_position": [120, 120, 121, 121],
        "indicator_id": ["X1", "X2", "X1", "X2"],
        "y_true": [1.0, 0.0, 0.0, 1.0],
        "p_up_selection_score": [0.8, 0.7, 0.6, 0.9],
        "level_c_ready": [True, True, True, True],
        "base_accepted": [True, False, True, False],
        "base_predicted_direction": ["Up", "Up", "Up", "Up"],
        "p_down_base": [0.2, 0.3, 0.4, 0.1],
        "p_down": [0.7, 0.7, 0.7, 0.7],
        "accepted": [False, True, False, True],
        "predicted_direction": ["Down", "Down", "Down", "Down"],
        "locked_evaluation_read": [False] * 4,
    })


def test_cache_key_is_canonical_for_nested_model_settings():
    first = _key()
    second = ReplayCacheKey(
        **{
            **first.payload(),
            "model_settings": {"nested": {"a": 1, "b": 2}, "depth": 2},
        }
    )
    assert first.digest() == second.digest()


def test_cache_key_changes_with_policy_or_source_artifact():
    first = _key()
    changed_policy = ReplayCacheKey(
        **{
            **first.payload(),
            "selected_parameters": {"threshold": 0.70},
        }
    )
    changed_source = ReplayCacheKey(
        **{
            **first.payload(),
            "source_artifact_hash": "different",
        }
    )
    assert first.digest() != changed_policy.digest()
    assert first.digest() != changed_source.digest()


def test_replay_cache_round_trip_separates_outcomes(tmp_path):
    inputs, outcomes = split_replay_source(
        _predictions(),
        locked_start=268,
    )
    assert "y_true" not in inputs
    assert inputs["accepted"].tolist() == [True, False, True, False]
    assert inputs["predicted_direction"].eq("Up").all()
    assert inputs["p_down"].tolist() == inputs["p_down_base"].tolist()
    assert "correctness_probability" not in inputs
    assert inputs["base_up_rank"].tolist() == [1.0, 2.0, 2.0, 1.0]

    write_replay_cache(
        tmp_path,
        _key(),
        inputs,
        outcomes,
        locked_start=268,
    )
    cached_inputs, cached_outcomes = load_replay_cache(
        tmp_path,
        _key(),
        locked_start=268,
    )
    pd.testing.assert_frame_equal(cached_inputs, inputs, check_dtype=False)
    pd.testing.assert_frame_equal(cached_outcomes, outcomes)
    attached = attach_replay_outcomes(cached_inputs, cached_outcomes)
    assert attached["y_true"].tolist() == outcomes["y_true"].tolist()


def test_replay_cache_rejects_locked_origins():
    predictions = _predictions()
    predictions.loc[0, "origin_position"] = 268
    with pytest.raises(ValueError, match="Locked origins"):
        split_replay_source(predictions, locked_start=268)


def test_replay_cache_rejects_mismatched_outcome_keys(tmp_path):
    inputs, outcomes = split_replay_source(
        _predictions(),
        locked_start=268,
    )
    outcomes = outcomes.iloc[:-1].copy()
    with pytest.raises(ValueError, match="keys do not match"):
        write_replay_cache(
            tmp_path,
            _key(),
            inputs,
            outcomes,
            locked_start=268,
        )


def test_experiment_ledger_upserts_by_experiment_id(tmp_path):
    path = tmp_path / "ledger.csv"
    write_experiment_ledger_row(
        path,
        {"experiment_id": "one", "hits": 2, "accepted": False},
    )
    write_experiment_ledger_row(
        path,
        {"experiment_id": "one", "hits": 3, "accepted": True},
    )
    ledger = pd.read_csv(path)
    assert ledger.columns.tolist() == LEDGER_COLUMNS
    assert len(ledger) == 1
    assert int(ledger.loc[0, "hits"]) == 3
    assert bool(ledger.loc[0, "accepted"])


def test_experiment_ledger_upsert_preserves_neighboring_rows(tmp_path):
    path = tmp_path / "ledger.csv"
    for experiment_id, hits in [("one", 1), ("two", 2), ("three", 3)]:
        write_experiment_ledger_row(
            path,
            {"experiment_id": experiment_id, "hits": hits},
        )

    write_experiment_ledger_row(
        path,
        {"experiment_id": "one", "hits": 10},
    )

    ledger = pd.read_csv(path).set_index("experiment_id")
    assert set(ledger.index) == {"one", "two", "three"}
    assert int(ledger.loc["one", "hits"]) == 10
    assert int(ledger.loc["two", "hits"]) == 2
    assert int(ledger.loc["three", "hits"]) == 3
    assert _key().payload()["model_settings"] == {
        "depth": 2,
        "nested": {"b": 2, "a": 1},
    }
