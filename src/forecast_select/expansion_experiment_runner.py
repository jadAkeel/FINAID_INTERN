from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .expansion_quality import (
    build_expansion_quality_panel,
    expansion_quality_cap_schedule,
    graduated_cap_schedule,
    paired_monthly_hit_bootstrap,
)
from .experiment_cache import (
    ReplayCacheKey,
    attach_replay_outcomes,
    load_replay_cache,
    replay_cache_generation,
    write_experiment_ledger_row,
)
from .indicator_selection import summarize_selected_predictions
from .io import atomic_write_json, sha256_file
from .regime_adaptive_pipeline import (
    _apply,
    _read_yaml,
    regime_adaptive_predictions_artifact,
    regime_adaptive_summary_path,
)
from .regime_experiment_runner import (
    _assert_replay_equivalent,
    _band_metrics,
    _metrics,
    _window,
    build_reference_replay,
)
from .uptrend_pipeline import ROOT


def _reference_bundle(root: Path) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
    dict,
    dict,
    ReplayCacheKey,
]:
    settings = _read_yaml(root / "configs/regime_adaptive_selector.yaml")
    project = _read_yaml(root / "configs/config.yaml")
    source = pd.read_parquet(regime_adaptive_predictions_artifact(root))
    summary = json.loads(
        regime_adaptive_summary_path(root).read_text(encoding="utf-8")
    )
    locked_start = int(settings["locked_origins"][0])
    if int(source["origin_position"].max()) >= locked_start:
        raise ValueError("Reference artifact includes locked origins")
    if source["locked_evaluation_read"].fillna(True).astype(bool).any():
        raise ValueError("Reference artifact reports that locked evidence was read")
    data_hash = str(source["data_hash"].dropna().astype(str).unique()[0])
    config_hash = str(source["config_hash"].dropna().astype(str).unique()[0])
    model_settings = {
        "selection": settings["selection"],
        "asset_group_overlay": settings["asset_group_overlay"],
        "generalized_correlation_overlay": settings[
            "generalized_correlation_overlay"
        ],
        "forward_regime": settings["forward_regime"],
        "stress": settings["stress"],
    }
    key = ReplayCacheKey(
        source_data_hash=data_hash,
        source_artifact_hash=sha256_file(
            regime_adaptive_predictions_artifact(root)
        ),
        relevant_config_hash=config_hash,
        origin_start=int(source["origin_position"].min()),
        origin_end=int(source["origin_position"].max()),
        feature_contract_version="regime_replay_inputs_with_base_up_rank_v1",
        model_settings=model_settings,
        selected_parameters=dict(summary["selected_parameters"]),
        release_identifier=str(settings["experiment_release"]),
    )
    try:
        inputs, outcomes = load_replay_cache(
            root,
            key,
            locked_start=locked_start,
        )
    except FileNotFoundError:
        build_reference_replay(root)
        inputs, outcomes = load_replay_cache(
            root,
            key,
            locked_start=locked_start,
        )
    replay = attach_replay_outcomes(inputs, outcomes)
    return replay, inputs, source, settings, project, summary, key


def _apply_schedule(
    replay: pd.DataFrame,
    settings: dict,
    params: dict,
    schedule: dict[int, int],
) -> tuple[pd.DataFrame, float]:
    started = perf_counter()
    predictions = _apply(
        replay,
        settings,
        params,
        cap=None,
        cap_schedule=schedule,
    )
    runtime = perf_counter() - started
    selected = predictions[predictions["accepted"].fillna(False).astype(bool)]
    monthly = selected.groupby("origin_position")["indicator_id"].agg(
        ["count", "nunique"]
    )
    if not monthly["count"].between(15, 20).all():
        raise AssertionError("Expansion policy violated the 15-20 call contract")
    if not monthly["count"].eq(monthly["nunique"]).all():
        raise AssertionError("Expansion policy selected duplicate indicators")
    expansion = selected[selected["base_up_rank"].gt(15)]
    if not expansion["predicted_direction"].eq("Up").all():
        raise AssertionError("Expansion-only calls must remain Up")
    return predictions, runtime


def _search_row(
    policy: str,
    thresholds: tuple[float, float],
    predictions: pd.DataFrame,
    baseline: pd.DataFrame,
) -> dict[str, float | int | str | bool]:
    row: dict[str, float | int | str | bool] = {
        "policy": policy,
        "lower_threshold": float(thresholds[0]),
        "upper_threshold": float(thresholds[1]),
    }
    stable = True
    for name, bounds in [
        ("early_tuning", [120, 149]),
        ("late_tuning", [150, 179]),
        ("tuning", [120, 179]),
    ]:
        candidate_metrics = _metrics(_window(predictions, bounds))
        baseline_metrics = _metrics(_window(baseline, bounds))
        row[f"{name}_calls"] = candidate_metrics["calls"]
        row[f"{name}_hits"] = candidate_metrics["hits"]
        row[f"{name}_accuracy"] = candidate_metrics["accuracy"]
        row[f"{name}_hit_delta"] = int(
            candidate_metrics["hits"] - baseline_metrics["hits"]
        )
        if name != "tuning":
            stable = bool(
                stable
                and candidate_metrics["hits"] >= baseline_metrics["hits"]
                and candidate_metrics["accuracy"]
                >= baseline_metrics["accuracy"]
            )
    row["stable_tuning_gate"] = stable
    row["coverage_delta"] = abs(
        int(row["tuning_calls"])
        - int(_metrics(_window(baseline, [120, 179]))["calls"])
    )
    return row


def _select_tuning_candidate(search: pd.DataFrame) -> pd.Series:
    ranked = search.sort_values(
        [
            "stable_tuning_gate",
            "tuning_hits",
            "tuning_accuracy",
            "coverage_delta",
            "lower_threshold",
            "upper_threshold",
        ],
        ascending=[False, False, False, True, True, True],
    )
    return ranked.iloc[0]


def _validation_bootstrap(
    predictions: pd.DataFrame,
    settings: dict,
    project: dict,
) -> dict[str, float | int]:
    return summarize_selected_predictions(
        _window(predictions, settings["validation_origins"]),
        block_months=int(project["bootstrap_blocks"]),
        bootstrap_replicates=int(project["bootstrap_replicates"]),
        seed=int(project["seed"]),
    )


def _bootstrap_gate(
    candidate: dict[str, float | int],
    baseline: dict[str, float | int],
    paired: dict[str, float | int],
) -> bool:
    return bool(
        candidate["bootstrap_p10"] >= baseline["bootstrap_p10"] - 0.005
        and paired["bootstrap_p10"] >= 0.0
    )


def _policy_evidence(
    predictions: pd.DataFrame,
    baseline: pd.DataFrame,
    search_row: pd.Series,
    settings: dict,
    project: dict,
) -> dict[str, object]:
    windows = {
        "early_tuning": [120, 149],
        "late_tuning": [150, 179],
        "tuning": settings["tuning_origins"],
        "validation": settings["validation_origins"],
        "confirmation_descriptive": settings["confirmation_origins"],
    }
    evidence = {name: _metrics(_window(predictions, bounds)) for name, bounds in windows.items()}
    baseline_windows = {
        name: _metrics(_window(baseline, bounds)) for name, bounds in windows.items()
    }
    validation = evidence["validation"]
    baseline_validation = baseline_windows["validation"]
    bootstrap = _validation_bootstrap(predictions, settings, project)
    baseline_bootstrap = _validation_bootstrap(baseline, settings, project)
    marginal = _band_metrics(
        _window(predictions, settings["validation_origins"]), 16, 20
    )
    validation_nonnegative = bool(
        validation["hits"] >= baseline_validation["hits"]
        and validation["accuracy"] >= baseline_validation["accuracy"]
    )
    marginal_non_destructive = bool(
        marginal["calls"] > 0 and marginal["accuracy"] >= 0.50
    )
    paired = {
        name: paired_monthly_hit_bootstrap(
            predictions,
            baseline,
            bounds,
            block_months=int(project["bootstrap_blocks"]),
            replicates=int(project["bootstrap_replicates"]),
            seed=int(project["seed"]),
        )
        for name, bounds in windows.items()
    }
    bootstrap_ok = _bootstrap_gate(
        bootstrap,
        baseline_bootstrap,
        paired["validation"],
    )
    accepted = bool(
        search_row["stable_tuning_gate"]
        and validation_nonnegative
        and marginal_non_destructive
        and bootstrap_ok
    )
    selected = predictions[predictions["accepted"].fillna(False).astype(bool)]
    caps = selected.groupby("origin_position").size().value_counts().sort_index()
    return {
        "accepted": accepted,
        "baseline_windows": baseline_windows,
        "bootstrap": bootstrap,
        "bootstrap_ok": bootstrap_ok,
        "cap_distribution": {
            str(int(cap)): int(count) for cap, count in caps.items()
        },
        "marginal_validation": marginal,
        "marginal_non_destructive": marginal_non_destructive,
        "paired_monthly_hit_delta": paired,
        "stable_tuning_gate": bool(search_row["stable_tuning_gate"]),
        "validation_nonnegative": validation_nonnegative,
        "windows": evidence,
    }


def _threshold_candidate_ledger_row(
    search_row: pd.Series,
    predictions: pd.DataFrame,
    baseline: pd.DataFrame,
    runtime_seconds: float,
    key: ReplayCacheKey,
    root: Path,
    project: dict,
) -> dict[str, object]:
    policy = str(search_row["policy"])
    lower = float(search_row["lower_threshold"])
    upper = float(search_row["upper_threshold"])
    tuning = _window(predictions, [120, 179])
    tuning_metrics = _metrics(tuning)
    core = _band_metrics(tuning, 1, 15)
    middle = _band_metrics(tuning, 16, 17)
    tail = _band_metrics(tuning, 18, 20)
    selected = tuning[tuning["accepted"].fillna(False).astype(bool)]
    down = selected[selected["predicted_direction"].eq("Down")]
    down_hits = int(down["y_true"].eq(0.0).sum())
    caps = (
        predictions[predictions["accepted"].fillna(False).astype(bool)]
        .groupby("origin_position")
        .size()
        .value_counts()
        .sort_index()
    )
    paired = paired_monthly_hit_bootstrap(
        predictions,
        baseline,
        [120, 179],
        block_months=int(project["bootstrap_blocks"]),
        replicates=int(project["bootstrap_replicates"]),
        seed=int(project["seed"]),
    )
    selected_for_validation = bool(search_row["selected"])
    threshold_id = f"{lower:g}_{upper:g}".replace(".", "p")
    rejection = []
    if not bool(search_row["stable_tuning_gate"]):
        rejection.append("stable_tuning_gate")
    if selected_for_validation:
        rejection.append("tuning_screen_selected_for_validation")
    else:
        rejection.append("not_selected_on_tuning")
    return {
        "experiment_id": f"phase1_{policy}_threshold_{threshold_id}",
        "hypothesis": "Bounded Tuning-only threshold candidate",
        "changed_variables": f"{policy}_thresholds=({lower}, {upper})",
        "selected_on": "tuning_120_179_only",
        "data_hash": key.source_data_hash,
        "config_hash": key.relevant_config_hash,
        **tuning_metrics,
        "early_tuning_calls": int(search_row["early_tuning_calls"]),
        "early_tuning_hits": int(search_row["early_tuning_hits"]),
        "early_tuning_accuracy": float(search_row["early_tuning_accuracy"]),
        "late_tuning_calls": int(search_row["late_tuning_calls"]),
        "late_tuning_hits": int(search_row["late_tuning_hits"]),
        "late_tuning_accuracy": float(search_row["late_tuning_accuracy"]),
        "validation_calls": None,
        "validation_hits": None,
        "validation_accuracy": None,
        "confirmation_calls": None,
        "confirmation_hits": None,
        "confirmation_accuracy": None,
        "cap_distribution": json.dumps(
            {str(int(cap)): int(count) for cap, count in caps.items()},
            sort_keys=True,
        ),
        "rank_1_15_calls": core["calls"],
        "rank_1_15_hits": core["hits"],
        "rank_1_15_accuracy": core["accuracy"],
        "rank_16_17_calls": middle["calls"],
        "rank_16_17_hits": middle["hits"],
        "rank_16_17_accuracy": middle["accuracy"],
        "rank_18_20_calls": tail["calls"],
        "rank_18_20_hits": tail["hits"],
        "rank_18_20_accuracy": tail["accuracy"],
        "down_calls": int(len(down)),
        "down_hits": down_hits,
        "down_precision": float(down_hits / len(down)) if len(down) else np.nan,
        "bootstrap_p10": paired["bootstrap_p10"],
        "bootstrap_p90": paired["bootstrap_p90"],
        "runtime_seconds": runtime_seconds,
        "cache_status": "hit",
        "cache_key": key.digest(),
        "cache_generation": replay_cache_generation(root, key),
        # The separately recorded full policy owns the final Validation gate.
        "accepted": False,
        "rejection_reason": "|".join(rejection),
    }


def _ledger_row(
    experiment_id: str,
    hypothesis: str,
    changed_variables: str,
    predictions: pd.DataFrame,
    evidence: dict[str, object],
    runtime_seconds: float,
    key: ReplayCacheKey,
    root: Path,
) -> dict[str, object]:
    all_metrics = _metrics(predictions)
    early = evidence["windows"]["early_tuning"]
    late = evidence["windows"]["late_tuning"]
    validation = evidence["windows"]["validation"]
    confirmation = evidence["windows"]["confirmation_descriptive"]
    core = _band_metrics(predictions, 1, 15)
    middle = _band_metrics(predictions, 16, 17)
    tail = _band_metrics(predictions, 18, 20)
    selected = predictions[predictions["accepted"].fillna(False).astype(bool)]
    down = selected[selected["predicted_direction"].eq("Down")]
    down_hits = int(down["y_true"].eq(0.0).sum())
    accepted = bool(evidence["accepted"])
    rejection_reasons = []
    if not evidence["stable_tuning_gate"]:
        rejection_reasons.append("stable_tuning_gate")
    for gate in [
        "validation_nonnegative",
        "marginal_non_destructive",
        "bootstrap_ok",
    ]:
        if not evidence[gate]:
            rejection_reasons.append(gate)
    return {
        "experiment_id": experiment_id,
        "hypothesis": hypothesis,
        "changed_variables": changed_variables,
        "selected_on": "tuning_120_179_only; validation_gate; confirmation_descriptive",
        "data_hash": key.source_data_hash,
        "config_hash": key.relevant_config_hash,
        **all_metrics,
        "early_tuning_calls": early["calls"],
        "early_tuning_hits": early["hits"],
        "early_tuning_accuracy": early["accuracy"],
        "late_tuning_calls": late["calls"],
        "late_tuning_hits": late["hits"],
        "late_tuning_accuracy": late["accuracy"],
        "validation_calls": validation["calls"],
        "validation_hits": validation["hits"],
        "validation_accuracy": validation["accuracy"],
        "confirmation_calls": confirmation["calls"],
        "confirmation_hits": confirmation["hits"],
        "confirmation_accuracy": confirmation["accuracy"],
        "cap_distribution": json.dumps(
            evidence["cap_distribution"], sort_keys=True
        ),
        "rank_1_15_calls": core["calls"],
        "rank_1_15_hits": core["hits"],
        "rank_1_15_accuracy": core["accuracy"],
        "rank_16_17_calls": middle["calls"],
        "rank_16_17_hits": middle["hits"],
        "rank_16_17_accuracy": middle["accuracy"],
        "rank_18_20_calls": tail["calls"],
        "rank_18_20_hits": tail["hits"],
        "rank_18_20_accuracy": tail["accuracy"],
        "down_calls": int(len(down)),
        "down_hits": down_hits,
        "down_precision": float(down_hits / len(down)) if len(down) else np.nan,
        "bootstrap_p10": evidence["bootstrap"]["bootstrap_p10"],
        "bootstrap_p90": evidence["bootstrap"]["bootstrap_p90"],
        "runtime_seconds": runtime_seconds,
        "cache_status": "hit",
        "cache_key": key.digest(),
        "cache_generation": replay_cache_generation(root, key),
        "accepted": accepted,
        "rejection_reason": "|".join(rejection_reasons),
    }


def build_expansion_policy_experiments(root: Path = ROOT) -> dict[str, object]:
    (
        replay,
        cached_inputs,
        source,
        settings,
        project,
        summary,
        key,
    ) = _reference_bundle(root)
    params = dict(summary["selected_parameters"])
    quality_settings = settings["expansion_quality"]
    quality_panel = build_expansion_quality_panel(
        replay,
        minimum_history=int(quality_settings["minimum_history_origins"]),
        ridge_alpha=float(quality_settings["ridge_alpha"]),
    )
    forward_cfg = settings.get("forward_regime", {})
    cap_mode = str(forward_cfg.get("cap_mode", "binary_15_or_20"))
    if cap_mode == "graduated_15_to_20":
        low = float(forward_cfg["graduated_low"])
        high = float(forward_cfg["graduated_high"])
        current_schedule = {}
        for row in quality_panel.itertuples(index=False):
            breadth = float(row.forecast_market_breadth)
            # same logic as cap_for_forward_breadth_graduated
            fraction = (breadth - low) / (high - low) if high != low else 0.0
            fraction = max(0.0, min(1.0, fraction))
            cap = int(round(15 + 5 * fraction))
            # clamp to 15-20
            cap = max(15, min(20, cap))
            current_schedule[int(row.origin_position)] = cap
    else:
        current_threshold = float(forward_cfg["expansion_threshold"])
        current_schedule = {
            int(row.origin_position): (
                20 if float(row.forecast_market_breadth) >= current_threshold else 15
            )
            for row in quality_panel.itertuples(index=False)
        }
    baseline, baseline_runtime = _apply_schedule(
        replay, settings, params, current_schedule
    )
    _assert_replay_equivalent(source, baseline, cached_inputs)

    search_rows = []
    predictions_by_key: dict[tuple[str, float, float], tuple[pd.DataFrame, float]] = {}
    for thresholds in quality_settings["graduated_threshold_pairs"]:
        pair = (float(thresholds[0]), float(thresholds[1]))
        schedule = graduated_cap_schedule(
            quality_panel,
            lower_threshold=pair[0],
            upper_threshold=pair[1],
        )
        predictions, runtime = _apply_schedule(replay, settings, params, schedule)
        predictions_by_key[("graduated", *pair)] = (predictions, runtime)
        search_rows.append(_search_row("graduated", pair, predictions, baseline))
    for thresholds in quality_settings["quality_threshold_pairs"]:
        pair = (float(thresholds[0]), float(thresholds[1]))
        schedule = expansion_quality_cap_schedule(
            quality_panel,
            rank_16_17_threshold=pair[0],
            rank_18_20_threshold=pair[1],
        )
        predictions, runtime = _apply_schedule(replay, settings, params, schedule)
        predictions_by_key[("quality_gate", *pair)] = (predictions, runtime)
        search_rows.append(_search_row("quality_gate", pair, predictions, baseline))

    search = pd.DataFrame(search_rows)
    selected_rows = {
        policy: _select_tuning_candidate(search[search["policy"].eq(policy)])
        for policy in ["graduated", "quality_gate"]
    }
    search["selected"] = False
    results = {}
    ledger_path = (
        root
        / "research/regime_adaptive_selector/metrics/experiment_ledger.csv"
    )
    baseline_search = pd.Series({"stable_tuning_gate": True})
    baseline_evidence = _policy_evidence(
        baseline,
        baseline,
        baseline_search,
        settings,
        project,
    )
    baseline_evidence["accepted"] = True
    if cap_mode == "graduated_15_to_20":
        baseline_desc = f"graduated_15_to_20 low={low} high={high}"
        baseline_id = "phase1_current_graduated_15_to_20"
        baseline_hyp = "Current graduated breadth gate is the matched reference"
    else:
        baseline_desc = f"cap=20 when forward_breadth>={current_threshold}"
        baseline_id = "phase1_current_binary_15_20"
        baseline_hyp = "Current binary breadth gate is the matched reference"
    baseline_row = _ledger_row(
        baseline_id,
        baseline_hyp,
        baseline_desc,
        baseline,
        baseline_evidence,
        baseline_runtime,
        key,
        root,
    )
    write_experiment_ledger_row(ledger_path, baseline_row)
    results["current_binary"] = baseline_evidence

    for row in search.itertuples(index=False):
        candidate_key = (
            str(row.policy),
            float(row.lower_threshold),
            float(row.upper_threshold),
        )
        predictions, runtime = predictions_by_key[candidate_key]
        threshold_row = pd.Series(row._asdict())
        threshold_row["selected"] = False
        selected_row = selected_rows[str(row.policy)]
        threshold_row["selected"] = bool(
            float(row.lower_threshold)
            == float(selected_row["lower_threshold"])
            and float(row.upper_threshold)
            == float(selected_row["upper_threshold"])
        )
        write_experiment_ledger_row(
            ledger_path,
            _threshold_candidate_ledger_row(
                threshold_row,
                predictions,
                baseline,
                runtime,
                key,
                root,
                project,
            ),
        )

    for policy, selected_row in selected_rows.items():
        pair = (
            float(selected_row["lower_threshold"]),
            float(selected_row["upper_threshold"]),
        )
        mask = (
            search["policy"].eq(policy)
            & search["lower_threshold"].eq(pair[0])
            & search["upper_threshold"].eq(pair[1])
        )
        search.loc[mask, "selected"] = True
        predictions, runtime = predictions_by_key[(policy, *pair)]
        evidence = _policy_evidence(
            predictions,
            baseline,
            selected_row,
            settings,
            project,
        )
        experiment_id = (
            "phase1_graduated_15_17_20"
            if policy == "graduated"
            else "phase1_expansion_quality_gate_15_17_20"
        )
        hypothesis = (
            "Graduated breadth thresholds improve marginal coverage"
            if policy == "graduated"
            else "Causal marginal-rank quality forecasts improve expansion"
        )
        ledger_row = _ledger_row(
            experiment_id,
            hypothesis,
            f"thresholds={pair}",
            predictions,
            evidence,
            runtime,
            key,
            root,
        )
        write_experiment_ledger_row(ledger_path, ledger_row)
        results[policy] = {
            **evidence,
            "selected_thresholds": list(pair),
            "stable_tuning_gate": bool(selected_row["stable_tuning_gate"]),
        }

    metrics_root = root / "research/regime_adaptive_selector/metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    search.to_csv(metrics_root / "phase1_threshold_search.csv", index=False)
    payload = {
        "experiment_phase": "direct_expansion_quality_gate",
        "active_model_changed": False,
        "confirmation_used_for_selection": False,
        "locked_evaluation_read": False,
        "locked_origins": settings["locked_origins"],
        "quality_feature_contract": quality_settings[
            "feature_contract_version"
        ],
        "quality_fit_through_max": int(
            quality_panel["quality_fit_through_origin"].dropna().max()
        ),
        "results": results,
    }
    atomic_write_json(payload, metrics_root / "phase1_summary.json")
    return payload


if __name__ == "__main__":
    print(
        json.dumps(
            build_expansion_policy_experiments(),
            indent=2,
            sort_keys=True,
        )
    )
