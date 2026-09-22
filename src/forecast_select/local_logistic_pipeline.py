"""Walk-forward driver for the Local / Interaction Logistic Up Selector study.

Everything downstream of the ``p_up`` signal (signed correlation graph,
trailing 48-month indicator prior, top-15 selection) is reused verbatim from
the production Uptrend Selector so that the only thing that varies between
candidates is the underlying logistic signal.

Locked-set guard
----------------
``configs/local_logistic_experiment.yaml`` pins ``maximum_readable_position``
to 267. The workbook is loaded with ``nrows`` capped at that value, so rows
268-316 - and therefore the locked origins 268-315 - are never read from disk
by this experiment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from .features import build_feature_panel
from .indicator_selection import (
    propagate_correlation_graph,
    select_top_indicators,
    without_self_loops,
)
from .io import load_workbook
from .local_logistic import (
    fit_interaction_logistic_model,
    fit_local_logistic_models,
    interaction_coefficients,
    local_coefficients,
    local_feature_columns,
    predict_interaction_probability,
    predict_local_probability,
)
from .targets import build_targets
from .uptrend_model import fit_uptrend_model, predict_uptrend_probability
from .validation import assert_target_history_available, causal_training_rows


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_CONFIG = "configs/local_logistic_experiment.yaml"


def read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_experiment_settings(root: Path = ROOT) -> dict:
    return read_yaml(root / EXPERIMENT_CONFIG)


# ---------------------------------------------------------------------------
# Locked-set guard
# ---------------------------------------------------------------------------


def assert_origins_unlocked(origins: Iterable[int], settings: dict) -> None:
    """Refuse any origin at or beyond the locked evaluation window."""
    locked_start, locked_end = (int(value) for value in settings["locked_origins"])
    maximum = int(settings["maximum_readable_position"])
    offending = sorted(
        {int(origin) for origin in origins if int(origin) >= locked_start}
    )
    if offending:
        raise AssertionError(
            "Locked origins "
            f"{locked_start}-{locked_end} must never be evaluated; "
            f"refused {offending[:5]}"
        )
    beyond = sorted({int(origin) for origin in origins if int(origin) > maximum - 1})
    if beyond:
        raise AssertionError(
            f"Origins beyond the readable horizon {maximum - 1}: {beyond[:5]}"
        )


# ---------------------------------------------------------------------------
# Panel preparation
# ---------------------------------------------------------------------------


@dataclass
class ExperimentPanel:
    frame: pd.DataFrame
    targets: pd.DataFrame
    panel: pd.DataFrame
    eligible: pd.DataFrame
    graph: pd.DataFrame
    config: dict
    settings: dict


def prepare_experiment_panel(root: Path = ROOT) -> ExperimentPanel:
    """Build the shared causal panel once, capped before the locked origins."""
    config = read_yaml(root / "configs/config.yaml")
    settings = load_experiment_settings(root)
    maximum_position = int(settings["maximum_readable_position"])
    locked_start = int(settings["locked_origins"][0])
    if maximum_position >= locked_start:
        raise AssertionError(
            "maximum_readable_position must stop before the locked origins"
        )
    frame = load_workbook(
        root / config["data_path"],
        maximum_position=maximum_position,
    )
    targets = build_targets(frame)
    lag = int(settings["availability_lag_months"])
    features = build_feature_panel(
        frame,
        availability_lag=lag,
        include_structured=True,
    )
    panel = features.merge(
        targets[[
            "origin_position", "indicator_id", "target_date", "y_true",
            "zero_change", "target_available", "value_t", "value_t1",
        ]],
        on=["origin_position", "indicator_id"],
        how="left",
        validate="one_to_one",
    )
    panel["eligible"] = (
        panel["target_available"]
        & panel["observed"].eq(1)
        & panel["origin_position"].gt(int(config["minimum_history_months"]))
    )
    panel["data_quality_ok"] = panel["eligible"]
    eligible = panel[panel["eligible"]].copy()

    graph_config = settings["graph"]
    indicators = [column for column in frame.columns if column.startswith("X")]
    graph = without_self_loops(frame[indicators].diff().iloc[
        :int(graph_config["estimation_end"])
    ].corr(min_periods=int(graph_config["minimum_pairs"])))
    return ExperimentPanel(
        frame=frame,
        targets=targets,
        panel=panel,
        eligible=eligible,
        graph=graph,
        config=config,
        settings=settings,
    )


# ---------------------------------------------------------------------------
# Shared selector stage (graph -> prior -> top 15)
# ---------------------------------------------------------------------------


def apply_selector(
    predictions: pd.DataFrame,
    probability_column: str,
    shared: ExperimentPanel,
) -> pd.DataFrame:
    """Run the frozen graph, prior blend and top-15 selector on one signal.

    ``predictions`` must carry ``origin_position``, ``indicator_id``,
    ``y_true`` and the named probability column. The returned frame adds the
    production selection columns (``accepted``, ``selection_rank``,
    ``predicted_direction``, ...).
    """
    settings = shared.settings
    graph_config = settings["graph"]
    lag = int(settings["availability_lag_months"])
    working = predictions.copy()
    working["p_up_raw"] = pd.to_numeric(
        working[probability_column], errors="coerce"
    ).to_numpy(dtype=float)
    working["p_up"] = working["p_up_raw"].to_numpy(dtype=float)

    graph_parts = []
    for _, group in working.groupby("origin_position", sort=True):
        current = group.copy()
        correlation = shared.graph.reindex(
            index=current["indicator_id"],
            columns=current["indicator_id"],
        ).fillna(0.0).to_numpy(dtype=float)
        current["p_up"] = propagate_correlation_graph(
            current["p_up"].to_numpy(dtype=float),
            correlation,
            alpha=float(graph_config["alpha"]),
        )
        current["predicted_direction"] = np.where(
            current["p_up"] >= 0.5, "Up", "Down"
        )
        graph_parts.append(current)
    graph_predictions = pd.concat(graph_parts, ignore_index=True)

    selection = settings["selection"]
    target_history = shared.targets.loc[
        shared.targets["target_available"] & shared.targets["y_true"].notna(),
        ["origin_position", "indicator_id", "y_true"],
    ]
    result = select_top_indicators(
        target_history,
        graph_predictions,
        cap=int(selection["monthly_selection_count"]),
        prior_window=int(selection["trailing_target_window"]),
        prior_weight=float(selection["indicator_prior_weight"]),
        minimum_history_months=int(selection["minimum_history_months"]),
        minimum_indicator_history=int(selection["minimum_indicator_history"]),
        availability_lag=lag,
    )
    if (result["calibration_fit_through_origin"] > result["origin_position"] - 2).any():
        raise AssertionError("Selector used an unavailable target")
    accepted = result[result["accepted"]]
    counts = accepted.groupby("origin_position")["indicator_id"].agg(
        ["count", "nunique"]
    )
    expected = int(selection["monthly_selection_count"])
    if not counts["count"].eq(expected).all() or not counts["nunique"].eq(expected).all():
        raise AssertionError(
            f"The selector must return {expected} unique indicators per month"
        )
    return result


# ---------------------------------------------------------------------------
# Walk-forward signal generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalSpec:
    """One (feature set, regularization) local-model configuration."""

    feature_set: str
    logistic_c: float

    @property
    def key(self) -> str:
        return f"{self.feature_set}_c{self.logistic_c:g}"


def run_walk_forward(
    shared: ExperimentPanel,
    origins: Iterable[int],
    local_specs: Iterable[LocalSpec],
    interaction_cs: Iterable[float] = (),
    collect_coefficients: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    """Fit every candidate at every origin and return aligned raw signals.

    All candidates share one causal training slice and one test slice per
    origin, so they are always compared on identical eligible rows.
    """
    settings = shared.settings
    origins = [int(origin) for origin in origins]
    assert_origins_unlocked(origins, settings)
    lag = int(settings["availability_lag_months"])
    seed = int(shared.config["seed"])
    global_config = settings["global_model"]
    local_config = settings["local_model"]
    interaction_config = settings["interaction_model"]
    minimum_grid_rows = int(min(local_config["minimum_local_rows_grid"]))
    specs = list(local_specs)
    interaction_cs = [float(value) for value in interaction_cs]

    frames: list[pd.DataFrame] = []
    local_coefficient_frames: list[pd.DataFrame] = []
    interaction_coefficient_frames: list[pd.DataFrame] = []
    model_size: dict[str, int] = {}
    started = time.perf_counter()

    for position, origin in enumerate(origins):
        train = causal_training_rows(shared.eligible, origin, availability_lag=lag)
        assert_target_history_available(train, origin, availability_lag=lag)
        if int(train["origin_position"].max()) > origin - lag - 1:
            raise AssertionError("Training rows violate the t-2 label rule")
        test = shared.eligible[
            shared.eligible["origin_position"].eq(origin)
        ].copy()
        if test.empty:
            continue

        global_model = fit_uptrend_model(
            train,
            seed=seed,
            logistic_c=float(global_config["logistic_c"]),
            max_iter=int(global_config["logistic_max_iter"]),
        )
        p_global = predict_uptrend_probability(global_model, test)
        model_size.setdefault(
            "global_coefficients",
            int(global_model.model.named_steps["classifier"].coef_.shape[1]),
        )

        record = test[[
            "origin_position", "origin_date", "target_date", "indicator_id",
            "y_true", "eligible", "data_quality_ok",
        ]].copy().reset_index(drop=True)
        record["p_global"] = p_global

        rows_recorded = False
        for spec in specs:
            columns = local_feature_columns(spec.feature_set)
            local_model = fit_local_logistic_models(
                train,
                feature_columns=columns,
                seed=seed,
                logistic_c=float(spec.logistic_c),
                max_iter=int(local_config["logistic_max_iter"]),
                minimum_local_rows=minimum_grid_rows,
                minimum_local_class_rows=int(
                    local_config["minimum_local_class_rows"]
                ),
                solver=str(local_config["solver"]),
            )
            probability, counts, fallback = predict_local_probability(
                local_model, test, p_global
            )
            record[f"p_local__{spec.key}"] = probability
            record[f"fallback__{spec.key}"] = fallback
            if not rows_recorded:
                record["local_training_rows"] = counts
                rows_recorded = True
            model_size.setdefault(
                f"local_coefficients__{spec.feature_set}", len(columns)
            )
            if collect_coefficients:
                coefficients = local_coefficients(local_model)
                if not coefficients.empty:
                    coefficients.insert(0, "origin_position", origin)
                    coefficients.insert(1, "spec", spec.key)
                    local_coefficient_frames.append(coefficients)

        for value in interaction_cs:
            interaction_model = fit_interaction_logistic_model(
                train,
                seed=seed,
                logistic_c=value,
                max_iter=int(interaction_config["logistic_max_iter"]),
                interaction_features=list(interaction_config["features"]),
            )
            record[f"p_interaction__c{value:g}"] = predict_interaction_probability(
                interaction_model, test
            )
            model_size.setdefault(
                "interaction_coefficients", interaction_model.n_coefficients
            )
            model_size.setdefault(
                "interaction_terms",
                len(interaction_model.interaction_columns)
                * len(interaction_model.encoder.categories_[0]),
            )
            if collect_coefficients:
                coefficients = interaction_coefficients(interaction_model)
                coefficients.insert(0, "origin_position", origin)
                coefficients.insert(1, "logistic_c", value)
                interaction_coefficient_frames.append(coefficients)

        frames.append(record)
        if progress and (position + 1) % 10 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"  origin {origin} ({position + 1}/{len(origins)}) "
                f"{elapsed:.1f}s",
                flush=True,
            )

    signals = pd.concat(frames, ignore_index=True)
    return {
        "signals": signals,
        "local_coefficients": (
            pd.concat(local_coefficient_frames, ignore_index=True)
            if local_coefficient_frames
            else pd.DataFrame()
        ),
        "interaction_coefficients": (
            pd.concat(interaction_coefficient_frames, ignore_index=True)
            if interaction_coefficient_frames
            else pd.DataFrame()
        ),
        "model_size": model_size,
        "runtime_seconds": float(time.perf_counter() - started),
    }


# ---------------------------------------------------------------------------
# Post-hoc candidate construction
# ---------------------------------------------------------------------------


def effective_local(
    signals: pd.DataFrame,
    spec_key: str,
    minimum_local_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a minimum-history threshold on top of the fitted local signal.

    Fitting used the smallest threshold in the grid, so a larger threshold is
    exactly equivalent to re-flagging those indicators as fallback rows.
    """
    p_global = signals["p_global"].to_numpy(dtype=float)
    p_local = signals[f"p_local__{spec_key}"].to_numpy(dtype=float)
    fallback = signals[f"fallback__{spec_key}"].to_numpy(dtype=bool)
    rows = signals["local_training_rows"].to_numpy(dtype=float)
    effective_fallback = fallback | (rows < float(minimum_local_rows))
    return np.where(effective_fallback, p_global, p_local), effective_fallback
