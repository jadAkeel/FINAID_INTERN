from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXPANSION_FEATURE_COLUMNS = [
    "forecast_market_breadth",
    "regime_stress",
    "regime_uncertainty",
    "top15_score_mean",
    "top15_score_min",
    "top15_score_dispersion",
    "rank_15_16_margin",
    "cutoff_score_slope",
    "top15_group_concentration",
    "down_model_disagreement",
    "ready_share",
    "graph_adjustment_concentration",
    "historical_marginal_quality",
]


def _first_numeric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else np.nan


def _band_target(
    ranked: pd.DataFrame,
    lower: int,
    upper: int,
) -> float:
    band = ranked[ranked["base_up_rank"].between(lower, upper)]
    values = pd.to_numeric(band["y_true"], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


def _origin_row(origin: int, frame: pd.DataFrame) -> dict[str, float | int]:
    ready = frame[
        frame["level_c_ready"].fillna(False).astype(bool)
        & frame["base_up_rank"].notna()
    ].copy()
    ready = ready.sort_values("base_up_rank")
    if len(ready) < 20:
        raise ValueError(f"Origin {origin} has fewer than 20 ranked candidates")
    ranks = pd.to_numeric(ready["base_up_rank"], errors="coerce")
    if ranks.isna().any() or not np.equal(ranks, np.floor(ranks)).all():
        raise ValueError(f"Origin {origin} has non-integral base ranks")
    if ranks.duplicated().any() or not set(range(1, 21)).issubset(
        set(ranks.astype(int))
    ):
        raise ValueError(f"Origin {origin} does not have unique base ranks 1-20")
    top15 = ready[ready["base_up_rank"].le(15)]
    scores = ready.set_index("base_up_rank")["p_up_selection_score"]
    score14 = float(scores.loc[14.0])
    score15 = float(scores.loc[15.0])
    score16 = float(scores.loc[16.0])
    score17 = float(scores.loc[17.0])

    down_columns = [
        column
        for column in [
            "p_down_global",
            "p_down_local",
            "p_down_pattern",
            "p_down_indicator_prior",
        ]
        if column in top15
    ]
    if down_columns:
        disagreement = float(
            top15[down_columns]
            .apply(pd.to_numeric, errors="coerce")
            .std(axis=1, ddof=0)
            .mean()
        )
    else:
        disagreement = np.nan
    if "asset_group" in top15:
        group_concentration = float(
            top15["asset_group"].value_counts(normalize=True).max()
        )
    else:
        group_concentration = np.nan
    if {
        "p_up_generalized_calibrated",
        "p_up_calibrated",
    }.issubset(top15.columns):
        graph_adjustment = float(
            (
                pd.to_numeric(
                    top15["p_up_generalized_calibrated"], errors="coerce"
                )
                - pd.to_numeric(top15["p_up_calibrated"], errors="coerce")
            )
            .abs()
            .mean()
        )
    else:
        graph_adjustment = np.nan
    stress = _first_numeric(frame, "regime_stress")
    return {
        "origin_position": int(origin),
        "forecast_market_breadth": _first_numeric(
            frame, "forecast_market_breadth"
        ),
        "regime_stress": stress,
        "regime_uncertainty": (
            1.0 - 2.0 * abs(stress - 0.5) if np.isfinite(stress) else np.nan
        ),
        "top15_score_mean": float(top15["p_up_selection_score"].mean()),
        "top15_score_min": float(top15["p_up_selection_score"].min()),
        "top15_score_dispersion": float(
            top15["p_up_selection_score"].std(ddof=0)
        ),
        "rank_15_16_margin": score15 - score16,
        "cutoff_score_slope": (score14 - score17) / 3.0,
        "top15_group_concentration": group_concentration,
        "down_model_disagreement": disagreement,
        "ready_share": float(len(ready) / len(frame)),
        "graph_adjustment_concentration": graph_adjustment,
        "quality_16_17": _band_target(ready, 16, 17),
        "quality_18_20": _band_target(ready, 18, 20),
        "quality_16_20": _band_target(ready, 16, 20),
    }


def _add_historical_quality(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    history_values = []
    for origin in result["origin_position"]:
        history = result[
            result["origin_position"].le(int(origin) - 2)
            & result["quality_16_20"].notna()
        ].tail(24)
        history_values.append(
            float(history["quality_16_20"].mean())
            if not history.empty
            else np.nan
        )
    result["historical_marginal_quality"] = history_values
    return result


def _walk_forward_quality_forecast(
    panel: pd.DataFrame,
    target_column: str,
    *,
    minimum_history: int,
    ridge_alpha: float,
) -> tuple[pd.Series, pd.Series]:
    forecasts = []
    fit_through = []
    for row in panel.itertuples(index=False):
        origin = int(row.origin_position)
        train = panel[
            panel["origin_position"].le(origin - 2)
            & panel[target_column].notna()
        ]
        fallback = float(np.clip(row.forecast_market_breadth, 0.0, 1.0))
        if len(train) < minimum_history:
            forecasts.append(fallback)
            fit_through.append(np.nan)
            continue
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=float(ridge_alpha))),
        ])
        model.fit(train[EXPANSION_FEATURE_COLUMNS], train[target_column])
        prediction = float(
            model.predict(
                panel.loc[
                    panel["origin_position"].eq(origin),
                    EXPANSION_FEATURE_COLUMNS,
                ]
            )[0]
        )
        forecasts.append(float(np.clip(prediction, 0.0, 1.0)))
        fit_through.append(int(train["origin_position"].max()))
    return (
        pd.Series(forecasts, index=panel.index, dtype=float),
        pd.Series(fit_through, index=panel.index, dtype=float),
    )


def build_expansion_quality_panel(
    replay_panel: pd.DataFrame,
    *,
    minimum_history: int = 12,
    ridge_alpha: float = 10.0,
) -> pd.DataFrame:
    """Forecast marginal-rank quality with labels available only through t-2."""
    required = {
        "origin_position",
        "indicator_id",
        "base_up_rank",
        "p_up_selection_score",
        "level_c_ready",
        "forecast_market_breadth",
        "regime_stress",
        "y_true",
    }
    missing = required.difference(replay_panel.columns)
    if missing:
        raise ValueError(f"Expansion panel is missing columns: {sorted(missing)}")
    rows = [
        _origin_row(int(origin), frame)
        for origin, frame in replay_panel.groupby("origin_position", sort=True)
    ]
    panel = _add_historical_quality(pd.DataFrame(rows))
    forecast_16_17, fit_16_17 = _walk_forward_quality_forecast(
        panel,
        "quality_16_17",
        minimum_history=minimum_history,
        ridge_alpha=ridge_alpha,
    )
    forecast_18_20, fit_18_20 = _walk_forward_quality_forecast(
        panel,
        "quality_18_20",
        minimum_history=minimum_history,
        ridge_alpha=ridge_alpha,
    )
    panel["forecast_quality_16_17"] = forecast_16_17
    panel["forecast_quality_18_20"] = forecast_18_20
    panel["quality_fit_through_origin"] = pd.concat(
        [fit_16_17, fit_18_20], axis=1
    ).max(axis=1)
    fitted = panel["quality_fit_through_origin"].notna()
    if (
        panel.loc[fitted, "quality_fit_through_origin"]
        > panel.loc[fitted, "origin_position"] - 2
    ).any():
        raise AssertionError("Expansion quality model used unavailable labels")
    panel["quality_model"] = "ridge_alpha_" + str(float(ridge_alpha))
    panel["quality_minimum_history"] = int(minimum_history)
    return panel


def graduated_cap_schedule(
    panel: pd.DataFrame,
    *,
    lower_threshold: float,
    upper_threshold: float,
) -> dict[int, int]:
    if not 0.0 <= lower_threshold < upper_threshold <= 1.0:
        raise ValueError("Graduated thresholds must be ordered inside [0, 1]")
    schedule = {}
    for row in panel.itertuples(index=False):
        score = float(row.forecast_market_breadth)
        schedule[int(row.origin_position)] = (
            20 if score >= upper_threshold else 17 if score >= lower_threshold else 15
        )
    return schedule


def expansion_quality_cap_schedule(
    panel: pd.DataFrame,
    *,
    rank_16_17_threshold: float,
    rank_18_20_threshold: float,
) -> dict[int, int]:
    if not 0.0 <= rank_16_17_threshold <= 1.0:
        raise ValueError("Rank 16-17 threshold must be inside [0, 1]")
    if not 0.0 <= rank_18_20_threshold <= 1.0:
        raise ValueError("Rank 18-20 threshold must be inside [0, 1]")
    schedule = {}
    for row in panel.itertuples(index=False):
        admit_17 = float(row.forecast_quality_16_17) >= rank_16_17_threshold
        admit_20 = (
            admit_17
            and float(row.forecast_quality_18_20) >= rank_18_20_threshold
        )
        schedule[int(row.origin_position)] = 20 if admit_20 else 17 if admit_17 else 15
    return schedule


def _monthly_hits(
    predictions: pd.DataFrame,
    bounds: Iterable[int],
) -> pd.Series:
    start, end = (int(value) for value in bounds)
    selected = predictions[
        predictions["accepted"].fillna(False).astype(bool)
        & predictions["origin_position"].between(start, end)
        & predictions["y_true"].notna()
    ].copy()
    selected["correct"] = (
        selected["predicted_direction"].eq("Up")
        & selected["y_true"].eq(1.0)
    ) | (
        selected["predicted_direction"].eq("Down")
        & selected["y_true"].eq(0.0)
    )
    origins = pd.Index(range(start, end + 1), name="origin_position")
    return selected.groupby("origin_position")["correct"].sum().reindex(
        origins, fill_value=0
    )


def paired_monthly_hit_bootstrap(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    bounds: Iterable[int],
    *,
    block_months: int,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    delta = (
        _monthly_hits(candidate, bounds) - _monthly_hits(baseline, bounds)
    ).to_numpy(dtype=float)
    if len(delta) < block_months:
        raise ValueError("Not enough months for the requested bootstrap block")
    rng = np.random.default_rng(seed)
    samples = []
    maximum_start = len(delta) - block_months
    for _ in range(replicates):
        sampled = []
        while len(sampled) < len(delta):
            start = int(rng.integers(0, maximum_start + 1))
            sampled.extend(delta[start : start + block_months])
        samples.append(float(np.mean(sampled[: len(delta)])))
    return {
        "months": int(len(delta)),
        "total_hit_delta": int(delta.sum()),
        "mean_monthly_hit_delta": float(delta.mean()),
        "bootstrap_p10": float(np.quantile(samples, 0.10)),
        "bootstrap_p90": float(np.quantile(samples, 0.90)),
    }
