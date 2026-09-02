from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_FAMILY_COLUMNS = {
    "trend_persistence": [
        "signed_run_length",
        "momentum_acceleration_3",
    ],
    "cross_sectional_dynamics": [
        "cross_section_rank_change_1",
        "cross_section_rank_change_3",
        "breadth_impulse_3",
        "dispersion_change_3",
    ],
    "risk_normalized": [
        "return_volatility_12",
        "risk_normalized_momentum_3",
        "risk_normalized_momentum_6",
    ],
}


def _stale_run(series: pd.Series) -> pd.Series:
    changed = series.diff().ne(0) & series.notna() & series.shift(1).notna()
    groups = changed.cumsum()
    return series.notna().groupby(groups).cumcount().astype(float).where(series.notna(), np.nan)


def _signed_run_length(changes: pd.Series) -> pd.Series:
    values = []
    previous_sign = 0.0
    run = 0
    for value in pd.to_numeric(changes, errors="coerce"):
        if not np.isfinite(value) or value == 0:
            previous_sign = 0.0
            run = 0
            values.append(np.nan if not np.isfinite(value) else 0.0)
            continue
        sign = float(np.sign(value))
        run = run + 1 if sign == previous_sign else 1
        previous_sign = sign
        values.append(sign * run)
    return pd.Series(values, index=changes.index, dtype=float)


def _canonicalize_component_signs(loadings: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Make rolling PCA component signs deterministic across refits."""
    for component in range(loadings.shape[0]):
        pivot = int(np.argmax(np.abs(loadings[component])))
        if loadings[component, pivot] < 0:
            loadings[component] *= -1
            scores[:, component] *= -1
    return loadings, scores


def build_structured_feature_panel(
    frame: pd.DataFrame,
    availability_lag: int = 1,
    pca_window: int = 60,
    pca_components: int = 2,
    correlation_window: int = 60,
    correlation_top_k: int = 3,
) -> pd.DataFrame:
    """Build optional past-only PCA, peer-correlation, and regime features."""
    if pca_window < 24 or correlation_window < 24:
        raise ValueError("Structured feature windows must be at least 24 months")
    if pca_components < 1 or correlation_top_k < 1:
        raise ValueError("Structured feature counts must be positive")
    indicators = [column for column in frame.columns if column.startswith("X")]
    source = frame[indicators].shift(availability_lag)
    changes = source.diff()
    cross_dispersion = changes.std(axis=1)
    cross_breadth = (changes > 0).where(changes.notna()).mean(axis=1)
    regime = pd.DataFrame({
        "origin_position": frame["position"],
        "regime_breadth_3": cross_breadth.rolling(3, min_periods=2).mean(),
        "regime_dispersion_12": cross_dispersion.rolling(12, min_periods=6).mean(),
        "regime_volatility_12": cross_dispersion.rolling(12, min_periods=6).std(),
    })
    rows = []
    for row_index in range(len(frame)):
        pca_history = changes.iloc[max(0, row_index - pca_window + 1):row_index + 1]
        eligible = pca_history.columns[pca_history.notna().sum().ge(24)].tolist()
        factor_scores = np.full(pca_components, np.nan)
        explained = np.full(pca_components, np.nan)
        loadings = {indicator: np.full(pca_components, np.nan) for indicator in indicators}
        if len(eligible) >= 2 and len(pca_history) >= 24:
            matrix = pca_history[eligible].copy()
            matrix = matrix.fillna(matrix.median()).fillna(0.0)
            scale = matrix.std(ddof=0).replace(0, np.nan)
            standardized = matrix.sub(matrix.mean()).div(scale).fillna(0.0).to_numpy(dtype=float)
            _, singular_values, right_vectors = np.linalg.svd(standardized, full_matrices=False)
            components = min(pca_components, right_vectors.shape[0])
            scores = standardized @ right_vectors[:components].T
            pca_loadings, scores = _canonicalize_component_signs(right_vectors[:components].copy(), scores)
            denominator = max(1, standardized.shape[0] - 1)
            variance = (singular_values[:components] ** 2) / denominator
            total_variance = float((singular_values ** 2).sum() / denominator)
            factor_scores[:components] = scores[-1]
            explained[:components] = variance / total_variance if total_variance > 0 else np.nan
            for index, indicator in enumerate(eligible):
                loadings[indicator][:components] = pca_loadings[:, index]
        correlation_history = changes.iloc[max(0, row_index - correlation_window + 1):row_index + 1]
        correlations = correlation_history.corr(min_periods=24)
        current_direction = changes.iloc[row_index]
        for indicator in indicators:
            peer_scores = correlations[indicator].drop(labels=indicator, errors="ignore").dropna() if indicator in correlations else pd.Series(dtype=float)
            selected = peer_scores.abs().sort_values(ascending=False).head(correlation_top_k)
            peers = selected.index.tolist()
            signed = peer_scores.loc[peers] if peers else pd.Series(dtype=float)
            directions = current_direction.reindex(peers)
            weights = signed.abs()
            consensus = np.nan
            if len(directions.dropna()) and float(weights.reindex(directions.dropna().index).sum()) > 0:
                consensus = float(np.average((directions.dropna() > 0).astype(float), weights=weights.reindex(directions.dropna().index)))
            row = {
                "origin_position": int(frame["position"].iloc[row_index]),
                "indicator_id": indicator,
                "peer_corr_abs_topk_mean": float(signed.abs().mean()) if len(signed) else np.nan,
                "peer_corr_signed_top1": float(signed.iloc[0]) if len(signed) else np.nan,
                "peer_direction_consensus": consensus,
                "peer_available_count": float(len(signed)),
            }
            for component in range(pca_components):
                row[f"pca_factor_{component + 1}"] = factor_scores[component]
                row[f"pca_loading_{component + 1}"] = loadings[indicator][component]
                row[f"pca_explained_variance_{component + 1}"] = explained[component]
            rows.append(row)
    structured = pd.DataFrame(rows).merge(regime, on="origin_position", how="left", validate="many_to_one")
    return structured


def build_feature_panel(
    frame: pd.DataFrame,
    availability_lag: int = 1,
    include_structured: bool = False,
    feature_families: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Build causal features. At origin t, feature values use observations through t-lag."""
    indicators = [c for c in frame.columns if c.startswith("X")]
    unknown_families = set(feature_families).difference(
        FEATURE_FAMILY_COLUMNS
    )
    if unknown_families:
        raise ValueError(
            f"Unknown feature families: {sorted(unknown_families)}"
        )
    source = frame[indicators].shift(availability_lag)
    feature_parts = []
    changes = source.diff()
    for indicator in indicators:
        s = source[indicator]
        d = changes[indicator]
        rolling_mean = s.rolling(12, min_periods=6).mean()
        rolling_std = s.rolling(12, min_periods=6).std()
        momentum_3 = s / s.shift(3).replace(0, np.nan) - 1
        momentum_6 = s / s.shift(6).replace(0, np.nan) - 1
        mad = s.rolling(12, min_periods=6).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
        out = pd.DataFrame({
            "origin_position": frame["position"],
            "origin_date": frame["Dates"],
            "indicator_id": indicator,
            "level": s,
            "diff_1": d,
            "pct_change_1": d / s.shift(1).replace(0, np.nan),
            "direction_1": (d > 0).where(d.notna()),
            "direction_lag_1": (d.shift(1) > 0).where(d.shift(1).notna()),
            "direction_lag_2": (d.shift(2) > 0).where(d.shift(2).notna()),
            "direction_lag_3": (d.shift(3) > 0).where(d.shift(3).notna()),
            "direction_lag_6": (d.shift(6) > 0).where(d.shift(6).notna()),
            "direction_lag_12": (d.shift(12) > 0).where(d.shift(12).notna()),
            "change_lag_1": d.shift(1),
            "change_lag_2": d.shift(2),
            "change_lag_3": d.shift(3),
            "change_lag_6": d.shift(6),
            "change_lag_12": d.shift(12),
            "momentum_3": momentum_3,
            "momentum_6": momentum_6,
            "momentum_9": s / s.shift(9).replace(0, np.nan) - 1,
            "momentum_12": s / s.shift(12).replace(0, np.nan) - 1,
            "rolling_mean_12": rolling_mean,
            "rolling_std_12": rolling_std,
            "rolling_mad_12": mad,
            "robust_z_12": (s - rolling_mean) / (1.4826 * mad.replace(0, np.nan)),
            "distance_mean_6": s / s.rolling(6, min_periods=3).mean().replace(0, np.nan) - 1,
            "distance_mean_12": s / rolling_mean.replace(0, np.nan) - 1,
            "stale_run": _stale_run(s),
            "observed": s.notna().astype(float),
            "time_since_observation": (~s.notna()).groupby(s.notna().cumsum()).cumcount().astype(float),
        })
        if "trend_persistence" in feature_families:
            out["signed_run_length"] = _signed_run_length(d)
            previous_momentum_3 = (
                s.shift(3) / s.shift(6).replace(0, np.nan) - 1
            )
            out["momentum_acceleration_3"] = (
                momentum_3 - previous_momentum_3
            )
        if "risk_normalized" in feature_families:
            returns = s.pct_change(fill_method=None)
            return_volatility = returns.rolling(12, min_periods=6).std()
            out["return_volatility_12"] = return_volatility
            out["risk_normalized_momentum_3"] = momentum_3.div(
                return_volatility.mul(np.sqrt(3.0)).replace(0, np.nan)
            )
            out["risk_normalized_momentum_6"] = momentum_6.div(
                return_volatility.mul(np.sqrt(6.0)).replace(0, np.nan)
            )
        feature_parts.append(out)
    panel = pd.concat(feature_parts, ignore_index=True)
    changes_long = changes.stack(future_stack=True).rename("current_change").reset_index()
    changes_long.columns = ["row_index", "indicator_id", "current_change"]
    stats = changes_long.groupby("row_index")["current_change"].agg(
        cross_section_median="median", cross_section_dispersion="std"
    )
    stats["cross_section_breadth"] = changes_long.assign(
        up=(changes_long["current_change"] > 0).where(changes_long["current_change"].notna())
    ).groupby("row_index")["up"].mean()
    cross_section_rank = changes_long.assign(
        cross_section_rank=changes_long.groupby("row_index")["current_change"].rank(pct=True)
    )[["row_index", "indicator_id", "cross_section_rank"]]
    panel["row_index"] = panel["origin_position"] - 1
    panel = panel.join(stats, on="row_index")
    panel = panel.merge(
        cross_section_rank,
        on=["row_index", "indicator_id"],
        how="left",
        validate="one_to_one",
    )
    panel = panel.drop(columns="row_index")
    if "cross_sectional_dynamics" in feature_families:
        panel = panel.sort_values(
            ["indicator_id", "origin_position"]
        ).reset_index(drop=True)
        grouped_rank = panel.groupby("indicator_id", sort=False)[
            "cross_section_rank"
        ]
        panel["cross_section_rank_change_1"] = (
            panel["cross_section_rank"] - grouped_rank.shift(1)
        )
        panel["cross_section_rank_change_3"] = (
            panel["cross_section_rank"] - grouped_rank.shift(3)
        )
        origin_stats = panel[[
            "origin_position",
            "cross_section_breadth",
            "cross_section_dispersion",
        ]].drop_duplicates("origin_position").sort_values("origin_position")
        origin_stats["breadth_impulse_3"] = (
            origin_stats["cross_section_breadth"]
            - origin_stats["cross_section_breadth"].shift(3)
        )
        origin_stats["dispersion_change_3"] = (
            origin_stats["cross_section_dispersion"]
            - origin_stats["cross_section_dispersion"].shift(3)
        )
        panel = panel.merge(
            origin_stats[[
                "origin_position",
                "breadth_impulse_3",
                "dispersion_change_3",
            ]],
            on="origin_position",
            how="left",
            validate="many_to_one",
        )
        panel = panel.sort_values(
            ["origin_position", "indicator_id"]
        ).reset_index(drop=True)
    if include_structured:
        structured = build_structured_feature_panel(frame, availability_lag=availability_lag)
        panel = panel.merge(structured, on=["origin_position", "indicator_id"], how="left", validate="one_to_one")
    return panel
