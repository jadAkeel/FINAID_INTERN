"""Build the full causal Down feature panel once and cache it.

This is the workhorse for all downstream Down-model research. It rebuilds
the same causal feature panel used by the frozen Directional Downside
experiment (features through t-1, labels through t-2) plus the blend p_down
and the regime columns needed for regime analysis.

Writes: research/down_v2/artifacts/down_panel.parquet
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from forecast_select.directional_downside import (  # noqa: E402
    build_directional_downside_features,
)
from forecast_select.io import load_workbook  # noqa: E402
from forecast_select.targets import build_targets  # noqa: E402
from forecast_select.validation import assert_target_alignment  # noqa: E402

OUT = ROOT / "research/down_v2/artifacts/down_panel.parquet"


def main() -> None:
    started = time.perf_counter()
    config = yaml.safe_load((ROOT / "configs/config.yaml").read_text(encoding="utf-8"))
    settings = yaml.safe_load(
        (ROOT / "configs/directional_downside_model.yaml").read_text(encoding="utf-8")
    )
    end = int(settings["confirmation_origins"][1])
    frame = load_workbook(ROOT / config["data_path"], maximum_position=end + 1)
    targets = build_targets(frame)
    assert_target_alignment(targets, frame)

    base_panel = targets[
        [
            "origin_position",
            "indicator_id",
            "origin_date",
            "target_date",
            "y_true",
            "target_available",
            "value_t",
        ]
    ].copy()
    base_panel["eligible"] = (
        base_panel["target_available"].fillna(False).astype(bool)
        & base_panel["value_t"].notna()
        & base_panel["origin_position"].gt(int(config["minimum_history_months"]))
    )
    base_panel["data_quality_ok"] = base_panel["eligible"]

    feat_settings = settings["features"]
    features = build_directional_downside_features(
        frame,
        availability_lag=int(settings["availability_lag_months"]),
        lead_correlation_window=int(feat_settings["lead_correlation_window"]),
        lead_minimum_pairs=int(feat_settings["lead_minimum_pairs"]),
        lead_top_k=int(feat_settings["lead_top_k"]),
    )
    panel = features.merge(
        base_panel[
            [
                "origin_position",
                "indicator_id",
                "origin_date",
                "target_date",
                "y_true",
                "eligible",
                "data_quality_ok",
            ]
        ],
        on=["origin_position", "indicator_id"],
        how="left",
        validate="one_to_one",
    )
    panel["down_target"] = 1.0 - panel["y_true"]
    panel["down_eligible"] = (
        panel["eligible"].fillna(False).astype(bool) & panel["down_target"].notna()
    )
    # Keep the FULL history: training at origin 120 needs rows from origins <= 118.
    # Evaluation scripts filter to the 120..266 evaluation range themselves.
    panel["evaluation_origin"] = panel["origin_position"].between(120, end)
    panel = panel[panel["origin_position"] <= end].copy()
    panel["data_hash"] = str(
        __import__("forecast_select.io", fromlist=["sha256_file"]).sha256_file(
            ROOT / config["data_path"]
        )
    )
    panel["locked_evaluation_read"] = False
    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT, index=False)

    print(f"rows={len(panel)} origins={panel['origin_position'].nunique()}")
    ev = panel[panel["evaluation_origin"]]
    print(f"evaluation origins 120..{end}: rows={len(ev)}")
    print(f"down_eligible rows={int(panel['down_eligible'].sum())}")
    print(f"down base rate (all)={panel.loc[panel['down_eligible'], 'down_target'].mean():.4f}")
    print(f"n feature columns={len([c for c in panel.columns if c.startswith('down_')])}")
    print(f"runtime={time.perf_counter() - started:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
