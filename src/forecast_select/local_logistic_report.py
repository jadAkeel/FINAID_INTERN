"""Render the Local / Interaction Logistic result tables from the artifacts.

    python -m forecast_select.local_logistic_report

Every number in the research report comes from this renderer so that nothing
is transcribed by hand. It reads only files produced by
``forecast_select.local_logistic_runner`` and writes
``research/local_logistic/metrics/tables.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .local_logistic_pipeline import ROOT


CANDIDATES = {
    "A_global": "A. Global Logistic (baseline)",
    "B_pure_local": "B. Pure Local Logistic",
    "C_fixed_shrinkage": "C. Global + Local fixed shrinkage",
    "D_sample_aware_shrinkage": "D. Global + Local sample-aware shrinkage",
    "E_interaction": "E. Global + indicator interactions",
}
PERIODS = ("tuning", "validation", "confirmation")


def _pct(value) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _num(value, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render(root: Path = ROOT) -> str:
    metrics = root / "research/local_logistic/metrics"
    summary = json.loads((metrics / "summary.json").read_text(encoding="utf-8"))
    frozen = summary["frozen_configuration"]
    parts: list[str] = []

    # ---- selector metrics ------------------------------------------------
    parts.append("### Top-15 selector accuracy by period\n")
    rows = []
    for key, label in CANDIDATES.items():
        row = [label]
        for period in PERIODS:
            block = summary["periods"][period]["candidates"][key]["selector"]
            row.append(f"{block['hits']}/{block['calls']} ({_pct(block['accuracy'])})")
        rows.append(row)
    parts.append(_table(rows, ["Candidate", *[p.capitalize() for p in PERIODS]]))

    parts.append("\n### Top-15 hit delta versus the Global baseline\n")
    rows = []
    for key, label in CANDIDATES.items():
        if key == "A_global":
            continue
        row = [label]
        for period in PERIODS:
            base = summary["periods"][period]["candidates"]["A_global"]["selector"]
            block = summary["periods"][period]["candidates"][key]["selector"]
            delta = block["hits"] - base["hits"]
            row.append(f"{delta:+d} hits ({(block['accuracy'] - base['accuracy']) * 100:+.2f} pp)")
        rows.append(row)
    parts.append(_table(rows, ["Candidate", *[p.capitalize() for p in PERIODS]]))

    # ---- monthly dispersion ---------------------------------------------
    parts.append("\n### Monthly selector accuracy dispersion\n")
    rows = []
    for key, label in CANDIDATES.items():
        for period in PERIODS:
            block = summary["periods"][period]["candidates"][key]["selector"]
            rows.append([
                label, period, str(block["months"]),
                _pct(block["monthly_mean"]), _pct(block["monthly_median"]),
                _num(block["monthly_std"]),
            ])
    parts.append(_table(
        rows,
        ["Candidate", "Period", "Months", "Monthly mean", "Monthly median", "Monthly SD"],
    ))

    # ---- raw model metrics ----------------------------------------------
    parts.append("\n### Raw `p_up` model quality (before graph and prior blending)\n")
    rows = []
    for key, label in CANDIDATES.items():
        for period in PERIODS:
            block = summary["periods"][period]["candidates"][key]["raw"]
            rows.append([
                label, period, str(block["rows"]), _pct(block["accuracy"]),
                _num(block["auc"]), _num(block["brier"]), _num(block["log_loss"]),
                _pct(block["base_rate"]),
            ])
    parts.append(_table(
        rows,
        ["Candidate", "Period", "Rows", "Accuracy", "AUC", "Brier", "Log loss",
         "Up base rate"],
    ))

    # ---- paired comparison ----------------------------------------------
    parts.append(
        "\n### Paired monthly comparison versus Global "
        "(circular block bootstrap, 6-month blocks, 500 replicates)\n"
    )
    rows = []
    for key, label in CANDIDATES.items():
        if key == "A_global":
            continue
        for period in PERIODS:
            block = summary["periods"][period]["candidates"][key].get(
                "paired_vs_global", {}
            )
            if not block:
                continue
            rows.append([
                label, period,
                f"{block['mean_monthly_difference'] * 100:+.2f} pp",
                f"{block['months_better']}/{block['months_worse']}/{block['months_equal']}",
                f"[{block['block_bootstrap_p05'] * 100:+.2f}, "
                f"{block['block_bootstrap_p95'] * 100:+.2f}] pp",
                "yes" if block["interval_excludes_zero"] else "no",
            ])
    parts.append(_table(
        rows,
        ["Candidate", "Period", "Mean monthly delta", "Better/worse/equal months",
         "90% interval", "Excludes zero"],
    ))

    # ---- selection overlap ----------------------------------------------
    parts.append("\n### Selection overlap with the Global baseline\n")
    rows = []
    for key, label in CANDIDATES.items():
        if key == "A_global":
            continue
        for period in PERIODS:
            block = summary["periods"][period]["candidates"][key].get(
                "overlap_vs_global", {}
            )
            if not block:
                continue
            rows.append([
                label, period,
                f"{block['mean_overlap_count']:.2f}/15",
                _pct(block["mean_overlap_pct"]),
                f"{block['added_correct']}/{block['added_calls']} "
                f"({_pct(block['added_accuracy'])})",
                f"{block['removed_correct']}/{block['removed_calls']} "
                f"({_pct(block['removed_accuracy'])})",
                f"{block['net_hit_change']:+d}",
            ])
    parts.append(_table(
        rows,
        ["Candidate", "Period", "Mean overlap", "Overlap %",
         "Added calls correct", "Removed calls correct", "Net hits"],
    ))

    # ---- fallback --------------------------------------------------------
    fallback = summary["fallback"]
    parts.append("\n### Local-model fallback frequency\n")
    parts.append(_table(
        [
            ["Pure Local (B)", _pct(fallback["pure_local_rate"])],
            ["Shrinkage candidates (C, D)", _pct(fallback["hybrid_rate"])],
            ["Tuning", _pct(fallback["by_period"]["tuning"])],
            ["Validation", _pct(fallback["by_period"]["validation"])],
            ["Confirmation", _pct(fallback["by_period"]["confirmation"])],
            ["Mean local training rows", _num(fallback["mean_local_training_rows"], 1)],
            ["Min / max local training rows",
             f"{fallback['min_local_training_rows']} / {fallback['max_local_training_rows']}"],
            ["Mean sample-aware weight", _num(fallback["mean_sample_aware_weight"], 4)],
        ],
        ["Measure", "Value"],
    ))

    # ---- model size ------------------------------------------------------
    size_path = root / "research/local_logistic/artifacts/model_size_frozen.json"
    if size_path.exists():
        size = json.loads(size_path.read_text(encoding="utf-8"))
        parts.append("\n### Model size\n")
        parts.append(_table(
            [[key.replace("_", " "), str(value)] for key, value in sorted(size.items())],
            ["Measure", "Count"],
        ))

    # ---- coefficient stability ------------------------------------------
    stability_path = metrics / "coefficient_stability.csv"
    if stability_path.exists():
        stability = pd.read_csv(stability_path)
        parts.append("\n### Local coefficient stability\n")
        parts.append(_table(
            [
                [
                    str(row.feature), _num(row.mean_coefficient),
                    _num(row.cross_indicator_std),
                    f"[{_num(row.cross_indicator_min)}, {_num(row.cross_indicator_max)}]",
                    _num(row.mean_within_indicator_std_over_origins),
                    _num(row.variance_ratio_within_over_between, 2),
                    _pct(row.mean_sign_consistency),
                    f"{int(row.indicators_positive)}/{int(row.indicators_negative)}",
                ]
                for row in stability.itertuples()
            ],
            ["Feature", "Mean slope", "Cross-indicator SD", "Cross-indicator range",
             "Within-indicator SD over origins", "Within/between variance ratio",
             "Sign consistency", "Indicators +/-"],
        ))

    # ---- per-indicator extremes -----------------------------------------
    per_indicator_path = metrics / "per_indicator.csv"
    if per_indicator_path.exists():
        frame = pd.read_csv(per_indicator_path)
        out_of_sample = frame[frame["period"].isin(["validation", "confirmation"])]
        grouped = out_of_sample.groupby("indicator_id").agg(
            eligible_rows=("eligible_rows", "sum"),
            local_rows=("mean_local_training_rows", "mean"),
            raw_global=("raw_accuracy_A_global", "mean"),
            raw_local=("raw_accuracy_B_pure_local", "mean"),
            raw_hybrid=("raw_accuracy_C_fixed_shrinkage", "mean"),
            raw_interaction=("raw_accuracy_E_interaction", "mean"),
            selected_global=("selected_A_global", "sum"),
            selected_hybrid=("selected_C_fixed_shrinkage", "sum"),
        ).reset_index()
        grouped["local_minus_global"] = grouped["raw_local"] - grouped["raw_global"]
        grouped = grouped.sort_values("local_minus_global")
        head = pd.concat([grouped.head(5), grouped.tail(5)])
        parts.append(
            "\n### Per-indicator raw accuracy, out-of-sample "
            "(validation + confirmation): five worst and five best for local modelling\n"
        )
        parts.append(_table(
            [
                [
                    str(row.indicator_id), str(int(row.eligible_rows)),
                    _num(row.local_rows, 0), _pct(row.raw_global), _pct(row.raw_local),
                    _pct(row.raw_hybrid), _pct(row.raw_interaction),
                    f"{row.local_minus_global * 100:+.2f} pp",
                    f"{int(row.selected_global)} / {int(row.selected_hybrid)}",
                ]
                for row in head.itertuples()
            ],
            ["Indicator", "Rows", "Mean local history", "Global", "Pure local",
             "Hybrid", "Interaction", "Local - Global", "Selected global/hybrid"],
        ))

    # ---- tuning search top rows -----------------------------------------
    search_path = metrics / "tuning_search.csv"
    if search_path.exists():
        search = pd.read_csv(search_path)
        parts.append("\n### Tuning search, best five rows per family\n")
        rows = []
        for family, group in search.groupby("family"):
            top = group.sort_values("accuracy", ascending=False).head(5)
            for row in top.itertuples():
                rows.append([
                    str(row.family), str(row.feature_set), _num(row.logistic_c, 2),
                    str(int(row.minimum_local_rows)),
                    "n/a" if pd.isna(row.local_weight) else _num(row.local_weight, 2),
                    f"{int(row.hits)}/{int(row.calls)}", _pct(row.accuracy),
                    f"{int(row.hits_vs_baseline):+d}",
                ])
        parts.append(_table(
            rows,
            ["Family", "Feature set", "C", "Min rows", "w", "Hits/calls",
             "Accuracy", "vs baseline"],
        ))

    parts.append("\n### Frozen configuration\n")
    parts.append("```json\n" + json.dumps(frozen, indent=2) + "\n```")
    parts.append(
        "\n### Locked-set status\n\n"
        f"- Locked origins: `{summary['locked_origins']}`\n"
        f"- Maximum workbook row position read: "
        f"`{summary['maximum_readable_position']}`\n"
        f"- Maximum origin evaluated: `{summary['maximum_origin_evaluated']}`\n"
        f"- Locked origins read: `{summary['locked_origins_read']}`\n"
    )
    return "\n".join(parts) + "\n"


def main() -> int:
    root = ROOT
    text = render(root)
    target = root / "research/local_logistic/metrics/tables.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
