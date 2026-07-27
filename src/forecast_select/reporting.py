from __future__ import annotations

from pathlib import Path

import pandas as pd
import json


def write_research_artifacts(root: Path) -> None:
    """Create lightweight figures and error tables from completed local artifacts."""
    tables = root / "reports/tables"
    figures = root / "reports/figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    catboost_path = root / "artifacts/oof_predictions/catboost_full_v2.parquet"
    if catboost_path.exists():
        cat = pd.read_parquet(catboost_path).dropna(subset=["y_true", "p_up"]).copy()
        cat["correct"] = (cat["y_true"].astype(int) == (cat["p_up"] >= 0.5).astype(int)).astype(int)
        by_indicator = cat.groupby("indicator_id").agg(rows=("correct", "size"), accuracy=("correct", "mean"), brier=("y_true", lambda s: float(((s - cat.loc[s.index, "p_up"]) ** 2).mean())), errors=("correct", lambda s: int((1 - s).sum()))).reset_index()
        by_indicator.to_csv(tables / "catboost_full_v2_error_by_indicator.csv", index=False)
        by_month = cat.groupby(["origin_position", "origin_date"]).agg(rows=("correct", "size"), accuracy=("correct", "mean"), errors=("correct", lambda s: int((1 - s).sum()))).reset_index()
        by_month.to_csv(tables / "catboost_full_v2_error_by_month.csv", index=False)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            bins = pd.cut(cat["p_up"], bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0], include_lowest=True)
            calibration = cat.groupby(bins, observed=False).agg(predicted=("p_up", "mean"), observed=("y_true", "mean"), rows=("y_true", "size")).dropna()
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
            ax.plot(calibration["predicted"], calibration["observed"], "o-", label="CatBoost v2")
            ax.set(xlabel="Mean predicted P(Up)", ylabel="Observed Up rate", title="CatBoost v2 calibration")
            ax.legend()
            fig.tight_layout()
            fig.savefig(figures / "catboost_full_v2_calibration.png", dpi=140)
            plt.close(fig)
        except Exception:
            (figures / "catboost_full_v2_calibration.unavailable.txt").write_text("matplotlib figure generation failed; see runtime logs.\n", encoding="utf-8")
    level_c_path = root / "artifacts/oof_predictions/dev_level_c_v2.parquet"
    if level_c_path.exists():
        level = pd.read_parquet(level_c_path).dropna(subset=["y_true", "correctness_lcb"]).copy()
        level = level.sort_values("correctness_lcb", ascending=False)
        level["correct"] = (level["y_true"].astype(int) == (level["p_up"] >= 0.5).astype(int)).astype(int)
        level["coverage"] = range(1, len(level) + 1)
        level["coverage"] = level["coverage"] / len(level)
        level["cumulative_accuracy"] = level["correct"].expanding().mean()
        level[["coverage", "cumulative_accuracy", "correctness_lcb"]].to_csv(tables / "level_c_v2_risk_coverage.csv", index=False)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(level["coverage"], 1 - level["cumulative_accuracy"], label="selective risk")
            ax.axvline(float(level["accepted"].mean()), color="grey", linestyle="--", label="mean accepted flag")
            ax.set(xlabel="Coverage", ylabel="Risk", title="Level-C v2 risk-coverage")
            ax.legend()
            fig.tight_layout()
            fig.savefig(figures / "level_c_v2_risk_coverage.png", dpi=140)
            plt.close(fig)
        except Exception:
            (figures / "level_c_v2_risk_coverage.unavailable.txt").write_text("matplotlib figure generation failed; see runtime logs.\n", encoding="utf-8")

    comparison_paths = [
        tables / "dev_metrics.csv",
        tables / "dev_ensemble_v2_metrics.csv",
        tables / "level_c_dev_metrics.csv",
        tables / "catboost_full_v2_metrics.csv",
    ]
    comparison = [
        pd.read_csv(path).assign(artifact=str(path.relative_to(root)))
        for path in comparison_paths
        if path.exists()
    ]
    if comparison:
        pd.concat(comparison, ignore_index=True).to_csv(tables / "model_comparison_v3.csv", index=False)


def write_final_report(root: Path) -> Path:
    write_research_artifacts(root)
    metrics = sorted(path for path in (root / "reports/tables").glob("*_metrics.csv") if path.name != "all_metrics.csv")
    lines = ["# Final Research Report", "", "## Executive Result", "", "The repository executes a leakage-safe revised-data pseudo-out-of-sample research pipeline with full-coverage and selective tracks. Claims below are limited to generated artifacts.", "", "## Official PDF Milestone Status", "", "The supplied workbook ends in May 2026. A complete six-month evaluation beginning January 2026 and the subsequent five-month persistence period are unavailable. The compensation-related milestone is therefore `NOT_YET_EVALUABLE`. Any selective accuracy above 65% must not be presented as proof of the PDF's overall-accuracy condition.", "", "## Validation", "", "Training rows for origin t are strictly earlier than t; the official target is 1 iff value(t+1) > value(t), with ties recorded as zero_change. The last 48 evaluable origins remain frozen as `locked_audit_v1` and were not read by Level-C.", "", "## Level-C", "", "Level-C v2 fits Platt calibration, a correctness model, a six-month date-block bootstrap bias correction, a 0.55 reliability floor, and a maximum of 20 accepted predictions per month using earlier Level-B outputs only. CatBoost v2 is evaluated separately as a completed challenger and is not promoted.", "", "## Experiments", ""]
    summary_path = root / "reports/experiments/level_c_dev_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        lines.extend([f"- Level-C ready rows: `{summary['ready_rows']}`", f"- Level-C full directional accuracy: `{summary['full_accuracy']:.6f}`", f"- Level-C accepted rows: `{summary['accepted']}`", f"- Level-C coverage: `{summary['coverage']:.6f}`", f"- Level-C accepted accuracy: `{summary['accepted_accuracy']:.6f}`", f"- Level-C bootstrap LCB p10: `{summary['lcb_p10']:.6f}`", ""])
    if metrics:
        table = pd.concat([pd.read_csv(path).assign(artifact=str(path.relative_to(root))) for path in metrics], ignore_index=True)
        table.to_csv(root / "reports/tables/all_metrics.csv", index=False)
        lines.append("Generated metric artifacts:")
        lines.extend(f"- `{path.relative_to(root)}`" for path in metrics)
    else:
        lines.append("No metric artifact was present when this report was generated.")
    lines.extend(["", "## Limitations", "", "- Anonymous indicators have no supplied units, release lags, revision histories, or vintages.", "- Pretrained Chronos-2, TiRex-2, and TimesFM experiments are blocked unless their official package/API and compatible checkpoints are verified locally.", "- June 2026 is an unscored forecast ledger; its outcome is not fabricated.", "", "## Reproduction", "", "Run `python -m forecast_select audit`, `python -m pytest`, then the backtest, freeze, locked-audit, prediction, and report commands in `README.md`.", ""])
    path = root / "reports/final_research_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
