from __future__ import annotations

from pathlib import Path

import pandas as pd
import json


def write_final_report(root: Path) -> Path:
    metrics = sorted((root / "reports/tables").glob("*_metrics.csv"))
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
