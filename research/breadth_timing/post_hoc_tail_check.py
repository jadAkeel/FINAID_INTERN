"""Post-hoc, exploratory: what happens in the months with the largest shock signals?

Written AFTER the pre-registered run, to explain why the shock signals' good
Validation/Confirmation AUC did not turn into hits. The quantiles are computed inside each
window with hindsight, so this is an upper bound for a flip rule, not a candidate.
"""
from pathlib import Path

import pandas as pd

METRICS = Path(__file__).resolve().parent / "metrics"


def main() -> pd.DataFrame:
    months = pd.read_csv(METRICS / "monthly_scores.csv", index_col=0)
    months["flip_gain"] = months["flipped_hits"] - months["production_hits"]
    rows = []
    for signal in ("shock_magnitude", "shock_dispersion"):
        for window, part in months.groupby("window", sort=False):
            for q in (0.1, 0.2, 0.3):
                top = part[part[signal] >= part[signal].quantile(1 - q)]
                rows.append({
                    "signal": signal, "window": window, "top_share": q, "months": len(top),
                    "flip_precision": round(float(top["flip_target"].mean()), 3),
                    "flip_gain": int(top["flip_gain"].sum()),
                    "production_accuracy": round(
                        float(top["production_hits"].sum() / top["calls"].sum()), 3
                    ),
                })
    table = pd.DataFrame(rows)
    table.to_csv(METRICS / "post_hoc_shock_tail.csv", index=False)
    return table


if __name__ == "__main__":
    print(main().to_string(index=False))
