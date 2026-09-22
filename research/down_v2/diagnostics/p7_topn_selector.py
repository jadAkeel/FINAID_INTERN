"""P7: Down model evaluated as a SELECTOR, not a classifier.

The product goal is a small number of strong Down calls, so AUC/Brier are not
enough. This ranks eligible indicators per origin by each candidate score and
reports precision at Top 1/3/5/10, plus counts and stability.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from json_output import strict_json_dumps  # noqa: E402
REFIT = ROOT / "research/down_v2/artifacts/down_refit_predictions.parquet"
OUT = ROOT / "research/down_v2/metrics/p7_topn_selector.json"

WINDOWS = {
    "tuning": (120, 179),
    "validation": (180, 219),
    "confirmation": (220, 266),
}


def topn_detail(frame: pd.DataFrame, score: str, n: int) -> dict[str, float | int]:
    top = frame.sort_values(
        ["origin_position", score], ascending=[True, False]
    ).groupby("origin_position").head(n)
    y = top["down_target"].to_numpy()
    return {
        "calls": int(len(top)),
        "months": int(top["origin_position"].nunique()),
        "hits": int(y.sum()),
        "precision": float(y.mean()),
        "calls_per_month": float(len(top) / max(1, top["origin_position"].nunique())),
    }


def main() -> None:
    pred = pd.read_parquet(REFIT)
    pred["down_target"] = pred["down_target"].astype(float)
    pred["p_down_blend"] = pred[
        ["p_down_global", "p_down_local", "p_down_pattern", "p_down_indicator_prior"]
    ].mean(axis=1)

    print("=" * 78)
    print("P7  Down model as a selector: Top-N precision")
    print("=" * 78)
    print(
        "Question: at each origin, rank all eligible indicators by the Down score\n"
        "and take the top N. How often are those actually Down?"
    )
    print()
    print(f"Down base rate among all eligible rows: {pred['down_target'].mean():.4f}")
    print()

    scores = {
        "p_down_global": pred["p_down_global"],
        "p_down_local": pred["p_down_local"],
        "p_down_pattern": pred["p_down_pattern"],
        "p_down_indicator_prior": pred["p_down_indicator_prior"],
        "p_down_blend": pred["p_down_blend"],
    }

    results = {}
    for sname, s in scores.items():
        tmp = pred.copy()
        tmp["s"] = s.to_numpy()
        per_score = {}
        for wname, (a, b) in WINDOWS.items():
            w = tmp[tmp["origin_position"].between(a, b)]
            per_score[wname] = {
                "auc": float(roc_auc_score(w["down_target"], w["s"])),
                **{f"top{n}": topn_detail(w, "s", n) for n in (1, 3, 5, 10)},
            }
        results[sname] = per_score

    print(f"{'score':24s} {'window':12s} {'AUC':>6s} " + " ".join(f"T{n}_prec/{'n':>4s}" for n in (1, 3, 5, 10)))
    for sname, per_score in results.items():
        for wname, wm in per_score.items():
            cells = []
            for n in (1, 3, 5, 10):
                d = wm[f"top{n}"]
                cells.append(f"{d['precision']:.4f}/{d['calls']:4d}")
            print(f"{sname:24s} {wname:12s} {wm['auc']:6.4f} " + " ".join(cells))
        print()

    print("--- Depth decay: does the signal survive beyond the top few? ---")
    tmp = pred.copy()
    tmp["s"] = pred["p_down_blend"].to_numpy()
    for wname, (a, b) in WINDOWS.items():
        w = tmp[tmp["origin_position"].between(a, b)]
        row = f"{wname:12s}"
        for n in (1, 2, 3, 5, 8, 10, 15, 20):
            d = topn_detail(w, "s", n)
            row += f" n={n}:{d['precision']:.4f}"
        print(row)
    print()

    print("--- Indicator prior depth decay (the simplest, best component) ---")
    tmp = pred.copy()
    tmp["s"] = pred["p_down_indicator_prior"].to_numpy()
    for wname, (a, b) in WINDOWS.items():
        w = tmp[tmp["origin_position"].between(a, b)]
        row = f"{wname:12s}"
        for n in (1, 3, 5, 10, 15):
            d = topn_detail(w, "s", n)
            row += f" n={n}:{d['precision']:.4f}"
        print(row)
    print()

    print("--- How concentrated are Down hits? ---")
    tmp = pred.copy()
    tmp["s"] = pred["p_down_blend"].to_numpy()
    top5 = tmp.sort_values(["origin_position", "s"], ascending=[True, False]).groupby(
        "origin_position"
    ).head(5)
    print(f"  top-5 Down calls that were correct: {int(top5['down_target'].sum())}/{len(top5)}")
    print(f"  months with >=1 correct top-5 Down call: "
          f"{int(top5.groupby('origin_position')['down_target'].max().sum())}/"
          f"{int(top5['origin_position'].nunique())}")
    print(f"  months with >=3 correct top-5 Down calls: "
          f"{int((top5.groupby('origin_position')['down_target'].sum() >= 3).sum())}/"
          f"{int(top5['origin_position'].nunique())}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(strict_json_dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
