"""Jev zero-shot on the consumed locked window, origins 268-315. See PROTOCOL.md.

The run is descriptive: the locked window was consumed on 2026-09-23 and its outcomes are
known. Everything below is fixed by the pre-registration and must not be tuned.

    python research/jev_zero_shot_locked/jev_locked.py              # dry run, sends nothing
    python research/jev_zero_shot_locked/jev_locked.py --send       # live run, once
    python research/jev_zero_shot_locked/jev_locked.py --score-only # re-score cached responses

The live run needs TYPESAFE_API_KEY (or AI_GATEWAY_API_KEY with --gateway vercel).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from forecast_select.io import load_workbook  # noqa: E402

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
METRICS = HERE / "metrics"
CACHE = ARTIFACTS / "responses.jsonl"
LOCKED_PREDICTIONS = ROOT / "research/final_locked_evaluation/metrics/locked_predictions.parquet"

LOCKED = (268, 315)
HISTORY = 36
SEED = 20260925
BLOCK_MONTHS = 6
REPLICATES = 5000

ROUTES = {
    "typesafe": ("https://api.typesafe.ai/v1/systemone", "TYPESAFE_API_KEY", "jev-latest"),
    "vercel": ("https://ai-gateway.vercel.sh/typesafe/v1/systemone", "AI_GATEWAY_API_KEY", "typesafe-ai/jev"),
}

INDICATOR_DESCRIPTION = (
    "Anonymous monthly economic or financial indicator. standardised_monthly_changes lists its "
    "last 36 month-over-month changes, oldest first, each divided by the standard deviation of "
    "those changes. The newest entry is the change into month M. directions gives the same "
    "months as U (rose), D (fell) or = (unchanged)."
)
INDICATOR_QUESTION = (
    "Month M+1 has not been observed yet. Will this indicator's value in month M+2 be strictly "
    "higher than its value in month M+1?"
)
MARKET_DESCRIPTION = (
    "A panel of 50 anonymous monthly economic and financial indicators. monthly_share_rising "
    "lists, for each of the last 36 months, oldest first, the share of indicators whose value "
    "rose in that month. The newest entry is month M."
)
MARKET_QUESTION = (
    "Month M+1 has not been observed yet. Between month M+1 and month M+2, will fewer than half "
    "of these indicators rise?"
)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = load_workbook(ROOT / "data/monthly_indicators.xlsx").set_index("position")
    levels = frame[[c for c in frame.columns if str(c).startswith("X")]]
    locked = pd.read_parquet(LOCKED_PREDICTIONS)
    universe = locked[
        locked["level_c_ready"].fillna(False).astype(bool)
        & ~locked["adaptive_data_quality_excluded"].fillna(False).astype(bool)
    ].copy()
    universe["accepted"] = universe["accepted"].fillna(False).astype(bool)
    lo, hi = LOCKED
    if sorted(universe["origin_position"].unique()) != list(range(lo, hi + 1)):
        raise ValueError("Locked universe does not cover every origin 268-315")
    return levels, universe


def check_targets(levels: pd.DataFrame, universe: pd.DataFrame) -> None:
    following = levels.shift(-1)
    labels = (following > levels).astype(float).where(following.notna() & levels.notna())
    recomputed = [labels.at[o, i] for o, i in zip(universe["origin_position"], universe["indicator_id"])]
    mismatch = int((np.asarray(recomputed) != universe["y_true"].to_numpy()).sum())
    if mismatch:
        raise ValueError(f"{mismatch} recomputed targets differ from the locked artifact")


def indicator_state(levels: pd.DataFrame, origin: int, indicator: str) -> dict:
    # Newest observation is origin - 1; month `origin` itself is unobserved.
    changes = levels[indicator].loc[: origin - 1].diff().dropna().tail(HISTORY)
    spread = float(changes.std(ddof=0))
    scaled = (changes / spread) if spread > 0 else changes * 0.0
    return {
        "description": INDICATOR_DESCRIPTION,
        "standardised_monthly_changes": [round(float(v), 2) for v in scaled],
        "directions": "".join("U" if v > 0 else "D" if v < 0 else "=" for v in changes),
        "share_of_months_rising": round(float((changes > 0).mean()), 3),
    }


def market_state(levels: pd.DataFrame, origin: int) -> dict:
    history = levels.loc[: origin - 1]
    rose = history.diff().gt(0).where(history.notna() & history.shift(1).notna())
    breadth = rose.mean(axis=1, skipna=True).dropna().tail(HISTORY)
    return {
        "description": MARKET_DESCRIPTION,
        "monthly_share_rising": [round(float(v), 3) for v in breadth],
    }


def build_requests(levels: pd.DataFrame, universe: pd.DataFrame, model: str) -> list[dict]:
    requests = []
    for origin, indicator in zip(universe["origin_position"], universe["indicator_id"]):
        requests.append({
            "kind": "indicator", "origin": int(origin), "indicator_id": indicator,
            "body": {
                "model": model,
                "state": json.dumps(indicator_state(levels, int(origin), indicator), sort_keys=True),
                "questions": {"rises": {"type": "noul", "instructions": INDICATOR_QUESTION}},
            },
        })
    for origin in sorted(universe["origin_position"].unique()):
        requests.append({
            "kind": "market", "origin": int(origin), "indicator_id": None,
            "body": {
                "model": model,
                "state": json.dumps(market_state(levels, int(origin)), sort_keys=True),
                "questions": {"broad_down": {"type": "noul", "instructions": MARKET_QUESTION}},
            },
        })
    return requests


def request_key(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def load_cache() -> dict[str, dict]:
    if not CACHE.exists():
        return {}
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            cache[record["key"]] = record
    return cache


def send_all(requests: list[dict], route: str) -> None:
    import requests as http

    url, key_env, _ = ROUTES[route]
    api_key = os.environ.get(key_env)
    if not api_key:
        raise SystemExit(f"{key_env} is not set; nothing was sent.")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    cache = load_cache()
    pending = [r for r in requests if request_key(r["body"]) not in cache]
    print(f"{len(cache)} cached, {len(pending)} to send via {route}")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with CACHE.open("a", encoding="utf-8") as sink:
        for n, item in enumerate(pending, start=1):
            for attempt in range(6):
                response = http.post(url, headers=headers, json=item["body"], timeout=60)
                if response.status_code in (429, 500, 502, 503, 529):
                    time.sleep(min(60, 2 ** attempt))
                    continue
                break
            if response.status_code != 200:
                raise SystemExit(
                    f"Stopped at request {n}: HTTP {response.status_code} {response.text[:300]}"
                )
            record = {
                "key": request_key(item["body"]), "kind": item["kind"], "origin": item["origin"],
                "indicator_id": item["indicator_id"], "response": response.json(),
            }
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            if n % 200 == 0:
                print(f"  sent {n}/{len(pending)}")
            time.sleep(0.06)  # stays under 1,200 requests per minute


def answers(requests: list[dict]) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    cache = load_cache()
    missing = [r for r in requests if request_key(r["body"]) not in cache]
    if missing:
        raise SystemExit(f"{len(missing)} responses are not cached; run with --send first.")
    rows, market, models = [], {}, set()
    for item in requests:
        response = cache[request_key(item["body"])]["response"]
        models.add(str(response.get("model")))
        if item["kind"] == "indicator":
            rows.append({"origin_position": item["origin"], "indicator_id": item["indicator_id"],
                         "p_up": float(response["answers"]["rises"]["noul"])})
        else:
            market[item["origin"]] = float(response["answers"]["broad_down"]["noul"])
    return pd.DataFrame(rows), pd.Series(market, name="p_broad_down"), sorted(models)


def auc(labels, scores) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        return None
    ranks = pd.Series(scores).rank().to_numpy()
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def paired_bootstrap(candidate: np.ndarray, reference: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    n = len(candidate)
    starts = rng.integers(0, n - BLOCK_MONTHS + 1, size=(REPLICATES, -(-n // BLOCK_MONTHS)))
    index = (starts[:, :, None] + np.arange(BLOCK_MONTHS)).reshape(REPLICATES, -1)[:, :n]
    delta = candidate[index].sum(1) - reference[index].sum(1)
    return {f"p{q}": float(np.quantile(delta, q / 100)) for q in (5, 50, 95)}


def select(frame: pd.DataFrame, calls: pd.Series, key: str, direction) -> pd.DataFrame:
    picks = []
    for origin, month in frame.groupby("origin_position", sort=True):
        ranked = month.sort_values([key, "indicator_id"], ascending=[False, True], kind="mergesort")
        picks.append(ranked.head(int(calls[origin])))
    chosen = pd.concat(picks, ignore_index=True)
    chosen["jev_direction"] = direction(chosen)
    chosen["hit"] = np.where(chosen["jev_direction"].eq("Up"), chosen["y_true"].eq(1), chosen["y_true"].eq(0))
    return chosen


def monthly_hits(frame: pd.DataFrame, origins: list[int]) -> np.ndarray:
    return frame.groupby("origin_position")["hit"].sum().reindex(origins, fill_value=0).to_numpy(float)


def score(universe: pd.DataFrame, jev: pd.DataFrame, p_broad_down: pd.Series, models: list[str]) -> dict:
    data = universe.merge(jev, on=["origin_position", "indicator_id"], how="left", validate="one_to_one")
    if data["p_up"].isna().any():
        raise ValueError("Missing Jev answers for some universe rows")
    origins = sorted(data["origin_position"].unique())
    production = data[data["accepted"]].copy()
    production["hit"] = np.where(production["predicted_direction"].eq("Up"),
                                 production["y_true"].eq(1), production["y_true"].eq(0))
    calls = production.groupby("origin_position").size()
    reference = monthly_hits(production, origins)

    p1 = select(data, calls, "p_up", lambda f: pd.Series("Up", index=f.index))
    data["certainty"] = (data["p_up"] - 0.5).abs()
    s2 = select(data, calls, "certainty", lambda f: np.where(f["p_up"] >= 0.5, "Up", "Down"))

    breadth = data.groupby("origin_position")["y_true"].mean()
    broad_down = (breadth < 0.5).reindex(origins)
    flip_months = [o for o in origins if p_broad_down[o] >= 0.5]
    flipped = production.copy()
    flip_mask = flipped["origin_position"].isin(flip_months)
    flipped.loc[flip_mask, "hit"] = np.where(
        flipped.loc[flip_mask, "predicted_direction"].eq("Up"),
        flipped.loc[flip_mask, "y_true"].eq(0), flipped.loc[flip_mask, "y_true"].eq(1))

    # Jev's probability for the direction the active model actually chose.
    chosen_probability = np.where(production["predicted_direction"].eq("Up"),
                                  production["p_up"], 1 - production["p_up"])

    def block(frame):
        hits = monthly_hits(frame, origins)
        return {"hits": int(hits.sum()), "calls": int(len(frame)),
                "accuracy": float(hits.sum() / len(frame)),
                "hit_delta_vs_active": int(hits.sum() - reference.sum()),
                "bootstrap_hit_delta": paired_bootstrap(hits, reference)}

    p_up = data["p_up"]
    return {
        "experiment": "jev_zero_shot_locked",
        "status": "descriptive; locked window consumed 2026-09-23, outcomes known",
        "origins": list(LOCKED),
        "jev_models_seen": models,
        "requests": {"indicator": int(len(data)), "market": int(len(origins))},
        "active_model": {"hits": int(reference.sum()), "calls": int(len(production)),
                         "accuracy": float(reference.sum() / len(production))},
        "P1_up_only_top_k": block(p1),
        "P2_market_timing": {
            "broad_down_months": int(broad_down.sum()),
            "auc": auc(broad_down.to_numpy(), p_broad_down.reindex(origins).to_numpy()),
            "flip_months": len(flip_months),
            "flip_months_that_were_broad_down": int(broad_down.loc[flip_months].sum()) if flip_months else 0,
            "flip_rule": block(flipped),
        },
        "P3_reliability_auc": auc(production["hit"].to_numpy(), chosen_probability),
        "S1_directional_auc": auc(data["y_true"].eq(1).to_numpy(), p_up.to_numpy()),
        "S2_direction_free_top_k": {**block(s2), "down_calls": int(s2["jev_direction"].eq("Down").sum())},
        "S3_p_up_distribution": {
            "mean": float(p_up.mean()), "p25": float(p_up.quantile(0.25)),
            "p50": float(p_up.median()), "p75": float(p_up.quantile(0.75)),
            "share_0_45_to_0_55": float(p_up.between(0.45, 0.55).mean()),
        },
    }


def interpret(result: dict) -> str:
    p1 = result["P1_up_only_top_k"]
    p2 = result["P2_market_timing"]
    signals = [
        p1["hit_delta_vs_active"] > 0 and p1["bootstrap_hit_delta"]["p5"] > 0,
        (p2["auc"] or 0) >= 0.60 and p2["flip_rule"]["hit_delta_vs_active"] > 0,
        (result["P3_reliability_auc"] or 0) >= 0.55,
    ]
    return "worth a forward test on new months" if any(signals) else "no evidence of added signal"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--send", action="store_true", help="send uncached requests to Jev")
    parser.add_argument("--score-only", action="store_true", help="score cached responses only")
    parser.add_argument("--gateway", choices=sorted(ROUTES), default="typesafe")
    args = parser.parse_args()

    levels, universe = load_inputs()
    check_targets(levels, universe)
    model = ROUTES[args.gateway][2]
    requests = build_requests(levels, universe, model)
    size = sum(len(json.dumps(r["body"])) for r in requests)
    print(f"targets verified on {len(universe)} rows; {len(requests)} requests, "
          f"~{size / 4 / 1e6:.2f}M input tokens (4 chars per token estimate)")

    if not args.send and not args.score_only:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        sample = {"indicator": requests[0]["body"], "market": requests[-1]["body"]}
        (ARTIFACTS / "sample_requests.json").write_text(json.dumps(sample, indent=2), encoding="utf-8")
        print(f"dry run: nothing sent. Sample payloads in {ARTIFACTS / 'sample_requests.json'}")
        return
    if args.send:
        send_all(requests, args.gateway)

    jev, p_broad_down, models = answers(requests)
    # From here on p_up means Jev's probability; the active model's is kept as p_up_active.
    universe = universe.rename(columns={"p_up": "p_up_active"})
    result = score(universe, jev, p_broad_down, models)
    result["interpretation"] = interpret(result)
    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
