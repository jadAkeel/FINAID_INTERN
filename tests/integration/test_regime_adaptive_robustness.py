import inspect
from pathlib import Path

import pandas as pd

from forecast_select.robustness_pipeline import build_regime_adaptive_robustness


def test_regime_adaptive_robustness_artifact_covers_requested_scenarios():
    path = Path("research/regime_adaptive_robustness/metrics/scenarios.csv")
    assert path.exists()
    scenarios = pd.read_csv(path)
    assert set(scenarios["maximum_replacements"]) == {0, 1, 2, 3}
    assert set(scenarios["window"]) == {"120_179", "180_219", "220_266"}
    assert set(scenarios["regime"]) == {"all", "calm", "mixed", "stressed"}
    assert {"normal_down_recall", "sudden_drop_recall"} <= set(scenarios.columns)


def test_regime_adaptive_robustness_does_not_read_locked_origins():
    path = Path("research/regime_adaptive_robustness/metrics/summary.json")
    payload = path.read_text(encoding="utf-8")
    assert '"locked_evaluation_read": false' in payload


def test_robustness_source_does_not_rank_confirmation_policies():
    source = inspect.getsource(build_regime_adaptive_robustness)
    assert '"best_confirmation"' not in source
    assert '"confirmation_used_for_selection": False' in source
