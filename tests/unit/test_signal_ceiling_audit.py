import pandas as pd
import pytest

from forecast_select import signal_ceiling_audit as audit


def _predictions(origins, locked_read=False):
    return pd.DataFrame(
        {
            "origin_position": origins,
            "locked_evaluation_read": [locked_read] * len(origins),
        }
    )


def test_load_nonlocked_predictions_pushes_down_causal_boundary(monkeypatch):
    calls = {}

    def fake_read_parquet(path, **kwargs):
        calls.update(kwargs)
        return _predictions([266])

    monkeypatch.setattr(audit.pd, "read_parquet", fake_read_parquet)

    result = audit._load_nonlocked_predictions()

    assert calls["filters"] == [("origin_position", "<=", 266)]
    assert int(result["origin_position"].max()) == 266


def test_load_nonlocked_predictions_rejects_rows_past_causal_boundary(monkeypatch):
    monkeypatch.setattr(audit.pd, "read_parquet", lambda *args, **kwargs: _predictions([267]))

    with pytest.raises(ValueError, match="locked-origin"):
        audit._load_nonlocked_predictions()


def test_load_nonlocked_predictions_rejects_locked_evaluation_flag(monkeypatch):
    monkeypatch.setattr(
        audit.pd,
        "read_parquet",
        lambda *args, **kwargs: _predictions([266], locked_read=True),
    )

    with pytest.raises(ValueError, match="locked evaluation"):
        audit._load_nonlocked_predictions()
