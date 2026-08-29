from pathlib import Path

from forecast_select import cli


def test_build_model_uses_promoted_active_model(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "build_active_model",
        lambda root: Path(root) / "artifacts/active/regime_adaptive_predictions.parquet",
    )

    assert cli.main(["build-model"]) == 0
    assert "regime_adaptive_predictions.parquet" in capsys.readouterr().out


def test_default_forecast_command_uses_regime_adaptive_writer(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        cli,
        "write_regime_adaptive_next_three_forecast",
        lambda root: Path(root) / "reports/regime_adaptive_next_three_forecast.json",
    )

    assert cli.main(["forecast-next-three"]) == 0
    assert "regime_adaptive_next_three_forecast.json" in capsys.readouterr().out
