from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from forecast_select import cli, project, research_cli


def test_cli_import_does_not_load_research_modules():
    code = (
        "import sys, json\n"
        "import forecast_select.cli\n"
        "research_mods = [\n"
        "    'forecast_select.contextual_pipeline',\n"
        "    'forecast_select.downside_pipeline',\n"
        "    'forecast_select.unified_pipeline',\n"
        "    'forecast_select.selection_score_v2_runner',\n"
        "    'forecast_select.directional_ranker_v1_runner',\n"
        "    'forecast_select.down_sensing_pipeline',\n"
        "    'forecast_select.robustness_pipeline',\n"
        "    'forecast_select.calibration_audit',\n"
        "]\n"
        "loaded = [m for m in research_mods if m in sys.modules]\n"
        "print(json.dumps(loaded))\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = json.loads(res.stdout.strip())
    assert loaded == [], f"Eagerly loaded research modules: {loaded}"


def test_project_import_does_not_load_research_pipelines():
    code = (
        "import sys, json\n"
        "import forecast_select.project\n"
        "research_mods = [\n"
        "    'forecast_select.contextual_pipeline',\n"
        "    'forecast_select.downside_pipeline',\n"
        "    'forecast_select.unified_pipeline',\n"
        "]\n"
        "loaded = [m for m in research_mods if m in sys.modules]\n"
        "print(json.dumps(loaded))\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = json.loads(res.stdout.strip())
    assert loaded == [], f"Eagerly loaded research pipelines: {loaded}"


def test_research_cli_import_does_not_load_research_pipelines():
    code = (
        "import sys, json\n"
        "import forecast_select.research_cli\n"
        "research_mods = [\n"
        "    'forecast_select.contextual_pipeline',\n"
        "    'forecast_select.downside_pipeline',\n"
        "    'forecast_select.unified_pipeline',\n"
        "    'forecast_select.selection_score_v2_runner',\n"
        "    'forecast_select.directional_ranker_v1_runner',\n"
        "    'forecast_select.down_sensing_pipeline',\n"
        "    'forecast_select.robustness_pipeline',\n"
        "    'forecast_select.calibration_audit',\n"
        "]\n"
        "loaded = [m for m in research_mods if m in sys.modules]\n"
        "print(json.dumps(loaded))\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = json.loads(res.stdout.strip())
    assert loaded == [], f"Eagerly loaded research modules: {loaded}"


def test_all_production_and_legacy_commands_registered_in_cli():
    parser = cli._parser()
    subparsers_action = next(
        action
        for action in parser._actions
        if isinstance(action, cli.argparse._SubParsersAction)
    )
    registered = set(subparsers_action.choices.keys())

    production_commands = {
        "audit-data",
        "build-model",
        "show-results",
        "build-uptrend-model",
        "show-uptrend-results",
        "forecast-next-three",
        "forecast-uptrend-next-three",
        "forecast-regime-next-three",
        "check-project",
    }
    assert production_commands.issubset(registered)
    assert research_cli.RESEARCH_COMMANDS.issubset(registered)


def test_research_cli_has_all_research_commands():
    parser = research_cli._parser()
    subparsers_action = next(
        action
        for action in parser._actions
        if isinstance(action, cli.argparse._SubParsersAction)
    )
    registered = set(subparsers_action.choices.keys())
    assert research_cli.RESEARCH_COMMANDS == registered


def test_core_project_checks_exclude_historical_research(monkeypatch, tmp_path):
    monkeypatch.setattr(project, "uptrend_model_status", lambda root: {})
    monkeypatch.setattr(project, "active_model_status", lambda root: {})
    monkeypatch.setattr(project, "active_model_artifact", lambda root: tmp_path / "active")
    monkeypatch.setattr(project, "uptrend_model_artifact", lambda root: tmp_path / "baseline")
    monkeypatch.setattr(
        project,
        "directional_downside_status",
        lambda root: {},
    )
    monkeypatch.setattr(
        project,
        "directional_downside_predictions_artifact",
        lambda root: tmp_path / "directional",
    )

    checks = project._core_project_checks(tmp_path)

    assert not any(key.startswith("contextual_") for key in checks)
    assert not any(key.startswith("downside_risk_") for key in checks)
    assert not any(key.startswith("unified_") for key in checks)


def test_registry_preserves_negative_and_quarantined_evidence():
    registry = (
        Path(__file__).resolve().parents[2] / "docs/EXPERIMENT_REGISTRY.md"
    ).read_text(encoding="utf-8")

    assert "SELECTION_GROUP_FAILED_REGISTRY.md" in registry
    assert "QUARANTINE_NOTICE.md" in registry
    assert "`contaminated`" in registry
    assert "planned output; absent" in registry
    assert "python -m forecast_select.cli" not in registry


def test_research_dispatch_executes_and_forwards_cap(monkeypatch, capsys):
    recorded = {}

    class MockRegimeModule:
        @staticmethod
        def build_regime_adaptive_selector(root, cap=None):
            recorded["root"] = root
            recorded["cap"] = cap
            return Path(root) / "artifacts/predictions.parquet"

    monkeypatch.setattr(
        "importlib.import_module",
        lambda name: MockRegimeModule if name == "forecast_select.regime_adaptive_pipeline" else None,
    )

    exit_code = cli.main(["build-regime-adaptive", "--cap", "15"])
    assert exit_code == 0
    assert recorded["cap"] == 15
    assert "predictions.parquet" in capsys.readouterr().out


def test_research_dispatch_formats_json(monkeypatch, capsys):
    class MockDownsideModule:
        @staticmethod
        def downside_risk_gate_status(root):
            return {"ready": True, "penalty": 0.0}

    monkeypatch.setattr(
        "importlib.import_module",
        lambda name: MockDownsideModule if name == "forecast_select.downside_pipeline" else None,
    )

    exit_code = cli.main(["show-risk-gate"])
    assert exit_code == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data == {"ready": True, "penalty": 0.0}


def test_production_monkeypatch_seams(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "build_active_model",
        lambda root: "mock_active_model_built",
    )
    monkeypatch.setattr(
        cli,
        "write_regime_adaptive_next_three_forecast",
        lambda root: "mock_forecast_written",
    )
    monkeypatch.setattr(
        cli,
        "build_uptrend_model",
        lambda root: "mock_uptrend_built",
    )
    monkeypatch.setattr(
        cli,
        "audit_data",
        lambda root: "mock_data_audited",
    )
    monkeypatch.setattr(
        cli,
        "check_project",
        lambda root: "mock_project_checked",
    )

    assert cli.main(["build-model"]) == 0
    assert "mock_active_model_built" in capsys.readouterr().out

    assert cli.main(["forecast-next-three"]) == 0
    assert "mock_forecast_written" in capsys.readouterr().out

    assert cli.main(["build-uptrend-model"]) == 0
    assert "mock_uptrend_built" in capsys.readouterr().out

    assert cli.main(["audit-data"]) == 0
    assert "mock_data_audited" in capsys.readouterr().out

    assert cli.main(["check-project"]) == 0
    assert "mock_project_checked" in capsys.readouterr().out
