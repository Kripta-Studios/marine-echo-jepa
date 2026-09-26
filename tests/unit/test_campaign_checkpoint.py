"""Blocked campaign invocation must preserve every recorded attempt and exposure."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from marine_echo.serving.cli import app

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("status", ["FAILED", "RUNNING", "BLOCKED", "COMPLETED"])
def test_blocked_campaign_does_not_overwrite_existing_evidence(tmp_path, monkeypatch, status):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/experiments.json").write_bytes(
        (ROOT / "configs/experiments.json").read_bytes()
    )
    registry = tmp_path / "reports/active/training_registry.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            {
                "completed_benchmark_runs": 0,
                "test_opened": True,
                "runs": [
                    {
                        "run_id": "direct-seed7",
                        "status": status,
                        "updates": 5,
                        "failure_log": "preserve.log",
                    }
                ],
            }
        )
    )
    original = registry.read_bytes()
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app, ["experiment", "complete-p0", "--protocol", "active", "--resume"]
    )
    assert result.exit_code == 2
    assert registry.read_bytes() == original


@pytest.mark.parametrize("status", ["FAILED", "RUNNING", "COMPLETED_SYNTHETIC_FIXTURE"])
def test_report_uses_recorded_attempts_and_exposure(tmp_path, monkeypatch, status):
    registry = tmp_path / "reports/active/training_registry.json"
    registry.parent.mkdir(parents=True)
    document = {
        "test_opened": True,
        "runs": [{"run_id": "control-seed7", "status": status, "failure_log": "failure.log"}],
    }
    registry.write_text(json.dumps(document))
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["report", "--protocol", "active"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["registry"] == document
    assert json.loads(result.stdout)["G2_EXPERIMENT"] == "INCOMPLETE"
