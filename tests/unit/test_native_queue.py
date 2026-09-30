"""SYNTHETIC_CORRECTNESS_ONLY serial admission and actual-result gating."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "native_queue", Path(__file__).resolve().parents[2] / "tools/run_reviewed_native_queue.py"
)
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


@pytest.mark.parametrize(
    "report,method",
    [
        ({"status": "COMPLETED", "config": {"method": "masked_ssl"}}, "masked_ssl"),
        (
            {
                "status": "COMPLETED_ZERO_SHOT_DEVELOPMENT",
                "weights_fitted_in_this_study": False,
                "cross_learning": False,
            },
            "chronos2",
        ),
    ],
)
def test_actual_report_schemas(monkeypatch, tmp_path, report, method):
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    job = {
        "id": "synthetic",
        "entrypoint": "tools/execute_native_ssl_job.py",
        "args": [],
        "report": "result.json",
        "method": method,
    }
    path = tmp_path / "queue.json"
    path.write_text(json.dumps({"kind": "reviewed_native_serial_queue_v1", "jobs": [job]}))
    monkeypatch.setattr("sys.argv", ["queue", str(path), "--receipt", str(tmp_path / "receipt")])
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        (tmp_path / "result.json").write_text(json.dumps(report))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(queue.subprocess, "run", run)
    queue.main()
    result = json.loads((tmp_path / "receipt/queue.json").read_text())
    assert len(calls) == 1
    assert result["status"] == "COMPLETED_REVIEWED_SERIAL_QUEUE"


@pytest.mark.parametrize("exit_code,reported_method", [(1, "direct"), (0, "masked_ssl")])
def test_failure_stops_before_next_scientific_command(
    monkeypatch, tmp_path, exit_code, reported_method
):
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    jobs = [
        {
            "id": name,
            "entrypoint": "tools/execute_native_ssl_job.py",
            "args": [],
            "report": "result.json",
            "method": "direct",
        }
        for name in ("first", "second")
    ]
    path = tmp_path / "queue.json"
    path.write_text(json.dumps({"kind": "reviewed_native_serial_queue_v1", "jobs": jobs}))
    monkeypatch.setattr("sys.argv", ["queue", str(path), "--receipt", str(tmp_path / "receipt")])
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        (tmp_path / "result.json").write_text(
            json.dumps({"status": "COMPLETED", "method": reported_method})
        )
        return SimpleNamespace(returncode=exit_code)

    monkeypatch.setattr(queue.subprocess, "run", run)
    with pytest.raises(SystemExit) as caught:
        queue.main()
    assert caught.value.code == 1
    assert len(calls) == 1
    assert (
        json.loads((tmp_path / "receipt/queue.json").read_text())["status"]
        == "STOPPED_AFTER_FAILURE"
    )
