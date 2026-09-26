"""Census eligibility uses full context and target coverage, not isolated good bins."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize(
    "field,value",
    [
        ("held_out_acoustic_payloads_processed", True),
        ("completed_benchmark_runs", 1),
        ("completed_benchmark_runs", False),
    ],
)
def test_resume_rejects_exposure_or_run_counter_tamper(monkeypatch, field, value):
    ledger = {
        "status": "RUNNING_TRAIN_ONLY_CENSUS",
        "identity": {},
        "days": {},
        "held_out_acoustic_payloads_processed": False,
        "completed_benchmark_runs": 0,
    }
    ledger[field] = value
    with pytest.raises(ValueError):
        module(monkeypatch).validate_ledger(ledger, {}, ["2020-02-17"])


def test_final_result_binds_final_execution_bytes(monkeypatch, tmp_path):
    import json

    census = module(monkeypatch)
    ledger = {"status": "RUNNING_TRAIN_ONLY_CENSUS"}
    execution = tmp_path / "execution.json"
    result_path = tmp_path / "result.json"
    census.finalize_reports(execution, ledger, result_path, {})
    assert json.loads(execution.read_text())["status"] == "COMPLETE_TRAIN_CENSUS_NOT_BENCHMARK"
    assert json.loads(result_path.read_text())["execution_report_sha256"] == census.sha(execution)


def module(monkeypatch):
    folder = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("train_census", folder / "train_census.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_second_generation_retains_integrity_and_support_rules(monkeypatch):
    folder = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("train_census_v2", folder / "train_census_v2.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    assert result.eligible_cutoffs(
        np.full((192, 45), 60), np.full(192, 60), ["config"] * 192
    ) == list(range(96, 169, 4))
    ledger = {
        "status": "RUNNING_TRAIN_ONLY_CENSUS",
        "identity": {},
        "days": {},
        "held_out_acoustic_payloads_processed": True,
        "completed_benchmark_runs": 0,
    }
    with pytest.raises(ValueError, match="Exposure"):
        result.validate_ledger(ledger, {}, ["2020-02-17"])


def test_two_complete_days_have_only_second_day_cutoffs(monkeypatch):
    result = module(monkeypatch).eligible_cutoffs(
        np.full((192, 45), 60), np.full(192, 60), ["config"] * 192
    )
    assert result == list(range(96, 169, 4))


def test_sparse_context_or_target_cannot_be_hidden(monkeypatch):
    valid = np.full((192, 45), 60)
    valid[:96] = 1
    assert module(monkeypatch).eligible_cutoffs(valid, np.full(192, 60), ["config"] * 192) == []


def test_configuration_boundary_blocks_window(monkeypatch):
    result = module(monkeypatch).eligible_cutoffs(
        np.full((192, 45), 60), np.full(192, 60), ["a"] * 96 + ["b"] * 96
    )
    assert result == []
