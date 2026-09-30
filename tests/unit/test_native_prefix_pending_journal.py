"""An idle ledger must not authorize a prefix process over another pending journal."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location("_prefix_pending_policy", ROOT / "tools/execute_native_prefix_job_v2.py")
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)


@pytest.mark.parametrize("suffix", [".assessment-pending", ".reconciliation-pending"])
def test_pending_journal_blocks_before_popen_outputs_or_scientific_initialization(monkeypatch, tmp_path, suffix):
    task_root = tmp_path / "SYNTHETIC_METADATA_ONLY"
    ledger_path = task_root / "orchestration/native_ssl_run_ledger_v1.json"
    output, receipt = task_root / "outputs/fit", task_root / "evidence/attempt"
    manifest_path, review_path = task_root / "manifest.json", task_root / "review.json"
    config = {"device": "cuda:0", "family": "cf", "mode": "frozen_readout", "method": "cf_jepa",
              "seed": 7, "prefix_days": 7, "updates": 2000, "cadence": 500}
    manifest = {"kind": "native_prefix_transfer_manifest_v1", "role": "prefix_transfer",
                "evidence_kind": "REVIEWED_PREFIX_TRANSFER", "suffix_numeric_path": None,
                "implementer_session_id": "SYNTHETIC_AUTHOR", "coordinator_session_id": "SYNTHETIC_ROOT", "config": config}
    required = [Path(wrapper.__file__).resolve(), task_root / "tools/execute_native_prefix_worker.py",
                task_root / "tools/native_reference_supervisor.py", manifest_path,
                task_root / "src/marine_echo/training/native_prefix_transfer.py",
                task_root / "src/marine_echo/training/native_resources.py",
                task_root / "orchestration/native_prefix_execution_scope_v1.json"]
    review = {**manifest, "status": "APPROVED_PREFIX_TRANSFER_PREFIT", "scope": "native_prefix_transfer_fit",
              "output_path": str(output), "receipt_path": str(receipt), "reviewer_session_id": "SYNTHETIC_REVIEWER",
              "bindings": {str(path): "a" * 64 for path in required}}
    documents = {manifest_path: manifest, review_path: review,
                 ledger_path: {"gpu_limit_hours": 96, "gpu_hours_spent_owned_scientific_jobs": 0, "runs": []}}
    original_read = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda path, *a, **k: json.dumps(documents[path]) if path in documents else original_read(path, *a, **k))
    monkeypatch.setattr(Path, "exists", lambda path: path == ledger_path.with_suffix(suffix))
    monkeypatch.setattr(wrapper, "ROOT", task_root)
    monkeypatch.setattr(wrapper, "LEDGER", ledger_path)
    monkeypatch.setattr(wrapper, "digest", lambda path: "a" * 64)
    monkeypatch.setattr(wrapper, "resource_budget", lambda *a: pytest.fail("Pending journal reached numerical/model admission phase"))
    monkeypatch.setattr(wrapper.subprocess, "Popen", lambda *a, **k: pytest.fail("Pending journal launched a process"))
    monkeypatch.setattr(Path, "mkdir", lambda *a, **k: pytest.fail("Pending journal created outputs or receipts"))
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("Pending journal initialized numerical output"))
    monkeypatch.setattr(sys, "argv", ["SYNTHETIC_METADATA_ONLY", "--manifest", str(manifest_path), "--review", str(review_path),
                                     "--output", str(output), "--receipt", str(receipt)])
    with pytest.raises(ValueError, match="prior scientific completion"):
        wrapper.main()
