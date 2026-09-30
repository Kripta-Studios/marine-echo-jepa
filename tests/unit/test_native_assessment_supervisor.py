"""Synthetic policy-only checks: no data decoding, processes, ledger or CUDA."""

import copy
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location(
    "native_assessment_supervisor_under_test", ROOT / "tools/execute_bounded_native_assessment.py"
)
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)


@pytest.fixture
def policy(monkeypatch):
    """Private fake digests exercise admission without reading artifact bytes."""
    manifest_path = ROOT / "evidence/SYNTHETIC_ASSESSMENT_POLICY_ONLY/manifest.json"
    output = ROOT / "evidence/SYNTHETIC_ASSESSMENT_POLICY_ONLY/output"
    manifest = {"role": "final_test", "evidence_kind": "REVIEWED_FROZEN_ASSESSMENT", "device": "cuda:0",
                "implementer_session_id": "synthetic-policy-implementer",
                "root_coordinator_session_id": "synthetic-policy-root"}
    paths = [Path(supervisor.__file__), ROOT / "tools/execute_native_assessment_worker.py",
             ROOT / "tools/native_reference_supervisor.py", manifest_path,
             ROOT / "src/marine_echo/evaluation/native_assessment.py",
             ROOT / "src/marine_echo/training/native_resources.py",
             ROOT / "src/marine_echo/training/native_ssl.py", ROOT / "evidence/synthetic-model.bin"]
    digest = lambda path: hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()
    monkeypatch.setattr(supervisor, "digest", digest)
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: pytest.fail("Started a process"))
    monkeypatch.setattr(supervisor, "save_ledger", lambda *a: pytest.fail("Wrote the live ledger"))
    review = {**manifest, "status": "APPROVED_FINAL_ASSESSMENT_EXECUTION",
              "scope": "model_only_frozen_assessment", "allowed_roles": ["final_test"],
              "reviewer_session_id": "synthetic-policy-distinct-reviewer", "output_path": str(output),
              "bindings": {str(path): digest(path) for path in paths}}
    return manifest, review, manifest_path, output


def test_policy_admission_is_metadata_only_and_side_effect_free(policy):
    supervisor.prelaunch(*policy)


@pytest.mark.parametrize("field,value", [
    ("status", "APPROVED_DEVELOPMENT_ASSESSMENT_EXECUTION"),
    ("allowed_roles", ["development"]), ("reviewer_session_id", "synthetic-policy-root"),
    ("reviewer_session_id", "SYNTHETIC-POLICY-IMPLEMENTER"), ("device", "cuda"),
    ("device", "cuda:1"), ("output_path", "some-other-output"), ("scope", "prefit_training"),
])
def test_changed_policy_is_rejected_before_process_or_ledger(policy, field, value):
    manifest, original, path, output = policy
    review = copy.deepcopy(original)
    review[field] = value
    with pytest.raises(ValueError):
        supervisor.prelaunch(manifest, review, path, output)


def test_every_extra_artifact_binding_is_checked_not_only_selected_sources(policy):
    manifest, review, path, output = policy
    review["bindings"][str(ROOT / "evidence/synthetic-model.bin")] = "0" * 64
    with pytest.raises(ValueError, match="Changed assessment binding"):
        supervisor.prelaunch(manifest, review, path, output)


@pytest.mark.parametrize("source", ["tools/execute_native_assessment_worker.py", "tools/native_reference_supervisor.py",
                                     "src/marine_echo/training/native_resources.py"])
def test_unbound_resource_component_is_rejected(policy, source):
    manifest, review, path, output = policy
    del review["bindings"][str(ROOT / source)]
    with pytest.raises(ValueError, match="bindings required"):
        supervisor.prelaunch(manifest, review, path, output)
