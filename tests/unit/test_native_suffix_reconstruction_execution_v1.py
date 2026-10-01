"""Pure launch/receipt/accounting checks; fictional authorization, no real data."""

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location(
    "suffix_reconstruction_wrapper",
    ROOT / "tools/execute_bounded_native_suffix_reconstruction_v1.py",
)
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)


def case():
    output = ROOT / "outputs/SYNTHETIC_unwritten_reconstruction.json"
    receipt = ROOT / "evidence/SYNTHETIC_unwritten_reconstruction"
    manifest = {
        "kind": "native_prefix_suffix_reconstruction_manifest_v1",
        "role": "adapted_suffix",
        "evidence_kind": "REVIEWED_PREFIX_SUFFIX_ASSESSMENT",
        "implementer_session_id": "fictional-author",
        "coordinator_session_id": "fictional-coordinator",
        "recipe": wrapper.RECIPE,
    }
    review = {
        "status": "APPROVED_PREFIX_SUFFIX_RECONSTRUCTION",
        "scope": "saved_adapted_suffix_reconstruction",
        "allowed_roles": ["adapted_suffix"],
        "allowed_uses": ["saved_adapted_suffix_reconstruction"],
        "evidence_kind": manifest["evidence_kind"],
        "implementer_session_id": manifest["implementer_session_id"],
        "coordinator_session_id": manifest["coordinator_session_id"],
        "reviewer_session_id": "fictional-distinct-reviewer",
        "runtime": wrapper.runtime(output, receipt),
    }
    return manifest, review, output, receipt


def test_exact_cpu_reconstruction_scope():
    wrapper.validate_scope(*case())


@pytest.mark.parametrize(
    "field,value",
    [
        ("role", "final_test"),
        ("evidence_kind", "SYNTHETIC_CORRECTNESS_ONLY"),
        ("kind", "native_prefix_suffix_assessment_manifest_v1"),
        ("implementer_session_id", ""),
        ("coordinator_session_id", None),
    ],
)
def test_no_other_scientific_or_synthetic_scope(field, value):
    args = case()
    args[0][field] = value
    with pytest.raises(ValueError):
        wrapper.validate_scope(*args)


@pytest.mark.parametrize(
    "damage",
    [
        "self",
        "coordinator",
        "status",
        "scope",
        "uses",
        "device",
        "ram",
        "deadline",
        "output",
        "receipt",
    ],
)
def test_no_scope_resource_or_review_waiver(damage):
    args = case()
    review = args[1]
    if damage in ("self", "coordinator"):
        review["reviewer_session_id"] = args[0][
            "implementer_session_id" if damage == "self" else "coordinator_session_id"
        ]
    elif damage == "status":
        review["status"] = "APPROVED_FINAL_ASSESSMENT"
    elif damage == "scope":
        review["scope"] = "prefix_suffix_assessment_execution"
    elif damage == "uses":
        review["allowed_uses"] = ["frozen_adapted_suffix_assessment"]
    else:
        field = {
            "device": "device",
            "ram": "ram_limit_bytes",
            "deadline": "deadline_seconds",
            "output": "output",
            "receipt": "receipt",
        }[damage]
        review["runtime"][field] = "cuda:0" if damage == "device" else 1
    with pytest.raises(ValueError):
        wrapper.validate_scope(*args)


def test_full_owned_cpu_time_does_not_change_gpu_or_fit_counters():
    ledger = {
        "gpu_hours_spent_owned_scientific_jobs": 17.802355808369175,
        "native_band_gpu_hours_spent_full_owned": 8.318836388888881,
        "cpu_prefix_hours_owned": 2,
        "cpu_reconstruction_hours_owned": 1,
    }
    wrapper.charge_attempt(ledger, 720)
    assert ledger == {
        "gpu_hours_spent_owned_scientific_jobs": 17.802355808369175,
        "native_band_gpu_hours_spent_full_owned": 8.318836388888881,
        "cpu_prefix_hours_owned": 2,
        "cpu_reconstruction_hours_owned": 1.2,
    }


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf"), "1"])
def test_no_nonfinite_or_negative_accounting(value):
    with pytest.raises(ValueError):
        wrapper.charge_attempt({}, value)
    with pytest.raises(ValueError):
        wrapper.cpu_hours({"cpu_reconstruction_hours_owned": value})


@pytest.mark.parametrize(
    "damage",
    [None, "exit", "timeout", "cleanup", "zero_shot", "kind", "evidence", "review", "manifest"],
)
def test_success_requires_own_exit_and_exact_reconstruction_receipt(tmp_path, damage):
    manifest, review, _, _ = case()
    output = tmp_path / "result.json"
    result = {
        "kind": "native_prefix_suffix_reconstruction_completion_v1",
        "status": "RECONSTRUCTED",
        "role": "adapted_suffix",
        "zero_shot": False,
        "evidence_kind": manifest["evidence_kind"],
        "independent_scientific_approval": False,
        "manifest_sha256": "a" * 64,
        "review_sha256": "b" * 64,
        "reviewer_session_id": review["reviewer_session_id"],
    }
    resources = {"exit_code": 0, "stopped_for": None, "owned_tree_cleanup_verified": True}
    if damage == "exit":
        resources["exit_code"] = 3
    elif damage == "timeout":
        resources["stopped_for"] = "FULL_OWNED_ATTEMPT_DEADLINE"
    elif damage == "cleanup":
        resources["owned_tree_cleanup_verified"] = False
    elif damage in ("zero_shot", "kind", "evidence", "review", "manifest"):
        field = {
            "zero_shot": "zero_shot",
            "kind": "kind",
            "evidence": "evidence_kind",
            "review": "review_sha256",
            "manifest": "manifest_sha256",
        }[damage]
        result[field] = True if damage == "zero_shot" else "wrong"
    output.write_text(json.dumps(result), encoding="utf-8")
    assert wrapper.verify_completion(output, manifest, review, "a" * 64, "b" * 64, resources) is (
        damage is None
    )


def test_parent_peak_is_inside_22gib_limit():
    assert wrapper.child_ram_allowance(2**30) == 21 * 2**30
    with pytest.raises(ValueError):
        wrapper.child_ram_allowance(22 * 2**30)


def test_destination_cannot_escape_named_root_through_dotdot(tmp_path):
    named_root = tmp_path / "outputs"
    named_root.mkdir()
    with pytest.raises(ValueError):
        wrapper.destination(named_root / ".." / "foreign.json", named_root)
    assert wrapper.destination(named_root / "result.json", named_root) == named_root / "result.json"


def test_owned_spawn_failure_has_closed_cpu_receipt_and_charge():
    ledger = {"gpu_hours_spent_owned_scientific_jobs": 17.802355808369175}
    record = {"status": "RUNNING_CPU_RECONSTRUCTION"}
    wrapper.close_spawn_failure(ledger, record, 3, OSError("synthetic spawn refusal"))
    assert record["status"] == "FAILED_CPU_RECONSTRUCTION_BEFORE_SPAWN"
    assert record["fitting"] is False
    assert record["owned_tree_cleanup_verified"] is True
    assert ledger["cpu_reconstruction_hours_owned"] == 3 / 3600
    assert ledger["gpu_hours_spent_owned_scientific_jobs"] == 17.802355808369175


@pytest.mark.parametrize("damage", [False, True])
def test_actual_numpy_worker_owned_exit_and_saved_result(tmp_path, damage):
    from test_native_suffix_reconstruction_v1 import case as reconstruction_case

    folder, _ = reconstruction_case(tmp_path)
    manifest_path, review_path = folder / "manifest.json", folder / "review.json"
    if damage:
        review = json.loads(review_path.read_text())
        review["status"] = "PENDING"
        review_path.write_text(json.dumps(review), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text())
    review = json.loads(review_path.read_text())
    output = folder / "owned-worker-result.json"
    command = wrapper.worker_command(manifest_path, review_path, output)
    with (folder / "owned-worker.log").open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        resources = wrapper.supervise_owned(
            child,
            started=time.monotonic(),
            deadline_seconds=30,
            rss_limit_bytes=wrapper.child_ram_allowance(2**30),
        )
    assert resources["owned_tree_cleanup_verified"] is True
    assert resources["stopped_for"] is None
    assert resources["exit_code"] == (1 if damage else 0)
    assert wrapper.verify_completion(
        output,
        manifest,
        review,
        wrapper.digest(manifest_path),
        wrapper.digest(review_path),
        resources,
    ) is (not damage)
    if damage:
        assert not output.exists()
