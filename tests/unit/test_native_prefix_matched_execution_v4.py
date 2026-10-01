"""Pure matched-prefix resource admission; no corpus, process or GPU access."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location(
    "matched_prefix_wrapper", ROOT / "tools/execute_native_prefix_matched_job_v4.py"
)
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)


def case(method="lightgbm", family="reference", mode="conventional", device="cpu"):
    output, receipt = ROOT / "outputs/SYNTHETIC_unwritten", ROOT / "evidence/SYNTHETIC_unwritten"
    config = {
        "method": method,
        "family": family,
        "mode": mode,
        "device": device,
        "seed": 7,
        "prefix_days": 1,
        "updates": 2000,
        "cadence": 500,
    }
    manifest = {
        "kind": "native_prefix_transfer_manifest_v1",
        "role": "prefix_transfer",
        "evidence_kind": "REVIEWED_PREFIX_TRANSFER",
        "implementer_session_id": "author",
        "coordinator_session_id": "coordinator",
        "config": config,
    }
    review = {
        "status": "APPROVED_PREFIX_TRANSFER_PREFIT",
        "scope": "native_prefix_transfer_fit",
        "role": "prefix_transfer",
        "evidence_kind": "REVIEWED_PREFIX_TRANSFER",
        "implementer_session_id": "author",
        "coordinator_session_id": "coordinator",
        "reviewer_session_id": "distinct-synthetic-reviewer",
        "config": config.copy(),
        "output_path": str(output),
        "receipt_path": str(receipt),
    }
    return manifest, review, output, receipt


def test_conventional_prefix_cpu_cell_admitted():
    args = case()
    assert wrapper.validate_scope(*args) == args[0]["config"]


@pytest.mark.parametrize(
    "method,mode",
    [
        ("cf_jepa", "frozen_readout"),
        ("cf_random_frozen", "frozen_readout"),
        ("cf_direct_supervised", "frozen_readout"),
        ("direct", "scratch_direct"),
    ],
)
def test_cf_neural_cells_keep_cuda_and_typed_modes(method, mode):
    args = case(method, "cf", mode, "cuda:0")
    assert wrapper.validate_scope(*args)["family"] == "cf"


@pytest.mark.parametrize(
    "field,value",
    [
        ("device", "cuda:0"),
        ("method", "cf_jepa"),
        ("mode", "frozen_readout"),
        ("seed", True),
        ("prefix_days", 2),
    ],
)
def test_conventional_confusion_refused(field, value):
    args = case()
    args[0]["config"][field] = value
    args[1]["config"] = args[0]["config"].copy()
    with pytest.raises(ValueError):
        wrapper.validate_scope(*args)


def test_cpu_full_attempt_does_not_inflate_gpu_budget():
    ledger = {
        "runs": [],
        "gpu_limit_hours": 96,
        "gpu_hours_spent_owned_scientific_jobs": 17.802355808369175,
    }
    wrapper.charge_attempt(
        ledger,
        device="cpu",
        elapsed_seconds=720,
        spent=17.802355808369175,
        band_spent=None,
        cpu_spent=2,
    )
    assert ledger["gpu_hours_spent_owned_scientific_jobs"] == 17.802355808369175
    assert ledger["cpu_prefix_hours_owned"] == pytest.approx(2.2)


def test_failed_gpu_attempt_includes_full_band_and_aggregate_lifetime():
    ledger = {"gpu_hours_spent_owned_scientific_jobs": 17.802355808369175}
    wrapper.charge_attempt(
        ledger,
        device="cuda:0",
        elapsed_seconds=720,
        spent=17.802355808369175,
        band_spent=8.318836388888881,
        cpu_spent=0,
    )
    assert ledger["gpu_hours_spent_owned_scientific_jobs"] == pytest.approx(18.002355808369175)
    assert ledger["native_band_gpu_hours_spent_full_owned"] == pytest.approx(8.518836388888881)


@pytest.mark.parametrize("bad", [True, -1, float("nan"), float("inf")])
def test_invalid_owned_time_refused_without_mutation(bad):
    ledger = {"gpu_hours_spent_owned_scientific_jobs": 17.802355808369175}
    before = ledger.copy()
    with pytest.raises(ValueError):
        wrapper.charge_attempt(
            ledger,
            device="cpu",
            elapsed_seconds=bad,
            spent=17.802355808369175,
            band_spent=None,
            cpu_spent=0,
        )
    assert ledger == before


def test_cpu_remains_possible_after_gpu_cap_but_rejects_active_job():
    ledger = {"gpu_limit_hours": 96, "gpu_hours_spent_owned_scientific_jobs": 96, "runs": []}
    assert wrapper.resource_budget(ledger, "reference") == (7200, 96, None)
    ledger["runs"].append({"status": "RUNNING_CUDA"})
    with pytest.raises(ValueError):
        wrapper.resource_budget(ledger, "reference")


def test_owned_ram_cap_includes_launcher_memory():
    assert wrapper.owned_child_ram_allowance(2 * 2**30) == 20 * 2**30
    with pytest.raises(ValueError):
        wrapper.owned_child_ram_allowance(22 * 2**30)


@pytest.mark.parametrize(
    "author", ["01a0f72d-927a-77a3-8b62-21d456f7ed85", "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1"]
)
def test_actual_authors_cannot_approve_even_with_different_declared_identities(author):
    args = case()
    args[1]["reviewer_session_id"] = author
    with pytest.raises(ValueError):
        wrapper.validate_scope(*args)
