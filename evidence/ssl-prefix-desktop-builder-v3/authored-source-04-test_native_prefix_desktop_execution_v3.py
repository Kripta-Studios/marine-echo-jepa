"""SYNTHETIC_CORRECTNESS_ONLY: pure desktop classifier and owned-prefix policy."""

from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import json
import sys
import uuid
from pathlib import Path

import pytest

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
sys.path.insert(0, str(MAIN / "src"))
sys.path.insert(0, str(MAIN / "tools"))
package = importlib.import_module("marine_echo.training")
package.__path__.append(str(BUILDER / "src/marine_echo/training"))
runtime = importlib.import_module("marine_echo.training.native_desktop_runtime_v3")
resources = runtime.native_resources
SPEC = importlib.util.spec_from_file_location(
    "desktop_prefix_wrapper_candidate", BUILDER / "tools/execute_native_prefix_desktop_job_v3.py"
)
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)
OWNER = json.loads(resources.OWNER_PATH.read_bytes())


def test_baseline_operational_vlc_owner_exception():
    result = runtime.original_resources.classify_gpu_processes(
        [(101, "python.exe"), (102, "vlc.exe")], driver_model="WDDM", owner_pid=101
    )
    assert 102 in result["blocking_pids"]
    admitted = resources.classify_gpu_processes(
        [(101, "python.exe"), (102, "vlc.exe")],
        driver_model="WDDM",
        owner_pid=101,
        owner_resolution=OWNER,
    )
    assert 102 not in admitted["blocking_pids"]


def private():
    path = (
        BUILDER
        / "evidence/ssl-prefix-desktop-builder-v3"
        / ("SYNTHETIC_CORRECTNESS_ONLY-" + str(uuid.uuid4()))
    )
    path.mkdir()
    return path


def ledger():
    return {"runs": [], "gpu_limit_hours": 96, "gpu_hours_spent_owned_scientific_jobs": 0.0}


@pytest.mark.parametrize("driver,vlc_blocked", [("WDDM", False), ("TCC", True), ("N/A", True)])
def test_exact_vlc_exception_retains_unknown_and_ml_blocking(driver, vlc_blocked):
    result = resources.classify_gpu_processes(
        [
            (101, "python.exe"),
            (102, "VLC.EXE"),
            (103, "python.exe"),
            (104, "vlc-helper.exe"),
            (105, ""),
        ],
        driver_model=driver,
        owner_pid=101,
        owner_resolution=OWNER,
    )
    assert (102 in result["blocking_pids"]) is vlc_blocked
    assert {103, 104, 105} <= set(result["blocking_pids"])
    assert 101 not in result["blocking_pids"]


def test_vlc_presence_is_optional_not_an_execution_gate():
    result = resources.classify_gpu_processes(
        [(101, "python.exe")], driver_model="WDDM", owner_pid=101, owner_resolution=OWNER
    )
    assert result["blocking_pids"] == []
    assert result["owner_authorized_additional_desktop_pids"] == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "APPROVED_PREFIT"),
        ("allowed_driver_model", "TCC"),
        ("allowed_additional_desktop_executables", ["vlc.exe", "python.exe"]),
        ("single_scientific_training_process_tree", False),
        ("unknown_or_other_ml_runtimes_blocked", False),
        ("gpu_allocated_reserved_bytes_strictly_less_than", 11 * 1024**3),
        ("owned_tree_ram_bytes_strictly_less_than", 23 * 1024**3),
        ("band_full_owned_gpu_hours", 13),
        ("aggregate_full_owned_gpu_hours", 97),
        ("foreign_process_termination_authorized", True),
        ("independent_prefit", "OPTIONAL"),
    ],
)
def test_owner_authority_cannot_relax_caps_or_review(field, value):
    owner = copy.deepcopy(OWNER)
    owner[field] = value
    with pytest.raises(ValueError):
        resources.classify_gpu_processes(
            [(101, "python.exe"), (102, "vlc.exe")],
            driver_model="WDDM",
            owner_pid=101,
            owner_resolution=owner,
        )


def test_runtime_resources_ast_is_exact_original_and_alias_is_source_bound():
    old = ast.parse((MAIN / "src/marine_echo/training/native_ssl.py").read_bytes())
    new = ast.parse(
        (BUILDER / "src/marine_echo/training/native_desktop_runtime_v3.py").read_bytes()
    )
    before = next(n for n in old.body if isinstance(n, ast.ClassDef) and n.name == "Resources")
    after = next(n for n in new.body if isinstance(n, ast.ClassDef) and n.name == "Resources")
    assert ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False)
    assert runtime.Resources is not runtime.original_core.Resources
    assert (
        Path(resources.__file__).resolve()
        == MAIN / "src/marine_echo/training/native_desktop_resources_v2.py"
    )
    runtime.validate_operational_authority()


@pytest.mark.parametrize(
    "blocker",
    [
        "lock",
        "RUNNING_CUDA",
        "RUNNING_CPU_FIT",
        "record_reconciliation",
        "ledger_reconciliation",
        "all.pending",
        ".prefix-pending",
        ".band-pending",
        ".assessment-pending",
        ".reconciliation-pending",
        "unknown_band",
        "invalid_total",
    ],
)
def test_active_unknown_or_pending_ownership_blocks_before_any_mutation(blocker):
    directory = private()
    path, lock = directory / "native_ssl_run_ledger_v1.json", directory / "gpu-owner.lock"
    value = ledger()
    if blocker == "lock":
        lock.open("x").close()
    elif blocker.startswith("RUNNING"):
        value["runs"] = [{"status": blocker}]
    elif blocker == "record_reconciliation":
        value["runs"] = [{"status": "FAILED", "requires_reconciliation": True}]
    elif blocker == "ledger_reconciliation":
        value["requires_reconciliation"] = True
    elif blocker.endswith("pending"):
        (directory / "all.pending" if blocker == "all.pending" else path.with_suffix(blocker)).open(
            "x"
        ).close()
    elif blocker == "unknown_band":
        value["runs"] = [{"status": "UNRECOGNIZED", "budget_family": "native_band_v1"}]
    else:
        value["gpu_hours_spent_owned_scientific_jobs"] = True
    with pytest.raises(ValueError):
        wrapper.validate_idle_desktop_ledger(value, path, lock)
    assert not (directory / "output").exists()


def test_full_owned_band_failures_and_resumes_share_twelve_inside_ninety_six():
    value = ledger()
    value["gpu_hours_spent_owned_scientific_jobs"] = 95.5
    value["runs"] = [
        {
            "status": status,
            "budget_family": "native_band_v1",
            "resources_full_attempt": {"elapsed_full_attempt_seconds": seconds},
        }
        for status, seconds in (
            ("PREFIX_TRANSFER_FAILED_OR_BLOCKED", 18000),
            ("COMPLETED_REAL_PREFIX_TRANSFER", 24300),
        )
    ]
    deadline, _, spent = wrapper.resource_budget(value, "band")
    assert spent == 11.75
    assert deadline == 900
    value["gpu_hours_spent_owned_scientific_jobs"] = 96
    with pytest.raises(ValueError):
        wrapper.resource_budget(value, "band")


def test_wrapper_worker_routes_and_admission_order_are_actual_ast():
    tree = ast.parse((BUILDER / "tools/execute_native_prefix_desktop_job_v3.py").read_bytes())
    names = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert "marine_echo.training.native_prefix_desktop_transfer_v3" in names
    assert "marine_echo.training.native_prefix_transfer" not in names
    worker = ast.parse((BUILDER / "tools/execute_native_prefix_desktop_worker_v3.py").read_bytes())
    assert any(
        isinstance(n, ast.ImportFrom)
        and n.module == "marine_echo.training.native_prefix_desktop_transfer_v3"
        for n in ast.walk(worker)
    )
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    guard = next(
        n.lineno
        for n in ast.walk(main)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "validate_idle_desktop_ledger"
    )
    mutation = min(
        n.lineno
        for n in ast.walk(main)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in {"mkdir", "Popen"}
    )
    assert guard < mutation
    strings = {
        n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    assert "tools/execute_native_prefix_desktop_worker_v3.py" in strings
    assert (
        "tools/execute_native_prefix_worker.py" in strings
    )  # historical binding, not command route
