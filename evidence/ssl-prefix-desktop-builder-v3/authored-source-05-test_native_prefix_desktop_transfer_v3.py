"""SYNTHETIC_CORRECTNESS_ONLY: additive prefix science and resource closure."""

from __future__ import annotations

import ast
import copy
import importlib
import json
import sys
import uuid
from pathlib import Path

import pytest
import torch

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
sys.path.insert(0, str(MAIN / "src"))
package = importlib.import_module("marine_echo.training")
if str(BUILDER / "src/marine_echo/training") not in package.__path__:
    package.__path__.append(str(BUILDER / "src/marine_echo/training"))
candidate = importlib.import_module("marine_echo.training.native_prefix_desktop_transfer_v3")
original = importlib.import_module("marine_echo.training.native_prefix_transfer")


class ResourceRouteOnly(ast.NodeTransformer):
    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if (
            node.attr == "Resources"
            and isinstance(node.value, ast.Name)
            and node.value.id == "desktop_runtime"
        ):
            node.value.id = "core"
        return node


def definitions(path):
    return {
        n.name: n
        for n in ast.parse(path.read_bytes()).body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef))
    }


def test_all_prefix_science_definitions_preserved_exactly():
    old = definitions(MAIN / "src/marine_echo/training/native_prefix_transfer.py")
    new = definitions(BUILDER / "src/marine_echo/training/native_prefix_desktop_transfer_v3.py")
    assert set(old) == set(new)
    for name in old:
        if name == "required_sources":
            continue
        after = ResourceRouteOnly().visit(copy.deepcopy(new[name]))
        assert ast.dump(old[name], include_attributes=False) == ast.dump(
            after, include_attributes=False
        ), name


def test_new_source_closure_includes_new_actual_routes_and_bound_owner():
    closure = set(candidate.required_sources())
    required = {
        BUILDER / "src/marine_echo/training/native_prefix_desktop_transfer_v3.py",
        BUILDER / "src/marine_echo/training/native_desktop_runtime_v3.py",
        BUILDER / "tools/execute_native_prefix_desktop_job_v3.py",
        BUILDER / "tools/execute_native_prefix_desktop_worker_v3.py",
        MAIN / "src/marine_echo/training/native_prefix_transfer.py",
        MAIN / "src/marine_echo/training/native_ssl.py",
        MAIN / "src/marine_echo/training/native_resources.py",
        MAIN / "src/marine_echo/training/native_desktop_resources_v2.py",
        MAIN / "tools/execute_native_prefix_job_v2.py",
        MAIN / "tools/execute_native_prefix_worker.py",
        MAIN / "tools/native_reference_supervisor.py",
        MAIN / "tools/native_band_budget_history_v2.py",
        MAIN / "orchestration/native_vlc_desktop_owner_resolution_v1.json",
        MAIN / "docs/adr/0024-owner-authorized-vlc-desktop-coexecution.md",
    }
    assert required <= closure
    assert candidate.core is original.core
    assert candidate.core.Resources is original.core.Resources
    assert candidate.desktop_runtime.Resources is not original.core.Resources


def test_real_config_sampling_and_checkpoint_policy_are_unchanged():
    old = original.PrefixConfig()
    new = candidate.PrefixConfig()
    assert vars(old) == vars(new)
    assert old.updates == new.updates == 2000
    assert old.cadence == new.cadence == 500
    assert old.history == new.history == 96


def test_actual_cpu_resources_do_not_initialize_cuda_and_write_unicode_receipt(monkeypatch):
    path = (
        BUILDER
        / "evidence/ssl-prefix-desktop-builder-v3"
        / ("SYNTHETIC_CORRECTNESS_ONLY-Recursos-Á-" + str(uuid.uuid4()))
    )
    path.mkdir()

    def forbidden(*args, **kwargs):
        pytest.fail("CPU synthetic resource check attempted CUDA initialization")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    with candidate.desktop_runtime.Resources("cpu", path) as resources:
        receipt = resources.snapshot()
    assert receipt["peak_rss_bytes"] < 22 * 1024**3
    assert receipt["peak_allocated_bytes"] == receipt["peak_reserved_bytes"] == 0
    assert receipt["elapsed_gpu_seconds"] == 0
    with (path / "resources.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {"evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY", "fitting": False, "resources": receipt},
            stream,
            indent=2,
        )
    assert json.loads((path / "resources.json").read_bytes())["fitting"] is False


@pytest.mark.skip(
    reason="ROOT-only physical prefix optimizer/save/resume fixture: preserve builder's prior denied optimizer/cache operations; no retry"
)
def test_root_only_physical_unicode_prefix_optimization_resume_and_forecast():
    # Executed through ROOT's delivered 600-second/22GiB owned-process fixture.
    # This skip preserves the known builder optimizer/cache denial without retry.
    pytest.skip(
        "ROOT command: evidence/ssl-prefix-desktop-builder-v3/root_fixture.py --output NEW_SYNTHETIC_PATH"
    )
