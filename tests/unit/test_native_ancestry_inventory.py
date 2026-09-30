"""SYNTHETIC_CORRECTNESS_ONLY: closed local ancestor graph guards."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

BUILDER = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "native_inventory_candidate",
    BUILDER / "src/marine_echo/evaluation/native_ancestry_inventory.py",
)
candidate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = candidate
SPEC.loader.exec_module(candidate)


def test_missing_parent_cannot_be_complete_local_ancestry():
    with pytest.raises(ValueError, match="parent"):
        candidate.validate_parent_graph({"child": {"parent": "absent"}})


def test_multigeneration_cycle_cannot_be_complete_local_ancestry():
    with pytest.raises(ValueError, match="[Cc]yclic"):
        candidate.validate_parent_graph(
            {"first": {"parent": "second"}, "second": {"parent": "first"}}
        )


def test_multigeneration_parents_are_derived_before_children():
    assert candidate.validate_parent_graph(
        {"child": {"parent": "root"}, "root": {"parent": None}}
    ) == ["root", "child"]
