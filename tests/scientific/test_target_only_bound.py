"""The impossibility bound must not depend on a stricter context policy."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module(monkeypatch):
    folder = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location(
        "target_only_bound", folder / "target_only_bound.py"
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_bound_counts_first_day_without_context(monkeypatch):
    assert module(monkeypatch).target_only_cutoffs(np.full((96, 45), 60), np.full(96, 60)) == list(
        range(0, 73, 4)
    )


def test_boundary_and_weighted_denominator(monkeypatch):
    counts = np.full((24, 45), 48)
    denominator = np.full(24, 60)
    fn = module(monkeypatch).target_only_cutoffs
    assert fn(counts, denominator) == [0]
    denominator[0] = 61
    assert fn(counts, denominator) == []


def test_all_three_targets_required(monkeypatch):
    counts = np.full((24, 45), 60)
    counts[20:] = 0
    assert module(monkeypatch).target_only_cutoffs(counts, np.full(24, 60)) == []


def test_corrupt_counts_fail_closed(monkeypatch):
    with pytest.raises(ValueError):
        module(monkeypatch).target_only_cutoffs(np.full((24, 45), 61), np.full(24, 60))
