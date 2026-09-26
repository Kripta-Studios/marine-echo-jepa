"""A repair must preserve its failed predecessor and exact unaffected outputs."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def module(monkeypatch):
    folder = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("train_census_v2", folder / "train_census_v2.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_tampered_previous_execution_fails_before_reuse(monkeypatch, tmp_path):
    census = module(monkeypatch)
    monkeypatch.setattr(census, "ROOT", tmp_path)
    folder = tmp_path / "evidence/continuation"
    folder.mkdir(parents=True)
    (folder / "train_census_execution.json").write_text('{"status":"tampered"}')
    (folder / "train_census_v2_contract.json").write_text(
        json.dumps({"prior_failure": {"sha256": "0" * 64}})
    )
    with pytest.raises(ValueError, match="prior"):
        census.previous_execution()


def test_changed_array_cannot_be_accepted_as_equivalent(monkeypatch, tmp_path):
    old, new = tmp_path / "old.npz", tmp_path / "new.npz"
    np.savez(old, valid_count=np.array([60, 60]), sv=np.array([1.0, np.nan]))
    np.savez(new, valid_count=np.array([60, 59]), sv=np.array([1.0, np.nan]))
    with pytest.raises(AssertionError):
        module(monkeypatch).assert_shards_equal(old, new)


def test_identical_nan_masks_are_equivalent(monkeypatch, tmp_path):
    old, new = tmp_path / "old.npz", tmp_path / "new.npz"
    np.savez(old, sv=np.array([1.0, np.nan]))
    np.savez(new, sv=np.array([1.0, np.nan]))
    assert module(monkeypatch).assert_shards_equal(old, new) == ["sv"]
