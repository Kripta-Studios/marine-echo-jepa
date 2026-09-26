"""Candidate entrypoint must reject protected dates before any input or output access."""

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "date,variant",
    [
        ("2020-07-07", "monthly-v1"),
        ("2020-02-17", "v2"),
        ("2020-05-27", "census-v1"),
        ("2020-07-07", "census-v1"),
    ],
)
def test_candidate_rejects_unregistered_dates(date, variant, monkeypatch):
    folder = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location(
        "train_candidate_scope", folder / "preprocess_train_candidate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError):
        module.main(date, variant)
