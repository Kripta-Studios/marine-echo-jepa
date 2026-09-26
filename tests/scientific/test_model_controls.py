"""Diagnostic controls cannot draw future targets from protected partitions."""

from __future__ import annotations

import numpy as np
import pytest

from marine_echo.training.controls import separated_train_permutation


def test_temporal_shuffle_stays_in_train_and_respects_gap() -> None:
    times = np.arange("2020-02-17", "2020-03-18", dtype="datetime64[D]")
    partitions = ["train"] * len(times)
    first = separated_train_permutation(times, partitions, seed=7, minimum_gap_hours=48)
    second = separated_train_permutation(
        times, partitions, seed=7, minimum_gap_hours=48
    )
    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == list(range(len(times)))
    assert np.all(
        np.abs((times[first] - times).astype("timedelta64[h]").astype(int)) >= 48
    )


def test_temporal_shuffle_rejects_protected_rows() -> None:
    times = np.arange("2020-02-17", "2020-02-21", dtype="datetime64[D]")
    with pytest.raises(ValueError):
        separated_train_permutation(
            times,
            ["train", "train", "validation", "train"],
            seed=7,
            minimum_gap_hours=24,
        )
