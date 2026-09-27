"""Raw-code model inputs cannot depend on future codes or future validity."""

import numpy as np

from marine_echo.training.raw_response_windows import build_raw_window


def test_future_response_perturbation_leaves_past_input_exact() -> None:
    sums = np.full((160, 4, 64), 120.0)
    counts = np.full((160, 4, 64), 3)
    target_sums = np.full(160, 1800.0)
    target_counts = np.full(160, 180)
    first = np.datetime64("2020-04-01T00:00:00", "ns")
    row = build_raw_window(first, sums, counts, target_sums, target_counts, cutoff_slot=96)
    changed_sums = sums.copy()
    changed_counts = counts.copy()
    changed_targets = target_sums.copy()
    changed_sums[96:] = -200
    changed_counts[96:] = 0
    changed_targets[96:] = -900
    changed = build_raw_window(first, changed_sums, changed_counts, changed_targets, target_counts, cutoff_slot=96)
    np.testing.assert_array_equal(row.context_codes, changed.context_codes)
    np.testing.assert_array_equal(row.context_mask, changed.context_mask)
    np.testing.assert_array_equal(row.context_valid_fraction, changed.context_valid_fraction)
    assert row.targets[0] != changed.targets[0]
    assert row.context_codes.shape == (96, 4, 64)
    assert row.target_times[0] == np.datetime64("2020-04-02T00:00:00", "ns")


def test_missing_future_target_is_nan_and_masked() -> None:
    sums = np.ones((160, 4, 64))
    counts = np.ones((160, 4, 64), dtype=int)
    target_sums = np.ones(160)
    target_counts = np.ones(160, dtype=int)
    target_counts[96:100] = 0
    row = build_raw_window(
        np.datetime64("2020-04-01T00:00:00", "ns"), sums, counts,
        target_sums, target_counts, cutoff_slot=96,
    )
    assert np.isnan(row.targets[0])
    assert not row.target_observed[0]


def test_near_partition_end_preserves_short_horizon_without_future_fill() -> None:
    sums = np.ones((104, 4, 64))
    counts = np.ones((104, 4, 64), dtype=int)
    target_sums = np.ones(104)
    target_counts = np.ones(104, dtype=int)
    row = build_raw_window(
        np.datetime64("2020-04-01T00:00:00", "ns"), sums, counts,
        target_sums, target_counts, cutoff_slot=100,
    )
    assert row.target_observed.tolist() == [True, False, False]
    assert np.isnan(row.targets[1:]).all()
