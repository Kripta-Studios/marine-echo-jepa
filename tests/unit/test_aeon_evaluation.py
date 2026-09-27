"""Synthetic fixtures for the prospective AEON scoring and calibration rules."""

from __future__ import annotations

import numpy as np
import pytest

from marine_echo.evaluation.aeon import (
    apply_interval_widening,
    calibrate_interval_widening,
    daily_pinball,
    eligible_source_dates,
    paired_48h_bootstrap,
)


def _forecast(n: int) -> np.ndarray:
    return np.broadcast_to(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), (n, 3, 5)).copy()


def _times(days: int, rows_per_day: int) -> np.ndarray:
    first = np.datetime64("2024-01-01T00:00", "m")
    hourly = first + np.arange(days * rows_per_day).astype("timedelta64[h]")
    return np.broadcast_to(hourly[:, None], (len(hourly), 3)).copy()


def test_eligible_dates_require_18_observed_issued_anchors_per_horizon() -> None:
    times = _times(3, 24)
    mask = np.ones((72, 3), dtype=bool)
    mask[0:7, 1] = False
    result = eligible_source_dates(times, mask, minimum_anchors=18)
    assert result == {1: 3, 3: 2, 6: 3}


def test_calibration_only_widens_extreme_quantiles_without_touching_middle() -> None:
    forecast = _forecast(5)
    truth = np.array([[0.0, 4.0, -4.0]] * 5)
    mask = np.ones((5, 3), dtype=bool)
    adjustment = calibrate_interval_widening(truth, forecast, mask, partition="calibration")
    np.testing.assert_array_equal(adjustment, [0.0, 2.0, 2.0])
    widened = apply_interval_widening(forecast, adjustment)
    np.testing.assert_array_equal(widened[..., 1:4], forecast[..., 1:4])
    assert (np.diff(widened, axis=-1) >= 0).all()
    with pytest.raises(ValueError, match="calibration"):
        calibrate_interval_widening(truth, forecast, mask, partition="test")


def test_daily_loss_uses_equal_source_dates_and_horizons() -> None:
    forecast = _forecast(48)
    truth = np.zeros((48, 3))
    truth[24:] = 2.0
    mask = np.ones((48, 3), dtype=bool)
    metrics = daily_pinball(truth, forecast, mask, _times(2, 24))
    assert metrics["issued_rows"] == 48
    assert metrics["scored_rows_per_horizon"] == [48, 48, 48]
    assert metrics["scored_days_per_horizon"] == [2, 2, 2]
    assert metrics["primary_daily_mean_pinball_db"] > 0
    mask[0, 1] = False
    masked = daily_pinball(truth, forecast, mask, _times(2, 24))
    assert masked["scored_rows_per_horizon"] == [48, 47, 48]


def test_paired_bootstrap_is_seeded_and_preserves_pairing() -> None:
    times = _times(8, 24)
    truth = np.zeros((192, 3))
    mask = np.ones_like(truth, dtype=bool)
    baseline = _forecast(192)
    candidate = baseline + 1.0
    first = paired_48h_bootstrap(truth, baseline, candidate, mask, times)
    second = paired_48h_bootstrap(truth, baseline, candidate, mask, times)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (2000,)
    assert np.isfinite(first).all()
    assert (first >= 0).all()
