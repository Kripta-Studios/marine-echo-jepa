import numpy as np
import pytest

from marine_echo.data.native_observations import aggregate_detected_intervals


def test_partial_edges_use_linear_length_weighting():
    result = aggregate_detected_intervals(
        np.array([[1.0, 9.0]]),
        np.array([[True, True]]),
        np.array([9.0, 11.0, 13.0]),
        np.array([10.0, 12.0]),
    )
    assert result["linear_sum"][0, 0] == 10
    assert result["detected_range_m"][0, 0] == 2
    assert result["available_range_m"][0] == 2


def test_detected_native_sample_survives_neighbour_censoring():
    result = aggregate_detected_intervals(
        np.array([[1.0, np.nan]]),
        np.array([[True, False]]),
        np.array([10.0, 11.0, 12.0]),
        np.array([10.0, 12.0]),
    )
    assert result["linear_sum"][0, 0] == 1
    assert result["detected_range_m"][0, 0] == 1
    assert result["available_range_m"][0] == 2


def test_empty_detection_is_missing_index_not_zero_truth():
    result = aggregate_detected_intervals(
        np.array([[np.nan]]),
        np.array([[False]]),
        np.array([10.0, 12.0]),
        np.array([10.0, 12.0]),
    )
    assert result["detected_range_m"].sum() == 0
    assert np.isnan(result["conditional_linear_mean"]).all()


def test_invalid_detected_values_fail():
    with pytest.raises(ValueError, match="positive"):
        aggregate_detected_intervals(
            np.array([[np.nan]]),
            np.array([[True]]),
            np.array([10.0, 12.0]),
            np.array([10.0, 12.0]),
        )
