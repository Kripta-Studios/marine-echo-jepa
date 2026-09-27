"""Instrument-code aggregation keeps zero/undefined and missing distinct."""

import numpy as np
import pytest

from marine_echo.data.raw_response import aggregate_code_response


def test_code_mean_uses_complete_target_pings_and_preserves_negative_codes() -> None:
    codes = np.full((4, 3, 235), 10.0)
    codes[0, 0, 20:200] = -2.0
    codes[0, 1, 20:200] = 6.0
    codes[0, 2, 20] = 0.0
    result = aggregate_code_response(codes)
    assert result["observed_pings"] == 3
    assert result["valid_target_pings"] == 2
    assert result["zero_target_pings"] == 1
    assert result["zero_affected_target_pings"] == 1
    assert result["nonfinite_target_pings"] == 0
    assert result["target_code_sum"] == pytest.approx(720.0)
    assert result["target_code_count"] == 360
    assert result["profile_code_count"][0].sum() == 539
    assert np.isfinite(result["profile_code_sum"]).all()


def test_nonfinite_has_disjoint_priority_and_no_implicit_zero_truth() -> None:
    codes = np.full((4, 2, 235), 3.0)
    codes[0, 0, 21] = np.nan
    codes[0, 0, 22] = 0.0
    codes[0, 1, 25] = np.inf
    result = aggregate_code_response(codes)
    assert result["valid_target_pings"] == 0
    assert result["nonfinite_target_pings"] == 2
    assert result["zero_target_pings"] == 0
    assert result["zero_affected_target_pings"] == 1
    assert result["target_code_count"] == 0
    assert result["target_code_sum"] == 0.0
    assert result["nonfinite_target_samples"] == 2
    assert result["zero_target_samples"] == 1


def test_malformed_acquisition_shape_rejected() -> None:
    with pytest.raises(ValueError):
        aggregate_code_response(np.ones((3, 1, 235)))
    with pytest.raises(ValueError):
        aggregate_code_response(np.ones((4, 1, 199)))
