"""Past-only candidate QC preserves support and cannot approve calibration."""

import numpy as np
import xarray as xr

from marine_echo.data.candidate_qc import clean_per_ping, sample_edges, support_denominator


def test_clock_jitter_never_inflates_support_or_hides_missing_pings():
    denominator = support_denominator(np.array([60, 60, 60]), np.array([59, 61, 61]))
    np.testing.assert_array_equal(denominator, [60, 61, 61])
    np.testing.assert_allclose(np.array([59, 61, 48]) / denominator, [59 / 60, 1, 48 / 61])


def synthetic_dataset():
    return xr.Dataset(
        {
            "Sv": (
                ("channel", "ping_time", "range_sample"),
                np.array(
                    [
                        [
                            [-60.0, -62.0, -65.0, -66.0, -70.0, -72.0],
                            [-59.0, -61.0, -64.0, -65.0, -69.0, -71.0],
                        ]
                    ]
                ),
            ),
            "echo_range": (
                ("channel", "ping_time", "range_sample"),
                np.broadcast_to(np.arange(10.0, 70.0, 10.0), (1, 2, 6)),
            ),
            "sound_absorption": (("channel",), [0.01]),
        },
        coords={
            "channel": ["38"],
            "ping_time": np.array(
                ["2020-03-03T00:00:01", "2020-03-03T00:00:16"], dtype="datetime64[ns]"
            ),
            "range_sample": np.arange(6),
        },
    )


def test_noise_does_not_depend_on_future_pings():
    data = synthetic_dataset()
    first = clean_per_ping(data, range_samples=2)
    changed = data.copy(deep=True)
    changed["Sv"].values[:, 1] += 80
    second = clean_per_ping(changed, range_samples=2)
    np.testing.assert_array_equal(first["Sv_noise"].values[:, 0], second["Sv_noise"].values[:, 0])
    np.testing.assert_array_equal(
        first["Sv_corrected"].values[:, 0], second["Sv_corrected"].values[:, 0]
    )


def test_corrected_values_are_linear_noise_subtraction():
    data = synthetic_dataset()
    result = clean_per_ping(data, range_samples=2)
    difference = 10 ** (data.Sv.values / 10) - 10 ** (result.Sv_noise.values / 10)
    valid = np.isfinite(result.Sv_corrected.values)
    assert valid.any()
    np.testing.assert_allclose(10 ** (result.Sv_corrected.values[valid] / 10), difference[valid])


def test_range_edges_preserve_sample_centres_without_extrapolation():
    np.testing.assert_allclose(sample_edges(np.array([1.0, 2.0, 3.0]), 1.0), [0.5, 1.5, 2.5, 3.5])
    import pytest

    with pytest.raises(ValueError):
        sample_edges(np.array([1.0, 2.0, 4.0]), 1.0)
