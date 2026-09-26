"""Synthetic contract tests for calibrated acoustic aggregation and windows."""

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from marine_echo.data.preprocessing import aggregate_calibrated_pings, build_window


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _pings(times: list[str], values: np.ndarray, **overrides: object):
    quarter_ids = np.array(times, dtype="datetime64[m]").astype(np.int64) // 15
    arguments = {
        "ping_times": np.array(times, dtype="datetime64[ns]"),
        "sv_linear": values,
        "valid_mask": np.ones(values.shape, dtype=bool),
        "frequency_hz": np.array([38000, 125000][: values.shape[1]]),
        "range_edges_m": np.array([0.0, 2.0, 4.0]),
        "supported_range": np.ones(values.shape[1:], dtype=bool),
        "configuration_ids": ["xml-a"] * len(times),
        "source_file_ids": ["train-file"] * len(times),
        "dataset_id": "synthetic-calibrated",
        "deployment_id": "synthetic-deployment",
        "instrument_id": "synthetic-instrument",
        "calibration_status": "VERIFIED_PHYSICAL_SV",
        "calibration_report_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "split_sha256": "c" * 64,
        "processing_sha256": "d" * 64,
        "expected_ping_count": np.ones(int(quarter_ids[-1] - quarter_ids[0] + 1), dtype=np.int64),
    }
    arguments.update(overrides)
    arguments["configuration_ids"] = [_digest(value) for value in arguments["configuration_ids"]]
    arguments["source_file_ids"] = [_digest(value) for value in arguments["source_file_ids"]]
    return aggregate_calibrated_pings(**arguments)


def _dense_window(*, context_valid: int = 60, target_valid: int = 60, unequal_target: bool = False):
    bin_starts = np.arange(
        np.datetime64("2020-01-01T00:00"),
        np.datetime64("2020-01-02T06:00"),
        np.timedelta64(15, "m"),
    )
    times = (
        (bin_starts[:, None] + np.arange(60) * np.timedelta64(15, "s"))
        .reshape(-1)
        .astype("datetime64[ns]")
    )
    values = np.full((len(times), 1, 2), 1e-6)
    mask = np.ones(values.shape, dtype=bool)
    mask[context_valid:60] = False
    for bin_index in range(96, 100):
        if unequal_target and bin_index != 96:
            continue
        mask[bin_index * 60 + target_valid : (bin_index + 1) * 60] = False
    if unequal_target:
        values[97 * 60 : 98 * 60] = 9e-6
    return _pings(
        [str(time) for time in times],
        values,
        valid_mask=mask,
        expected_ping_count=np.full(120, 60, dtype=np.int64),
    )


def _dense_window_args():
    return {
        "cutoff": np.datetime64("2020-01-02T00:00"),
        "split_start": np.datetime64("2020-01-01T00:00"),
        "split_end": np.datetime64("2020-01-02T06:00"),
        "partition": "train",
        "allowed_source_file_ids": {_digest("train-file")},
        "expected_dataset_id": "synthetic-calibrated",
        "expected_deployment_id": "synthetic-deployment",
        "expected_instrument_id": "synthetic-instrument",
        "expected_calibration_report_sha256": "a" * 64,
        "expected_config_sha256": "b" * 64,
        "expected_split_sha256": "c" * 64,
        "expected_processing_sha256": "d" * 64,
        "analysis_frequency_hz": 38000,
        "analysis_range_m": (0.0, 4.0),
        "availability_basis": "zero_latency_replay",
    }


@pytest.mark.parametrize("count", [1, 47])
def test_context_fractional_ping_support_fails_below_48_of_60(count: int) -> None:
    series = _dense_window(context_valid=count)
    assert series.expected_ping_count[0] == 60
    assert series.valid_ping_count[0, 0, 0] == count
    with pytest.raises(ValueError, match="context.*support"):
        build_window(series, **_dense_window_args())


def test_target_fractional_ping_support_47_fails_48_passes() -> None:
    insufficient = _dense_window(target_valid=47)
    with pytest.raises(ValueError, match="Target support"):
        build_window(insufficient, **_dense_window_args())
    sufficient = _dense_window(target_valid=48)
    window = build_window(sufficient, **_dense_window_args())
    assert window.target_support[0] == pytest.approx(0.8)


def test_target_linear_mean_weights_unequal_valid_ping_counts() -> None:
    series = _dense_window(target_valid=48, unequal_target=True)
    window = build_window(series, **_dense_window_args())
    assert window.target_support[0] == pytest.approx(228 / 240)
    assert window.target_index_linear[0] == pytest.approx((48 + 60 * 9 + 60 + 60) / 228 * 1e-6)


def test_window_rejects_corrupt_fractional_count_metadata() -> None:
    series = _dense_window()
    corrupt = replace(
        series,
        support_denominator_ping_count=np.ones_like(series.support_denominator_ping_count),
    )
    with pytest.raises(ValueError, match="ping support counts"):
        build_window(corrupt, **_dense_window_args())


@pytest.mark.parametrize(
    "observed,valid,expected_fraction",
    [
        (59, 59, 59 / 60),
        (61, 61, 1.0),
        (61, 48, 48 / 61),
    ],
)
def test_effective_support_denominator_preserves_clock_jitter(
    observed: int, valid: int, expected_fraction: float
) -> None:
    times = np.datetime64("2020-01-01T00:00:00") + np.arange(observed) * np.timedelta64(14, "s")
    values = np.full((observed, 1, 2), 1e-6)
    mask = np.ones(values.shape, dtype=bool)
    mask[valid:] = False
    series = _pings(
        [str(time) for time in times],
        values,
        valid_mask=mask,
        expected_ping_count=np.array([60], dtype=np.int64),
    )
    assert series.ping_count.tolist() == [observed]
    assert series.expected_ping_count.tolist() == [60]
    assert series.support_denominator_ping_count.tolist() == [max(60, observed)]
    assert series.excess_ping_count.tolist() == [max(0, observed - 60)]
    assert series.valid_ping_count[0, 0, 0] == valid
    assert series.valid_ping_count[0, 0, 0] / series.support_denominator_ping_count[0] == (
        pytest.approx(expected_fraction)
    )


def test_duplicate_ping_timestamp_rejected_before_support_accounting() -> None:
    times = ["2020-01-01T00:00:00", "2020-01-01T00:00:00"]
    with pytest.raises(ValueError, match="unique and increasing"):
        _pings(times, np.full((2, 1, 2), 1e-6), expected_ping_count=np.array([60]))


def test_linear_aggregation_trailing_edges_and_frequency_support() -> None:
    values = np.array(
        [
            [[1e-6, 1e-6], [2e-6, 2e-6]],
            [[9e-6, 9e-6], [4e-6, 4e-6]],
            [[7e-6, 7e-6], [8e-6, 8e-6]],
        ]
    )
    series = _pings(
        ["2020-01-01T00:00:00", "2020-01-01T00:14:59", "2020-01-01T00:15:00"],
        values,
        supported_range=np.array([[True, True], [True, False]]),
    )
    assert series.bin_start[0] == np.datetime64("2020-01-01T00:00:00")
    assert series.bin_end[0] == np.datetime64("2020-01-01T00:15:00")
    assert series.ping_count.tolist() == [2, 1]
    assert series.sv_linear[0, 0, 0] == pytest.approx(5e-6)
    assert series.sv_db[0, 0, 0] == pytest.approx(10 * np.log10(5e-6))
    assert not series.valid_mask[:, 1, 1].any()
    assert np.isnan(series.sv_db[:, 1, 1]).all()


def test_calibration_and_identity_fail_closed() -> None:
    values = np.ones((1, 1, 2)) * 1e-6
    with pytest.raises(ValueError, match="verified physical calibration"):
        _pings(["2020-01-01T00:00:00"], values, calibration_status="RAW_COUNTS_ONLY")
    with pytest.raises(ValueError, match="SHA-256 calibration"):
        _pings(["2020-01-01T00:00:00"], values, calibration_report_sha256="")
    with pytest.raises(ValueError, match="nonnegative"):
        _pings(["2020-01-01T00:00:00"], -values)
    with pytest.raises(ValueError, match="measured policy"):
        _pings(
            ["2020-01-01T00:00:00"],
            values,
            ping_available_times=np.array(["2020-01-01T00:00:00"], dtype="datetime64[ns]"),
        )
    with pytest.raises(ValueError, match="at most 32 hours"):
        _pings(
            ["2020-01-01T00:00:00", "2020-01-02T09:00:00"],
            np.ones((2, 1, 2)) * 1e-6,
        )


def test_gap_and_mixed_configuration_cannot_form_valid_windows() -> None:
    values = np.ones((3, 1, 2)) * 1e-6
    series = _pings(
        ["2020-01-01T00:00:00", "2020-01-01T00:15:00", "2020-01-01T00:45:00"],
        values,
        configuration_ids=["xml-a", "xml-b", "xml-b"],
    )
    assert series.ping_count.tolist() == [1, 1, 0, 1]
    assert series.configuration_ids == (
        _digest("xml-a"),
        _digest("xml-b"),
        None,
        _digest("xml-b"),
    )
    assert series.configuration_boundary.tolist() == [False, True, True, True]
    assert not series.valid_mask[2].any()


def test_same_bin_configuration_change_invalidates_aggregation() -> None:
    series = _pings(
        ["2020-01-01T00:00:00", "2020-01-01T00:05:00"],
        np.full((2, 1, 2), 1e-6),
        configuration_ids=["xml-a", "xml-b"],
    )
    assert series.configuration_ids == (None,)
    assert not series.valid_mask.any()


def test_window_containment_availability_and_future_mutation() -> None:
    times = np.arange(
        np.datetime64("2020-01-01T00:00"),
        np.datetime64("2020-01-02T06:00"),
        np.timedelta64(15, "m"),
    ).astype("datetime64[ns]")
    values = np.full((len(times), 1, 2), 1e-6)
    availability = times.copy()
    kwargs = {
        "ping_available_times": availability,
        "availability_policy": "measured",
    }
    series = _pings([str(time) for time in times], values, **kwargs)
    window = build_window(
        series,
        cutoff=np.datetime64("2020-01-02T00:00"),
        split_start=np.datetime64("2020-01-01T00:00"),
        split_end=np.datetime64("2020-01-02T06:00"),
        partition="train",
        allowed_source_file_ids={_digest("train-file")},
        expected_dataset_id="synthetic-calibrated",
        expected_deployment_id="synthetic-deployment",
        expected_instrument_id="synthetic-instrument",
        expected_calibration_report_sha256="a" * 64,
        expected_config_sha256="b" * 64,
        expected_split_sha256="c" * 64,
        expected_processing_sha256="d" * 64,
        analysis_frequency_hz=38000,
        analysis_range_m=(0.0, 4.0),
        availability_basis="measured",
    )
    assert window.context_mask.shape == (96, 1, 2)
    assert window.target_support[0] == pytest.approx(1.0)
    assert len(window.row_id) == 64
    changed = values.copy()
    changed[96:] = 9e-6
    mutated = _pings([str(time) for time in times], changed, **kwargs)
    second = build_window(
        mutated,
        cutoff=np.datetime64("2020-01-02T00:00"),
        split_start=np.datetime64("2020-01-01T00:00"),
        split_end=np.datetime64("2020-01-02T06:00"),
        partition="train",
        allowed_source_file_ids={_digest("train-file")},
        expected_dataset_id="synthetic-calibrated",
        expected_deployment_id="synthetic-deployment",
        expected_instrument_id="synthetic-instrument",
        expected_calibration_report_sha256="a" * 64,
        expected_config_sha256="b" * 64,
        expected_split_sha256="c" * 64,
        expected_processing_sha256="d" * 64,
        analysis_frequency_hz=38000,
        analysis_range_m=(0.0, 4.0),
        availability_basis="measured",
    )
    np.testing.assert_array_equal(window.context_linear, second.context_linear)
    assert window.row_id == second.row_id
    assert second.target_index_linear[0] != window.target_index_linear[0]
    late = availability.copy()
    late[95] = np.datetime64("2020-01-02T02:00")
    late_series = _pings(
        [str(time) for time in times],
        values,
        ping_available_times=late,
        availability_policy="measured",
    )
    late_kwargs = {
        "cutoff": np.datetime64("2020-01-02T00:00"),
        "split_start": np.datetime64("2020-01-01T00:00"),
        "split_end": np.datetime64("2020-01-02T06:00"),
        "partition": "train",
        "allowed_source_file_ids": {_digest("train-file")},
        "expected_dataset_id": "synthetic-calibrated",
        "expected_deployment_id": "synthetic-deployment",
        "expected_instrument_id": "synthetic-instrument",
        "expected_calibration_report_sha256": "a" * 64,
        "expected_config_sha256": "b" * 64,
        "expected_split_sha256": "c" * 64,
        "expected_processing_sha256": "d" * 64,
        "analysis_frequency_hz": 38000,
        "analysis_range_m": (0.0, 4.0),
        "availability_basis": "measured",
    }
    with pytest.raises(ValueError, match="context bin"):
        build_window(late_series, **late_kwargs)
    with pytest.raises(ValueError, match="split"):
        build_window(
            series,
            cutoff=np.datetime64("2020-01-02T00:00"),
            split_start=np.datetime64("2020-01-01T00:15"),
            split_end=np.datetime64("2020-01-02T06:00"),
            partition="train",
            allowed_source_file_ids={_digest("train-file")},
            expected_dataset_id="synthetic-calibrated",
            expected_deployment_id="synthetic-deployment",
            expected_instrument_id="synthetic-instrument",
            expected_calibration_report_sha256="a" * 64,
            expected_config_sha256="b" * 64,
            expected_split_sha256="c" * 64,
            expected_processing_sha256="d" * 64,
            analysis_frequency_hz=38000,
            analysis_range_m=(0.0, 4.0),
            availability_basis="measured",
        )


def test_window_rejects_source_mismatch_and_insufficient_target_support() -> None:
    times = np.arange(
        np.datetime64("2020-01-01T00:00"),
        np.datetime64("2020-01-02T06:00"),
        np.timedelta64(15, "m"),
    ).astype("datetime64[ns]")
    values = np.full((len(times), 1, 2), 1e-6)
    mask = np.ones(values.shape, dtype=bool)
    mask[96, 0, :] = False
    series = _pings([str(time) for time in times], values, valid_mask=mask)
    window_args = {
        "cutoff": np.datetime64("2020-01-02T00:00"),
        "split_start": np.datetime64("2020-01-01T00:00"),
        "split_end": np.datetime64("2020-01-02T06:00"),
        "partition": "train",
        "allowed_source_file_ids": {_digest("train-file")},
        "expected_dataset_id": "synthetic-calibrated",
        "expected_deployment_id": "synthetic-deployment",
        "expected_instrument_id": "synthetic-instrument",
        "expected_calibration_report_sha256": "a" * 64,
        "expected_config_sha256": "b" * 64,
        "expected_split_sha256": "c" * 64,
        "expected_processing_sha256": "d" * 64,
        "analysis_frequency_hz": 38000,
        "analysis_range_m": (0.0, 4.0),
        "availability_basis": "zero_latency_replay",
    }
    with pytest.raises(ValueError, match="support"):
        build_window(series, **window_args)
    mask[96, 0, :] = True
    complete = _pings([str(time) for time in times], values, valid_mask=mask)
    with pytest.raises(ValueError, match="source identities"):
        build_window(
            complete,
            **(window_args | {"allowed_source_file_ids": {_digest("other-file")}}),
        )
    with pytest.raises(ValueError, match="Source identity"):
        build_window(complete, **(window_args | {"expected_instrument_id": "other-instrument"}))
    with pytest.raises(ValueError, match="Protected test truth"):
        build_window(complete, **(window_args | {"partition": "test"}))
    for relaxed in (
        {"context_bins": 4},
        {"horizons": (1,)},
        {"minimum_target_support": 0.1},
    ):
        with pytest.raises(ValueError, match="frozen protocol"):
            build_window(complete, **(window_args | relaxed))
    mask[95, 0, :] = False
    incomplete_context = _pings([str(time) for time in times], values, valid_mask=mask)
    with pytest.raises(ValueError, match="context bin"):
        build_window(incomplete_context, **window_args)
