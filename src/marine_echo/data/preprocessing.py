"""Conservative aggregation of verified physical Sv and past-only window construction.

This module starts after upstream acoustic calibration. It cannot turn digitizer
counts into Sv or establish whether environmental inputs apply to a deployment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_BIN_MINUTES = 15
_DB_FLOOR = 1e-18
_MAX_BINS_PER_BATCH = 128


@dataclass(frozen=True)
class BinnedAcoustics:
    bin_start: NDArray[np.datetime64]
    bin_end: NDArray[np.datetime64]
    available_time: NDArray[np.datetime64]
    sv_linear: NDArray[np.float64]
    sv_db: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    ping_count: NDArray[np.int64]
    frequency_hz: NDArray[np.int64]
    range_edges_m: NDArray[np.float64]
    supported_range: NDArray[np.bool_]
    configuration_ids: tuple[str | None, ...]
    configuration_boundary: NDArray[np.bool_]
    source_file_ids: tuple[frozenset[str], ...]
    dataset_id: str
    deployment_id: str
    instrument_id: str
    calibration_report_sha256: str
    availability_policy: str


@dataclass(frozen=True)
class AcousticWindow:
    cutoff: np.datetime64
    partition: str
    context_linear: NDArray[np.float64]
    context_mask: NDArray[np.bool_]
    target_index_linear: NDArray[np.float64]
    target_index_db: NDArray[np.float64]
    target_support: NDArray[np.float64]
    source_file_ids: frozenset[str]


def _time_ns(values: NDArray[np.datetime64], name: str) -> NDArray[np.datetime64]:
    if values.ndim != 1 or values.dtype.kind != "M" or np.isnat(values).any():
        raise ValueError(f"{name} requires finite one-dimensional timestamps.")
    return values.astype("datetime64[ns]")


def aggregate_calibrated_pings(
    *,
    ping_times: NDArray[np.datetime64],
    sv_linear: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    frequency_hz: NDArray[np.int64],
    range_edges_m: NDArray[np.float64],
    supported_range: NDArray[np.bool_],
    configuration_ids: list[str],
    source_file_ids: list[str],
    dataset_id: str,
    deployment_id: str,
    instrument_id: str,
    calibration_status: str,
    calibration_report_sha256: str,
    ping_available_times: NDArray[np.datetime64] | None = None,
    availability_policy: str = "zero_latency_replay",
) -> BinnedAcoustics:
    """Average verified calibrated sv in fixed trailing bins without filling gaps.

    The caller must supply physical-unit values and a reviewed calibration report
    digest. Configuration changes within one bin invalidate that bin.
    """
    if calibration_status != "VERIFIED_PHYSICAL_SV":
        raise ValueError("Input lacks verified physical calibration to sv.")
    if len(calibration_report_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in calibration_report_sha256
    ):
        raise ValueError("A SHA-256 calibration report identity is required.")
    if not all((dataset_id, deployment_id, instrument_id)):
        raise ValueError("Dataset, deployment and instrument identities are required.")
    times = _time_ns(ping_times, "ping_times")
    if not len(times) or (np.diff(times) <= np.timedelta64(0, "ns")).any():
        raise ValueError("Ping timestamps must be nonempty, unique and increasing.")
    if sv_linear.ndim != 3 or sv_linear.shape != valid_mask.shape:
        raise ValueError("Calibrated sv and mask must align as [ping,frequency,range].")
    pings, frequencies, ranges = sv_linear.shape
    if pings != len(times) or not frequencies or not ranges:
        raise ValueError("Calibrated array dimensions must match ping times.")
    if (
        frequency_hz.shape != (frequencies,)
        or frequency_hz.dtype.kind not in "iu"
        or (frequency_hz <= 0).any()
        or len(np.unique(frequency_hz)) != frequencies
    ):
        raise ValueError("Frequency identities must be unique positive values in Hz.")
    if (
        range_edges_m.shape != (ranges + 1,)
        or not np.isfinite(range_edges_m).all()
        or (np.diff(range_edges_m) <= 0).any()
        or range_edges_m[0] < 0
    ):
        raise ValueError("Physical range edges must be finite and increasing.")
    if (
        supported_range.shape != (frequencies, ranges)
        or supported_range.dtype.kind != "b"
    ):
        raise ValueError("Frequency-specific physical range support is required.")
    if valid_mask.dtype.kind != "b" or not np.isfinite(sv_linear[valid_mask]).all():
        raise ValueError("Valid calibrated sv values must be finite.")
    if (sv_linear[valid_mask] < 0).any():
        raise ValueError("Linear sv must be nonnegative.")
    if (
        len(configuration_ids) != pings
        or len(source_file_ids) != pings
        or not all(configuration_ids)
        or not all(source_file_ids)
    ):
        raise ValueError("Every ping requires a configuration and source identity.")
    if availability_policy not in ("measured", "zero_latency_replay"):
        raise ValueError("An explicit availability policy is required.")
    if ping_available_times is None:
        if availability_policy == "measured":
            raise ValueError("Measured availability requires observed timestamps.")
        available = times
    else:
        if availability_policy != "measured":
            raise ValueError("Observed availability requires the measured policy.")
        available = _time_ns(ping_available_times, "ping_available_times")
        if len(available) != pings or (available < times).any():
            raise ValueError("Availability must follow acquisition for every ping.")

    quarter_ids = times.astype("datetime64[m]").astype(np.int64) // _BIN_MINUTES
    first, last = int(quarter_ids[0]), int(quarter_ids[-1])
    count = last - first + 1
    if count > _MAX_BINS_PER_BATCH:
        raise ValueError("A bounded aggregation batch may cover at most 32 hours.")
    starts = (np.arange(first, last + 1, dtype=np.int64) * _BIN_MINUTES).astype(
        "datetime64[m]"
    )
    ends = starts + np.timedelta64(_BIN_MINUTES, "m")
    binned = np.full((count, frequencies, ranges), np.nan, dtype=np.float64)
    masks = np.zeros((count, frequencies, ranges), dtype=bool)
    availability = np.full(count, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    ping_counts = np.zeros(count, dtype=np.int64)
    configurations: list[str | None] = []
    sources: list[frozenset[str]] = []
    for offset, quarter in enumerate(range(first, last + 1)):
        indices = np.flatnonzero(quarter_ids == quarter)
        ping_counts[offset] = len(indices)
        sources.append(frozenset(source_file_ids[index] for index in indices))
        configs = {configuration_ids[index] for index in indices}
        configurations.append(next(iter(configs)) if len(configs) == 1 else None)
        if not len(indices):
            continue
        availability[offset] = available[indices].max()
        if len(configs) != 1:
            continue
        selected_valid = valid_mask[indices] & supported_range[None, :, :]
        support = selected_valid.sum(axis=0)
        total = np.where(selected_valid, sv_linear[indices], 0.0).sum(axis=0)
        np.divide(total, support, out=binned[offset], where=support > 0)
        masks[offset] = support > 0
    db = np.full_like(binned, np.nan)
    db[masks] = 10.0 * np.log10(np.maximum(binned[masks], _DB_FLOOR))
    boundaries = np.array(
        [False]
        + [
            configurations[index] != configurations[index - 1]
            for index in range(1, count)
        ],
        dtype=bool,
    )
    return BinnedAcoustics(
        bin_start=starts.astype("datetime64[ns]"),
        bin_end=ends.astype("datetime64[ns]"),
        available_time=availability,
        sv_linear=binned,
        sv_db=db,
        valid_mask=masks,
        ping_count=ping_counts,
        frequency_hz=frequency_hz.copy(),
        range_edges_m=range_edges_m.copy(),
        supported_range=supported_range.copy(),
        configuration_ids=tuple(configurations),
        configuration_boundary=boundaries,
        source_file_ids=tuple(sources),
        dataset_id=dataset_id,
        deployment_id=deployment_id,
        instrument_id=instrument_id,
        calibration_report_sha256=calibration_report_sha256,
        availability_policy=availability_policy,
    )


def build_window(
    series: BinnedAcoustics,
    *,
    cutoff: np.datetime64,
    split_start: np.datetime64,
    split_end: np.datetime64,
    partition: str,
    allowed_source_file_ids: set[str],
    expected_dataset_id: str,
    expected_deployment_id: str,
    expected_instrument_id: str,
    analysis_frequency_hz: int,
    analysis_range_m: tuple[float, float],
    context_bins: int = 96,
    horizons: tuple[int, ...] = (1, 3, 6),
    minimum_target_support: float = 0.8,
) -> AcousticWindow:
    """Select a complete split-contained window and its supported future indices."""
    if partition not in ("train", "validation", "calibration", "test"):
        raise ValueError("A declared chronological partition is required.")
    if (
        series.dataset_id != expected_dataset_id
        or series.deployment_id != expected_deployment_id
        or series.instrument_id != expected_instrument_id
    ):
        raise ValueError("Source identity differs from the split manifest.")
    if not allowed_source_file_ids:
        raise ValueError("A split-specific source allowlist is required.")
    if context_bins <= 0 or not horizons or any(h not in (1, 3, 6) for h in horizons):
        raise ValueError(
            "Context length or horizon differs from the declared protocol."
        )
    if not 0 < minimum_target_support <= 1:
        raise ValueError("Target support threshold is invalid.")
    cutoff_ns = np.datetime64(cutoff, "ns")
    start_ns = np.datetime64(split_start, "ns")
    end_ns = np.datetime64(split_end, "ns")
    if np.isnat(cutoff_ns) or np.isnat(start_ns) or np.isnat(end_ns):
        raise ValueError("Finite UTC bin and split timestamps are required.")
    context_start = cutoff_ns - context_bins * np.timedelta64(_BIN_MINUTES, "m")
    target_end = cutoff_ns + max(horizons) * np.timedelta64(1, "h")
    if context_start < start_ns or target_end > end_ns:
        raise ValueError("Window crosses its chronological split boundary.")
    context = np.flatnonzero(
        (series.bin_end > context_start) & (series.bin_end <= cutoff_ns)
    )
    expected_context_ends = context_start + np.arange(
        1, context_bins + 1
    ) * np.timedelta64(_BIN_MINUTES, "m")
    if len(context) != context_bins or not np.array_equal(
        series.bin_end[context], expected_context_ends
    ):
        raise ValueError("Context has missing time bins.")
    target_groups = []
    for horizon in horizons:
        interval_start = cutoff_ns + (horizon - 1) * np.timedelta64(1, "h")
        interval_end = interval_start + np.timedelta64(1, "h")
        indices = np.flatnonzero(
            (series.bin_end > interval_start) & (series.bin_end <= interval_end)
        )
        if len(indices) != 4 or series.bin_end[indices[-1]] != interval_end:
            raise ValueError("Target has missing time bins.")
        target_groups.append(indices)
    extent = np.flatnonzero(
        (series.bin_end > context_start) & (series.bin_end <= target_end)
    )
    if (
        any(series.configuration_ids[index] is None for index in extent)
        or len({series.configuration_ids[index] for index in extent}) != 1
    ):
        raise ValueError("Window crosses a configuration boundary or missing segment.")
    sources = frozenset().union(*(series.source_file_ids[index] for index in extent))
    if not sources or not sources.issubset(allowed_source_file_ids):
        raise ValueError("Window source identities differ from the split allowlist.")
    channel = np.flatnonzero(series.frequency_hz == analysis_frequency_hz)
    if len(channel) != 1:
        raise ValueError("Analysis frequency is not an exact available channel.")
    lower, upper = analysis_range_m
    ranges = np.flatnonzero(
        (series.range_edges_m[:-1] >= lower) & (series.range_edges_m[1:] <= upper)
    )
    if (
        not len(ranges)
        or lower >= upper
        or not series.supported_range[channel[0], ranges].all()
    ):
        raise ValueError("Analysis band includes unsupported physical range.")
    # A bin is input only if it was available at the issue time. The future
    # target is read separately and cannot influence context construction.
    available_context = (~np.isnat(series.available_time[context])) & (
        series.available_time[context] <= cutoff_ns
    )
    context_mask = series.valid_mask[context] & available_context[:, None, None]
    context_linear = np.where(context_mask, series.sv_linear[context], np.nan)
    if not context_mask.any():
        raise ValueError("No context was available at forecast issue time.")
    widths = np.diff(series.range_edges_m)[ranges]
    targets: list[float] = []
    support_fractions: list[float] = []
    for indices in target_groups:
        values = series.sv_linear[np.ix_(indices, channel, ranges)][:, 0, :]
        mask = series.valid_mask[np.ix_(indices, channel, ranges)][:, 0, :]
        weights = np.broadcast_to(widths, mask.shape)
        denominator = float(weights.sum())
        support = float(weights[mask].sum() / denominator)
        if support < minimum_target_support:
            raise ValueError("Target support is below the frozen threshold.")
        targets.append(
            float((values[mask] * weights[mask]).sum() / weights[mask].sum())
        )
        support_fractions.append(support)
    linear = np.array(targets, dtype=np.float64)
    return AcousticWindow(
        cutoff=cutoff_ns,
        partition=partition,
        context_linear=context_linear,
        context_mask=context_mask,
        target_index_linear=linear,
        target_index_db=10.0 * np.log10(np.maximum(linear, _DB_FLOOR)),
        target_support=np.array(support_fractions, dtype=np.float64),
        source_file_ids=sources,
    )
