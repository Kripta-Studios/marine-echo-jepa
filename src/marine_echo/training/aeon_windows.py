"""Fixed 24x4 past-only AEON FullDepth hourly forecast rows."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from marine_echo.training.aeon_corpus import AeonHourlySlot


@dataclass(frozen=True)
class AeonWindowPlan:
    context_hours: int = 24
    horizons_hours: tuple[int, ...] = (1, 3, 6)

    def __post_init__(self) -> None:
        if self.context_hours != 24 or self.horizons_hours != (1, 3, 6):
            raise ValueError("AEON FullDepth study fixes 24 predecessors and 1/3/6 steps.")


@dataclass(frozen=True)
class AeonHourlyWindow:
    row_id: str
    partition: str
    cutoff_source_timestamp: np.datetime64
    cutoff_interval_id: int
    context_db: NDArray[np.float64]
    context_mask: NDArray[np.bool_]
    context_interval_ids: NDArray[np.int64]
    context_source_timestamps: NDArray[np.datetime64]
    target_interval_ids: NDArray[np.int64]
    target_source_timestamps: NDArray[np.datetime64]
    target_db: NDArray[np.float64]
    target_mask: NDArray[np.bool_]
    target_qc_status: tuple[str, str, str]
    source_archive_sha256: str
    past_members: tuple[str, ...]
    target_members: tuple[str, ...]


def _roughly_hourly(left: np.datetime64, right: np.datetime64, steps: int = 1) -> bool:
    delta = right - left
    return bool(np.timedelta64(55 * steps, "m") <= delta <= np.timedelta64(65 * steps, "m"))


def iter_aeon_windows(
    slots: Iterable[AeonHourlySlot],
    *,
    plan: AeonWindowPlan,
    partition: str,
    partition_start: str,
    partition_end_exclusive: str,
) -> Iterator[AeonHourlyWindow]:
    """Issue from complete predecessors, irrespective of future target availability."""
    if partition not in ("train", "validation", "development_fit", "development_assessment"):
        raise ValueError("Only reviewed AEON development partitions may form windows.")
    start = np.datetime64(partition_start, "us")
    end = np.datetime64(partition_end_exclusive, "us")
    if np.isnat(start) or np.isnat(end) or end <= start:
        raise ValueError("AEON partition bounds are invalid.")
    sequence = sorted((slot for slot in slots if start <= slot.source_timestamp < end), key=lambda x: x.interval_id)
    if not sequence:
        return
    by_id = {slot.interval_id: slot for slot in sequence}
    if len(by_id) != len(sequence):
        raise ValueError("AEON joined source interval IDs are duplicated.")
    archive_hashes = {slot.archive_sha256 for slot in sequence}
    if len(archive_hashes) != 1:
        raise ValueError("AEON partition mixes source archives.")
    last_interval_id = sequence[-1].interval_id
    for cutoff in sequence:
        if cutoff.interval_id + 6 > last_interval_id:
            continue
        prior = [by_id.get(cutoff.interval_id - offset) for offset in range(23, -1, -1)]
        if any(item is None for item in prior):
            continue
        context = [item for item in prior if item is not None]
        if any(not item.observed_mask[0] for item in context):
            continue
        if any(
            not _roughly_hourly(left.source_timestamp, right.source_timestamp)
            for left, right in zip(context, context[1:])
        ):
            continue
        context_values = np.stack([item.sv_db for item in context])
        context_mask = np.stack([item.observed_mask for item in context])
        targets = [by_id.get(cutoff.interval_id + offset) for offset in plan.horizons_hours]
        target_values = np.full(3, np.nan)
        target_mask = np.zeros(3, dtype=bool)
        target_times = np.full(3, np.datetime64("NaT", "us"), dtype="datetime64[us]")
        target_status = []
        target_members: set[str] = set()
        for index, future in enumerate(targets):
            if future is None:
                target_status.append("MISSING_INTERVAL")
                continue
            target_times[index] = future.source_timestamp
            target_members.update(future.member_names)
            if not _roughly_hourly(cutoff.source_timestamp, future.source_timestamp, plan.horizons_hours[index]):
                target_status.append("SOURCE_TIME_DISCONTINUITY")
                continue
            target_status.append(future.qc_status[0])
            if future.observed_mask[0]:
                target_values[index] = future.sv_db[0]
                target_mask[index] = True
        row_id = hashlib.sha256(
            f"aeon-full-depth:{cutoff.archive_sha256}:{cutoff.interval_id}:24:1,3,6".encode()
        ).hexdigest()
        yield AeonHourlyWindow(
            row_id=row_id,
            partition=partition,
            cutoff_source_timestamp=cutoff.source_timestamp,
            cutoff_interval_id=cutoff.interval_id,
            context_db=context_values,
            context_mask=context_mask,
            context_interval_ids=np.asarray([item.interval_id for item in context]),
            context_source_timestamps=np.asarray(
                [item.source_timestamp for item in context], dtype="datetime64[us]"
            ),
            target_interval_ids=np.asarray(
                [cutoff.interval_id + offset for offset in plan.horizons_hours]
            ),
            target_source_timestamps=target_times,
            target_db=target_values,
            target_mask=target_mask,
            target_qc_status=tuple(target_status),  # type: ignore[arg-type]
            source_archive_sha256=cutoff.archive_sha256,
            past_members=tuple(sorted({name for item in context for name in item.member_names})),
            target_members=tuple(sorted(target_members)),
        )
