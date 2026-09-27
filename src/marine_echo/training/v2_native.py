"""Candidate-2 windows from detected native range-length sufficient statistics."""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from marine_echo.training.v2_stream import HourlyWindow


@dataclass(frozen=True)
class NativeObservationSlot:
    start: np.datetime64
    end: np.datetime64
    linear_sv: NDArray[np.float64]
    detected_range_ping_m: NDArray[np.float64]
    observed_pings: int
    expected_pings: int
    configuration_id: str | None
    source_sha256: tuple[str, ...]
    processed_sha256: str

    def validate(self) -> None:
        weight = self.detected_range_ping_m
        if (
            self.linear_sv.shape != (4, 64)
            or weight.shape != (4, 64)
            or self.start.astype("datetime64[ns]") + np.timedelta64(15, "m") != self.end
            or self.expected_pings <= 0
            or self.observed_pings < 0
            or not np.isfinite(weight).all()
            or np.any(weight < 0)
            or np.any(weight[:, :5] != 0)
            or np.any(weight[:, 50:] != 0)
            or np.any(weight[1:] != 0)
            or float(weight[0, 5:50].sum()) > 90 * self.observed_pings + 1e-6
            or not np.isfinite(self.linear_sv[weight > 0]).all()
            or np.any(self.linear_sv[weight > 0] <= 0)
        ):
            raise ValueError("Native slot violates the fixed weighted-observation contract.")


def _known_configuration(slots: list[NativeObservationSlot]) -> bool:
    observed = [item for item in slots if item.observed_pings > 0]
    return (
        bool(observed)
        and all(item.configuration_id is not None for item in observed)
        and len({item.configuration_id for item in observed}) == 1
    )


def iter_native_hourly_windows(
    slots: Iterable[NativeObservationSlot], *, partition: str
) -> Iterator[HourlyWindow]:
    """Issue on past eligibility; score fixed future hours on their own masks."""
    if partition != "train":
        raise ValueError("Only TRAIN development may read candidate-2 acoustic values.")
    buffer: deque[NativeObservationSlot] = deque(maxlen=120)
    for slot in slots:
        slot.validate()
        buffer.append(slot)
        if len(buffer) < 120:
            continue
        sequence = list(buffer)
        context_slots = sequence[:96]
        cutoff = context_slots[-1].end
        if cutoff.astype("datetime64[m]").astype(int) % 60:
            continue
        if any(left.end != right.start for left, right in pairwise(sequence)):
            continue
        if not _known_configuration(context_slots):
            continue
        observed_context = np.array([item.observed_pings for item in context_slots])
        if (
            observed_context[-4:].sum() < 120
            or np.count_nonzero(observed_context.reshape(24, 4).sum(axis=1)) < 12
        ):
            continue
        weight_context = np.stack([item.detected_range_ping_m for item in context_slots])
        if weight_context[-4:, 0, 5:50].sum() == 0:
            continue
        linear_context = np.stack([item.linear_sv for item in context_slots])
        context_mask = weight_context > 0
        context = np.full(linear_context.shape, np.nan)
        context[context_mask] = 10 * np.log10(linear_context[context_mask])
        expected_context = np.array([item.expected_pings for item in context_slots])
        acquisition = observed_context / expected_context
        detection = np.divide(
            weight_context[:, 0, 5:50].sum(axis=1),
            90 * observed_context,
            out=np.full(96, np.nan),
            where=observed_context > 0,
        )
        context_index = np.full(96, np.nan)
        for index in range(96):
            weights = weight_context[index, 0, 5:50]
            if weights.sum() > 0:
                values = linear_context[index, 0, 5:50]
                context_index[index] = 10 * np.log10(
                    np.sum(values * weights, where=weights > 0) / weights.sum()
                )
        target_groups = [sequence[first : first + 4] for first in (96, 104, 116)]
        future_linear = np.stack([[item.linear_sv for item in group] for group in target_groups])
        future_weight = np.stack(
            [[item.detected_range_ping_m for item in group] for group in target_groups]
        )
        future_mask = future_weight > 0
        future_db = np.full(future_linear.shape, np.nan)
        future_db[future_mask] = 10 * np.log10(future_linear[future_mask])
        target_db = np.full(3, np.nan)
        target_mask = np.zeros(3, dtype=bool)
        target_detection = np.full(3, np.nan)
        target_detection_mask = np.zeros(3, dtype=bool)
        target_acquisition = np.full(3, np.nan)
        for horizon, group in enumerate(target_groups):
            observed = sum(item.observed_pings for item in group)
            expected = sum(item.expected_pings for item in group)
            weights = future_weight[horizon, :, 0, 5:50]
            values = future_linear[horizon, :, 0, 5:50]
            detected_range_ping_m = weights.sum()
            target_acquisition[horizon] = observed / expected
            if observed >= 120 and _known_configuration(group):
                target_detection_mask[horizon] = True
                target_detection[horizon] = detected_range_ping_m / (90 * observed)
                if detected_range_ping_m > 0 and target_detection[horizon] >= 0.1:
                    numerator = np.sum(values * weights, where=weights > 0)
                    target_db[horizon] = 10 * np.log10(numerator / detected_range_ping_m)
                    target_mask[horizon] = True
        sources = tuple(sorted({digest for item in sequence for digest in item.source_sha256}))
        past_sources = tuple(
            sorted({digest for item in context_slots for digest in item.source_sha256})
        )
        target_sources = tuple(
            sorted(
                {
                    digest
                    for group in target_groups
                    for item in group
                    for digest in item.source_sha256
                }
            )
        )
        row_id = hashlib.sha256((str(cutoff) + ":native-sampled-v2").encode()).hexdigest()
        yield HourlyWindow(
            row_id=row_id,
            partition=partition,
            cutoff=cutoff,
            context=context,
            context_mask=context_mask,
            context_acquisition_fraction=acquisition,
            context_detection_fraction=detection,
            context_age_minutes=np.array(
                [(cutoff - item.end) / np.timedelta64(1, "m") for item in context_slots]
            ),
            target_interval_start=np.array([group[0].start for group in target_groups]),
            target_interval_end=np.array([group[-1].end for group in target_groups]),
            target_db=target_db,
            target_mask=target_mask,
            target_detection_fraction=target_detection,
            target_detection_mask=target_detection_mask,
            target_acquisition_fraction=target_acquisition,
            future_train_db=future_db,
            future_train_mask=future_mask,
            source_sha256=sources,
            context_index_db=context_index,
            past_source_sha256=past_sources,
            target_source_sha256=target_sources,
        )
