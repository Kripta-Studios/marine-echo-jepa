"""Causal 24-hour windows over transformed AZFP response-code summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RawWindow:
    cutoff: np.datetime64
    context_codes: NDArray[np.float64]
    context_mask: NDArray[np.bool_]
    context_valid_fraction: NDArray[np.float64]
    targets: NDArray[np.float64]
    target_observed: NDArray[np.bool_]
    target_times: NDArray[np.datetime64]


def build_raw_window(
    first_slot_start: np.datetime64,
    profile_code_sum: NDArray[np.float64],
    profile_code_count: NDArray[np.int64],
    target_code_sum: NDArray[np.float64],
    target_code_count: NDArray[np.int64],
    *,
    cutoff_slot: int,
) -> RawWindow:
    """Use exactly 96 past quarter-hours and fixed +1/+3/+6 UTC hours."""
    shape = profile_code_sum.shape
    if (
        profile_code_sum.ndim != 3
        or shape[1:] != (4, 64)
        or profile_code_count.shape != shape
        or target_code_sum.shape != (shape[0],)
        or target_code_count.shape != (shape[0],)
        or not 96 <= cutoff_slot
        or cutoff_slot + 4 > shape[0]
        or (profile_code_count < 0).any()
        or (target_code_count < 0).any()
    ):
        raise ValueError("Complete causal raw-response window arrays required")
    start = cutoff_slot - 96
    past_sum = profile_code_sum[start:cutoff_slot]
    past_count = profile_code_count[start:cutoff_slot]
    mask = past_count > 0
    context = np.full((96, 4, 64), np.nan, dtype=np.float64)
    np.divide(past_sum, past_count, out=context, where=mask)
    if not np.isfinite(context[mask]).all():
        raise ValueError("Observed past response code is nonfinite")
    targets = np.full(3, np.nan, dtype=np.float64)
    observed = np.zeros(3, dtype=np.bool_)
    times = np.empty(3, dtype="datetime64[ns]")
    for h, horizon in enumerate((1, 3, 6)):
        target_start = cutoff_slot + 4 * (horizon - 1)
        if target_start + 4 <= shape[0]:
            count = int(target_code_count[target_start : target_start + 4].sum())
            if count:
                targets[h] = float(target_code_sum[target_start : target_start + 4].sum()) / count
                observed[h] = True
        times[h] = first_slot_start + target_start * np.timedelta64(15, "m")
    return RawWindow(
        cutoff=first_slot_start + cutoff_slot * np.timedelta64(15, "m"),
        context_codes=context,
        context_mask=mask,
        context_valid_fraction=mask.mean(axis=(1, 2)),
        targets=targets,
        target_observed=observed,
        target_times=times,
    )
