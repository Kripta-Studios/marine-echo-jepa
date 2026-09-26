"""Diagnostic binning of uncalibrated AZFP sample-index counts."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np


def aggregate_raw_count_file(
    ping_times: np.ndarray,
    counts: np.ndarray,
    *,
    sample_bins: int = 64,
) -> list[dict[str, Any]]:
    """Aggregate one sorted raw file into trailing 15-minute UTC bins.

    The last dimension remains a sample index. It is not physical range or Sv.
    """
    if counts.ndim != 3 or ping_times.ndim != 1 or counts.shape[1] != len(ping_times):
        raise ValueError(
            "Expected counts[channel, ping, sample] and one time per ping."
        )
    if not 1 <= sample_bins <= counts.shape[2]:
        raise ValueError("Sample-bin count exceeds available samples.")
    times_ns = ping_times.astype("datetime64[ns]")
    if np.isnat(times_ns).any() or np.any(times_ns[1:] <= times_ns[:-1]):
        raise ValueError("Ping times must be finite and strictly increasing.")
    quarters = times_ns.astype("datetime64[m]").astype(np.int64) // 15
    edges = np.linspace(0, counts.shape[2], sample_bins + 1, dtype=int)
    rows: list[dict[str, Any]] = []
    for quarter in np.unique(quarters):
        selected = counts[:, quarters == quarter, :]
        channel_values: list[list[float | None]] = []
        for channel in selected:
            values: list[float | None] = []
            for start, end in pairwise(edges):
                segment = channel[:, start:end]
                finite = segment[np.isfinite(segment)]
                values.append(float(finite.mean()) if finite.size else None)
            channel_values.append(values)
        start_time = np.datetime64(int(quarter * 15), "m").astype("datetime64[s]")
        end_time = start_time + np.timedelta64(15, "m")
        rows.append(
            {
                "bin_start_utc": str(start_time) + "Z",
                "bin_end_utc": str(end_time) + "Z",
                "ping_count": int(np.count_nonzero(quarters == quarter)),
                "counts": channel_values,
            }
        )
    return rows
