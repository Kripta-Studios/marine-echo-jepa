"""Aggregate transformed AZFP response codes, not calibrated acoustic Sv."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def aggregate_code_response(codes: NDArray[np.float64]) -> dict:
    """Summarize fixed averaged bins 20..199 without inventing zero truth."""
    if codes.ndim != 3 or codes.shape[0] != 4 or codes.shape[2] < 200:
        raise ValueError("Four channels with at least 200 averaged bins are required")
    selected = codes[:, :, 20:200]
    finite = np.isfinite(selected)
    zero = finite & (selected == 0)
    valid = finite & ~zero
    target_finite = finite[0]
    target_zero = zero[0]
    target_valid_ping = valid[0].all(axis=1)
    nonfinite_ping = (~target_finite).any(axis=1)
    zero_ping = ~nonfinite_ping & target_zero.any(axis=1)
    sums = np.zeros((4, 64), dtype=np.float64)
    counts = np.zeros((4, 64), dtype=np.int64)
    groups = (np.arange(180) * 64) // 180
    for channel in range(4):
        for group in range(64):
            mask = valid[channel, :, groups == group]
            values = selected[channel, :, groups == group]
            sums[channel, group] = np.where(mask, values, 0).sum(dtype=np.float64)
            counts[channel, group] = mask.sum(dtype=np.int64)
    target_values = selected[0, target_valid_ping]
    return {
        "observed_pings": int(codes.shape[1]),
        "valid_target_pings": int(target_valid_ping.sum()),
        "nonfinite_target_pings": int(nonfinite_ping.sum()),
        "zero_target_pings": int(zero_ping.sum()),
        "zero_affected_target_pings": int(target_zero.any(axis=1).sum()),
        "nonfinite_affected_target_pings": int(nonfinite_ping.sum()),
        "nonfinite_target_samples": int((~target_finite).sum()),
        "zero_target_samples": int(target_zero.sum()),
        "target_code_sum": float(target_values.sum(dtype=np.float64)),
        "target_code_count": int(target_values.size),
        "profile_code_sum": sums,
        "profile_code_count": counts,
    }
