"""Detection-conditioned native interval summaries; never full-cell v1 support."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def aggregate_detected_intervals(
    linear_sv: NDArray[np.float64],
    detected: NDArray[np.bool_],
    native_edges: NDArray[np.float64],
    target_edges: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Conserve detected range length and linear Sv, including partial band edges."""
    if linear_sv.ndim != 2 or detected.shape != linear_sv.shape:
        raise ValueError("Aligned ping/native-sample values and detection mask required")
    if detected.dtype != np.bool_:
        raise ValueError("Boolean detection mask required")
    if len(native_edges) != linear_sv.shape[1] + 1:
        raise ValueError("Native edge count differs from samples")
    for edges in (native_edges, target_edges):
        if (
            edges.ndim != 1
            or len(edges) < 2
            or not np.isfinite(edges).all()
            or (np.diff(edges) <= 0).any()
        ):
            raise ValueError("Finite increasing range edges required")
    if not np.isfinite(linear_sv[detected]).all() or (linear_sv[detected] <= 0).any():
        raise ValueError("Detected values must be finite and positive")
    overlap = np.maximum(
        0,
        np.minimum(native_edges[1:, None], target_edges[None, 1:])
        - np.maximum(native_edges[:-1, None], target_edges[None, :-1]),
    )
    weights = detected.astype(np.float64) @ overlap
    total = np.where(detected, linear_sv, 0.0) @ overlap
    mean = np.full(total.shape, np.nan)
    np.divide(total, weights, out=mean, where=weights > 0)
    return {
        "linear_sum": total,
        "detected_range_m": weights,
        "available_range_m": overlap.sum(axis=0),
        "conditional_linear_mean": mean,
    }
