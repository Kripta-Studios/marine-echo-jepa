"""TRAIN-only candidate numerical QC; no corpus or calibration approval is implied."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def support_denominator(
    expected: NDArray[np.int64], observed: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Retain cadence-derived missingness while conservatively accounting for clock jitter."""
    if expected.shape != observed.shape or expected.ndim != 1:
        raise ValueError("Matching per-bin counts required")
    if expected.dtype.kind not in "iu" or observed.dtype.kind not in "iu":
        raise ValueError("Integer acquisition counts required")
    if (expected <= 0).any() or (observed < 0).any():
        raise ValueError("Positive expected and nonnegative observed counts required")
    return np.maximum(expected, observed)


def sample_edges(centres: NDArray[np.float64], spacing_m: float) -> NDArray[np.float64]:
    """Bound manufacturer-defined uniform sampling volumes around upstream centres."""
    if centres.ndim != 1 or len(centres) < 2 or not np.isfinite(centres).all():
        raise ValueError("Finite one-dimensional sample centres required")
    if spacing_m <= 0 or not np.isfinite(spacing_m):
        raise ValueError("Positive manufacturer-derived spacing required")
    if not np.allclose(np.diff(centres), spacing_m, rtol=1e-10, atol=1e-10):
        raise ValueError("Range centres disagree with the declared sample spacing")
    edges = np.concatenate([centres - spacing_m / 2, [centres[-1] + spacing_m / 2]])
    if edges[0] < 0:
        raise ValueError("A source sampling volume extends before zero range")
    return edges


def clean_per_ping(dataset: Any, *, range_samples: int = 10) -> Any:
    """Pinned upstream linear noise subtraction uses no other ping, past or future."""
    import echopype as ep  # type: ignore[import-untyped]

    if ep.__version__ != "0.11.1":
        raise ValueError("Candidate QC requires pinned Echopype0.11.1")
    if not isinstance(range_samples, int) or range_samples <= 0:
        raise ValueError("A positive fixed range block is required")
    return ep.clean.remove_background_noise(
        dataset.copy(deep=True),
        ping_num=1,
        range_sample_num=range_samples,
        SNR_threshold="3dB",
    )
