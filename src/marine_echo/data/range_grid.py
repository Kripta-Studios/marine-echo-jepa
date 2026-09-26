"""Conservative linear-sv overlap integration on explicitly supplied physical edges.

No sample-centre to edge inference, calibration, noise rule or depth conversion is
performed here. A physical adapter must justify those separately. Each invocation
handles one channel with a fixed original geometry and at most 4096 pings.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RangeGrid:
    sv_linear: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    support_fraction: NDArray[np.float64]
    supported_range: NDArray[np.bool_]


def regrid_linear_sv(
    values: NDArray[np.float64],
    valid: NDArray[np.bool_],
    source_edges_m: NDArray[np.float64],
    target_edges_m: NDArray[np.float64],
) -> RangeGrid:
    """Integrate piecewise-constant linear sv, retaining only fully supported cells.

    Support fractions retain the full target-cell denominator, including absent
    or geometrically unsupported source range. Partial cells remain NaN/masked;
    they are never extrapolated or silently treated as completely observed.
    """
    if (
        values.ndim != 2
        or values.shape != valid.shape
        or valid.dtype.kind != "b"
        or not 0 < values.shape[0] <= 4096
        or not 0 < values.shape[1] <= 8192
        or target_edges_m.size > 1025
    ):
        raise ValueError("Expected bounded [ping,range] values and Boolean mask.")
    for edges in (source_edges_m, target_edges_m):
        if (
            edges.ndim != 1
            or len(edges) < 2
            or not np.isfinite(edges).all()
            or edges[0] < 0
            or (np.diff(edges) <= 0).any()
        ):
            raise ValueError("Physical range edges must be finite, nonnegative and increasing.")
    if len(source_edges_m) != values.shape[1] + 1:
        raise ValueError("Source edges must match the original samples.")
    if not np.isfinite(values[valid]).all() or (values[valid] < 0).any():
        raise ValueError("Valid physical linear sv must be finite and nonnegative.")
    widths = np.diff(target_edges_m)
    overlap = np.maximum(
        0.0,
        np.minimum(source_edges_m[1:, None], target_edges_m[None, 1:])
        - np.maximum(source_edges_m[:-1, None], target_edges_m[None, :-1]),
    )
    observed_width = valid.astype(np.float64) @ overlap
    support = np.clip(observed_width / widths, 0.0, 1.0)
    geometric = overlap.sum(axis=0)
    # Tolerance is limited to floating arithmetic, not a relaxed support policy.
    supported = np.isclose(geometric, widths, rtol=1e-12, atol=0.0)
    mask = np.isclose(observed_width, widths, rtol=1e-12, atol=0.0) & supported
    integrated = np.where(valid, values, 0.0).astype(np.float64) @ overlap
    result = np.full(observed_width.shape, np.nan)
    np.divide(integrated, widths, out=result, where=mask)
    return RangeGrid(result, mask, support, supported)
