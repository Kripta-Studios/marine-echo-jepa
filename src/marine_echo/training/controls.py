"""Bounded train-only diagnostic control permutations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def separated_train_permutation(
    anchor_times: NDArray[np.datetime64],
    partitions: list[str],
    *,
    seed: int,
    minimum_gap_hours: int,
) -> NDArray[np.int64]:
    """Circularly permute train future targets with a minimum time gap.

    This is a diagnostic association-breaking control, not an alternative
    chronological split. An infeasible requested gap fails visibly.
    """
    if len(anchor_times) != len(partitions) or len(anchor_times) < 2 or len(anchor_times) > 4096:
        raise ValueError("Control batch size or partition metadata is invalid.")
    if set(partitions) != {"train"}:
        raise ValueError("Control permutation may read training rows only.")
    if (
        minimum_gap_hours <= 0
        or np.isnat(anchor_times).any()
        or np.any(np.diff(anchor_times) <= np.timedelta64(0, "s"))
    ):
        raise ValueError("Control times or separation are invalid.")
    generator = np.random.default_rng(seed)
    indices = np.arange(len(anchor_times))
    for offset in generator.permutation(np.arange(1, len(anchor_times))):
        permutation = (indices + offset) % len(anchor_times)
        gaps = np.abs(
            (anchor_times[permutation] - anchor_times).astype("timedelta64[s]").astype(np.int64)
        )
        if np.all(gaps >= minimum_gap_hours * 3600):
            return permutation.astype(np.int64)
    raise ValueError("No bounded circular control permutation satisfies the temporal gap.")
