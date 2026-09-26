"""Features retain source timestamp membership."""

import numpy as np
from numpy.typing import NDArray


def context_indices(bin_ends: NDArray[np.datetime64], cutoff: np.datetime64) -> NDArray[np.int64]:
    validate_times(bin_ends)
    return np.flatnonzero((bin_ends > cutoff - np.timedelta64(24, "h")) & (bin_ends <= cutoff))


def target_indices(
    bin_ends: NDArray[np.datetime64], cutoff: np.datetime64, horizon: int
) -> NDArray[np.int64]:
    validate_times(bin_ends)
    if horizon not in (1, 3, 6):
        raise ValueError("Unsupported horizon.")
    start = cutoff + np.timedelta64(horizon - 1, "h")
    return np.flatnonzero((bin_ends > start) & (bin_ends <= start + np.timedelta64(1, "h")))


def validate_times(times: NDArray[np.datetime64]) -> None:
    if np.isnat(times).any() or (np.diff(times) <= np.timedelta64(0, "s")).any():
        raise ValueError("Timestamps must be unique, finite and increasing.")


def permitted_anchor(cutoff: np.datetime64, start: np.datetime64, end: np.datetime64) -> bool:
    return bool(
        cutoff - np.timedelta64(24, "h") >= start and cutoff + np.timedelta64(6, "h") <= end
    )


def past_features(
    values: NDArray[np.float64], bin_ends: NDArray[np.datetime64], cutoff: np.datetime64
) -> NDArray[np.float64]:
    """Summaries use only complete past bins; unavailable columns stay explicitly masked."""
    if len(values) != len(bin_ends):
        raise ValueError("Value/timestamp length mismatch.")
    past = values[context_indices(bin_ends, cutoff)]
    if len(past) == 0:
        raise ValueError("No observations before cutoff.")
    flat = past.reshape(len(past), -1)
    valid = np.isfinite(flat)
    count = valid.sum(axis=0)
    total = np.where(valid, flat, 0).sum(axis=0)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    last = np.zeros(flat.shape[1])
    for column in range(flat.shape[1]):
        available = np.flatnonzero(valid[:, column])
        if len(available):
            last[column] = flat[available[-1], column]
    return np.concatenate([last, mean, count / 96])


class TrainScaler:
    def fit(self, values: NDArray[np.float64], partition: str) -> "TrainScaler":
        if partition != "train":
            raise ValueError("Scaler may fit training data only.")
        if not np.isfinite(values).all():
            raise ValueError("Imputation must be explicit before scaling.")
        self.mean = values.mean(axis=0)
        self.scale = np.maximum(values.std(axis=0), 1e-6)
        return self

    def transform(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return (values - self.mean) / self.scale


def require_training_partition(partitions: list[str]) -> None:
    if not partitions or set(partitions) != {"train"}:
        raise ValueError("SSL can read training windows only.")
