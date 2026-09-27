"""Prospective source-date scoring rules for the AEON hourly Sv study.

These pure functions do not read a partition or open an artifact. The caller
must enforce the independent calibration/test freeze and source-row lineage.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


QUANTILES = np.asarray([0.05, 0.25, 0.5, 0.75, 0.95], dtype=np.float64)
HORIZONS = (1, 3, 6)


def _validate(
    truth: NDArray[np.float64],
    forecast: NDArray[np.float64],
    observed: NDArray[np.bool_],
    source_times: NDArray[np.datetime64] | None = None,
) -> None:
    if (
        truth.ndim != 2
        or truth.shape[1] != 3
        or forecast.shape != (*truth.shape, 5)
        or observed.shape != truth.shape
        or observed.dtype.kind != "b"
        or not np.isfinite(forecast).all()
        or (np.diff(forecast, axis=-1) < 0).any()
        or not np.isfinite(truth[observed]).all()
    ):
        raise ValueError("AEON truth, issued forecasts, masks or quantiles are invalid.")
    if source_times is not None and (
        source_times.shape != truth.shape
        or source_times.dtype.kind != "M"
        or np.isnat(source_times[observed]).any()
    ):
        raise ValueError("Observed AEON outcomes need aligned source timestamps.")


def eligible_source_dates(
    source_times: NDArray[np.datetime64],
    observed: NDArray[np.bool_],
    *,
    minimum_anchors: int = 18,
) -> dict[int, int]:
    """Count source dates with the prespecified scored-anchor floor per horizon."""
    if (
        source_times.shape != observed.shape
        or observed.ndim != 2
        or observed.shape[1] != 3
        or observed.dtype.kind != "b"
        or source_times.dtype.kind != "M"
        or np.isnat(source_times[observed]).any()
        or minimum_anchors != 18
    ):
        raise ValueError("AEON source-date eligibility inputs differ from protocol.")
    dates = source_times.astype("datetime64[D]")
    result = {}
    for index, horizon in enumerate(HORIZONS):
        valid_dates = dates[observed[:, index], index]
        _, counts = np.unique(valid_dates, return_counts=True)
        result[horizon] = int(np.count_nonzero(counts >= minimum_anchors))
    return result


def _row_pinball(truth: NDArray[np.float64], forecast: NDArray[np.float64]) -> NDArray[np.float64]:
    residual = truth[..., None] - forecast
    return np.maximum(QUANTILES * residual, (QUANTILES - 1) * residual).mean(axis=-1)


def daily_pinball(
    truth: NDArray[np.float64],
    forecast: NDArray[np.float64],
    observed: NDArray[np.bool_],
    source_times: NDArray[np.datetime64],
) -> dict[str, object]:
    """Average scored rows per source date, then dates, then three horizons."""
    _validate(truth, forecast, observed, source_times)
    dates = source_times.astype("datetime64[D]")
    row_loss = _row_pinball(truth, forecast)
    daily_means: list[float] = []
    scored_days: list[int] = []
    for index in range(3):
        valid = observed[:, index]
        days = np.unique(dates[valid, index])
        if not len(days):
            raise ValueError("Every AEON horizon needs observed issued outcomes.")
        daily = [float(row_loss[valid & (dates[:, index] == day), index].mean()) for day in days]
        daily_means.append(float(np.mean(daily)))
        scored_days.append(len(days))
    return {
        "issued_rows": len(truth),
        "scored_rows_per_horizon": observed.sum(axis=0).astype(int).tolist(),
        "scored_days_per_horizon": scored_days,
        "daily_mean_pinball_db_per_horizon": daily_means,
        "primary_daily_mean_pinball_db": float(np.mean(daily_means)),
    }


def calibrate_interval_widening(
    truth: NDArray[np.float64],
    forecast: NDArray[np.float64],
    observed: NDArray[np.bool_],
    *,
    partition: str,
) -> NDArray[np.float64]:
    """Fit nonnegative 90% interval widening on calibration outcomes only."""
    if partition != "calibration":
        raise ValueError("AEON interval widening may fit calibration only.")
    _validate(truth, forecast, observed)
    adjustments = []
    for horizon in range(3):
        valid = observed[:, horizon]
        if not valid.any():
            raise ValueError("Every AEON calibration horizon needs observed outcomes.")
        score = np.maximum(
            forecast[valid, horizon, 0] - truth[valid, horizon],
            truth[valid, horizon] - forecast[valid, horizon, 4],
        )
        adjustments.append(max(0.0, float(np.quantile(score, 0.9, method="higher"))))
    return np.asarray(adjustments, dtype=np.float64)


def apply_interval_widening(
    forecast: NDArray[np.float64], adjustment: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Change only the frozen 90% interval endpoints."""
    if (
        forecast.ndim != 3
        or forecast.shape[1:] != (3, 5)
        or adjustment.shape != (3,)
        or not np.isfinite(forecast).all()
        or not np.isfinite(adjustment).all()
        or (adjustment < 0).any()
        or (np.diff(forecast, axis=-1) < 0).any()
    ):
        raise ValueError("AEON interval adjustment or forecast is invalid.")
    widened = forecast.copy()
    widened[:, :, 0] -= adjustment
    widened[:, :, 4] += adjustment
    return widened


def paired_48h_bootstrap(
    truth: NDArray[np.float64],
    baseline: NDArray[np.float64],
    candidate: NDArray[np.float64],
    observed: NDArray[np.bool_],
    source_times: NDArray[np.datetime64],
    *,
    draws: int = 2000,
    seed: int = 20260926,
) -> NDArray[np.float64]:
    """Candidate-minus-baseline primary loss under paired 48-hour source-date blocks."""
    if draws != 2000 or seed != 20260926:
        raise ValueError("AEON bootstrap draws and seed are fixed before test evaluation.")
    _validate(truth, baseline, observed, source_times)
    _validate(truth, candidate, observed, source_times)
    if not observed.all(axis=1).any():
        raise ValueError("AEON paired comparison needs shared observed outcomes.")
    dates = source_times.astype("datetime64[D]")
    valid_dates = np.unique(dates[observed])
    if not len(valid_dates):
        raise ValueError("AEON paired comparison needs source dates.")
    all_dates = np.arange(valid_dates[0], valid_dates[-1] + np.timedelta64(1, "D"))
    base_loss = _row_pinball(truth, baseline)
    candidate_loss = _row_pinball(truth, candidate)
    daily_difference = np.full((len(all_dates), 3), np.nan)
    for day_index, day in enumerate(all_dates):
        for horizon in range(3):
            valid = observed[:, horizon] & (dates[:, horizon] == day)
            if valid.any():
                daily_difference[day_index, horizon] = float(
                    (candidate_loss[valid, horizon] - base_loss[valid, horizon]).mean()
                )
    blocks = [daily_difference[start : start + 2] for start in range(0, len(all_dates), 2)]
    rng = np.random.default_rng(seed)
    results = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        sample = np.concatenate([blocks[index] for index in chosen], axis=0)
        if np.isnan(sample).all(axis=0).any():
            raise ValueError("A bootstrap sample has no scored day for a horizon.")
        results[draw] = float(np.nanmean(sample, axis=0).mean())
    return results
