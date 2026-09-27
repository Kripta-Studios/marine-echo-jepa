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


def _eligible_dates_by_horizon(
    source_times: NDArray[np.datetime64], observed: NDArray[np.bool_]
) -> list[NDArray[np.datetime64]]:
    dates = source_times.astype("datetime64[D]")
    result = []
    for horizon in range(3):
        unique, counts = np.unique(dates[observed[:, horizon], horizon], return_counts=True)
        result.append(unique[counts >= 18])
    return result


def daily_pinball(
    truth: NDArray[np.float64],
    forecast: NDArray[np.float64],
    observed: NDArray[np.bool_],
    source_times: NDArray[np.datetime64],
) -> dict[str, object]:
    """Average scored rows per source date, then dates, then three horizons."""
    _validate(truth, forecast, observed, source_times)
    dates = source_times.astype("datetime64[D]")
    residual = truth[..., None] - forecast
    quantile_loss = np.maximum(QUANTILES * residual, (QUANTILES - 1) * residual)
    eligible_days = _eligible_dates_by_horizon(source_times, observed)
    daily_means: list[float] = []
    quantile_means: list[list[float]] = []
    scored_days: list[int] = []
    eligible_rows: list[int] = []
    median_mae: list[float] = []
    coverage90: list[float] = []
    width90: list[float] = []
    for index in range(3):
        days = eligible_days[index]
        if not len(days):
            raise ValueError("Every AEON horizon needs an eligible source date.")
        valid = observed[:, index] & np.isin(dates[:, index], days)
        by_day = np.stack(
            [quantile_loss[valid & (dates[:, index] == day), index].mean(axis=0) for day in days]
        )
        by_quantile = by_day.mean(axis=0)
        quantile_means.append(by_quantile.astype(float).tolist())
        daily_means.append(float(by_quantile.mean()))
        scored_days.append(len(days))
        eligible_rows.append(int(valid.sum()))
        median_mae.append(float(np.abs(truth[valid, index] - forecast[valid, index, 2]).mean()))
        coverage90.append(
            float(
                (
                    (truth[valid, index] >= forecast[valid, index, 0])
                    & (truth[valid, index] <= forecast[valid, index, 4])
                ).mean()
            )
        )
        width90.append(float((forecast[valid, index, 4] - forecast[valid, index, 0]).mean()))
    return {
        "issued_rows": len(truth),
        "scored_rows_per_horizon": observed.sum(axis=0).astype(int).tolist(),
        "scored_fraction_per_horizon": observed.mean(axis=0).astype(float).tolist(),
        "eligible_scored_rows_per_horizon": eligible_rows,
        "eligible_days_per_horizon": scored_days,
        "scored_days_per_horizon": scored_days,
        "eligible_source_dates_by_horizon": [days.astype(str).tolist() for days in eligible_days],
        "daily_mean_pinball_db_by_horizon_quantile": quantile_means,
        "daily_mean_pinball_db_per_horizon": daily_means,
        "primary_daily_mean_pinball_db": float(np.mean(daily_means)),
        "median_mae_db_per_horizon": median_mae,
        "coverage90_per_horizon": coverage90,
        "width90_db_per_horizon": width90,
        "profile_mae": "NOT_APPLICABLE_FULL_DEPTH_SCALAR_STUDY",
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
    if not observed.any(axis=0).all():
        raise ValueError("AEON paired comparison needs support at every horizon.")
    dates = source_times.astype("datetime64[D]")
    eligible_days = _eligible_dates_by_horizon(source_times, observed)
    if any(not len(days) for days in eligible_days):
        raise ValueError("AEON paired comparison needs eligible dates at every horizon.")
    first = min(days[0] for days in eligible_days)
    last = max(days[-1] for days in eligible_days)
    all_dates = np.arange(first, last + np.timedelta64(1, "D"))
    base_loss = _row_pinball(truth, baseline)
    candidate_loss = _row_pinball(truth, candidate)
    daily_difference = np.full((len(all_dates), 3), np.nan)
    for day_index, day in enumerate(all_dates):
        for horizon in range(3):
            valid = (
                observed[:, horizon]
                & (dates[:, horizon] == day)
                & np.isin(day, eligible_days[horizon])
            )
            if valid.any():
                daily_difference[day_index, horizon] = float(
                    (candidate_loss[valid, horizon] - base_loss[valid, horizon]).mean()
                )
    blocks = [
        block
        for start in range(0, len(all_dates), 2)
        if np.isfinite(block := daily_difference[start : start + 2]).any()
    ]
    rng = np.random.default_rng(seed)
    results = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        for _ in range(10_000):
            chosen = rng.integers(0, len(blocks), size=len(blocks))
            sample = np.concatenate([blocks[index] for index in chosen], axis=0)
            if not np.isnan(sample).all(axis=0).any():
                break
        else:
            raise ValueError("Could not draw all-horizon AEON paired bootstrap support.")
        results[draw] = float(np.nanmean(sample, axis=0).mean())
    return results
