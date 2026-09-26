"""Daily-weighted predictive scores and paired contiguous-block uncertainty."""

import numpy as np
from numpy.typing import NDArray

QUANTILES = np.array([0.05, 0.25, 0.5, 0.75, 0.95])


def pinball(truth: NDArray[np.float64], prediction: NDArray[np.float64]) -> NDArray[np.float64]:
    residual = truth[..., None] - prediction
    if prediction.shape != (*truth.shape, 5):
        raise ValueError("Expected one prediction per declared quantile.")
    if np.any(np.diff(prediction, axis=-1) < 0):
        raise ValueError("Quantiles must be monotone.")
    return (QUANTILES - (residual < 0)) * residual


def daily_metrics(
    truth: NDArray[np.float64],
    prediction: NDArray[np.float64],
    times: NDArray[np.datetime64],
    *,
    shared_support: NDArray[np.bool_] | None = None,
) -> dict[str, float | int | None]:
    if len(truth) != len(times) or len(truth) == 0 or times.dtype.kind != "M":
        raise ValueError("Nonempty aligned UTC acquisition timestamps are required.")
    if np.isnat(times).any() or (np.diff(times) <= np.timedelta64(0, "s")).any():
        raise ValueError("Unique chronological timestamps required.")
    days = times.astype("datetime64[D]")
    losses = pinball(truth, prediction)
    valid = np.isfinite(truth).all(axis=1) & np.isfinite(prediction).all(axis=(1, 2))
    result: dict[str, float | int | None] = {
        "eligible_rows": len(truth),
        "predicted_rows": int(valid.sum()),
        "prediction_coverage": float(valid.mean()),
        "daily_mean_pinball_db": None,
        "median_mae_db": None,
        "coverage90": None,
        "width90_db": None,
    }
    scoring = np.ones(len(truth), dtype=bool) if shared_support is None else shared_support
    if scoring.shape != (len(truth),) or scoring.dtype.kind != "b":
        raise ValueError("Shared support must be a frozen aligned boolean mask.")
    result["shared_support_rows"] = int(scoring.sum())
    # No implicit per-model deletion. Any absent prediction on common support invalidates
    # its score; callers must freeze one common comparison support for every model.
    if not scoring.any() or not valid[scoring].all():
        return result
    valid = scoring
    result["daily_mean_pinball_db"] = float(
        np.mean([losses[valid & (days == day)].mean() for day in np.unique(days[valid])])
    )
    result["median_mae_db"] = float(np.abs(truth[valid] - prediction[valid, :, 2]).mean())
    result["coverage90"] = float(
        (
            (truth[valid] >= prediction[valid, :, 0]) & (truth[valid] <= prediction[valid, :, 4])
        ).mean()
    )
    result["width90_db"] = float((prediction[valid, :, 4] - prediction[valid, :, 0]).mean())
    return result


def interval_widening(
    truth: NDArray[np.float64], prediction: NDArray[np.float64], partition: str
) -> NDArray[np.float64]:
    if partition != "calibration":
        raise ValueError("Interval adjustment uses calibration only.")
    score = np.maximum(prediction[:, :, 0] - truth, truth - prediction[:, :, -1])
    if not np.isfinite(score).all() or len(score) == 0:
        raise ValueError("Calibration scores must be finite and nonempty.")
    return np.maximum(0, np.quantile(score, 0.9, axis=0, method="higher"))


def paired_blocks(
    times: NDArray[np.datetime64],
    errors: NDArray[np.float64],
    *,
    draws: int = 2000,
    seed: int = 20260926,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Resample common 48-hour calendar blocks, retaining every model's paired rows."""
    if len(times) != len(errors) or len(times) == 0 or errors.ndim != 2:
        raise ValueError("Aligned rows and model columns are required.")
    if (
        draws != 2000
        or seed != 20260926
        or np.isnat(times).any()
        or np.any(np.diff(times) <= np.timedelta64(0, "s"))
    ):
        raise ValueError("Chronological finite timestamps required.")
    if not np.isfinite(errors).all():
        raise ValueError("Compare models on explicitly shared valid support.")
    # First midnight is fixed by the evaluation timestamps, not model errors.
    days = times.astype("datetime64[D]")
    unique_days = np.unique(days)
    daily = np.stack([errors[days == day].mean(axis=0) for day in unique_days])
    block_ids = ((unique_days - unique_days[0]) / np.timedelta64(2, "D")).astype(int)
    blocks = np.unique(block_ids)
    sums = np.stack([daily[block_ids == block].sum(axis=0) for block in blocks])
    counts = np.array([(block_ids == block).sum() for block in blocks])
    chosen = np.random.default_rng(seed).integers(0, len(blocks), (draws, len(blocks)))
    return chosen, sums[chosen].sum(axis=1) / counts[chosen].sum(axis=1)[:, None]
