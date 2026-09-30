"""Reconstruct source-native daily quantile scores from saved row artifacts."""

from __future__ import annotations

import numpy as np

QUANTILES = np.asarray([0.05, 0.25, 0.5, 0.75, 0.95])


def native_scores(
    predictions: np.ndarray,
    targets: np.ndarray,
    observed: np.ndarray,
    target_dates: np.ndarray,
    deployments: np.ndarray,
    minimum_daily_rows: int = 18,
) -> dict:
    if (
        predictions.shape != (*targets.shape, 5)
        or targets.shape != observed.shape
        or targets.shape != target_dates.shape
    ):
        raise ValueError("Native score row/horizon/quantile shape differs.")
    if targets.ndim != 2 or targets.shape[1] != 3 or len(deployments) != len(targets):
        raise ValueError("Native score requires three horizons and deployment identity.")
    if minimum_daily_rows < 1 or np.any(np.diff(predictions, axis=-1) < 0):
        raise ValueError("Daily floor or ordered quantiles are invalid.")
    if not np.isfinite(predictions).all() or not np.isfinite(targets[observed]).all():
        raise ValueError(
            "All forecasts and observed targets must be finite; support cannot shrink."
        )
    valid = observed
    error = targets[..., None] - predictions
    losses = np.maximum(QUANTILES * error, (QUANTILES - 1) * error).mean(axis=-1)
    daily, per_deployment = [], {}
    for deployment in sorted(set(deployments.tolist())):
        horizon_scores = []
        for index, horizon in enumerate((1, 3, 6)):
            indices = (deployments == deployment) & valid[:, index]
            values = []
            for date in sorted(set(target_dates[indices, index].tolist())):
                selected = indices & (target_dates[:, index] == date)
                count = int(selected.sum())
                if date and count >= minimum_daily_rows:
                    score = float(losses[selected, index].mean())
                    values.append(score)
                    daily.append(
                        {
                            "deployment": deployment,
                            "horizon_intervals": horizon,
                            "target_source_date": date,
                            "rows": count,
                            "pinball_db": score,
                        }
                    )
            horizon_scores.append(float(np.mean(values)) if values else None)
        per_deployment[deployment] = {
            "horizons_pinball_db": horizon_scores,
            "primary_pinball_db": float(np.mean(horizon_scores))
            if all(v is not None for v in horizon_scores)
            else None,
        }
    available = [result["primary_pinball_db"] for result in per_deployment.values()]
    return {
        "primary_pinball_db": float(np.mean(available))
        if available and all(v is not None for v in available)
        else None,
        "per_deployment": per_deployment,
        "daily_rows": daily,
        "issued_rows": len(targets),
        "scored_per_horizon": valid.sum(axis=0).tolist(),
        "eligible_source_dates_per_horizon": [
            sum(row["horizon_intervals"] == h for row in daily) for h in (1, 3, 6)
        ],
        "quantiles": QUANTILES.tolist(),
        "minimum_daily_rows": minimum_daily_rows,
        "aggregation": "equal_target_source_date_then_horizon_then_deployment",
        "unit": "source_reported_conditioned_native_product_dB",
    }
