"""Conventional calibrated acoustic baseline ladder B0 through B3.

Inputs are canonical Sv in dB, with an already frozen two-metre grid.
Raw digitizer counts are explicitly rejected. Fitting is train-only here;
validation selection and interval calibration are separate protocol steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
HORIZONS = (1, 3, 6)
_BAND = slice(5, 50)


@dataclass
class BaselineForecast:
    quantiles: NDArray[np.float64]
    eligible: NDArray[np.bool_]
    profile: NDArray[np.float64] | None = None


def linear_average_db(
    values: NDArray[np.float64],
    mask: NDArray[np.bool_],
    *,
    axis: int | tuple[int, ...] | None = None,
) -> NDArray[np.float64] | float:
    """Take the linear-Sv mean, then convert to dB with missing support."""
    if values.shape != mask.shape:
        raise ValueError("Value and validity-mask shapes differ.")
    if not np.isfinite(values[mask]).all():
        raise ValueError("A valid Sv value is non-finite.")
    linear = np.where(mask, np.power(10.0, np.clip(values, -180, 80) / 10.0), 0.0)
    support = mask.sum(axis=axis)
    mean = np.divide(
        linear.sum(axis=axis),
        support,
        out=np.full_like(linear.sum(axis=axis), np.nan, dtype=np.float64),
        where=support > 0,
    )
    result = 10.0 * np.log10(np.maximum(mean, 1e-18))
    return float(result) if np.ndim(result) == 0 else result


def _check_context(values: NDArray[np.float64], mask: NDArray[np.bool_], units: str) -> None:
    if units != "sv_db":
        raise ValueError("Physical baselines require calibrated sv_db input.")
    if values.ndim != 4 or values.shape[1:] != (96, 4, 64) or mask.shape != values.shape:
        raise ValueError("Expected [N,96,4,64] values and mask.")
    if not np.isfinite(values[mask]).all():
        raise ValueError("Valid acoustic values must be finite.")


def _profile_mean(
    values: NDArray[np.float64], mask: NDArray[np.bool_], positions: NDArray[np.int64]
) -> NDArray[np.float64]:
    return np.asarray(
        linear_average_db(values[positions], mask[positions], axis=0), dtype=np.float64
    )


class _FixedBaseline:
    def __init__(self) -> None:
        self._residual: NDArray[np.float64] | None = None

    def predict_point(
        self,
        values: NDArray[np.float64],
        mask: NDArray[np.bool_],
        *,
        units: str = "sv_db",
    ) -> NDArray[np.float64]:
        raise NotImplementedError

    def fit(
        self,
        values: NDArray[np.float64],
        mask: NDArray[np.bool_],
        target: NDArray[np.float64],
        *,
        partition: str,
        units: str = "sv_db",
    ) -> _FixedBaseline:
        if partition != "train":
            raise ValueError("Residual distribution may fit training data only.")
        point = self.predict_point(values, mask, units=units)
        if target.shape != point.shape:
            raise ValueError("Target shape must be [N,3].")
        residual = target - point
        self._residual = np.stack(
            [
                np.quantile(residual[np.isfinite(residual[:, h]), h], QUANTILES)
                if np.isfinite(residual[:, h]).any()
                else np.full(5, np.nan)
                for h in range(3)
            ]
        )
        return self

    def predict(
        self,
        values: NDArray[np.float64],
        mask: NDArray[np.bool_],
        *,
        units: str = "sv_db",
    ) -> BaselineForecast:
        if self._residual is None:
            raise RuntimeError("Fit the training residual distribution first.")
        point = self.predict_point(values, mask, units=units)
        quantiles = point[..., None] + self._residual[None, :, :]
        return BaselineForecast(quantiles=np.sort(quantiles, axis=-1), eligible=np.isfinite(point))


class PersistenceBaseline(_FixedBaseline):
    """B0: last four available 15-minute bins within the registered 3-hour limit."""

    def __init__(self, max_age_bins: int = 12) -> None:
        super().__init__()
        if not 1 <= max_age_bins <= 96:
            raise ValueError("Invalid persistence age limit.")
        self.max_age_bins = max_age_bins

    def predict_point(
        self,
        values: NDArray[np.float64],
        mask: NDArray[np.bool_],
        *,
        units: str = "sv_db",
    ) -> NDArray[np.float64]:
        _check_context(values, mask, units)
        point = np.full((len(values), 3), np.nan)
        for sample in range(len(values)):
            valid_bins = np.flatnonzero(mask[sample, -self.max_age_bins :, 0, _BAND].any(axis=1))
            if not valid_bins.size:
                continue
            positions = (96 - self.max_age_bins + valid_bins[-4:]).astype(np.int64)
            scalar = linear_average_db(
                values[sample, positions, 0, _BAND], mask[sample, positions, 0, _BAND]
            )
            point[sample] = scalar
        return point


class DailySeasonalBaseline(_FixedBaseline):
    """B1: target hour at the same UTC time on the previous day."""

    def predict_point(
        self,
        values: NDArray[np.float64],
        mask: NDArray[np.bool_],
        *,
        units: str = "sv_db",
    ) -> NDArray[np.float64]:
        _check_context(values, mask, units)
        point = np.full((len(values), 3), np.nan)
        for horizon, first in enumerate((0, 8, 20)):
            point[:, horizon] = np.asarray(
                linear_average_db(
                    values[:, first : first + 4, 0, _BAND],
                    mask[:, first : first + 4, 0, _BAND],
                    axis=(1, 2),
                ),
                dtype=np.float64,
            )
        return point


def engineered_history(
    values: NDArray[np.float64], mask: NDArray[np.bool_], *, units: str = "sv_db"
) -> NDArray[np.float64]:
    """Fixed lags, trends, support and variability from past channels only."""
    _check_context(values, mask, units)
    channel_db = np.asarray(linear_average_db(values, mask, axis=3), dtype=np.float64)
    features: list[NDArray[np.float64]] = [
        channel_db[:, position, :] for position in (0, 20, 72, 84, 92, 95)
    ]
    for start, end in ((0, 4), (8, 12), (20, 24), (72, 96), (92, 96)):
        part = channel_db[:, start:end, :]
        valid = np.isfinite(part)
        count = valid.sum(axis=1)
        mean = np.divide(
            np.where(valid, part, 0).sum(axis=1),
            count,
            out=np.full((len(values), 4), np.nan),
            where=count > 0,
        )
        features.extend((mean, count / (end - start)))
    return np.concatenate(features, axis=1)


class RidgeQuantileBaseline:
    """B2: train-only ridge median plus empirical residual quantiles."""

    def __init__(self, alpha: float = 10.0) -> None:
        self.models = [
            make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                StandardScaler(),
                Ridge(alpha=alpha),
            )
            for _ in HORIZONS
        ]
        self.residual: NDArray[np.float64] | None = None

    def fit(
        self,
        values: NDArray[np.float64],
        mask: NDArray[np.bool_],
        target: NDArray[np.float64],
        *,
        partition: str,
        units: str = "sv_db",
    ) -> RidgeQuantileBaseline:
        if partition != "train":
            raise ValueError("Ridge may fit training data only.")
        x = engineered_history(values, mask, units=units)
        if target.shape != (len(values), 3) or not np.isfinite(target).all():
            raise ValueError("Ridge requires finite [N,3] targets.")
        residuals = []
        for horizon, model in enumerate(self.models):
            model.fit(x, target[:, horizon])
            residuals.append(np.quantile(target[:, horizon] - model.predict(x), QUANTILES))
        self.residual = np.stack(residuals)
        return self

    def predict(
        self,
        values: NDArray[np.float64],
        mask: NDArray[np.bool_],
        *,
        units: str = "sv_db",
    ) -> BaselineForecast:
        if self.residual is None:
            raise RuntimeError("Fit ridge on train first.")
        x = engineered_history(values, mask, units=units)
        median = np.stack([model.predict(x) for model in self.models], axis=1)
        quantiles = np.sort(median[..., None] + self.residual[None, :, :], axis=-1)
        return BaselineForecast(quantiles=quantiles, eligible=np.ones((len(values), 3), dtype=bool))


class TreeQuantileBaseline:
    """B3: histogram boosting with a separate fitted head per horizon/quantile."""

    def __init__(
        self,
        *,
        max_iter: int = 150,
        max_leaf_nodes: int = 15,
        min_samples_leaf: int = 8,
    ) -> None:
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.models = [
            HistGradientBoostingRegressor(
                loss="quantile",
                quantile=quantile,
                max_iter=max_iter,
                max_leaf_nodes=max_leaf_nodes,
                min_samples_leaf=min_samples_leaf,
                learning_rate=0.05,
                random_state=7,
            )
            for _ in HORIZONS
            for quantile in QUANTILES
        ]
        self._fitted = False

    def fit(
        self,
        values: NDArray[np.float64],
        mask: NDArray[np.bool_],
        target: NDArray[np.float64],
        *,
        partition: str,
        units: str = "sv_db",
    ) -> TreeQuantileBaseline:
        if partition != "train":
            raise ValueError("Tree may fit training data only.")
        if target.shape != (len(values), 3) or not np.isfinite(target).all():
            raise ValueError("Tree requires finite [N,3] targets.")
        x = self.imputer.fit_transform(engineered_history(values, mask, units=units))
        for index, model in enumerate(self.models):
            model.fit(x, target[:, index // len(QUANTILES)])
        self._fitted = True
        return self

    def predict(
        self,
        values: NDArray[np.float64],
        mask: NDArray[np.bool_],
        *,
        units: str = "sv_db",
    ) -> BaselineForecast:
        if not self._fitted:
            raise RuntimeError("Fit tree on train first.")
        x = self.imputer.transform(engineered_history(values, mask, units=units))
        quantiles = np.stack([model.predict(x) for model in self.models], axis=1).reshape(
            len(values), 3, 5
        )
        return BaselineForecast(
            quantiles=np.sort(quantiles, axis=-1),
            eligible=np.ones((len(values), 3), dtype=bool),
        )
