"""Matched TRAIN-development ridge and direct heads for the proposed v2 target."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from marine_echo.models.compact import DirectForecaster, ModelConfig
from marine_echo.training.v2_stream import HourlyWindow

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


@dataclass(frozen=True)
class JointPrediction:
    quantiles_db: NDArray[np.float64]
    detection_fraction: NDArray[np.float64]


def _finite_mean(values: NDArray[np.float64], *, axis: int) -> NDArray[np.float64]:
    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    total = np.where(finite, values, 0).sum(axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)


def past_features(row: HourlyWindow) -> NDArray[np.float64]:
    """Fixed past summaries; no future field is accessed."""
    if row.context.shape != (96, 4, 64) or row.context_mask.shape != row.context.shape:
        raise ValueError("Expected the frozen 96-bin acoustic context.")
    valid = row.context_mask[:, 0, 5:50]
    profile = row.context[:, 0, 5:50]
    linear = np.where(valid, np.power(10.0, np.clip(profile, -180, 80) / 10.0), 0)
    count = valid.sum(axis=1)
    mean_linear = np.divide(linear.sum(axis=1), count, out=np.full(96, np.nan), where=count > 0)
    index = 10 * np.log10(np.maximum(mean_linear, 1e-18))
    sections = (slice(0, 96), slice(48, 96), slice(80, 96), slice(92, 96))
    parts: list[float] = []
    for series in (
        index,
        row.context_acquisition_fraction,
        row.context_detection_fraction,
        row.context_age_minutes,
    ):
        if series.shape != (96,):
            raise ValueError("Past auxiliary series must have 96 bins.")
        parts.extend(float(_finite_mean(series[section], axis=0)) for section in sections)
        parts.append(float(series[-1]))
    parts.extend((float(np.isfinite(index).mean()), float(row.context_mask[:, 0, 5:50].mean())))
    return np.asarray(parts, dtype=np.float64)


class DevelopmentRidge:
    """Fixed alpha=1 joint ridge with TRAIN residual index quantiles."""

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha != 1.0:
            raise ValueError("The first v2 development ridge alpha is frozen at 1.")
        self.index_models = [self._model() for _ in range(3)]
        self.fraction_models = [self._model() for _ in range(3)]
        self.residual: NDArray[np.float64] | None = None

    @staticmethod
    def _model() -> Pipeline:
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=1.0),
        )

    def fit(self, rows: list[HourlyWindow]) -> DevelopmentRidge:
        if not rows or any(row.partition != "train" for row in rows):
            raise ValueError("TRAIN rows alone may fit the development ridge.")
        if len({row.row_id for row in rows}) != len(rows):
            raise ValueError("Duplicate development row ID.")
        x = np.stack([past_features(row) for row in rows])
        target = np.stack([row.target_db for row in rows])
        target_mask = np.stack([row.target_mask for row in rows])
        fraction = np.stack([row.target_detection_fraction for row in rows])
        fraction_mask = np.stack([row.target_detection_mask for row in rows])
        residual = []
        for horizon in range(3):
            scalar_valid = target_mask[:, horizon]
            fraction_valid = fraction_mask[:, horizon]
            if not scalar_valid.any() or not fraction_valid.any():
                raise ValueError("Each horizon needs TRAIN index and detection targets.")
            if (
                not np.isfinite(target[scalar_valid, horizon]).all()
                or not np.isfinite(fraction[fraction_valid, horizon]).all()
            ):
                raise ValueError("Valid TRAIN target contains a non-finite value.")
            self.index_models[horizon].fit(x[scalar_valid], target[scalar_valid, horizon])
            self.fraction_models[horizon].fit(x[fraction_valid], fraction[fraction_valid, horizon])
            fitted = self.index_models[horizon].predict(x[scalar_valid])
            residual.append(np.quantile(target[scalar_valid, horizon] - fitted, QUANTILES))
        self.residual = np.stack(residual)
        return self

    def predict(self, rows: list[HourlyWindow]) -> JointPrediction:
        if self.residual is None:
            raise RuntimeError("Fit the development ridge on TRAIN first.")
        x = np.stack([past_features(row) for row in rows])
        median = np.stack([model.predict(x) for model in self.index_models], axis=1)
        fraction = np.stack([model.predict(x) for model in self.fraction_models], axis=1)
        return JointPrediction(
            quantiles_db=np.sort(median[..., None] + self.residual[None, :, :], axis=-1),
            detection_fraction=np.clip(fraction, 0.0, 1.0),
        )


@dataclass(frozen=True)
class JointTorchPrediction:
    quantiles: torch.Tensor
    detection_fraction: torch.Tensor
    eligible: torch.Tensor


class JointDirectForecaster(nn.Module):
    """Existing matched compact encoder plus past quality features and fraction head."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.base = DirectForecaster(config)
        self.aux_project = nn.Linear(96 * 4, config.width)
        self.fraction_head = nn.Linear(config.width, 1)

    def forward(
        self, context: torch.Tensor, context_mask: torch.Tensor, past_aux: torch.Tensor
    ) -> JointTorchPrediction:
        if past_aux.shape != (len(context), 96, 4):
            raise ValueError("Expected four past-only auxiliary features per context bin.")
        encoded, valid = self.base.encoder(context, context_mask)
        predicted = self.base.predictor(encoded, valid)
        auxiliary = torch.where(torch.isfinite(past_aux), past_aux, 0).flatten(start_dim=1)
        predicted = predicted + self.aux_project(auxiliary)[:, None, None, :]
        forecast = self.base.head(predicted, valid.any(dim=1))
        fraction = torch.sigmoid(self.fraction_head(predicted.mean(dim=2)).squeeze(-1))
        return JointTorchPrediction(forecast.quantiles, fraction, forecast.eligible)


def joint_supervised_loss(
    forecast: JointTorchPrediction,
    target_index: torch.Tensor,
    target_index_mask: torch.Tensor,
    target_fraction: torch.Tensor,
    target_fraction_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean masked normalized pinball plus fraction MSE, each over its valid horizons."""
    shape = forecast.quantiles.shape[:2]
    if (
        target_index.shape != shape
        or target_index_mask.shape != shape
        or target_fraction.shape != shape
        or target_fraction_mask.shape != shape
        or forecast.detection_fraction.shape != shape
    ):
        raise ValueError("Joint target and prediction dimensions differ.")
    losses: list[torch.Tensor] = []
    index_valid = target_index_mask & forecast.eligible[:, None]
    fraction_valid = target_fraction_mask & forecast.eligible[:, None]
    if index_valid.any():
        if not torch.isfinite(target_index[index_valid]).all():
            raise ValueError("Observed index target is non-finite.")
        q = forecast.quantiles.new_tensor(QUANTILES)
        horizon_losses = []
        for horizon in range(shape[1]):
            valid = index_valid[:, horizon]
            if valid.any():
                error = target_index[valid, horizon, None] - forecast.quantiles[valid, horizon]
                horizon_losses.append(torch.maximum(q * error, (q - 1.0) * error).mean())
        losses.append(torch.stack(horizon_losses).mean())
    if fraction_valid.any():
        observed = target_fraction[fraction_valid]
        if not torch.isfinite(observed).all() or ((observed < 0) | (observed > 1)).any():
            raise ValueError("Observed detection fraction is invalid.")
        horizon_losses = []
        for horizon in range(shape[1]):
            valid = fraction_valid[:, horizon]
            if valid.any():
                horizon_losses.append(
                    (forecast.detection_fraction[valid, horizon] - target_fraction[valid, horizon])
                    .square()
                    .mean()
                )
        losses.append(torch.stack(horizon_losses).mean())
    if not losses:
        raise ValueError("A joint batch needs at least one observed target component.")
    return sum(losses)
