"""Frozen JEPA latent plus identical past raw features with masked joint ridge heads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from marine_echo.models.v2_development import QUANTILES, JointPrediction


@dataclass
class NativeHybridRidge:
    alpha: float = 1.0

    def fit(
        self,
        x: NDArray[np.float64],
        index: NDArray[np.float64],
        index_mask: NDArray[np.bool_],
        fraction: NDArray[np.float64],
        fraction_mask: NDArray[np.bool_],
        *,
        partition: str,
    ) -> NativeHybridRidge:
        if partition != "train" or self.alpha != 1.0:
            raise ValueError("Hybrid fit requires TRAIN and the fixed alpha=1 budget.")
        if (
            x.ndim != 2
            or len(x) == 0
            or index.shape != (len(x), 3)
            or index_mask.shape != index.shape
            or fraction.shape != index.shape
            or fraction_mask.shape != index.shape
        ):
            raise ValueError("Hybrid TRAIN feature and masked target shapes differ.")
        finite = np.isfinite(x)
        self.median = np.array(
            [
                np.median(x[finite[:, col], col]) if finite[:, col].any() else 0.0
                for col in range(x.shape[1])
            ]
        )
        imputed = np.where(finite, x, self.median)
        self.mean = imputed.mean(axis=0)
        self.scale = np.maximum(imputed.std(axis=0), 1e-6)
        normalized = (imputed - self.mean) / self.scale
        self.index_coefficients = np.empty((3, normalized.shape[1] + 1))
        self.fraction_coefficients = np.empty_like(self.index_coefficients)
        self.residual = np.empty((3, 5))
        for horizon in range(3):
            for mask, target, destination in (
                (index_mask, index, self.index_coefficients),
                (fraction_mask, fraction, self.fraction_coefficients),
            ):
                valid = mask[:, horizon]
                if not valid.any() or not np.isfinite(target[valid, horizon]).all():
                    raise ValueError("Hybrid requires observed TRAIN target for every horizon.")
                design = np.column_stack((np.ones(valid.sum()), normalized[valid]))
                penalty = np.eye(design.shape[1]) * self.alpha
                penalty[0, 0] = 0
                destination[horizon] = np.linalg.solve(
                    design.T @ design + penalty, design.T @ target[valid, horizon]
                )
            valid = index_mask[:, horizon]
            fitted = (
                self.index_coefficients[horizon, 0]
                + normalized[valid] @ self.index_coefficients[horizon, 1:]
            )
            self.residual[horizon] = np.quantile(index[valid, horizon] - fitted, QUANTILES)
        return self

    def predict(self, x: NDArray[np.float64]) -> JointPrediction:
        if not hasattr(self, "index_coefficients"):
            raise RuntimeError("Fit the hybrid on TRAIN first.")
        normalized = (np.where(np.isfinite(x), x, self.median) - self.mean) / self.scale
        medians = normalized @ self.index_coefficients[:, 1:].T + self.index_coefficients[:, 0]
        fractions = (
            normalized @ self.fraction_coefficients[:, 1:].T + self.fraction_coefficients[:, 0]
        )
        return JointPrediction(
            quantiles_db=np.sort(medians[..., None] + self.residual[None], axis=-1),
            detection_fraction=np.clip(fractions, 0, 1),
        )

    def save(self, path: Path) -> None:
        if not hasattr(self, "index_coefficients"):
            raise RuntimeError("Fit the hybrid on TRAIN first.")
        with path.open("xb") as stream:
            np.savez_compressed(
                stream,
                median=self.median,
                mean=self.mean,
                scale=self.scale,
                index_coefficients=self.index_coefficients,
                fraction_coefficients=self.fraction_coefficients,
                residual=self.residual,
            )
