"""Train-only ridge and raw-plus-frozen-latent quantile heads."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .baselines import BaselineForecast, HORIZONS, QUANTILES, engineered_history
from .compact import TemporalJEPA


def frozen_context_features(
    model: TemporalJEPA,
    values: NDArray[np.float64],
    mask: NDArray[np.bool_],
    *,
    batch_size: int = 16,
) -> NDArray[np.float64]:
    """Embed past-only windows; neither future observations nor gradients enter."""
    if values.ndim != 4 or values.shape[1:] != (96, 4, 64) or mask.shape != values.shape:
        raise ValueError("Expected matching [N,96,4,64] past windows and masks.")
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    features = []
    try:
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                context = torch.as_tensor(values[start : start + batch_size], dtype=torch.float32, device=device)
                valid = torch.as_tensor(mask[start : start + batch_size], dtype=torch.bool, device=device)
                tokens, token_valid = model.encoder(context, valid)
                weighted = torch.where(token_valid[..., None], tokens, torch.zeros_like(tokens))
                current = weighted.sum(dim=1) / token_valid.sum(dim=1, keepdim=True).clamp_min(1)
                future = model.predictor(tokens, token_valid).mean(dim=2).flatten(start_dim=1)
                features.append(torch.cat((current, future), dim=1).cpu().numpy().astype(np.float64))
    finally:
        model.train(was_training)
    return np.concatenate(features, axis=0) if features else np.empty((0, 4 * model.encoder.config.width))


class FrozenLatentRidge:
    """Diagnostic ridge on frozen current/future latent summaries."""

    def __init__(self, encoder: TemporalJEPA, alpha: float = 10.0) -> None:
        self.encoder = encoder
        self.models = [
            make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=alpha))
            for _ in HORIZONS
        ]
        self.residual: NDArray[np.float64] | None = None

    def fit(
        self, values: NDArray[np.float64], mask: NDArray[np.bool_],
        targets: NDArray[np.float64], *, partition: str,
    ) -> FrozenLatentRidge:
        if partition != "train":
            raise ValueError("Downstream ridge may fit training targets only.")
        if targets.shape != (len(values), 3) or not np.isfinite(targets).all():
            raise ValueError("Expected finite [N,3] targets.")
        x = frozen_context_features(self.encoder, values, mask)
        residuals = []
        for horizon, model in enumerate(self.models):
            model.fit(x, targets[:, horizon])
            residuals.append(np.quantile(targets[:, horizon] - model.predict(x), QUANTILES))
        self.residual = np.stack(residuals)
        return self

    def predict(self, values: NDArray[np.float64], mask: NDArray[np.bool_]) -> BaselineForecast:
        if self.residual is None:
            raise RuntimeError("Fit on training windows first.")
        x = frozen_context_features(self.encoder, values, mask)
        medians = np.stack([model.predict(x) for model in self.models], axis=1)
        eligible = mask.any(axis=(1, 2, 3))[:, None].repeat(3, axis=1)
        return BaselineForecast(np.sort(medians[..., None] + self.residual[None, :, :], axis=-1), eligible)


class RawLatentTreeQuantiles:
    """B3-budget tree heads on identical raw features plus frozen latents."""

    def __init__(self, encoder: TemporalJEPA, *, max_iter: int = 150) -> None:
        self.encoder = encoder
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.models = [
            HistGradientBoostingRegressor(
                loss="quantile", quantile=quantile, max_iter=max_iter,
                max_leaf_nodes=15, min_samples_leaf=8, learning_rate=0.05, random_state=7,
            )
            for _ in HORIZONS for quantile in QUANTILES
        ]
        self._fitted = False

    def _features(self, values: NDArray[np.float64], mask: NDArray[np.bool_]) -> NDArray[np.float64]:
        raw = engineered_history(values, mask, units="sv_db")
        latent = frozen_context_features(self.encoder, values, mask)
        return np.concatenate((raw, latent), axis=1)

    def fit(
        self, values: NDArray[np.float64], mask: NDArray[np.bool_],
        targets: NDArray[np.float64], *, partition: str,
    ) -> RawLatentTreeQuantiles:
        if partition != "train":
            raise ValueError("Hybrid heads may fit training targets only.")
        if targets.shape != (len(values), 3) or not np.isfinite(targets).all():
            raise ValueError("Expected finite [N,3] targets.")
        x = self.imputer.fit_transform(self._features(values, mask))
        for index, model in enumerate(self.models):
            model.fit(x, targets[:, index // len(QUANTILES)])
        self._fitted = True
        return self

    def predict(self, values: NDArray[np.float64], mask: NDArray[np.bool_]) -> BaselineForecast:
        if not self._fitted:
            raise RuntimeError("Fit hybrid on training windows first.")
        x = self.imputer.transform(self._features(values, mask))
        predictions = np.stack([model.predict(x) for model in self.models], axis=1).reshape(len(values), 3, 5)
        eligible = mask.any(axis=(1, 2, 3))[:, None].repeat(3, axis=1)
        return BaselineForecast(np.sort(predictions, axis=-1), eligible)
