"""Past-only conventional forecasts for native detection-conditioned v2 rows."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

from marine_echo.models.v2_development import (
    QUANTILES,
    DevelopmentRidge,
    JointPrediction,
    past_features,
)
from marine_echo.training.v2_stream import HourlyWindow

ConventionalFamily = Literal["persistence", "seasonal", "ridge", "hist_gradient_boosting"]


def _linear_mean_db(values: np.ndarray) -> float:
    if not len(values):
        return float("nan")
    return float(10 * np.log10(np.power(10.0, values / 10.0).mean()))


class NativeConventional:
    """Fixed persistence/seasonal/boosting models with TRAIN-only calibration."""

    def __init__(self, family: ConventionalFamily, *, tree_max_iter: int = 150) -> None:
        if family not in ("persistence", "seasonal", "ridge", "hist_gradient_boosting"):
            raise ValueError("Unknown native conventional family.")
        if not 1 <= tree_max_iter <= 150:
            raise ValueError("Tree iteration budget exceeds the frozen configuration.")
        self.family = family
        self.tree_max_iter = tree_max_iter
        self.fallback_count = 0
        self.fitted = False

    def _fixed_points(self, rows: list[HourlyWindow]) -> tuple[np.ndarray, np.ndarray, int]:
        point = np.full((len(rows), 3), np.nan)
        fraction = np.full((len(rows), 3), np.nan)
        fallback = 0
        for index, row in enumerate(rows):
            if row.context_index_db is None or row.context_index_db.shape != (96,):
                raise ValueError("Native conventional models require exact past conditional index.")
            if self.family == "persistence":
                positions = np.flatnonzero(np.isfinite(row.context_index_db[-12:]))[-4:] + 84
                for horizon in range(3):
                    selected = row.context_index_db[positions]
                    point[index, horizon] = _linear_mean_db(selected)
                    available = row.context_detection_fraction[positions]
                    fraction[index, horizon] = (
                        float(np.nanmean(available)) if np.isfinite(available).any() else np.nan
                    )
            else:
                for horizon, start in enumerate((0, 8, 20)):
                    selected = row.context_index_db[start : start + 4]
                    point[index, horizon] = _linear_mean_db(selected[np.isfinite(selected)])
                    available = row.context_detection_fraction[start : start + 4]
                    fraction[index, horizon] = (
                        float(np.nanmean(available)) if np.isfinite(available).any() else np.nan
                    )
        missing_index = ~np.isfinite(point)
        missing_fraction = ~np.isfinite(fraction)
        fallback = int((missing_index | missing_fraction).sum())
        point = np.where(missing_index, self.index_fallback[None], point)
        fraction = np.where(missing_fraction, self.fraction_fallback[None], fraction)
        return point, fraction, fallback

    def fit(self, rows: list[HourlyWindow]) -> NativeConventional:
        if not rows or any(row.partition != "train" for row in rows):
            raise ValueError("Only TRAIN rows may fit native conventional models.")
        target = np.stack([row.target_db for row in rows])
        mask = np.stack([row.target_mask for row in rows])
        fraction = np.stack([row.target_detection_fraction for row in rows])
        fraction_mask = np.stack([row.target_detection_mask for row in rows])
        if any(not mask[:, h].any() or not fraction_mask[:, h].any() for h in range(3)):
            raise ValueError("Every horizon needs TRAIN index and detection targets.")
        if not np.isfinite(target[mask]).all() or not np.isfinite(fraction[fraction_mask]).all():
            raise ValueError("Observed TRAIN targets must be finite.")
        self.index_fallback = np.array([np.median(target[mask[:, h], h]) for h in range(3)])
        self.fraction_fallback = np.array(
            [np.median(fraction[fraction_mask[:, h], h]) for h in range(3)]
        )
        if self.family == "ridge":
            self.ridge = DevelopmentRidge(alpha=1.0).fit(rows)
        elif self.family != "hist_gradient_boosting":
            point, _, _ = self._fixed_points(rows)
            self.residual = np.stack(
                [
                    np.quantile(target[mask[:, h], h] - point[mask[:, h], h], QUANTILES)
                    for h in range(3)
                ]
            )
        else:
            features = np.stack([past_features(row) for row in rows])
            self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            x = self.imputer.fit_transform(features)
            self.index_models = []
            self.fraction_models = []
            for horizon in range(3):
                heads = []
                for quantile in QUANTILES:
                    head = HistGradientBoostingRegressor(
                        loss="quantile",
                        quantile=quantile,
                        max_iter=self.tree_max_iter,
                        max_leaf_nodes=15,
                        min_samples_leaf=8,
                        learning_rate=0.05,
                        random_state=7,
                    )
                    head.fit(x[mask[:, horizon]], target[mask[:, horizon], horizon])
                    heads.append(head)
                self.index_models.append(heads)
                head = HistGradientBoostingRegressor(
                    loss="squared_error",
                    max_iter=self.tree_max_iter,
                    max_leaf_nodes=15,
                    min_samples_leaf=8,
                    learning_rate=0.05,
                    random_state=7,
                )
                head.fit(x[fraction_mask[:, horizon]], fraction[fraction_mask[:, horizon], horizon])
                self.fraction_models.append(head)
        self.fitted = True
        return self

    def predict(self, rows: list[HourlyWindow]) -> JointPrediction:
        if not self.fitted:
            raise RuntimeError("Fit native conventional model on TRAIN first.")
        if not rows:
            raise ValueError("Prediction rows must be nonempty.")
        if self.family == "ridge":
            self.fallback_count = 0
            return self.ridge.predict(rows)
        if self.family != "hist_gradient_boosting":
            point, fraction, self.fallback_count = self._fixed_points(rows)
            quantiles = point[..., None] + self.residual[None]
        else:
            x = self.imputer.transform(np.stack([past_features(row) for row in rows]))
            quantiles = np.stack(
                [
                    np.stack([head.predict(x) for head in heads], axis=1)
                    for heads in self.index_models
                ],
                axis=1,
            )
            fraction = np.stack([head.predict(x) for head in self.fraction_models], axis=1)
            self.fallback_count = 0
        return JointPrediction(
            quantiles_db=np.sort(quantiles, axis=-1),
            detection_fraction=np.clip(fraction, 0, 1),
        )
