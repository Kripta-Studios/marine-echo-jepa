"""Hand-calculated and train-only expectations for conventional baselines."""

from __future__ import annotations

import numpy as np
import pytest

from marine_echo.models.baselines import (
    DailySeasonalBaseline,
    PersistenceBaseline,
    RidgeQuantileBaseline,
    TreeQuantileBaseline,
    linear_average_db,
)


def _context(value: float = 0.0, samples: int = 1) -> tuple[np.ndarray, np.ndarray]:
    values = np.full((samples, 96, 4, 64), value, dtype=np.float64)
    return values, np.ones_like(values, dtype=bool)


def test_linear_sv_mean_of_zero_and_ten_db() -> None:
    result = linear_average_db(np.array([0.0, 10.0]), np.array([True, True]))
    assert result == pytest.approx(10 * np.log10(5.5))


def test_persistence_and_previous_day_use_distinct_past_hours() -> None:
    values, mask = _context()
    values[:, 92:96, 0, 5:50] = np.array([0.0, 10.0, 0.0, 10.0])[None, :, None]
    values[:, 0:4, 0, 5:50] = 20.0
    values[:, 8:12, 0, 5:50] = 30.0
    values[:, 20:24, 0, 5:50] = 40.0
    persistence = PersistenceBaseline().predict_point(values, mask)
    previous_day = DailySeasonalBaseline().predict_point(values, mask)
    assert persistence.shape == (1, 3)
    assert persistence[0, 0] == pytest.approx(10 * np.log10(5.5))
    assert previous_day[0].tolist() == pytest.approx([20.0, 30.0, 40.0])


def test_learned_baselines_refuse_nontrain_fit_and_output_quantiles() -> None:
    values, mask = _context(samples=24)
    values += np.arange(24)[:, None, None, None] * 0.1
    target = np.repeat(np.arange(24)[:, None] * 0.1, 3, axis=1)
    for model in (RidgeQuantileBaseline(), TreeQuantileBaseline(max_iter=4)):
        with pytest.raises(ValueError):
            model.fit(values, mask, target, partition="test")
        model.fit(values, mask, target, partition="train")
        forecast = model.predict(values[:2], mask[:2])
        assert forecast.quantiles.shape == (2, 3, 5)
        assert np.isfinite(forecast.quantiles).all()
        assert np.all(np.diff(forecast.quantiles, axis=-1) >= 0)


def test_raw_counts_cannot_enter_physical_baseline() -> None:
    values, mask = _context()
    with pytest.raises(ValueError):
        PersistenceBaseline().predict_point(values, mask, units="raw_counts")
