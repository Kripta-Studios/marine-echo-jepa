"""Contract tests for the post-hoc AEON supervised tree challenger."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from marine_echo.training.aeon_sota_supervised import _features, _fit_quantiles, _recipe_gate


def _row(past: np.ndarray, observed: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(context_db=past, context_mask=observed)


def test_features_use_only_past_values_and_masks() -> None:
    past = np.arange(96, dtype=np.float64).reshape(24, 4)
    observed = np.ones((24, 4), dtype=bool)
    observed[0, 1] = False
    past[0, 1] = np.nan
    features = _features([_row(past, observed)])
    assert features.shape == (1, 210)
    assert np.isfinite(features).all()
    assert features[0, 18 + 1] == 0
    assert features[0, 18 + 96 + 1] == 0
    changed = past.copy()
    changed[-1, 0] += 5
    assert not np.array_equal(features, _features([_row(changed, observed)]))


def test_fifteen_train_only_quantile_heads() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(180, 210))
    y = np.stack([x[:, 0] + h for h in (1, 3, 6)], axis=1)
    mask = np.ones_like(y, dtype=bool)
    mask[::7, 1] = False
    recipe = {
        "objective": "quantile", "n_estimators": 5, "max_depth": 4,
        "num_leaves": 15, "min_child_samples": 40,
        "learning_rate": 0.03, "colsample_bytree": 0.8,
        "reg_lambda": 2.0, "random_state": 7, "n_jobs": 2,
    }
    predictions, models = _fit_quantiles(x, y, mask, x[:3], recipe)
    assert predictions.shape == (3, 3, 5)
    assert np.isfinite(predictions).all()
    assert np.all(np.diff(predictions, axis=-1) >= 0)
    assert len(models) == 3 and all(len(heads) == 5 for heads in models)


def test_recipe_gate_rejects_extra_validation_choice() -> None:
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    recipe = json.loads((root / "configs/aeon_sota_supervised.json").read_text(encoding="utf-8"))
    _recipe_gate(recipe)
    recipe["tree"]["n_estimators"] = [128, 256]
    with pytest.raises(ValueError, match="fixed post-hoc"):
        _recipe_gate(recipe)
