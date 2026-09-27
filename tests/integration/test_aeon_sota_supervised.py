"""Contract tests for the post-hoc AEON supervised tree challenger."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from marine_echo.evaluation.aeon import daily_pinball
from marine_echo.training.aeon_development import _sha256
from marine_echo.training.aeon_sota_supervised import (
    _features,
    _fit_quantiles,
    _recipe_gate,
    _verified_b3_score,
)


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


def test_b3_comparison_recomputes_score_and_binds_evaluator() -> None:
    from pathlib import Path

    truth = np.zeros((24, 3))
    forecast = np.zeros((24, 3, 5))
    observed = np.ones((24, 3), dtype=bool)
    hours = np.datetime64("2024-10-09T00:00") + np.arange(24).astype("timedelta64[h]")
    times = np.repeat(hours[:, None], 3, axis=1)
    expected = daily_pinball(truth, forecast, observed, times)
    rescore = {
        "evaluation_code_sha256": _sha256(Path(daily_pinball.__code__.co_filename)),
        "slots": {"hist_gradient_boosting": {"protocol_validation_metrics": expected}},
    }
    assert _verified_b3_score(truth, forecast, observed, times, rescore) == expected
    rescore["evaluation_code_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="evaluator"):
        _verified_b3_score(truth, forecast, observed, times, rescore)
    rescore["evaluation_code_sha256"] = _sha256(Path(daily_pinball.__code__.co_filename))
    with pytest.raises(ValueError, match="B3 score"):
        _verified_b3_score(truth, forecast + 1, observed, times, rescore)
