"""Joint v2 predictors use past values and separate masked targets."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from marine_echo.models.compact import ModelConfig
from marine_echo.models.v2_development import (
    DevelopmentRidge,
    JointDirectForecaster,
    joint_supervised_loss,
)
from marine_echo.training.v2_stream import HourlyWindow


def _rows(n: int = 8) -> list[HourlyWindow]:
    result = []
    for index in range(n):
        context = np.full((96, 4, 64), np.nan)
        context[:, 0, 5:50] = -80 + index
        mask = np.isfinite(context)
        result.append(
            HourlyWindow(
                row_id=f"row-{index}",
                partition="train",
                cutoff=np.datetime64("2020-02-18T00:00") + index * np.timedelta64(1, "h"),
                context=context,
                context_mask=mask,
                context_acquisition_fraction=np.full(96, 0.7),
                context_detection_fraction=np.full(96, 0.5),
                context_age_minutes=np.arange(95, -1, -1) * 15.0,
                target_interval_start=np.array([np.datetime64("2020-02-18T01:00")] * 3),
                target_interval_end=np.array([np.datetime64("2020-02-18T02:00")] * 3),
                target_db=np.full(3, -79 + index, dtype=float),
                target_mask=np.ones(3, dtype=bool),
                target_detection_fraction=np.full(3, 0.5 + index * 0.01),
                target_detection_mask=np.ones(3, dtype=bool),
                target_acquisition_fraction=np.full(3, 0.7),
                future_train_db=np.zeros((3, 4, 4, 64)),
                future_train_mask=np.zeros((3, 4, 4, 64), dtype=bool),
                source_sha256=("a" * 64,),
            )
        )
    return result


def test_ridge_fit_uses_train_and_predict_ignores_future_truth() -> None:
    rows = _rows()
    model = DevelopmentRidge(alpha=1.0).fit(rows)
    first = model.predict(rows)
    altered = [
        replace(row, target_db=np.full(3, 1000.0), target_detection_fraction=np.zeros(3))
        for row in rows
    ]
    second = model.predict(altered)
    assert np.array_equal(first.quantiles_db, second.quantiles_db)
    assert np.array_equal(first.detection_fraction, second.detection_fraction)
    assert first.quantiles_db.shape == (len(rows), 3, 5)
    assert np.all(np.diff(first.quantiles_db, axis=-1) >= 0)
    assert np.all((first.detection_fraction >= 0) & (first.detection_fraction <= 1))
    with pytest.raises(ValueError, match="TRAIN"):
        DevelopmentRidge().fit([replace(row, partition="validation") for row in rows])


def test_joint_direct_masks_missing_index_but_learns_fraction() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = JointDirectForecaster(ModelConfig(width=16, layers=1, heads=4))
    context = torch.zeros((2, 96, 4, 64))
    mask = torch.zeros_like(context, dtype=torch.bool)
    mask[:, :, 0, 5:50] = True
    aux = torch.zeros((2, 96, 4))
    prediction = model(context, mask, aux)
    target = torch.tensor([[-1.0, float("nan"), 1.0], [0.0, float("nan"), 2.0]])
    target_mask = torch.tensor([[True, False, True], [True, False, True]])
    fraction = torch.full((2, 3), 0.5)
    fraction_mask = torch.ones((2, 3), dtype=torch.bool)
    loss = joint_supervised_loss(prediction, target, target_mask, fraction, fraction_mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.fraction_head.weight.grad is not None
    assert model.fraction_head.weight.grad.abs().sum() > 0
    target[:, 1] = 1000
    changed = joint_supervised_loss(prediction, target, target_mask, fraction, fraction_mask)
    assert torch.equal(loss.detach(), changed.detach())
