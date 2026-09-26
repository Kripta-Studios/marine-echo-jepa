"""Deterministic CPU training and safe resume diagnostics on synthetic fixtures."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from marine_echo.models.compact import DirectForecaster, ModelConfig
from marine_echo.training.loop import (
    TrainBatch,
    load_checkpoint,
    run_steps,
    save_checkpoint,
    supervised_loss,
)


def _batch() -> TrainBatch:
    torch.manual_seed(11)
    context = torch.randn(2, 96, 4, 64)
    return TrainBatch(
        context=context,
        context_mask=torch.ones_like(context, dtype=torch.bool),
        target_index=torch.zeros(2, 3),
        target_profile=torch.zeros(2, 3, 4, 64),
        target_profile_mask=torch.ones(2, 3, 4, 64, dtype=torch.bool),
    )


def test_resume_matches_uninterrupted_cpu_updates(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = ModelConfig(width=16, layers=1, heads=4)
    initial = DirectForecaster(config)
    continuous, partial = deepcopy(initial), deepcopy(initial)
    first_optimizer = torch.optim.AdamW(continuous.parameters(), lr=1e-3)
    second_optimizer = torch.optim.AdamW(partial.parameters(), lr=1e-3)
    batch = _batch()
    run_steps(
        continuous,
        first_optimizer,
        [batch],
        family="direct",
        start_step=0,
        updates=4,
        partition="train",
    )
    run_steps(
        partial,
        second_optimizer,
        [batch],
        family="direct",
        start_step=0,
        updates=2,
        partition="train",
    )
    checkpoint = tmp_path / "step_2.pt"
    save_checkpoint(checkpoint, partial, second_optimizer, step=2, protocol_sha256="a" * 64)
    resumed = DirectForecaster(config)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    assert load_checkpoint(checkpoint, resumed, resumed_optimizer, protocol_sha256="a" * 64) == 2
    run_steps(
        resumed,
        resumed_optimizer,
        [batch],
        family="direct",
        start_step=2,
        updates=2,
        partition="train",
    )
    for key, reference in continuous.state_dict().items():
        assert torch.equal(reference, resumed.state_dict()[key]), key
    with pytest.raises(ValueError):
        load_checkpoint(checkpoint, resumed, resumed_optimizer, protocol_sha256="b" * 64)


def test_fixed_synthetic_microoverfit_and_train_guard() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(23)
    model = DirectForecaster(ModelConfig(width=16, layers=1, heads=4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = _batch()
    first = float(supervised_loss(model(batch.context, batch.context_mask), batch).item())
    with pytest.raises(ValueError):
        run_steps(
            model,
            optimizer,
            [batch],
            family="direct",
            start_step=0,
            updates=1,
            partition="test",
        )
    run_steps(
        model,
        optimizer,
        [batch],
        family="direct",
        start_step=0,
        updates=20,
        partition="train",
    )
    last = float(supervised_loss(model(batch.context, batch.context_mask), batch).item())
    assert last < first


def test_real_lineage_ids_require_train_allowlist() -> None:
    torch.set_num_threads(1)
    model = DirectForecaster(ModelConfig(width=16, layers=1, heads=4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = replace(_batch(), source_file_ids=("heldout.01A",))
    with pytest.raises(ValueError):
        run_steps(
            model,
            optimizer,
            [batch],
            family="direct",
            start_step=0,
            updates=1,
            partition="train",
            allowed_train_file_ids={"train.01A"},
        )
