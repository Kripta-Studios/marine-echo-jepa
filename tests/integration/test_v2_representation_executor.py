"""Synthetic rows exercise finite JEPA, controls and frozen-latent hybrid executors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from marine_echo.models.compact import ModelConfig
from marine_echo.training.v2_representation import execute_representation
from marine_echo.training.v2_stream import HourlyWindow


def _rows(first_day: str, *, step_days: int) -> list[HourlyWindow]:
    rows = []
    for index in range(8):
        cutoff = np.datetime64(first_day) + index * np.timedelta64(step_days, "D")
        context = np.full((96, 4, 64), np.nan)
        context[:, 0, 5:50] = -90 + index
        future = np.full((3, 4, 4, 64), np.nan)
        future[:, :, 0, 5:50] = -88 + index
        rows.append(
            HourlyWindow(
                row_id=f"{first_day}-{index}",
                partition="train",
                cutoff=cutoff,
                context=context,
                context_mask=np.isfinite(context),
                context_acquisition_fraction=np.full(96, 0.7),
                context_detection_fraction=np.full(96, 0.4),
                context_age_minutes=np.arange(95, -1, -1) * 15.0,
                target_interval_start=np.array(
                    [cutoff + np.timedelta64(h, "h") for h in (0, 2, 5)]
                ),
                target_interval_end=np.array([cutoff + np.timedelta64(h, "h") for h in (1, 3, 6)]),
                target_db=np.full(3, -88 + index, dtype=float),
                target_mask=np.ones(3, dtype=bool),
                target_detection_fraction=np.full(3, 0.4 + index * 0.01),
                target_detection_mask=np.ones(3, dtype=bool),
                target_acquisition_fraction=np.full(3, 0.7),
                future_train_db=future,
                future_train_mask=np.isfinite(future),
                source_sha256=("a" * 64,),
            )
        )
    return rows


@pytest.mark.parametrize("family", ["ema_jepa", "shared_sigreg"])
def test_representation_pretrain_probe_and_hybrid_write_rows(family: str, tmp_path: Path) -> None:
    torch.set_num_threads(1)
    result = execute_representation(
        _rows("2020-02-18", step_days=2),
        _rows("2020-04-02", step_days=1),
        tmp_path / family,
        family=family,
        protocol_sha256="b" * 64,
        fixture_only=True,
        model_config=ModelConfig(width=16, layers=1, heads=4),
        pretrain_updates=2,
        probe_updates=2,
        batch_size=4,
        device="cpu",
        hybrid_slot=True,
    )
    assert result["status"] == "COMPLETED_SYNTHETIC_FIXTURE"
    assert result["pretrain_updates"] == 2
    assert result["probe_updates"] == 2
    assert Path(result["pretrain_checkpoint"]).is_file()
    assert Path(result["probe_checkpoint"]).is_file()
    for name in ("representation", "hybrid"):
        with np.load(result[name]["predictions"], allow_pickle=False) as saved:
            assert saved["row_ids"].shape == (8,)
            assert saved["quantiles_db"].shape == (8, 3, 5)
            assert saved["detection_fraction"].shape == (8, 3)


def test_hybrid_requires_explicit_slot(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    result = execute_representation(
        _rows("2020-02-18", step_days=2),
        _rows("2020-04-02", step_days=1),
        tmp_path / "probe-only",
        family="ema_jepa",
        protocol_sha256="b" * 64,
        fixture_only=True,
        model_config=ModelConfig(width=16, layers=1, heads=4),
        pretrain_updates=1,
        probe_updates=1,
        batch_size=4,
        device="cpu",
    )
    assert "hybrid" not in result


@pytest.mark.parametrize("control", ["random_encoder", "shuffled_future"])
def test_representation_controls_are_explicit(control: str, tmp_path: Path) -> None:
    torch.set_num_threads(1)
    result = execute_representation(
        _rows("2020-02-18", step_days=2),
        _rows("2020-04-02", step_days=1),
        tmp_path / control,
        family="ema_jepa",
        control=control,
        protocol_sha256="b" * 64,
        fixture_only=True,
        model_config=ModelConfig(width=16, layers=1, heads=4),
        pretrain_updates=2,
        probe_updates=2,
        batch_size=4,
        device="cpu",
    )
    assert result["control"] == control
    assert result["pretrain_updates"] == (0 if control == "random_encoder" else 2)
    assert Path(result["probe_checkpoint"]).is_file()
