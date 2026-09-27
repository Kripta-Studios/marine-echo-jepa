"""Artificial rows exercise immutable ridge/direct development artifacts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from marine_echo.models.compact import ModelConfig
from marine_echo.training.v2_executor import execute_development
from marine_echo.training.v2_stream import HourlyWindow


def _rows(date: str) -> list[HourlyWindow]:
    rows = []
    for index in range(8):
        cutoff = np.datetime64(date) + index * np.timedelta64(1, "h")
        context = np.full((96, 4, 64), np.nan)
        context[:, 0, 5:50] = -90 + index
        rows.append(
            HourlyWindow(
                row_id=f"{date}-{index}",
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
                target_db=np.array([-89 + index, -88 + index, -87 + index], dtype=float),
                target_mask=np.array([True, index % 2 == 0, True]),
                target_detection_fraction=np.full(3, 0.4 + index * 0.01),
                target_detection_mask=np.ones(3, dtype=bool),
                target_acquisition_fraction=np.full(3, 0.7),
                future_train_db=np.full((3, 4, 4, 64), np.nan),
                future_train_mask=np.zeros((3, 4, 4, 64), dtype=bool),
                source_sha256=("a" * 64,),
            )
        )
    return rows


def test_fixture_development_writes_real_updates_and_aligned_rows(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    fit = _rows("2020-02-18")
    assess = _rows("2020-04-02")
    result = execute_development(
        fit,
        assess,
        tmp_path / "runs",
        protocol_sha256="b" * 64,
        fixture_only=True,
        model_config=ModelConfig(width=16, layers=1, heads=4),
        updates=2,
        batch_size=4,
        device="cpu",
    )
    assert result["status"] == "COMPLETED_SYNTHETIC_FIXTURE"
    assert result["direct"]["updates"] == 2
    assert Path(result["direct"]["checkpoint"]).is_file()
    for family in ("ridge", "direct"):
        with np.load(result[family]["predictions"], allow_pickle=False) as saved:
            assert saved["row_ids"].tolist() == [row.row_id for row in assess]
            assert saved["quantiles_db"].shape == (8, 3, 5)
            assert saved["target_mask"].shape == (8, 3)
            assert saved["detection_fraction"].shape == (8, 3)
    with pytest.raises(FileExistsError):
        execute_development(
            fit,
            assess,
            tmp_path / "runs",
            protocol_sha256="b" * 64,
            fixture_only=True,
            model_config=ModelConfig(width=16, layers=1, heads=4),
            updates=2,
            batch_size=4,
            device="cpu",
        )


def test_development_rejects_overlapping_or_protected_rows(tmp_path: Path) -> None:
    fit = _rows("2020-02-18")
    with pytest.raises(ValueError, match="overlap"):
        execute_development(
            fit,
            _rows("2020-02-19"),
            tmp_path / "overlap",
            protocol_sha256="b" * 64,
            fixture_only=True,
            updates=2,
        )
    with pytest.raises(ValueError, match="TRAIN"):
        execute_development(
            fit,
            [replace(row, partition="test") for row in _rows("2020-04-02")],
            tmp_path / "protected",
            protocol_sha256="b" * 64,
            fixture_only=True,
            updates=2,
        )
