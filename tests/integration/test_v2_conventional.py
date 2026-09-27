"""Native conventional families use past-only inputs and masked TRAIN outcomes."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from marine_echo.models.v2_conventional import NativeConventional
from marine_echo.training.v2_conventional import execute_conventional
from marine_echo.training.v2_stream import HourlyWindow


def _rows(first_day: str) -> list[HourlyWindow]:
    rows = []
    for index in range(12):
        cutoff = np.datetime64(first_day) + index * np.timedelta64(2, "D")
        context = np.full((96, 4, 64), np.nan)
        context[:, 0, 5:50] = -90.0 + index
        past_index = np.full(96, -90.0 + index)
        past_index[0:4] = np.nan if index % 2 else past_index[0:4]
        future = np.full((3, 4, 4, 64), np.nan)
        future[:, :, 0, 5:50] = -88.0 + index
        rows.append(
            HourlyWindow(
                row_id=f"{first_day}-{index}",
                partition="train",
                cutoff=cutoff,
                context=context,
                context_mask=np.isfinite(context),
                context_acquisition_fraction=np.full(96, 0.7),
                context_detection_fraction=np.full(96, 0.4 + index * 0.01),
                context_age_minutes=np.arange(95, -1, -1) * 15.0,
                target_interval_start=np.array(
                    [cutoff + np.timedelta64(h, "h") for h in (0, 2, 5)]
                ),
                target_interval_end=np.array([cutoff + np.timedelta64(h, "h") for h in (1, 3, 6)]),
                target_db=np.full(3, -88.0 + index),
                target_mask=np.array([True, index % 3 != 0, True]),
                target_detection_fraction=np.full(3, 0.45 + index * 0.01),
                target_detection_mask=np.ones(3, dtype=bool),
                target_acquisition_fraction=np.full(3, 0.7),
                future_train_db=future,
                future_train_mask=np.isfinite(future),
                source_sha256=("a" * 64,),
                context_index_db=past_index,
            )
        )
    return rows


@pytest.mark.parametrize("family", ["persistence", "seasonal", "hist_gradient_boosting"])
def test_conventional_prediction_never_reads_future(family: str) -> None:
    fit, assess = _rows("2020-02-18"), _rows("2020-04-02")
    model = NativeConventional(family, tree_max_iter=3).fit(fit)
    first = model.predict(assess)
    changed = [
        replace(
            row,
            target_db=np.full(3, 999.0),
            target_detection_fraction=np.zeros(3),
            future_train_db=np.full_like(row.future_train_db, 999.0),
        )
        for row in assess
    ]
    second = model.predict(changed)
    np.testing.assert_array_equal(first.quantiles_db, second.quantiles_db)
    np.testing.assert_array_equal(first.detection_fraction, second.detection_fraction)
    assert np.isfinite(first.quantiles_db).all()
    assert ((first.detection_fraction >= 0) & (first.detection_fraction <= 1)).all()


def test_conventional_executor_writes_recomputable_rows(tmp_path: Path) -> None:
    result = execute_conventional(
        _rows("2020-02-18"),
        _rows("2020-04-02"),
        tmp_path / "seasonal",
        family="seasonal",
        protocol_sha256="b" * 64,
        fixture_only=True,
    )
    assert result["status"] == "COMPLETED_SYNTHETIC_FIXTURE"
    assert result["family"] == "seasonal"
    assert Path(result["predictions"]).is_file()
    assert Path(result["model"]).is_file()
    assert result["metrics"]["issued_rows"] == 12
    assert result["fallback_count"] > 0
    with pytest.raises(FileExistsError):
        execute_conventional(
            _rows("2020-02-18"),
            _rows("2020-04-02"),
            tmp_path / "seasonal",
            family="seasonal",
            protocol_sha256="b" * 64,
            fixture_only=True,
        )
