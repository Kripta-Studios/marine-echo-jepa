"""Artificial rows exercise immutable ridge/direct development artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from marine_echo.models.compact import ModelConfig
from marine_echo.models.v2_development import JointPrediction
from marine_echo.training.v2_executor import (
    V2_PROTOCOL_SHA256,
    _code_digest,
    _review_gate,
    _score,
    _validate_rows,
    _verify_predictions,
    execute_development,
)
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
    assert result["direct"]["resume_equivalent"] is True
    assert result["direct"]["resume_verified_updates"] == 1
    assert result["direct"]["peak_process_rss_bytes"] > 0
    assert Path(result["direct"]["checkpoint"]).is_file()
    for family in ("ridge", "direct"):
        with np.load(result[family]["predictions"], allow_pickle=False) as saved:
            assert saved["row_ids"].tolist() == [row.row_id for row in assess]
            assert saved["quantiles_db"].shape == (8, 3, 5)
            assert saved["target_mask"].shape == (8, 3)
            assert saved["detection_fraction"].shape == (8, 3)
        assert _verify_predictions(Path(result[family]["predictions"]), assess)["issued_rows"] == 8
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


def test_real_development_requires_exact_batch_and_model_config(tmp_path: Path) -> None:
    fit, assess = _rows("2020-02-18"), _rows("2020-04-02")
    with pytest.raises(ValueError, match="batch size 16"):
        execute_development(
            fit,
            assess,
            tmp_path / "batch",
            protocol_sha256=V2_PROTOCOL_SHA256,
            batch_size=4,
        )
    with pytest.raises(ValueError, match="model configuration"):
        execute_development(
            fit,
            assess,
            tmp_path / "model",
            protocol_sha256=V2_PROTOCOL_SHA256,
            model_config=ModelConfig(width=16, layers=1, heads=4),
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


def test_input_digest_includes_age_and_ssl_future() -> None:
    fit, assess = _rows("2020-02-18"), _rows("2020-04-02")
    first, _ = _validate_rows(fit, assess, fixture_only=True)
    changed_age = list(fit)
    changed_age[0] = replace(fit[0], context_age_minutes=fit[0].context_age_minutes + 1)
    second, _ = _validate_rows(changed_age, assess, fixture_only=True)
    assert first != second
    changed_future = list(fit)
    changed_future[0] = replace(fit[0], future_train_mask=np.ones_like(fit[0].future_train_mask))
    third, _ = _validate_rows(changed_future, assess, fixture_only=True)
    assert first != third


def test_real_review_binds_exact_inputs_code_and_sources(tmp_path: Path) -> None:
    review = {
        "status": "APPROVED_V2_TRAIN_DEVELOPMENT",
        "candidate_id": "v2_candidate2",
        "protocol_sha256": V2_PROTOCOL_SHA256,
        "input_sha256": "b" * 64,
        "code_sha256": _code_digest(),
        "native_index_sha256": "c" * 64,
        "support_report_sha256": "d" * 64,
        "source_sha256": ["a" * 64],
    }
    path = tmp_path / "review.json"

    def check() -> None:
        path.write_text(json.dumps(review), encoding="utf-8")
        _review_gate(
            fixture_only=False,
            review_path=path,
            review_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            protocol_sha256=V2_PROTOCOL_SHA256,
            source_hashes=("a" * 64,),
            input_sha256="b" * 64,
            native_index_sha256="c" * 64,
            support_report_sha256="d" * 64,
        )

    check()
    review["source_sha256"].append("e" * 64)
    with pytest.raises(ValueError, match="does not authorize"):
        check()
    review["source_sha256"] = ["a" * 64]
    review["input_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="does not authorize"):
        check()


def test_daily_pinball_groups_by_future_target_day() -> None:
    rows = []
    for hour in ("2020-04-01T18", "2020-04-01T19", "2020-04-02T01"):
        row = _rows("2020-04-02")[0]
        cutoff = np.datetime64(hour)
        rows.append(
            replace(
                row,
                row_id=hour,
                cutoff=cutoff,
                target_interval_start=np.array(
                    [cutoff + np.timedelta64(h, "h") for h in (0, 2, 5)]
                ),
                target_interval_end=np.array([cutoff + np.timedelta64(h, "h") for h in (1, 3, 6)]),
                target_db=np.zeros(3),
                target_mask=np.array([False, False, True]),
                target_detection_mask=np.zeros(3, dtype=bool),
            )
        )
    quantiles = np.zeros((3, 3, 5))
    quantiles[2, 2, :] = 40.0
    metrics = _score(rows, JointPrediction(quantiles, np.zeros((3, 3))))
    assert metrics["horizons"][2]["index_days"] == 2
    assert metrics["horizons"][2]["daily_mean_pinball_db"] == 5.0
