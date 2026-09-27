"""Finite forward EMA campaign controls without real fitting."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from marine_echo.models.aeon_forward_ssl import AeonForwardSSL
from marine_echo.training.aeon_forward_run import (
    FORWARD_SLOTS, _forward_representation_diagnostics, _shuffled_pair_indices,
)


def test_forward_slots_and_shuffled_pair_distances() -> None:
    assert FORWARD_SLOTS == (
        "forward_ema_seed7", "forward_ema_seed13", "forward_ema_seed23",
        "random_encoder_seed7", "temporally_shuffled_future_target_seed7",
    )
    interval_ids = np.arange(500000, 500000 + 24 * 128, 24)
    selected = _shuffled_pair_indices(interval_ids, np.random.default_rng(7), 64)
    assert len(selected) == 64
    assert len(set(selected.tolist())) == 64
    assert np.min(np.diff(np.sort(interval_ids[selected]))) >= 24
    with pytest.raises(ValueError, match="separated"):
        _shuffled_pair_indices(np.arange(63), np.random.default_rng(7), 64)


def test_forward_train_diagnostic_reports_collapse_without_validation() -> None:
    torch.manual_seed(7)
    model = AeonForwardSSL(width=16, layers=2)
    past = torch.randn(8, 24, 4)
    past_mask = torch.ones_like(past, dtype=torch.bool)
    future = torch.randn(8, 6, 4)
    future_mask = torch.ones_like(future, dtype=torch.bool)
    diagnostics = _forward_representation_diagnostics(
        model, past, past_mask, future, future_mask,
        row_ids=[f"{index:064x}" for index in range(8)],
        cutoff_interval_ids=np.arange(8), source_sha256="a" * 64,
    )
    assert diagnostics["source_partition"] == "train"
    assert diagnostics["row_count"] == 8
    assert 0 <= diagnostics["target_effective_rank"] <= 16
    assert np.isfinite(diagnostics["target_dimension_variance_mean"])
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    collapsed = _forward_representation_diagnostics(
        model, past, past_mask, future, future_mask,
        row_ids=[f"{index:064x}" for index in range(8)],
        cutoff_interval_ids=np.arange(8), source_sha256="a" * 64,
    )
    assert collapsed["target_effective_rank"] == 0
    assert collapsed["target_dimension_variance_mean"] == 0
    assert collapsed["prediction_target_rms_ratio"] is None

@pytest.mark.parametrize("slot_id", (
    "forward_ema_seed7", "random_encoder_seed7",
    "temporally_shuffled_future_target_seed7",
))
def test_forward_slot_executes_one_step_fixture_with_real_artifacts(tmp_path, slot_id: str) -> None:
    from dataclasses import replace

    from marine_echo.training.aeon_corpus import AeonHourlySlot
    from marine_echo.training.aeon_forward import build_forward_pairs
    from marine_echo.training.aeon_forward_run import _train_slot
    from marine_echo.training.aeon_windows import AeonWindowPlan, iter_aeon_windows

    first = np.datetime64("2024-03-06T00:52:00", "us")
    slots = [AeonHourlySlot(
        interval_id=500000 + index, source_timestamp=first + index * np.timedelta64(1, "h"),
        sv_db=np.full(4, -80.0 + index / 100), observed_mask=np.ones(4, dtype=bool),
        qc_status=("OBSERVED_SOURCE_PRODUCT",) * 4,
        member_names=("train.csv",), archive_sha256="a" * 64,
    ) for index in range(50)]
    fit = list(iter_aeon_windows(
        slots, plan=AeonWindowPlan(), partition="train",
        partition_start="2024-03-06", partition_end_exclusive="2024-03-09",
    ))
    pairs = build_forward_pairs(fit, slots)
    if slot_id == "temporally_shuffled_future_target_seed7":
        pairs = replace(pairs, cutoff_interval_ids=np.arange(len(pairs.row_ids)) * 24)
    assess = [replace(row, partition="validation", row_id="f" + row.row_id[1:])
              for row in fit[1:21]]
    config = {
        "source_sha256": "a" * 64, "protocol_sha256": "b" * 64,
        "pair_id_and_interval_sha256": "d" * 64,
        "model": {
            "encoder_width": 16, "encoder_layers": 2, "batch_size": 8,
            "variance_floor_std": 1.0, "variance_epsilon": 1e-4,
            "prediction_loss_weight": 25.0,
            "variance_weight_mean_two_online_views": 25.0,
            "covariance_weight_mean_two_online_views": 1.0,
            "learning_rate": 3e-4, "weight_decay": 1e-4,
            "gradient_clip_norm": 1.0, "ema_teacher_momentum": 0.996,
            "pretrain_updates": 1, "supervised_updates": 1,
            "checkpoint_every_updates": 1,
        },
        "device_policy": {"peak_process_rss_limit_bytes": 22 * 1024**3,
                          "peak_gpu_reserved_limit_bytes": 10 * 1024**3},
    }
    result = _train_slot(
        slot_id, fit, assess, pairs, config, tmp_path, "cpu",
        config_sha256="c" * 64,
    )
    is_random = slot_id == "random_encoder_seed7"
    assert result["pretrain_updates"] == (0 if is_random else 1)
    assert result["supervised_updates"] == 1
    assert result["classification"] == "POST_HOC_DEVELOPMENT_NOT_FINAL_EVALUATION"
    assert result["sota_claim"] == "NOT_ESTABLISHED"
    assert len(result["checkpoints"]) == (1 if is_random else 2)
    assert (tmp_path / result["prediction_path"]).exists()
    assert (tmp_path / result["final_checkpoint_path"]).exists()
    diagnostics = result["train_representation_diagnostics"]
    assert set(diagnostics["branches"]) == {
        "online_context", "online_future_target", "ema_future_teacher", "predictor"
    }
    assert result["protocol_validation_metrics"]["eligible_days_per_horizon"] == [1, 1, 1]
