"""Post-campaign TRAIN-only frozen-encoder representation check."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from marine_echo.models.aeon_ssl import AeonTemporalSSL
from marine_echo.training.aeon_random_diagnostics import _checkpoint_diagnostic
from marine_echo.training.aeon_windows import AeonHourlyWindow


def _train_rows() -> list[AeonHourlyWindow]:
    rows = []
    for index in range(8):
        cutoff = np.datetime64("2024-03-08T00:00:00", "us") + index * np.timedelta64(1, "h")
        rows.append(AeonHourlyWindow(
            row_id=f"{index:064x}", partition="train", cutoff_source_timestamp=cutoff,
            cutoff_interval_id=480120 + index,
            context_db=np.full((24, 4), -80.0 + index),
            context_mask=np.ones((24, 4), dtype=bool),
            context_interval_ids=np.arange(480097, 480121) + index,
            context_source_timestamps=cutoff - np.arange(23, -1, -1) * np.timedelta64(1, "h"),
            target_interval_ids=np.array([480121, 480123, 480126]) + index,
            target_source_timestamps=cutoff + np.array([1, 3, 6]) * np.timedelta64(1, "h"),
            target_db=np.full(3, -80.0 + index), target_mask=np.ones(3, dtype=bool),
            target_qc_status=("OBSERVED_SOURCE_PRODUCT",) * 3,
            source_archive_sha256="a" * 64, past_members=("train.csv",),
            target_members=("train.csv",),
        ))
    return rows


def test_random_checkpoint_diagnostic_uses_train_rows_and_frozen_encoder(
    tmp_path: Path,
) -> None:
    rows = _train_rows()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = AeonTemporalSSL(mode="ema", width=16, layers=2, regularizer_weight=0.03)
    checkpoint = tmp_path / "model.pt"
    torch.save({
        "phase": "supervised", "step": 1500, "family": "random_encoder_ema",
        "seed": 7, "model_state_dict": model.state_dict(),
        "scaler_fit_only": (-80.0, 1.0, -80.0, 1.0),
        "source_sha256": "a" * 64, "protocol_sha256": "b" * 64,
    }, checkpoint)
    config = {"neural": {"encoder_width": 16, "encoder_layers": 2,
                         "ema_sigreg_weight": 0.03, "shared_sigreg_weight": 0.04}}
    with pytest.raises(ValueError, match="TRAIN scaler"):
        _checkpoint_diagnostic(checkpoint, "random_encoder_ema", rows, config,
                               expected_source="a" * 64, expected_protocol="b" * 64)
    from marine_echo.training.aeon_development import _normalizer

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    saved["scaler_fit_only"] = _normalizer(rows)
    torch.save(saved, checkpoint)
    report = _checkpoint_diagnostic(checkpoint, "random_encoder_ema", rows, config,
                                    expected_source="a" * 64, expected_protocol="b" * 64)
    assert report["source_partition"] == "train"
    assert report["row_ids"] == [row.row_id for row in rows]
    assert np.isfinite(report["target_effective_rank"])
    assert report["row_count"] == 8

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    saved["model_state_dict"]["encoder.network.0.weight"][0, 0] += 0.1
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="representation weights changed"):
        _checkpoint_diagnostic(checkpoint, "random_encoder_ema", rows, config,
                               expected_source="a" * 64, expected_protocol="b" * 64)
