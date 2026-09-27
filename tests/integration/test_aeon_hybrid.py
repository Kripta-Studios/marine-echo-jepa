"""Frozen-pretrain latent and raw AEON hybrid contract fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from marine_echo.models.aeon_ssl import AeonTemporalSSL
from marine_echo.training.aeon_hybrid import (
    HYBRID_SLOTS,
    _fit_quantile_heads,
    _hybrid_gate,
    _pretrain_encoder,
    _source_spec,
)


def test_hybrid_uses_only_final_pretrain_checkpoint(tmp_path: Path) -> None:
    model = AeonTemporalSSL(mode="ema", width=16, layers=2, regularizer_weight=0.03)
    checkpoint = tmp_path / "checkpoint-pretrain-1500.pt"
    record = {
        "phase": "pretrain", "step": 1500, "family": "ema_jepa", "seed": 7,
        "model_state_dict": model.state_dict(), "scaler_fit_only": (-80.0, 1.0, -80.0, 1.0),
        "source_sha256": "a" * 64, "protocol_sha256": "b" * 64,
    }
    torch.save(record, checkpoint)
    config = {"neural": {"encoder_width": 16, "encoder_layers": 2,
                         "ema_sigreg_weight": 0.03, "shared_sigreg_weight": 0.04}}
    loaded = _pretrain_encoder(checkpoint, "ema", config, (-80.0, 1.0, -80.0, 1.0),
                               source_sha256="a" * 64, protocol_sha256="b" * 64)
    assert torch.equal(loaded.encoder.network[0].weight, model.encoder.network[0].weight)
    record["phase"] = "supervised"
    torch.save(record, checkpoint)
    with pytest.raises(ValueError, match="final pretrain"):
        _pretrain_encoder(checkpoint, "ema", config, (-80.0, 1.0, -80.0, 1.0),
                          source_sha256="a" * 64, protocol_sha256="b" * 64)


def test_hybrid_fits_fifteen_train_only_quantile_heads() -> None:
    rng = np.random.default_rng(7)
    train_x = rng.normal(size=(60, 18))
    train_y = np.stack([train_x[:, 0] + horizon for horizon in range(3)], axis=1)
    observed = np.ones((60, 3), dtype=bool)
    assess_x = rng.normal(size=(3, 18))
    config = {"conventional": {
        "hist_gradient_boosting_max_iter": 2,
        "hist_gradient_boosting_max_depth": 4,
        "hist_gradient_boosting_learning_rate": 0.05,
        "random_state": 7, "max_threads": 2,
    }}
    forecast, models = _fit_quantile_heads(train_x, train_y, observed, assess_x, config)
    assert forecast.shape == (3, 3, 5)
    assert all(len(heads) == 5 for heads in models)
    assert all(head.loss == "quantile" for heads in models for head in heads)
    assert np.all(np.diff(forecast, axis=-1) >= 0)


def test_hybrid_config_freezes_ten_sources_and_rejects_control_deletion() -> None:
    import json

    root = Path(__file__).resolve().parents[2]
    hybrid = json.loads((root / "configs/aeon_hybrid.json").read_text(encoding="utf-8"))
    core = json.loads((root / "configs/aeon_campaign.json").read_text(encoding="utf-8"))
    _hybrid_gate(hybrid, core)
    assert tuple(hybrid["slots"]) == HYBRID_SLOTS
    assert [_source_spec(slot)[3] for slot in HYBRID_SLOTS].count("supervised") == 2
    assert [_source_spec(slot)[3] for slot in HYBRID_SLOTS].count("pretrain") == 8
    assert _source_spec("ema_pretrain_raw_latent_hgb_seed23") == (
        "ema_jepa_seed23", "ema", 23, "pretrain"
    )
    assert _source_spec("shuffled_target_shared_sigreg_raw_latent_hgb_seed7") == (
        "temporally_shuffled_pretrain_target_shared_sigreg_seed7",
        "shared_sigreg", 7, "pretrain",
    )
    assert _source_spec("random_encoder_ema_raw_latent_hgb_seed7") == (
        "random_encoder_ema_seed7", "ema", 7, "supervised"
    )
    expected_sources = (
        ("ema_jepa_seed7", "ema", 7, "pretrain"),
        ("ema_jepa_seed13", "ema", 13, "pretrain"),
        ("ema_jepa_seed23", "ema", 23, "pretrain"),
        ("shared_sigreg_seed7", "shared_sigreg", 7, "pretrain"),
        ("shared_sigreg_seed13", "shared_sigreg", 13, "pretrain"),
        ("shared_sigreg_seed23", "shared_sigreg", 23, "pretrain"),
        ("random_encoder_ema_seed7", "ema", 7, "supervised"),
        ("random_encoder_shared_sigreg_seed7", "shared_sigreg", 7, "supervised"),
        ("temporally_shuffled_pretrain_target_ema_seed7", "ema", 7, "pretrain"),
        ("temporally_shuffled_pretrain_target_shared_sigreg_seed7", "shared_sigreg", 7, "pretrain"),
    )
    assert tuple(_source_spec(slot) for slot in HYBRID_SLOTS) == expected_sources
    hybrid["slots"].pop()
    with pytest.raises(ValueError, match="fixed TRAIN/validation"):
        _hybrid_gate(hybrid, core)


def test_hybrid_random_control_rejects_changed_frozen_encoder(tmp_path: Path) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = AeonTemporalSSL(mode="shared_sigreg", width=16, layers=2,
                                regularizer_weight=0.04)
    checkpoint = tmp_path / "random-supervised.pt"
    record = {
        "phase": "supervised", "step": 1500,
        "family": "random_encoder_shared_sigreg", "seed": 7,
        "model_state_dict": model.state_dict(), "scaler_fit_only": (-80.0, 1.0, -80.0, 1.0),
        "source_sha256": "a" * 64, "protocol_sha256": "b" * 64,
    }
    torch.save(record, checkpoint)
    config = {"neural": {"encoder_width": 16, "encoder_layers": 2,
                         "ema_sigreg_weight": 0.03, "shared_sigreg_weight": 0.04}}
    _pretrain_encoder(checkpoint, "shared_sigreg", config, (-80.0, 1.0, -80.0, 1.0),
                      source_sha256="a" * 64, protocol_sha256="b" * 64,
                      family="random_encoder_shared_sigreg", phase="supervised")
    record["model_state_dict"]["encoder.network.0.weight"][0, 0] += 0.1
    torch.save(record, checkpoint)
    with pytest.raises(ValueError, match="random encoder differs"):
        _pretrain_encoder(checkpoint, "shared_sigreg", config, (-80.0, 1.0, -80.0, 1.0),
                          source_sha256="a" * 64, protocol_sha256="b" * 64,
                          family="random_encoder_shared_sigreg", phase="supervised")


def test_hybrid_rejects_checkpoint_family_mismatch(tmp_path: Path) -> None:
    model = AeonTemporalSSL(mode="ema", width=16, layers=2, regularizer_weight=0.03)
    checkpoint = tmp_path / "wrong-family.pt"
    torch.save({
        "phase": "pretrain", "step": 1500, "family": "shared_sigreg", "seed": 7,
        "model_state_dict": model.state_dict(), "scaler_fit_only": (-80.0, 1.0, -80.0, 1.0),
        "source_sha256": "a" * 64, "protocol_sha256": "b" * 64,
    }, checkpoint)
    config = {"neural": {"encoder_width": 16, "encoder_layers": 2,
                         "ema_sigreg_weight": 0.03, "shared_sigreg_weight": 0.04}}
    with pytest.raises(ValueError, match="exact final pretrain"):
        _pretrain_encoder(checkpoint, "ema", config, (-80.0, 1.0, -80.0, 1.0),
                          source_sha256="a" * 64, protocol_sha256="b" * 64,
                          family="ema_jepa", seed=7, phase="pretrain")
