"""Admit only the trainer's known opaque review hash without loosening schema guards."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "evidence/ssl-band-replication-builder-v2")
)
from replication_test_support import artifact, codec

from marine_echo.inference.native_band_replication_acoustic import NativeBandAcousticPredictor


def test_known_trainer_review_hash_is_accepted_without_rng_or_provenance_io():
    saved, _ = artifact()
    saved["review_sha256"] = "b" * 64
    rng = torch.get_rng_state().clone()
    predictor = NativeBandAcousticPredictor(codec(saved))
    assert predictor.config.seed == 13
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("bad", [None, True, 3, "b" * 63, "G" * 64, {"execute": "unsafe"}])
def test_invalid_review_hash_rejected_before_model_setup(bad):
    saved, _ = artifact()
    saved["review_sha256"] = bad
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="review hash"):
        NativeBandAcousticPredictor(codec(saved))
    assert torch.equal(rng, torch.get_rng_state())
