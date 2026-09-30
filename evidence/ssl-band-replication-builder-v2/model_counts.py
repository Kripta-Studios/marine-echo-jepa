"""SYNTHETIC_CORRECTNESS_ONLY meta templates; no weights, optimizer or fitting."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replication_test_support  # noqa: F401
import torch

from marine_echo.models.native_band_temporal import NativeBandTemporalModel

before = torch.get_rng_state().clone()
with torch.device("meta"):
    model = NativeBandTemporalModel(192, 64, 4, 4).float()
if not torch.equal(before, torch.get_rng_state()):
    raise ValueError("Meta parameter-count construction consumed CPU RNG.")
print(
    json.dumps(
        {
            "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
            "architecture": "nonlinear_frequency_conditioned_v1",
            "total_trainable_parameters": sum(p.numel() for p in model.parameters()),
            "encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
            "readout_parameters": sum(p.numel() for p in model.readout.parameters()),
            "predictor_parameters": sum(p.numel() for p in model.predictor.parameters()),
            "dtype": str(next(model.parameters()).dtype),
            "cpu_rng_unchanged": True,
        },
        indent=2,
    )
)
