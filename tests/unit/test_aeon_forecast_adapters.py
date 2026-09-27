"""Synthetic-only tests for frozen AEON forecast adapters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib  # type: ignore[import-untyped]
import numpy as np
import pytest
import torch

from marine_echo.models.aeon_forward_ssl import AeonForwardSSL
from marine_echo.models.aeon_ssl import AeonDirect, AeonTemporalSSL
from marine_echo.training import aeon_forecast_adapters
from marine_echo.training.aeon_forecast_adapters import (
    FrozenArtifact,
    adapt_chronos2,
    adapt_conventional,
    adapt_core_neural_ensemble,
    adapt_forward_ensemble,
    adapt_lightgbm,
    emit_forecast_only,
)
from marine_echo.training.aeon_windows import AeonHourlyWindow


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(count: int = 3) -> list[AeonHourlyWindow]:
    rows = []
    for index in range(count):
        cutoff = np.datetime64("2024-12-02T00", "us") + np.timedelta64(index, "h")
        values = np.arange(96, dtype=np.float64).reshape(24, 4) / 10 + index
        mask = np.ones((24, 4), dtype=bool)
        mask[2, 2] = False
        rows.append(AeonHourlyWindow(
            row_id=hashlib.sha256(f"fixture:{index}".encode()).hexdigest(),
            partition="calibration", cutoff_source_timestamp=cutoff,
            cutoff_interval_id=1000 + index, context_db=values, context_mask=mask,
            context_interval_ids=np.arange(977 + index, 1001 + index),
            context_source_timestamps=cutoff - np.arange(23, -1, -1).astype("timedelta64[h]"),
            target_interval_ids=np.asarray([1001, 1003, 1006]) + index,
            target_source_timestamps=np.full(3, np.datetime64("NaT", "us")),
            target_db=np.full(3, np.nan), target_mask=np.zeros(3, dtype=bool),
            target_qc_status=("UNOPENED",) * 3, source_archive_sha256="a" * 64,
            past_members=("past",), target_members=(),
        ))
    return rows


class _Head:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.full(len(values), self.value)


class _Chronos:
    def predict_quantiles(self, inputs: np.ndarray, **_: object):
        values = []
        for _row in inputs:
            forecast = np.empty((4, 6, 5), dtype=np.float64)
            forecast[:] = np.arange(5)
            values.append(forecast)
        return values, []


def _artifact(path: Path) -> FrozenArtifact:
    return FrozenArtifact(path=path, sha256=_sha(path))


def test_conventional_and_lightgbm_reload_without_targets(tmp_path: Path) -> None:
    rows = _rows()
    persistence = tmp_path / "persistence.joblib"
    joblib.dump({"family": "persistence", "context_indices_by_horizon": (-1, -1, -1),
                 "train_residual_quantiles": [[-2, -1, 0, 1, 2]] * 3}, persistence)
    conventional = adapt_conventional(rows, _artifact(persistence), family="persistence")
    expected = np.repeat(
        np.stack([r.context_db[-1, 0] for r in rows])[:, None], 3, axis=1
    )
    np.testing.assert_allclose(conventional[:, :, 2], expected)

    lightgbm = tmp_path / "lightgbm.joblib"
    heads = [[_Head(horizon * 10 + quantile) for quantile in range(5)] for horizon in range(3)]
    joblib.dump({"family": "lightgbm_full_past_quantile", "models": heads,
                 "recipe_sha256": "b" * 64}, lightgbm)
    prediction = adapt_lightgbm(rows, _artifact(lightgbm), recipe_sha256="b" * 64)
    assert prediction.shape == (3, 3, 5)
    assert prediction[0, 2].tolist() == [20, 21, 22, 23, 24]
    with pytest.raises(ValueError, match="digest differs"):
        adapt_lightgbm(
            rows, FrozenArtifact(lightgbm, "0" * 64), recipe_sha256="b" * 64
        )


def _core_checkpoint(path: Path, family: str, seed: int, median: float) -> FrozenArtifact:
    torch.manual_seed(seed)
    if family == "direct":
        model = AeonDirect(width=128, layers=3)
        step = 3000
    else:
        model = AeonTemporalSSL(mode="ema", width=128, layers=3, regularizer_weight=0.03)
        step = 1500
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.copy_(torch.arange(15).reshape(3, 5).flatten() + median)
    torch.save({"phase": "supervised", "step": step, "family": family, "seed": seed,
                "model_state_dict": model.state_dict(),
                "scaler_fit_only": (0.0, 1.0, 0.0, 1.0),
                "source_sha256": "4e72dd4dbec707b6bf15168e51f380cbe9145a78b595ef886d78cc3806c0ecde",
                "protocol_sha256": "d83392832a1bde3ae3e096de0a664ca9cfc27762a18bb10fb5ace3a97e787435"}, path)
    return _artifact(path)


def test_core_direct_and_ema_ensembles_reload_exact_three_seeds(tmp_path: Path) -> None:
    rows = _rows()
    for family in ("direct", "ema_jepa"):
        artifacts = [
            _core_checkpoint(tmp_path / f"{family}-{seed}.pt", family, seed, float(index))
            for index, seed in enumerate((7, 13, 23))
        ]
        prediction = adapt_core_neural_ensemble(rows, artifacts, family=family, device="cpu")
        assert prediction.shape == (3, 3, 5)
        np.testing.assert_allclose(prediction[0, 0], np.arange(5) + 1)


def _forward_checkpoint(path: Path, seed: int, offset: float) -> FrozenArtifact:
    model = AeonForwardSSL()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.copy_(torch.arange(15).reshape(3, 5).flatten() + offset)
    torch.save({"phase": "supervised", "step": 1500, "slot_id": f"forward_ema_seed{seed}",
                "seed": seed, "model_state_dict": model.state_dict(),
                "scaler_fit_only": (0.0, 1.0, 0.0, 1.0),
                "source_sha256": "4e72dd4dbec707b6bf15168e51f380cbe9145a78b595ef886d78cc3806c0ecde",
                "protocol_sha256": "d83392832a1bde3ae3e096de0a664ca9cfc27762a18bb10fb5ace3a97e787435",
                "pair_id_and_interval_sha256": "9a3afbb17c6cc9694062b65c66b70f95df258c467814f23c75e40ecccce0b7bc",
                "config_sha256": "c" * 64}, path)
    return _artifact(path)


def test_forward_ema_ensemble_and_forecast_only_schema(tmp_path: Path) -> None:
    rows = _rows()
    artifacts = [
        _forward_checkpoint(tmp_path / f"forward-{seed}.pt", seed, float(index))
        for index, seed in enumerate((7, 13, 23))
    ]
    prediction = adapt_forward_ensemble(
        rows, artifacts, config_sha256="c" * 64, device="cpu"
    )
    output = tmp_path / "forecast.npz"
    record = emit_forecast_only(
        output, rows, prediction, model_id="forward_ema_ensemble",
        family="forward_ema", components=artifacts,
    )
    with np.load(output, allow_pickle=False) as saved:
        assert set(saved.files) == {"row_ids", "quantiles_db"}
    assert set(json.loads(output.with_suffix(".json").read_text())) >= {
        "adapter_code_sha256", "component_artifacts", "forecast_sha256"
    }
    assert record["forecast_sha256"] == _sha(output)


def test_chronos_fixture_uses_multivariate_mask_and_exact_output(tmp_path: Path) -> None:
    prediction = adapt_chronos2(_rows(), pipeline=_Chronos(), fixture_only=True)
    assert prediction.shape == (3, 3, 5)
    np.testing.assert_array_equal(prediction[0, 0], np.arange(5))


def test_chronos_snapshot_rejects_extra_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    names = (".gitattributes", "README.md", "config.json", "model.safetensors")
    for name in names:
        (snapshot / name).write_text(name, encoding="utf-8")
    expected = {name: _sha(snapshot / name) for name in names}
    monkeypatch.setattr(aeon_forecast_adapters, "CHRONOS_SNAPSHOT_SHA256", expected)
    (snapshot / "unexpected.bin").write_bytes(b"extra")
    with pytest.raises(ValueError, match="extra files"):
        adapt_chronos2(
            _rows(), snapshot=snapshot, snapshot_files_sha256=expected, device="cpu"
        )
