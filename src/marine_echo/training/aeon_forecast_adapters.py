"""Reload frozen AEON models and emit forecast-only evaluator artifacts.

The caller owns reviewed CAL/TEST access and supplies already-issued windows.
These adapters inspect past context fields only. They do not open a corpus,
select a model, fit a parameter, calibrate an interval, or score an outcome.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence

import joblib  # type: ignore[import-untyped]
import numpy as np
import torch
from numpy.typing import NDArray

from marine_echo.models.aeon_forward_ssl import AeonForwardSSL
from marine_echo.models.aeon_ssl import AeonDirect, AeonTemporalSSL
from marine_echo.training.aeon_campaign import _past_features
from marine_echo.training.aeon_chronos import (
    CHRONOS_QUANTILES,
    HORIZON_INDICES,
    MODEL_REVISION,
    _load_official_pipeline,
)
from marine_echo.training.aeon_development import _context_tensors
from marine_echo.training.aeon_sota_supervised import _features
from marine_echo.training.aeon_windows import AeonHourlyWindow
from marine_echo.training import (
    aeon_campaign, aeon_chronos, aeon_development, aeon_hybrid,
    aeon_sota_supervised, aeon_windows,
)
from marine_echo.models import aeon_forward_ssl, aeon_ssl, sigreg


SOURCE_SHA256 = "4e72dd4dbec707b6bf15168e51f380cbe9145a78b595ef886d78cc3806c0ecde"
PROTOCOL_SHA256 = "d83392832a1bde3ae3e096de0a664ca9cfc27762a18bb10fb5ace3a97e787435"
FORWARD_PAIR_SHA256 = "9a3afbb17c6cc9694062b65c66b70f95df258c467814f23c75e40ecccce0b7bc"
SEEDS = (7, 13, 23)
CHRONOS_SNAPSHOT_SHA256 = {
    ".gitattributes": "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
    "README.md": "c7b29bc88f5bc3ebc6e7213b6353be80090d671fefa4aa74ed20498838c16558",
    "config.json": "ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031",
    "model.safetensors": "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42",
}


@dataclass(frozen=True)
class FrozenArtifact:
    """Path plus the independently frozen byte digest."""

    path: Path
    sha256: str


class _Predictor(Protocol):
    def predict(self, values: NDArray[np.float64]) -> object: ...


class _ChronosLike(Protocol):
    def predict_quantiles(
        self, inputs: NDArray[np.float64], **kwargs: object
    ) -> tuple[Sequence[object], Sequence[object]]: ...


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def adapter_composite_sha256() -> str:
    """Bind all repository inference dependencies used by any supported adapter."""
    modules = (
        aeon_campaign, aeon_chronos, aeon_development, aeon_hybrid,
        aeon_sota_supervised, aeon_windows, aeon_forward_ssl, aeon_ssl, sigreg,
    )
    digest = hashlib.sha256()
    digest.update(Path(__file__).name.encode("ascii"))
    digest.update(Path(__file__).read_bytes())
    for module in modules:
        module_file = module.__file__
        if module_file is None:
            raise RuntimeError("AEON adapter dependency lacks a filesystem source path.")
        path = Path(module_file).resolve(strict=True)
        digest.update(path.name.encode("ascii"))
        digest.update(path.read_bytes())
    for distribution in (
        "numpy", "torch", "scikit-learn", "joblib", "lightgbm", "chronos-forecasting",
    ):
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = "NOT_INSTALLED"
        digest.update(f"{distribution}=={version}\n".encode("ascii"))
    return digest.hexdigest()


def _checked(artifact: FrozenArtifact) -> Path:
    path = artifact.path.resolve(strict=True)
    if len(artifact.sha256) != 64 or _sha256(path) != artifact.sha256:
        raise ValueError("AEON frozen model artifact digest differs.")
    return path


def _validate_rows(rows: Sequence[AeonHourlyWindow]) -> None:
    if not rows:
        raise ValueError("AEON forecast adapter needs at least one issued window.")
    row_ids: set[str] = set()
    previous: np.datetime64 | None = None
    for row in rows:
        if (
            row.partition not in ("calibration", "test")
            or not isinstance(row.row_id, str)
            or len(row.row_id) != 64
            or row.row_id in row_ids
            or row.context_db.shape != (24, 4)
            or row.context_mask.shape != (24, 4)
            or row.context_mask.dtype.kind != "b"
            or not row.context_mask[:, 0].all()
            or not np.isfinite(row.context_db[row.context_mask]).all()
            or not np.array_equal(np.diff(row.context_interval_ids), np.ones(23, dtype=np.int64))
            or row.context_interval_ids[-1] != row.cutoff_interval_id
            or (previous is not None and row.cutoff_source_timestamp <= previous)
        ):
            raise ValueError("AEON issued-window past context differs from the frozen contract.")
        row_ids.add(row.row_id)
        previous = row.cutoff_source_timestamp


def _validated_prediction(
    prediction: NDArray[np.float64], count: int
) -> NDArray[np.float64]:
    result = np.asarray(prediction, dtype=np.float64)
    if (
        result.shape != (count, 3, 5)
        or not np.isfinite(result).all()
        or (np.diff(result, axis=-1) < 0).any()
    ):
        raise ValueError("AEON adapter forecast is nonfinite, nonmonotone or mis-shaped.")
    return result


def adapt_conventional(
    rows: Sequence[AeonHourlyWindow], artifact: FrozenArtifact, *, family: str,
) -> NDArray[np.float64]:
    """Reload a core persistence, seasonal, ridge or HGB artifact."""
    _validate_rows(rows)
    saved = joblib.load(_checked(artifact))
    if not isinstance(saved, dict) or saved.get("family") != family:
        raise ValueError("AEON conventional artifact family differs.")
    row_list = list(rows)
    if family in ("persistence", "seasonal_24_source_intervals"):
        expected_indices = (-1, -1, -1) if family == "persistence" else (0, 2, 5)
        if tuple(saved.get("context_indices_by_horizon", ())) != expected_indices:
            raise ValueError("AEON conventional context indices differ.")
        residual = np.asarray(saved.get("train_residual_quantiles"), dtype=np.float64)
        base = np.stack([
            [row.context_db[index, 0] for index in expected_indices] for row in rows
        ])
        prediction = base[:, :, None] + residual[None, :, :]
    elif family == "ridge":
        models = saved.get("models")
        residual = np.asarray(saved.get("train_residual_quantiles"), dtype=np.float64)
        if not isinstance(models, list) or len(models) != 3 or residual.shape != (3, 5):
            raise ValueError("AEON ridge artifact geometry differs.")
        features = _past_features(row_list)
        base = np.stack([np.asarray(model.predict(features)) for model in models], axis=1)
        prediction = base[:, :, None] + residual[None, :, :]
    elif family == "hist_gradient_boosting":
        prediction = _predict_heads(saved.get("models"), _past_features(row_list))
    else:
        raise ValueError("Unsupported AEON core conventional family.")
    return _validated_prediction(np.sort(prediction, axis=-1), len(rows))


def _predict_heads(models: object, features: NDArray[np.float64]) -> NDArray[np.float64]:
    if (
        not isinstance(models, list)
        or len(models) != 3
        or any(not isinstance(heads, list) or len(heads) != 5 for heads in models)
    ):
        raise ValueError("AEON quantile-head artifact geometry differs.")
    prediction: NDArray[np.float64] = np.empty(
        (len(features), 3, 5), dtype=np.float64
    )
    for horizon, heads in enumerate(models):
        for quantile, model in enumerate(heads):
            if not hasattr(model, "predict"):
                raise ValueError("AEON quantile artifact contains a non-predictor head.")
            prediction[:, horizon, quantile] = np.asarray(
                model.predict(features), dtype=np.float64
            )
    prediction.sort(axis=-1)
    return prediction


def adapt_lightgbm(
    rows: Sequence[AeonHourlyWindow], artifact: FrozenArtifact, *, recipe_sha256: str,
) -> NDArray[np.float64]:
    """Reload the reviewed post-hoc 210-feature LightGBM challenger."""
    _validate_rows(rows)
    saved = joblib.load(_checked(artifact))
    if (
        not isinstance(saved, dict)
        or saved.get("family") != "lightgbm_full_past_quantile"
        or saved.get("recipe_sha256") != recipe_sha256
        or len(recipe_sha256) != 64
    ):
        raise ValueError("AEON LightGBM artifact or recipe binding differs.")
    prediction = _predict_heads(saved.get("models"), _features(list(rows)))
    return _validated_prediction(prediction, len(rows))


def _scaler(saved: dict[str, Any]) -> tuple[float, float, float, float]:
    raw = saved.get("scaler_fit_only")
    if not isinstance(raw, (tuple, list)) or len(raw) != 4:
        raise ValueError("AEON checkpoint lacks its TRAIN-only scaler.")
    scaler = tuple(float(value) for value in raw)
    if not np.isfinite(scaler).all() or scaler[1] <= 0 or scaler[3] <= 0:
        raise ValueError("AEON TRAIN-only checkpoint scaler is invalid.")
    return scaler  # type: ignore[return-value]


def _torch_checkpoint(artifact: FrozenArtifact) -> dict[str, Any]:
    saved = torch.load(_checked(artifact), map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or not isinstance(saved.get("model_state_dict"), dict):
        raise ValueError("AEON PyTorch checkpoint schema differs.")
    return saved


def _infer_torch(
    rows: Sequence[AeonHourlyWindow], model: torch.nn.Module,
    scaler: tuple[float, float, float, float], *, device: str,
) -> NDArray[np.float64]:
    if device not in ("cpu", "cuda") or device == "cuda" and not torch.cuda.is_available():
        raise ValueError("AEON adapter device is unavailable or unsupported.")
    values, mask = _context_tensors(list(rows), scaler)
    model.to(device).eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(rows), 64):
            output = model(values[start : start + 64].to(device), mask[start : start + 64].to(device))
            parts.append(output.cpu().numpy() * scaler[3] + scaler[2])
    model.to("cpu")
    return _validated_prediction(np.concatenate(parts), len(rows))


def adapt_core_neural_ensemble(
    rows: Sequence[AeonHourlyWindow], artifacts: Sequence[FrozenArtifact], *,
    family: str, device: str = "cpu",
) -> NDArray[np.float64]:
    """Reload and equally average the three reviewed core neural seeds."""
    _validate_rows(rows)
    if family not in ("direct", "ema_jepa", "shared_sigreg") or len(artifacts) != 3:
        raise ValueError("AEON core ensemble family or member count differs.")
    forecasts = []
    for expected_seed, artifact in zip(SEEDS, artifacts, strict=True):
        saved = _torch_checkpoint(artifact)
        expected_step = 3000 if family == "direct" else 1500
        if (
            saved.get("phase") != "supervised"
            or saved.get("step") != expected_step
            or saved.get("family") != family
            or saved.get("seed") != expected_seed
            or saved.get("source_sha256") != SOURCE_SHA256
            or saved.get("protocol_sha256") != PROTOCOL_SHA256
        ):
            raise ValueError("AEON core checkpoint identity or endpoint differs.")
        if family == "direct":
            model: torch.nn.Module = AeonDirect(width=128, layers=3)
        else:
            mode: Literal["ema", "shared_sigreg"] = (
                "ema" if family == "ema_jepa" else "shared_sigreg"
            )
            weight = 0.03 if mode == "ema" else 0.04
            model = AeonTemporalSSL(mode=mode, width=128, layers=3, regularizer_weight=weight)
        model.load_state_dict(saved["model_state_dict"], strict=True)
        forecasts.append(_infer_torch(rows, model, _scaler(saved), device=device))
    return _validated_prediction(np.sort(np.mean(forecasts, axis=0), axis=-1), len(rows))


def adapt_forward_ensemble(
    rows: Sequence[AeonHourlyWindow], artifacts: Sequence[FrozenArtifact], *,
    config_sha256: str, device: str = "cpu",
) -> NDArray[np.float64]:
    """Reload and equally average the three reviewed forward-EMA seeds."""
    _validate_rows(rows)
    if len(artifacts) != 3 or len(config_sha256) != 64:
        raise ValueError("AEON forward ensemble member count or config digest differs.")
    forecasts = []
    for seed, artifact in zip(SEEDS, artifacts, strict=True):
        saved = _torch_checkpoint(artifact)
        if (
            saved.get("phase") != "supervised"
            or saved.get("step") != 1500
            or saved.get("slot_id") != f"forward_ema_seed{seed}"
            or saved.get("seed") != seed
            or saved.get("source_sha256") != SOURCE_SHA256
            or saved.get("protocol_sha256") != PROTOCOL_SHA256
            or saved.get("pair_id_and_interval_sha256") != FORWARD_PAIR_SHA256
            or saved.get("config_sha256") != config_sha256
        ):
            raise ValueError("AEON forward checkpoint identity or endpoint differs.")
        model = AeonForwardSSL()
        model.load_state_dict(saved["model_state_dict"], strict=True)
        forecasts.append(_infer_torch(rows, model, _scaler(saved), device=device))
    return _validated_prediction(np.sort(np.mean(forecasts, axis=0), axis=-1), len(rows))


def _chronos_context(rows: Sequence[AeonHourlyWindow]) -> NDArray[np.float64]:
    values = np.stack([row.context_db for row in rows]).astype(np.float64, copy=True)
    mask = np.stack([row.context_mask for row in rows])
    values[~mask] = np.nan
    return values.transpose(0, 2, 1)


def adapt_chronos2(
    rows: Sequence[AeonHourlyWindow], *, snapshot: Path | None = None,
    snapshot_files_sha256: dict[str, str] | None = None,
    pipeline: _ChronosLike | None = None, fixture_only: bool = False,
    device: str = "cpu",
) -> NDArray[np.float64]:
    """Run the exact frozen Chronos-2 revision without fitting."""
    _validate_rows(rows)
    if pipeline is not None and not fixture_only:
        raise ValueError("Injected Chronos pipelines are permitted for synthetic fixtures only.")
    if pipeline is None:
        if fixture_only or snapshot is None or snapshot_files_sha256 is None:
            raise ValueError("Real Chronos adaptation needs the exact reviewed local snapshot.")
        if snapshot_files_sha256 != CHRONOS_SNAPSHOT_SHA256:
            raise ValueError("Chronos-2 snapshot differs from exact revision " + MODEL_REVISION + ".")
        snapshot = snapshot.resolve(strict=True)
        present = {path.name for path in snapshot.iterdir()}
        if present != set(CHRONOS_SNAPSHOT_SHA256) or any(
            not (snapshot / name).is_file() for name in present
        ):
            raise ValueError("Chronos-2 snapshot contains missing or extra files.")
        if any(_sha256(snapshot / name) != digest for name, digest in snapshot_files_sha256.items()):
            raise ValueError("Chronos-2 snapshot digest differs.")
        pipeline = _load_official_pipeline(snapshot, device=device)
    quantiles, _ = pipeline.predict_quantiles(
        inputs=_chronos_context(rows), prediction_length=6,
        quantile_levels=CHRONOS_QUANTILES.tolist(), batch_size=64,
        context_length=24, cross_learning=False, limit_prediction_length=False,
    )
    if len(quantiles) != len(rows):
        raise ValueError("Chronos-2 returned a different number of issued rows.")
    prediction: NDArray[np.float64] = np.empty((len(rows), 3, 5), dtype=np.float64)
    for index, item in enumerate(quantiles):
        value: object = item
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (4, 6, 5):
            raise ValueError("Chronos-2 returned invalid multivariate geometry.")
        prediction[index] = array[0, HORIZON_INDICES]
    return _validated_prediction(np.sort(prediction, axis=-1), len(rows))


def emit_forecast_only(
    output: Path, rows: Sequence[AeonHourlyWindow], prediction: NDArray[np.float64], *,
    model_id: str, family: str, components: Sequence[FrozenArtifact],
) -> dict[str, Any]:
    """Write the evaluator's exact NPZ schema and a truth-free provenance sidecar."""
    _validate_rows(rows)
    forecast = _validated_prediction(prediction, len(rows))
    output = output.resolve()
    sidecar = output.with_suffix(".json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("AEON forecast-only output already exists.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(
            stream, row_ids=np.asarray([row.row_id for row in rows]), quantiles_db=forecast
        )
    record: dict[str, Any] = {
        "status": "COMPLETED_AEON_FORECAST_ONLY_ADAPTER",
        "model_id": model_id,
        "family": family,
        "model_revision": MODEL_REVISION if family == "chronos2" else None,
        "adapter_code_sha256": _sha256(Path(__file__)),
        "adapter_composite_sha256": adapter_composite_sha256(),
        "component_artifacts": [
            {"path": artifact.path.name, "sha256": artifact.sha256} for artifact in components
        ],
        "issued_rows": len(rows),
        "row_ids_sha256": hashlib.sha256(
            ("\n".join(row.row_id for row in rows) + "\n").encode("ascii")
        ).hexdigest(),
        "forecast_path": output.name,
        "forecast_sha256": _sha256(output),
        "schema": ["row_ids", "quantiles_db"],
        "truth_fields": "ABSENT",
        "fit_performed": False,
    }
    sidecar.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return record
