"""Small declared synthetic shards exercise the real baseline execution path."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from marine_echo.data.canonical_store import CanonicalStore, StoreIdentity, split_bounds_sha256
from marine_echo.data.preprocessing import aggregate_calibrated_pings
from marine_echo.evaluation.baseline_runner import run_baselines
from marine_echo.training.data_adapter import CanonicalWindowAdapter, TrainOnlyScaler


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _bounds():
    return {
        "train": ("2020-01-01T00:00:00Z", "2020-01-04T00:00:00Z"),
        "validation": ("2020-01-04T00:00:00Z", "2020-01-07T00:00:00Z"),
        "calibration": ("2020-01-07T00:00:00Z", "2020-01-08T00:00:00Z"),
        "test": ("2020-01-08T00:00:00Z", "2020-01-09T00:00:00Z"),
    }


def _store(tmp_path: Path, *, altered_future: bool = False) -> CanonicalStore:
    tmp_path.mkdir(parents=True, exist_ok=True)
    bounds = _bounds()
    calibration = {
        "calibration_status": "VERIFIED_PHYSICAL_SV",
        "dataset_id": "synthetic-four-channel",
        "deployment_id": "synthetic-deployment",
        "instrument_id": "synthetic-instrument",
        "source_inventory_sha256": "b" * 64,
        "config_sha256": "c" * 64,
        "scope": "synthetic-fixture-only",
    }
    report = tmp_path / "synthetic-calibration.json"
    report.write_text(json.dumps(calibration, sort_keys=True) + "\n", encoding="utf-8")
    sources = tuple(
        _digest(f"{part}-{offset}")
        for part in bounds
        for offset in range(3 if part in ("train", "validation") else 1)
    )
    identity = StoreIdentity(
        dataset_id="synthetic-four-channel",
        deployment_id="synthetic-deployment",
        instrument_id="synthetic-instrument",
        source_inventory_sha256="b" * 64,
        source_file_ids=sources,
        config_sha256="c" * 64,
        calibration_report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        processing_sha256="d" * 64,
        split_sha256=split_bounds_sha256(bounds),
    )
    store = CanonicalStore(tmp_path / "canonical", identity, calibration_manifest=report)
    for part, (start, _) in bounds.items():
        first = np.datetime64(start[:10], "D")
        days = 3 if part in ("train", "validation") else 1
        for offset in range(days):
            day = first + np.timedelta64(offset, "D")
            times = day.astype("datetime64[m]") + np.arange(96) * np.timedelta64(15, "m")
            base = 1e-6 * (1 + offset * 0.2 + (part == "validation") * 0.1)
            if altered_future and part == "validation" and offset == 1:
                base *= 3
            values = np.full((96, 4, 64), base, dtype=np.float64)
            values[:, 1] *= 1.1
            values[:, 2] *= 1.2
            values[:, 3] *= 1.3
            series = aggregate_calibrated_pings(
                ping_times=times.astype("datetime64[ns]"),
                sv_linear=values,
                valid_mask=np.ones(values.shape, dtype=bool),
                frequency_hz=np.array([38000, 125000, 200000, 455000]),
                range_edges_m=np.arange(65, dtype=float) * 2,
                supported_range=np.ones((4, 64), dtype=bool),
                configuration_ids=[_digest("xml-a")] * 96,
                source_file_ids=[_digest(f"{part}-{offset}")] * 96,
                dataset_id=identity.dataset_id,
                deployment_id=identity.deployment_id,
                instrument_id=identity.instrument_id,
                calibration_status="VERIFIED_PHYSICAL_SV",
                calibration_report_sha256=identity.calibration_report_sha256,
                config_sha256=identity.config_sha256,
                split_sha256=identity.split_sha256,
                processing_sha256=identity.processing_sha256,
            )
            store.write_day(series, partition=part)
    return store


def test_adapter_fixed_geometry_future_mutation_and_partition_refusal(tmp_path: Path) -> None:
    first = CanonicalWindowAdapter(_store(tmp_path / "first"), _bounds(), fixture_only=True)
    second = CanonicalWindowAdapter(
        _store(tmp_path / "second", altered_future=True), _bounds(), fixture_only=True
    )
    base = first.materialize("validation", max_windows=128)
    changed = second.materialize("validation", max_windows=128)
    assert base.context_db.shape[1:] == (96, 4, 64)
    assert base.target_db.shape[1:] == (3,)
    assert len(base.row_ids) > 0
    first_row = np.flatnonzero(base.cutoffs == np.datetime64("2020-01-05T00:00"))[0]
    np.testing.assert_array_equal(base.context_db[first_row], changed.context_db[first_row])
    assert not np.array_equal(base.target_db[first_row], changed.target_db[first_row])
    assert base.row_ids[first_row] == changed.row_ids[first_row]
    with pytest.raises(ValueError, match="protected"):
        first.materialize("test", max_windows=128)
    with pytest.raises(ValueError, match="protected"):
        first.materialize("calibration", max_windows=128)
    with pytest.raises(ValueError, match="bounded"):
        first.materialize("train", max_windows=1)
    assert first.peak_cached_days <= 3


def test_scaler_train_only_and_provenance(tmp_path: Path) -> None:
    adapter = CanonicalWindowAdapter(_store(tmp_path), _bounds(), fixture_only=True)
    train = adapter.materialize("train", max_windows=128)
    validation = adapter.materialize("validation", max_windows=128)
    scaler = TrainOnlyScaler().fit(train)
    assert scaler.provenance["partition"] == "train"
    assert scaler.provenance["split_sha256"] == adapter.store.identity.split_sha256
    assert scaler.transform(validation).shape == validation.context_db.shape
    with pytest.raises(ValueError, match="train"):
        TrainOnlyScaler().fit(validation)
    with pytest.raises(ValueError, match="R0/R1"):
        CanonicalWindowAdapter(adapter.store, _bounds(), fixture_only=False)
    with pytest.raises(ValueError, match="bounded"):
        adapter.materialize("train", max_windows=1025)
    with pytest.raises(ValueError, match="eligible"):
        CanonicalWindowAdapter(
            adapter.store, _bounds(), fixture_only=True, availability_basis="measured"
        ).materialize("train", max_windows=128)


def test_adapter_rejects_cross_day_grid_mutation(tmp_path: Path, monkeypatch) -> None:
    adapter = CanonicalWindowAdapter(_store(tmp_path), _bounds(), fixture_only=True)
    original = adapter.store.load_day

    def altered(day: str):
        series = original(day)
        if day == "2020-01-02":
            return replace(series, frequency_hz=np.array([38000, 126000, 200000, 455000]))
        return series

    monkeypatch.setattr(adapter.store, "load_day", altered)
    with pytest.raises(ValueError, match="grids or provenance"):
        adapter.materialize("train", max_windows=128)


def test_adapter_rejects_cross_partition_source_registration(tmp_path: Path) -> None:
    store = _store(tmp_path)
    manifest_path = tmp_path / "canonical/processing_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_source = manifest["shards"]["2020-01-01"]["source_file_ids"][0]
    manifest["shards"]["2020-01-04"]["source_file_ids"].append(train_source)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="across train and validation"):
        CanonicalWindowAdapter(store, _bounds(), fixture_only=True)


def test_four_baselines_write_immutable_fitted_predictions_from_synthetic_rows(
    tmp_path: Path,
) -> None:
    adapter = CanonicalWindowAdapter(_store(tmp_path), _bounds(), fixture_only=True)
    output = tmp_path / "runs"
    first = run_baselines(adapter, output, max_windows=128, tree_max_iter=12)
    assert set(first) == {"persistence", "seasonal", "ridge", "hist_gradient_boosting"}
    for family, result in first.items():
        assert result["status"] == "COMPLETED_SYNTHETIC_FIXTURE"
        assert result["metrics"]["daily_mean_pinball_db"] is not None
        predictions = output / family / "validation_predictions.npz"
        with np.load(predictions, allow_pickle=False) as loaded:
            assert loaded["quantiles"].shape[1:] == (3, 5)
            assert loaded["truth_db"].shape[1:] == (3,)
            assert len(loaded["row_ids"]) > 0
    resumed = run_baselines(adapter, output, max_windows=128, tree_max_iter=12)
    assert all(result["disposition"] == "SKIPPED_VERIFIED" for result in resumed.values())
    prediction = output / "ridge" / "validation_predictions.npz"
    prediction.write_bytes(prediction.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        run_baselines(adapter, output, max_windows=128, tree_max_iter=12)
