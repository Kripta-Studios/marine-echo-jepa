"""Synthetic persistence and metadata-only eligibility contract tests."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from marine_echo.data.canonical_store import (
    CanonicalStore,
    StoreIdentity,
    audit_eligibility,
    split_bounds_sha256,
)
from marine_echo.data.preprocessing import aggregate_calibrated_pings, build_window


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


_CALIBRATION_REPORT = {
    "calibration_status": "VERIFIED_PHYSICAL_SV",
    "dataset_id": "synthetic-calibrated",
    "deployment_id": "synthetic-deployment",
    "instrument_id": "synthetic-instrument",
    "source_inventory_sha256": "b" * 64,
    "config_sha256": "c" * 64,
    "scope": "synthetic-fixture-only",
}
_CALIBRATION_BYTES = (json.dumps(_CALIBRATION_REPORT, sort_keys=True) + "\n").encode()
_CALIBRATION_SHA256 = hashlib.sha256(_CALIBRATION_BYTES).hexdigest()


def _calibration_path(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic-calibration.json"
    path.write_bytes(_CALIBRATION_BYTES)
    return path


def _series(day: str, source: str = "train-a", *, support: np.ndarray | None = None):
    start = np.datetime64(day, "m")
    times = start + np.arange(96) * np.timedelta64(15, "m")
    ranges = 2 if support is None else support.shape[1]
    values = np.full((96, 1, ranges), 1e-6)
    return aggregate_calibrated_pings(
        ping_times=times.astype("datetime64[ns]"),
        sv_linear=values,
        valid_mask=np.ones(values.shape, dtype=bool),
        frequency_hz=np.array([38000]),
        range_edges_m=np.arange(ranges + 1, dtype=float) * 2,
        supported_range=np.ones((1, ranges), dtype=bool) if support is None else support,
        configuration_ids=[_digest("xml-a")] * len(times),
        source_file_ids=[_digest(source)] * len(times),
        dataset_id="synthetic-calibrated",
        deployment_id="synthetic-deployment",
        instrument_id="synthetic-instrument",
        calibration_status="VERIFIED_PHYSICAL_SV",
        calibration_report_sha256=_CALIBRATION_SHA256,
        config_sha256="c" * 64,
        split_sha256=split_bounds_sha256(_bounds()),
        processing_sha256="d" * 64,
    )


def _bounds():
    return {
        "train": ("2020-01-01T00:00:00Z", "2020-01-04T00:00:00Z"),
        "validation": ("2020-01-04T00:00:00Z", "2020-01-07T00:00:00Z"),
        "calibration": ("2020-01-07T00:00:00Z", "2020-01-10T00:00:00Z"),
        "test": ("2020-01-10T00:00:00Z", "2020-01-13T00:00:00Z"),
    }


def _identity(source_ids: tuple[str, ...], bounds=None):
    return StoreIdentity(
        dataset_id="synthetic-calibrated",
        deployment_id="synthetic-deployment",
        instrument_id="synthetic-instrument",
        source_inventory_sha256="b" * 64,
        source_file_ids=tuple(_digest(value) for value in source_ids),
        config_sha256="c" * 64,
        calibration_report_sha256=_CALIBRATION_SHA256,
        processing_sha256="d" * 64,
        split_sha256=split_bounds_sha256(_bounds() if bounds is None else bounds),
    )


def test_day_shard_resume_checksum_and_no_pickle(tmp_path: Path) -> None:
    store = CanonicalStore(
        tmp_path / "canonical",
        _identity(("train-a",)),
        calibration_manifest=_calibration_path(tmp_path),
    )
    day = _series("2020-01-01")
    assert store.write_day(day, partition="train") == "WRITTEN"
    shard = tmp_path / "canonical/shards/2020-01-01.npz"
    with np.load(shard, allow_pickle=False) as loaded:
        assert loaded["sv_linear"].dtype.kind == "f"
        assert loaded["source_file_ids"].dtype.kind == "U"
    resumed = CanonicalStore(
        tmp_path / "canonical",
        _identity(("train-a",)),
        calibration_manifest=tmp_path / "synthetic-calibration.json",
    )
    assert resumed.write_day(day, partition="train") == "SKIPPED_VERIFIED"
    changed_values = day.sv_linear.copy()
    changed_db = day.sv_db.copy()
    changed_values[0, 0, 0] = 2e-6
    changed_db[0, 0, 0] = 10 * np.log10(2e-6)
    with pytest.raises(ValueError, match="resume input"):
        resumed.write_day(
            replace(day, sv_linear=changed_values, sv_db=changed_db), partition="train"
        )
    assert store.load_day_metadata("2020-01-01").valid_mask.shape == (96, 1, 2)
    np.testing.assert_array_equal(store.load_day("2020-01-01").sv_linear, day.sv_linear)
    shard.write_bytes(shard.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        store.write_day(day, partition="train")


def test_resume_rejects_identity_change_and_orphan_shard(tmp_path: Path) -> None:
    root = tmp_path / "canonical"
    calibration_path = _calibration_path(tmp_path)
    store = CanonicalStore(root, _identity(("train-a",)), calibration_manifest=calibration_path)
    store.write_day(_series("2020-01-01"), partition="train")
    with pytest.raises(ValueError, match="identity"):
        CanonicalStore(
            root,
            replace(_identity(("train-a",)), processing_sha256="e" * 64),
            calibration_manifest=calibration_path,
        )
    calibration_path.write_text('{"calibration_status":"RAW_COUNTS_ONLY"}', encoding="utf-8")
    with pytest.raises(ValueError, match="calibration manifest"):
        CanonicalStore(root, _identity(("train-a",)), calibration_manifest=calibration_path)
    calibration_path.write_bytes(_CALIBRATION_BYTES)
    orphan = root / "shards/2020-01-02.npz"
    orphan.write_bytes(b"do not overwrite")
    with pytest.raises(ValueError, match="unregistered"):
        store.write_day(_series("2020-01-02"), partition="train")
    assert orphan.read_bytes() == b"do not overwrite"


def test_store_rejects_corrupted_series_and_wrong_source(tmp_path: Path) -> None:
    store = CanonicalStore(
        tmp_path / "canonical",
        _identity(("train-a",)),
        calibration_manifest=_calibration_path(tmp_path),
    )
    series = _series("2020-01-01")
    with pytest.raises(ValueError, match="shape"):
        store.write_day(
            replace(series, valid_mask=np.ones((1, 1, 2), dtype=bool)),
            partition="train",
        )
    with pytest.raises(ValueError, match="bin interval"):
        store.write_day(
            replace(series, bin_end=series.bin_end + np.timedelta64(1, "m")),
            partition="train",
        )
    with pytest.raises(ValueError, match="source"):
        store.write_day(_series("2020-01-01", "unregistered"), partition="train")
    with pytest.raises(ValueError, match="bin counts"):
        altered_counts = series.ping_count.copy()
        altered_counts[0] = 0
        altered_counts[1] = 2
        store.write_day(replace(series, ping_count=altered_counts), partition="train")
    with pytest.raises(ValueError, match="integer"):
        store.write_day(
            replace(series, ping_count=series.ping_count.astype(float)), partition="train"
        )


def test_audit_uses_test_masks_only_and_keeps_frozen_minima(tmp_path: Path, monkeypatch) -> None:
    bounds = _bounds()
    sources = tuple(f"{part}-{day}" for part in bounds for day in range(3))
    store = CanonicalStore(
        tmp_path / "canonical",
        _identity(sources, bounds),
        calibration_manifest=_calibration_path(tmp_path),
    )
    for part, (start, _) in bounds.items():
        first = np.datetime64(start[:10], "D")
        for offset in range(3):
            day = str(first + np.timedelta64(offset, "D"))
            store.write_day(_series(day, f"{part}-{offset}"), partition=part)

    def forbidden_values(_day):
        raise AssertionError("Audit must not load protected acoustic values")

    monkeypatch.setattr(store, "load_day", forbidden_values)
    with monkeypatch.context() as guard:
        original_getitem = np.lib.npyio.NpzFile.__getitem__

        def metadata_only(archive, key):
            if key in ("sv_linear", "sv_db"):
                raise AssertionError("Audit opened protected acoustic outcomes")
            return original_getitem(archive, key)

        guard.setattr(np.lib.npyio.NpzFile, "__getitem__", metadata_only)
        result = audit_eligibility(
            store,
            bounds,
            analysis_frequency_hz=38000,
            analysis_range_m=(0.0, 4.0),
            availability_basis="zero_latency_replay",
        )
    assert result["status"] == "INELIGIBLE"
    assert result["required_days"] == {"overall": 90, "calibration": 12, "test": 20}
    assert result["before_qc_days"] == {part: 3 for part in bounds}
    assert result["after_qc_days"]["train"] >= 1
    assert result["protected_partition_inspection"] == "timestamps_masks_provenance_only"
    assert result["test_outcomes_opened"] is False
    assert result["test_metadata_opened"] is True
    assert result["availability_basis"] == "zero_latency_replay"
    assert result["availability_claim"] == "simulation_only"
    assert (tmp_path / "canonical" / result["exposure_record"]).is_file()
    with pytest.raises(ValueError, match="Protected test outcomes"):
        CanonicalStore.load_day(store, "2020-01-10")


def test_audit_rejects_cross_partition_source_membership(tmp_path: Path) -> None:
    bounds = _bounds()
    store = CanonicalStore(
        tmp_path / "canonical",
        _identity(("shared",), bounds),
        calibration_manifest=_calibration_path(tmp_path),
    )
    store.write_day(_series("2020-01-01", "shared"), partition="train")
    store.write_day(_series("2020-01-10", "shared"), partition="test")
    with pytest.raises(ValueError, match="across partitions"):
        audit_eligibility(
            store,
            bounds,
            analysis_frequency_hz=38000,
            analysis_range_m=(0.0, 4.0),
            availability_basis="zero_latency_replay",
        )


def test_original_band_denominator_and_off_grid_band() -> None:
    # One fifth of the primary band's original physical width is unsupported.
    support = np.array([[True, True, True, True, False]])
    day = _series("2020-01-01", support=support)
    next_day = _series("2020-01-02", support=support)
    # Two day shards are joined here only to exercise the same window math.
    joined = replace(
        day,
        bin_start=np.concatenate([day.bin_start, next_day.bin_start]),
        bin_end=np.concatenate([day.bin_end, next_day.bin_end]),
        observed_available_time=np.concatenate(
            [day.observed_available_time, next_day.observed_available_time]
        ),
        replay_available_time=np.concatenate(
            [day.replay_available_time, next_day.replay_available_time]
        ),
        sv_linear=np.concatenate([day.sv_linear, next_day.sv_linear]),
        sv_db=np.concatenate([day.sv_db, next_day.sv_db]),
        valid_mask=np.concatenate([day.valid_mask, next_day.valid_mask]),
        ping_count=np.concatenate([day.ping_count, next_day.ping_count]),
        configuration_ids=day.configuration_ids + next_day.configuration_ids,
        configuration_boundary=np.concatenate(
            [day.configuration_boundary, next_day.configuration_boundary]
        ),
        source_file_ids=day.source_file_ids + next_day.source_file_ids,
        raw_ping_times=np.concatenate([day.raw_ping_times, next_day.raw_ping_times]),
    )
    args = {
        "cutoff": np.datetime64("2020-01-02T00:00"),
        "split_start": np.datetime64("2020-01-01T00:00"),
        "split_end": np.datetime64("2020-01-03T00:00"),
        "partition": "train",
        "allowed_source_file_ids": {_digest("train-a")},
        "expected_dataset_id": "synthetic-calibrated",
        "expected_deployment_id": "synthetic-deployment",
        "expected_instrument_id": "synthetic-instrument",
        "expected_calibration_report_sha256": _CALIBRATION_SHA256,
        "expected_config_sha256": "c" * 64,
        "expected_split_sha256": split_bounds_sha256(_bounds()),
        "expected_processing_sha256": "d" * 64,
        "analysis_frequency_hz": 38000,
        "analysis_range_m": (0.0, 10.0),
        "availability_basis": "zero_latency_replay",
    }
    assert build_window(joined, **args).target_support[0] == pytest.approx(0.8)
    with pytest.raises(ValueError, match="exact range edges"):
        build_window(joined, **(args | {"analysis_range_m": (1.0, 10.0)}))


def test_split_gap_and_pruned_shard_fail_closed(tmp_path: Path) -> None:
    bounds = _bounds()
    gap = bounds | {"validation": ("2020-01-05T00:00:00Z", bounds["validation"][1])}
    with pytest.raises(ValueError, match="adjacent"):
        split_bounds_sha256(gap)
    store = CanonicalStore(
        tmp_path / "canonical",
        _identity(("train-a",), bounds),
        calibration_manifest=_calibration_path(tmp_path),
    )
    store.write_day(_series("2020-01-01"), partition="train")
    manifest_path = tmp_path / "canonical/processing_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shards"].clear()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unregistered"):
        audit_eligibility(
            store,
            bounds,
            analysis_frequency_hz=38000,
            analysis_range_m=(0.0, 4.0),
            availability_basis="zero_latency_replay",
        )


def test_test_metadata_exposure_is_durable_before_loading(tmp_path: Path, monkeypatch) -> None:
    bounds = _bounds()
    store = CanonicalStore(
        tmp_path / "canonical",
        _identity(("test-a",), bounds),
        calibration_manifest=_calibration_path(tmp_path),
    )
    store.write_day(_series("2020-01-10", "test-a"), partition="test")
    original = store.load_day_metadata

    def inspect(day: str, **kwargs):
        if day == "2020-01-10":
            records = list((tmp_path / "canonical/exposure").glob("*.json"))
            assert len(records) == 1
            evidence = json.loads(records[0].read_text(encoding="utf-8"))
            assert evidence["event"] == "TEST_METADATA_MASKS_INSPECTED"
            assert (
                evidence["processing_manifest_sha256"]
                == hashlib.sha256(
                    (tmp_path / "canonical/processing_manifest.json").read_bytes()
                ).hexdigest()
            )
            assert evidence["test_acoustic_values_opened"] is False
            assert evidence["fields"] == ["timestamps", "masks", "availability", "provenance"]
        return original(day, **kwargs)

    monkeypatch.setattr(store, "load_day_metadata", inspect)
    result = audit_eligibility(
        store,
        bounds,
        analysis_frequency_hz=38000,
        analysis_range_m=(0.0, 4.0),
        availability_basis="zero_latency_replay",
    )
    assert result["test_metadata_opened"] is True
    assert len(list((tmp_path / "canonical/exposure").glob("*.json"))) == 1
    monkeypatch.setattr(store, "load_day_metadata", original)
    audit_eligibility(
        store,
        bounds,
        analysis_frequency_hz=38000,
        analysis_range_m=(0.0, 4.0),
        availability_basis="measured",
    )
    assert len(list((tmp_path / "canonical/exposure").glob("*.json"))) == 2


def test_test_metadata_read_refuses_unbound_exposure_record(tmp_path: Path) -> None:
    bounds = _bounds()
    store = CanonicalStore(
        tmp_path / "canonical",
        _identity(("test-a",), bounds),
        calibration_manifest=_calibration_path(tmp_path),
    )
    store.write_day(_series("2020-01-10", "test-a"), partition="test")
    foreign = tmp_path / "foreign.json"
    foreign.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="exposure record"):
        store.load_day_metadata(
            "2020-01-10",
            exposure_record=foreign,
            availability_basis="zero_latency_replay",
        )
    assert not (tmp_path / "canonical/exposure").exists()


def test_test_metadata_read_aborts_when_exposure_cannot_be_published(tmp_path: Path) -> None:
    bounds = _bounds()
    store = CanonicalStore(
        tmp_path / "canonical",
        _identity(("test-a",), bounds),
        calibration_manifest=_calibration_path(tmp_path),
    )
    store.write_day(_series("2020-01-10", "test-a"), partition="test")
    (tmp_path / "canonical/exposure").write_text("blocked", encoding="utf-8")
    with pytest.raises(FileExistsError):
        store.load_day_metadata("2020-01-10")


def test_test_metadata_exposure_rejects_linked_container(tmp_path: Path, monkeypatch) -> None:
    store = CanonicalStore(
        tmp_path / "canonical",
        _identity(("test-a",)),
        calibration_manifest=_calibration_path(tmp_path),
    )
    store.write_day(_series("2020-01-10", "test-a"), partition="test")
    container = tmp_path / "canonical/exposure"
    container.mkdir()
    original = Path.is_junction
    monkeypatch.setattr(Path, "is_junction", lambda path: path == container or original(path))
    with pytest.raises(ValueError, match="exposure container"):
        store.load_day_metadata("2020-01-10")
    assert list(container.iterdir()) == []


def test_test_metadata_exposure_reuse_requires_exact_schema(tmp_path: Path) -> None:
    store = CanonicalStore(
        tmp_path / "canonical",
        _identity(("test-a",)),
        calibration_manifest=_calibration_path(tmp_path),
    )
    store.write_day(_series("2020-01-10", "test-a"), partition="test")
    record = store._record_test_metadata_exposure(store._manifest(), availability_basis="measured")
    contents = json.loads(record.read_text(encoding="utf-8"))
    contents["fields"] = ["timestamps"]
    record.write_text(json.dumps(contents), encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind"):
        store.load_day_metadata("2020-01-10", exposure_record=record, availability_basis="measured")


def test_unknown_measured_availability_cannot_be_eligible(tmp_path: Path) -> None:
    bounds = _bounds()
    sources = tuple(f"{part}-{day}" for part in bounds for day in range(3))
    store = CanonicalStore(
        tmp_path / "canonical",
        _identity(sources, bounds),
        calibration_manifest=_calibration_path(tmp_path),
    )
    for part, (start, _) in bounds.items():
        first = np.datetime64(start[:10], "D")
        for offset in range(3):
            store.write_day(
                _series(str(first + np.timedelta64(offset, "D")), f"{part}-{offset}"),
                partition=part,
            )
    result = audit_eligibility(
        store,
        bounds,
        analysis_frequency_hz=38000,
        analysis_range_m=(0.0, 4.0),
        availability_basis="measured",
    )
    assert result["after_qc_days"] == {part: 0 for part in bounds}
    assert result["availability_claim"] == "observed_only"
