"""Immutable day shards and a protected, metadata-only eligibility audit."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from marine_echo.data.preprocessing import BinnedAcoustics, valid_sha256

PARTITIONS = ("train", "validation", "calibration", "test")
REQUIRED_DAYS = {"overall": 90, "calibration": 12, "test": 20}
PROMOTION_STATUS = "NONPROMOTABLE_ENGINEERING_FIXTURE"
_UTC_MIDNIGHT = re.compile(r"^\d{4}-\d{2}-\d{2}T00:00:00Z$")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def split_bounds_sha256(bounds: dict[str, tuple[str, str]]) -> str:
    if set(bounds) != set(PARTITIONS):
        raise ValueError("Split bounds must declare all four partitions.")
    previous_end: datetime | None = None
    for partition in PARTITIONS:
        start, end = bounds[partition]
        if not _UTC_MIDNIGHT.fullmatch(start) or not _UTC_MIDNIGHT.fullmatch(end):
            raise ValueError("Split boundaries require canonical UTC-midnight calendar days.")
        start_time = datetime.fromisoformat(start)
        end_time = datetime.fromisoformat(end)
        if start_time >= end_time or (previous_end is not None and start_time != previous_end):
            raise ValueError("Chronological split boundaries must be exactly adjacent.")
        previous_end = end_time
    return _sha256_bytes(_json_bytes(bounds))


def _identity_document(identity: StoreIdentity) -> dict[str, Any]:
    return json.loads(_json_bytes(asdict(identity)))


@dataclass(frozen=True)
class StoreIdentity:
    dataset_id: str
    deployment_id: str
    instrument_id: str
    source_inventory_sha256: str
    source_file_ids: tuple[str, ...]
    config_sha256: str
    calibration_report_sha256: str
    processing_sha256: str
    split_sha256: str

    def validate(self) -> None:
        if not all((self.dataset_id, self.deployment_id, self.instrument_id)):
            raise ValueError("Physical dataset, deployment and instrument identities are required.")
        digests = (
            self.source_inventory_sha256,
            self.config_sha256,
            self.calibration_report_sha256,
            self.processing_sha256,
            self.split_sha256,
        )
        if not all(valid_sha256(value) for value in digests):
            raise ValueError("Store provenance requires SHA-256 identities.")
        if not self.source_file_ids or any(not valid_sha256(x) for x in self.source_file_ids):
            raise ValueError("Registered source files require SHA-256 identities.")
        if len(set(self.source_file_ids)) != len(self.source_file_ids):
            raise ValueError("Registered source identities must be unique.")


@dataclass(frozen=True)
class DayMetadata:
    bin_start: NDArray[np.datetime64]
    bin_end: NDArray[np.datetime64]
    observed_available_time: NDArray[np.datetime64]
    replay_available_time: NDArray[np.datetime64]
    valid_mask: NDArray[np.bool_]
    ping_count: NDArray[np.int64]
    frequency_hz: NDArray[np.int64]
    range_edges_m: NDArray[np.float64]
    supported_range: NDArray[np.bool_]
    configuration_ids: tuple[str | None, ...]
    source_file_ids: tuple[frozenset[str], ...]
    raw_ping_times: NDArray[np.datetime64]


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    descriptor, staging_name = tempfile.mkstemp(prefix=path.name + ".stage.", dir=path.parent)
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(document))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def _series_digest(series: BinnedAcoustics) -> str:
    digest = hashlib.sha256()
    arrays = (
        series.bin_start,
        series.bin_end,
        series.observed_available_time,
        series.replay_available_time,
        series.sv_linear,
        series.sv_db,
        series.valid_mask,
        series.ping_count,
        series.frequency_hz,
        series.range_edges_m,
        series.supported_range,
        series.configuration_boundary,
        series.raw_ping_times,
    )
    for array in arrays:
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(
        _json_bytes(
            {
                "configurations": series.configuration_ids,
                "sources": [sorted(source) for source in series.source_file_ids],
                "identities": {
                    "dataset_id": series.dataset_id,
                    "deployment_id": series.deployment_id,
                    "instrument_id": series.instrument_id,
                    "calibration_report_sha256": series.calibration_report_sha256,
                    "config_sha256": series.config_sha256,
                    "split_sha256": series.split_sha256,
                    "processing_sha256": series.processing_sha256,
                    "availability_policy": series.availability_policy,
                },
            }
        )
    )
    return digest.hexdigest()


class CanonicalStore:
    """One-writer immutable shard catalog; resume validates both input and stored bytes."""

    def __init__(self, root: Path, identity: StoreIdentity, *, calibration_manifest: Path) -> None:
        identity.validate()
        if (
            not calibration_manifest.is_file()
            or _sha256_file(calibration_manifest) != identity.calibration_report_sha256
        ):
            raise ValueError(
                "Verified calibration manifest file and SHA-256 identity are required."
            )
        calibration = json.loads(calibration_manifest.read_text(encoding="utf-8"))
        expected = {
            "calibration_status": "VERIFIED_PHYSICAL_SV",
            "dataset_id": identity.dataset_id,
            "deployment_id": identity.deployment_id,
            "instrument_id": identity.instrument_id,
            "source_inventory_sha256": identity.source_inventory_sha256,
            "config_sha256": identity.config_sha256,
        }
        if any(calibration.get(key) != value for key, value in expected.items()):
            raise ValueError("Calibration manifest physical status or provenance differs.")
        if calibration.get("scope") != "synthetic-fixture-only":
            raise ValueError(
                "This nonpromotable store accepts synthetic calibration fixtures only."
            )
        self.root = root
        self.identity = identity
        self.calibration_manifest = calibration_manifest
        self.manifest_path = root / "processing_manifest.json"
        self.shards = root / "shards"
        if root.exists() and not self.manifest_path.is_file() and any(root.iterdir()):
            raise FileExistsError(
                "Unregistered existing canonical store; no user artifacts overwritten."
            )
        self.shards.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            current = self._manifest()
            if current.get("identity") != _identity_document(identity):
                raise ValueError("Canonical store identity differs on resume.")
        else:
            _atomic_json(
                self.manifest_path,
                {
                    "schema_version": "1.0",
                    "promotion_status": PROMOTION_STATUS,
                    "identity": _identity_document(identity),
                    "shards": {},
                },
            )

    def _manifest(self) -> dict[str, Any]:
        if _sha256_file(self.calibration_manifest) != self.identity.calibration_report_sha256:
            raise ValueError("Calibration manifest changed after store initialization.")
        document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if (
            document.get("schema_version") != "1.0"
            or document.get("promotion_status") != PROMOTION_STATUS
            or document.get("identity") != _identity_document(self.identity)
            or not isinstance(document.get("shards"), dict)
        ):
            raise ValueError("Canonical manifest schema or identity is invalid.")
        if self.shards.is_symlink() or not self.shards.is_dir():
            raise ValueError("Canonical shard directory is missing or linked.")
        expected = {f"{day}.npz" for day in document["shards"]}
        observed = set()
        for path in self.shards.iterdir():
            if path.is_symlink() or not path.is_file() or path.suffix != ".npz":
                raise ValueError("Canonical shard directory contains a linked or unexpected entry.")
            observed.add(path.name)
        if observed != expected:
            raise ValueError("Canonical shard directory has missing or unregistered files.")
        return document

    def _validate_series(self, series: BinnedAcoustics) -> str:
        identity = self.identity
        if (
            series.dataset_id != identity.dataset_id
            or series.deployment_id != identity.deployment_id
            or series.instrument_id != identity.instrument_id
            or series.calibration_report_sha256 != identity.calibration_report_sha256
            or series.config_sha256 != identity.config_sha256
            or series.split_sha256 != identity.split_sha256
            or series.processing_sha256 != identity.processing_sha256
        ):
            raise ValueError("Calibrated series identity differs from the store identity.")
        n = len(series.bin_start)
        if (
            not n
            or len(series.bin_end) != n
            or len(series.observed_available_time) != n
            or len(series.replay_available_time) != n
            or len(series.ping_count) != n
            or len(series.configuration_ids) != n
            or len(series.configuration_boundary) != n
            or len(series.source_file_ids) != n
            or series.sv_linear.shape != series.valid_mask.shape
            or series.sv_linear.shape != series.sv_db.shape
            or series.sv_linear.shape[0] != n
            or series.supported_range.shape != series.sv_linear.shape[1:]
            or series.frequency_hz.shape != (series.sv_linear.shape[1],)
            or series.range_edges_m.shape != (series.sv_linear.shape[2] + 1,)
        ):
            raise ValueError("Canonical series shape is inconsistent.")
        if (
            np.isnat(series.bin_start).any()
            or np.isnat(series.bin_end).any()
            or np.any(series.bin_end - series.bin_start != np.timedelta64(15, "m"))
            or not np.array_equal(series.bin_start[1:], series.bin_end[:-1])
        ):
            raise ValueError("Canonical bin interval is invalid or discontinuous.")
        days = np.unique(series.bin_start.astype("datetime64[D]"))
        if len(days) != 1 or series.bin_end[-1] > days[0] + np.timedelta64(1, "D"):
            raise ValueError("One shard must stay inside one UTC calendar day.")
        day = str(days[0])
        if series.ping_count.dtype.kind not in "iu" or (series.ping_count < 0).any():
            raise ValueError("Ping counts must be nonnegative integers.")
        if (
            len(series.raw_ping_times) != int(series.ping_count.sum())
            or np.isnat(series.raw_ping_times).any()
            or (np.diff(series.raw_ping_times) <= np.timedelta64(0, "ns")).any()
            or (series.raw_ping_times < series.bin_start[0]).any()
            or (series.raw_ping_times >= series.bin_end[-1]).any()
        ):
            raise ValueError("Raw ping timestamp provenance is inconsistent.")
        bin_index = np.searchsorted(series.bin_end, series.raw_ping_times, side="right")
        if (
            (bin_index >= n).any()
            or not np.array_equal(np.bincount(bin_index, minlength=n), series.ping_count)
            or (series.raw_ping_times < series.bin_start[bin_index]).any()
        ):
            raise ValueError("Raw ping timestamps do not match the canonical bin counts.")
        if (
            series.frequency_hz.dtype.kind not in "iu"
            or (series.frequency_hz <= 0).any()
            or len(np.unique(series.frequency_hz)) != len(series.frequency_hz)
            or not np.isfinite(series.range_edges_m).all()
            or (np.diff(series.range_edges_m) <= 0).any()
            or series.supported_range.dtype.kind != "b"
            or series.configuration_boundary.dtype.kind != "b"
            or any(x is not None and not valid_sha256(x) for x in series.configuration_ids)
        ):
            raise ValueError("Canonical frequency, range or configuration metadata is invalid.")
        for index in range(n):
            if series.ping_count[index] == 0:
                if (
                    not np.isnat(series.observed_available_time[index])
                    or not np.isnat(series.replay_available_time[index])
                    or series.source_file_ids[index]
                ):
                    raise ValueError("Empty bin availability or source metadata is inconsistent.")
            else:
                if series.replay_available_time[index] != series.bin_end[index]:
                    raise ValueError("Replay availability must equal the trailing bin end.")
                observed = series.observed_available_time[index]
                if (
                    not np.isnat(observed)
                    and observed < series.raw_ping_times[bin_index == index].max()
                ):
                    raise ValueError(
                        "Measured availability precedes acquisition in a populated bin."
                    )
        if not np.array_equal(
            series.configuration_boundary[1:],
            np.array(
                [
                    series.configuration_ids[i] != series.configuration_ids[i - 1]
                    for i in range(1, n)
                ],
                dtype=bool,
            ),
        ) or bool(series.configuration_boundary[0]):
            raise ValueError("Configuration boundary flags differ from configuration identities.")
        source_ids = set().union(*series.source_file_ids)
        if (
            not source_ids
            or any(not valid_sha256(x) for x in source_ids)
            or not source_ids.issubset(set(identity.source_file_ids))
        ):
            raise ValueError("Series source identities are not registered.")
        if (
            series.valid_mask.dtype.kind != "b"
            or not np.isfinite(series.sv_linear[series.valid_mask]).all()
            or (series.sv_linear[series.valid_mask] < 0).any()
            or np.any(series.valid_mask & ~series.supported_range[None, :, :])
            or not np.allclose(
                series.sv_db[series.valid_mask],
                10.0 * np.log10(np.maximum(series.sv_linear[series.valid_mask], 1e-18)),
            )
            or not np.isnan(series.sv_linear[~series.valid_mask]).all()
            or not np.isnan(series.sv_db[~series.valid_mask]).all()
        ):
            raise ValueError("Canonical physical values or frequency support are inconsistent.")
        return day

    def _verified_entry(self, day: str, entry: dict[str, Any]) -> Path:
        path = self.shards / f"{day}.npz"
        if not path.is_file() or _sha256_file(path) != entry.get("sha256"):
            raise ValueError("Stored day shard checksum differs from the manifest.")
        return path

    def write_day(self, series: BinnedAcoustics, *, partition: str) -> str:
        if partition not in PARTITIONS:
            raise ValueError("A declared partition is required.")
        day = self._validate_series(series)
        candidate_digest = _series_digest(series)
        document = self._manifest()
        old = document["shards"].get(day)
        if old is not None:
            self._verified_entry(day, old)
            if old.get("partition") != partition or old.get("input_sha256") != candidate_digest:
                raise ValueError("Existing day shard differs from the requested resume input.")
            return "SKIPPED_VERIFIED"
        path = self.shards / f"{day}.npz"
        if path.exists() or path.is_symlink():
            raise FileExistsError("An unregistered shard exists; no user artifacts overwritten.")
        descriptor, staging_name = tempfile.mkstemp(prefix=f"{day}.stage.", dir=self.shards)
        staging = Path(staging_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                np.savez_compressed(
                    stream,
                    bin_start=series.bin_start,
                    bin_end=series.bin_end,
                    observed_available_time=series.observed_available_time,
                    replay_available_time=series.replay_available_time,
                    sv_linear=series.sv_linear,
                    sv_db=series.sv_db,
                    valid_mask=series.valid_mask,
                    ping_count=series.ping_count,
                    frequency_hz=series.frequency_hz,
                    range_edges_m=series.range_edges_m,
                    supported_range=series.supported_range,
                    configuration_ids=np.array([x or "" for x in series.configuration_ids]),
                    configuration_boundary=series.configuration_boundary,
                    source_file_ids=np.array(
                        [json.dumps(sorted(x)) for x in series.source_file_ids]
                    ),
                    raw_ping_times=series.raw_ping_times,
                    availability_policy=np.array(series.availability_policy),
                    identity_sha256=np.array(_sha256_bytes(_json_bytes(asdict(self.identity)))),
                )
                stream.flush()
                os.fsync(stream.fileno())
            if path.exists() or path.is_symlink():
                raise FileExistsError(
                    "Day shard appeared during publication; no overwrite allowed."
                )
            os.replace(staging, path)
        finally:
            staging.unlink(missing_ok=True)
        document["shards"][day] = {
            "partition": partition,
            "sha256": _sha256_file(path),
            "input_sha256": candidate_digest,
            "bins": len(series.bin_start),
            "raw_pings": len(series.raw_ping_times),
            "source_file_ids": sorted(set().union(*series.source_file_ids)),
        }
        _atomic_json(self.manifest_path, document)
        return "WRITTEN"

    def _record_test_metadata_exposure(
        self, manifest: dict[str, Any], *, availability_basis: str
    ) -> Path:
        directory = self.root / "exposure"
        directory.mkdir(parents=True, exist_ok=True)
        if directory.is_symlink() or directory.is_junction() or not directory.is_dir():
            raise ValueError("Test metadata exposure container is linked or invalid.")
        path = directory / f"test-metadata-{uuid.uuid4().hex}.json"
        record = {
            "schema_version": "1.0",
            "event": "TEST_METADATA_MASKS_INSPECTED",
            "recorded_before_npz_access": True,
            "recorded_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "processing_manifest_sha256": _sha256_file(self.manifest_path),
            "split_sha256": self.identity.split_sha256,
            "test_shard_sha256": {
                day: entry["sha256"]
                for day, entry in sorted(manifest["shards"].items())
                if entry["partition"] == "test"
            },
            "fields": ["timestamps", "masks", "availability", "provenance"],
            "availability_basis": availability_basis,
            "test_metadata_opened": True,
            "test_acoustic_values_opened": False,
            "promotion_status": PROMOTION_STATUS,
        }
        with path.open("xb") as stream:
            stream.write(_json_bytes(record))
            stream.flush()
            os.fsync(stream.fileno())
        return path

    def load_day_metadata(
        self,
        day: str,
        *,
        exposure_record: Path | None = None,
        availability_basis: str = "unspecified_direct_metadata_read",
    ) -> DayMetadata:
        manifest = self._manifest()
        entry = manifest["shards"].get(day)
        if entry is None:
            raise FileNotFoundError("Day is not registered in the canonical manifest.")
        if entry["partition"] == "test":
            if exposure_record is None:
                exposure_record = self._record_test_metadata_exposure(
                    manifest, availability_basis=availability_basis
                )
            else:
                if (
                    exposure_record.parent != self.root / "exposure"
                    or exposure_record.parent.is_symlink()
                    or exposure_record.parent.is_junction()
                    or exposure_record.is_symlink()
                    or exposure_record.is_junction()
                    or not exposure_record.is_file()
                ):
                    raise ValueError("Durable test metadata exposure record is missing.")
                recorded = json.loads(exposure_record.read_text(encoding="utf-8"))
                expected_shards = {
                    name: shard["sha256"]
                    for name, shard in sorted(manifest["shards"].items())
                    if shard["partition"] == "test"
                }
                if (
                    recorded.get("schema_version") != "1.0"
                    or recorded.get("event") != "TEST_METADATA_MASKS_INSPECTED"
                    or recorded.get("recorded_before_npz_access") is not True
                    or recorded.get("processing_manifest_sha256")
                    != _sha256_file(self.manifest_path)
                    or recorded.get("split_sha256") != self.identity.split_sha256
                    or recorded.get("test_shard_sha256") != expected_shards
                    or recorded.get("fields")
                    != ["timestamps", "masks", "availability", "provenance"]
                    or recorded.get("availability_basis") != availability_basis
                    or recorded.get("test_metadata_opened") is not True
                    or recorded.get("test_acoustic_values_opened") is not False
                    or recorded.get("promotion_status") != PROMOTION_STATUS
                ):
                    raise ValueError("Test metadata exposure record does not bind this audit.")
        path = self._verified_entry(day, entry)
        with np.load(path, allow_pickle=False) as loaded:
            if str(loaded["identity_sha256"]) != _sha256_bytes(_json_bytes(asdict(self.identity))):
                raise ValueError("Shard identity digest differs from the manifest.")
            # Deliberately omit sv_linear and sv_db. Test partition acoustic
            # outcomes remain unopened during eligibility auditing.
            return DayMetadata(
                bin_start=loaded["bin_start"],
                bin_end=loaded["bin_end"],
                observed_available_time=loaded["observed_available_time"],
                replay_available_time=loaded["replay_available_time"],
                valid_mask=loaded["valid_mask"],
                ping_count=loaded["ping_count"],
                frequency_hz=loaded["frequency_hz"],
                range_edges_m=loaded["range_edges_m"],
                supported_range=loaded["supported_range"],
                configuration_ids=tuple(x or None for x in loaded["configuration_ids"]),
                source_file_ids=tuple(frozenset(json.loads(x)) for x in loaded["source_file_ids"]),
                raw_ping_times=loaded["raw_ping_times"],
            )

    def load_day(self, day: str) -> BinnedAcoustics:
        entry = self._manifest()["shards"].get(day)
        if entry is None:
            raise FileNotFoundError("Day is not registered in the canonical manifest.")
        if entry["partition"] == "test":
            raise ValueError("Protected test outcomes require a separately reviewed R2 path.")
        meta = self.load_day_metadata(day)
        path = self._verified_entry(day, entry)
        with np.load(path, allow_pickle=False) as loaded:
            return BinnedAcoustics(
                bin_start=meta.bin_start,
                bin_end=meta.bin_end,
                observed_available_time=meta.observed_available_time,
                replay_available_time=meta.replay_available_time,
                sv_linear=loaded["sv_linear"],
                sv_db=loaded["sv_db"],
                valid_mask=meta.valid_mask,
                ping_count=meta.ping_count,
                frequency_hz=meta.frequency_hz,
                range_edges_m=meta.range_edges_m,
                supported_range=meta.supported_range,
                configuration_ids=meta.configuration_ids,
                configuration_boundary=loaded["configuration_boundary"],
                source_file_ids=meta.source_file_ids,
                raw_ping_times=meta.raw_ping_times,
                dataset_id=self.identity.dataset_id,
                deployment_id=self.identity.deployment_id,
                instrument_id=self.identity.instrument_id,
                calibration_report_sha256=self.identity.calibration_report_sha256,
                config_sha256=self.identity.config_sha256,
                split_sha256=self.identity.split_sha256,
                processing_sha256=self.identity.processing_sha256,
                availability_policy=str(loaded["availability_policy"]),
            )


def audit_eligibility(
    store: CanonicalStore,
    bounds: dict[str, tuple[str, str]],
    *,
    analysis_frequency_hz: int,
    analysis_range_m: tuple[float, float],
    availability_basis: str,
) -> dict[str, Any]:
    """Audit masks and provenance across partitions without loading Sv values."""
    if availability_basis not in ("measured", "zero_latency_replay"):
        raise ValueError("Availability basis must be measured or zero_latency_replay.")
    if split_bounds_sha256(bounds) != store.identity.split_sha256:
        raise ValueError("Split manifest digest differs from the canonical store.")
    split_times = {
        part: (np.datetime64(start[:-1], "ns"), np.datetime64(end[:-1], "ns"))
        for part, (start, end) in bounds.items()
    }
    previous_end: np.datetime64 | None = None
    for part in PARTITIONS:
        start, end = split_times[part]
        if (
            np.isnat(start)
            or np.isnat(end)
            or start >= end
            or (previous_end is not None and start != previous_end)
        ):
            raise ValueError("Chronological split boundaries overlap or are invalid.")
        previous_end = end
    manifest = store._manifest()
    entries = manifest["shards"]
    exposure_record = (
        store._record_test_metadata_exposure(manifest, availability_basis=availability_basis)
        if any(entry["partition"] == "test" for entry in entries.values())
        else None
    )
    grouped: dict[str, list[DayMetadata]] = {part: [] for part in PARTITIONS}
    source_partition: dict[str, str] = {}
    raw_time_partition: dict[int, str] = {}
    bin_time_partition: dict[int, str] = {}
    before_qc = {part: 0 for part in PARTITIONS}
    reference_frequency: NDArray[np.int64] | None = None
    reference_edges: NDArray[np.float64] | None = None
    for day, entry in sorted(entries.items()):
        part = entry.get("partition")
        if part not in PARTITIONS:
            raise ValueError("Shard partition is invalid.")
        meta = store.load_day_metadata(
            day, exposure_record=exposure_record, availability_basis=availability_basis
        )
        if (
            str(meta.bin_start[0].astype("datetime64[D]")) != day
            or meta.bin_start[0] < split_times[part][0]
            or meta.bin_end[-1] > split_times[part][1]
            or len(meta.bin_start) != entry.get("bins")
            or len(meta.raw_ping_times) != entry.get("raw_pings")
        ):
            raise ValueError("Shard metadata crosses its split or day boundary.")
        if reference_frequency is None:
            reference_frequency, reference_edges = meta.frequency_hz, meta.range_edges_m
        else:
            assert reference_edges is not None
            if not np.array_equal(reference_frequency, meta.frequency_hz) or not np.array_equal(
                reference_edges, meta.range_edges_m
            ):
                raise ValueError("Canonical frequency/range grid differs across shards.")
        for source in set().union(*meta.source_file_ids):
            previous = source_partition.setdefault(source, part)
            if previous != part:
                raise ValueError("A raw source identity appears across partitions.")
        for time in meta.raw_ping_times.astype("datetime64[ns]").astype(np.int64):
            if int(time) in raw_time_partition:
                raise ValueError("A raw timestamp appears across partitions or shards.")
            raw_time_partition[int(time)] = part
        for time in meta.bin_end.astype("datetime64[ns]").astype(np.int64):
            if int(time) in bin_time_partition:
                raise ValueError("A bin timestamp appears across partitions or shards.")
            bin_time_partition[int(time)] = part
        grouped[part].append(meta)
        before_qc[part] += 1

    after_qc = {part: 0 for part in PARTITIONS}
    eligible_windows = {part: 0 for part in PARTITIONS}
    for part in PARTITIONS:
        shards = grouped[part]
        if not shards:
            continue
        ends = np.concatenate([item.bin_end for item in shards])
        available = np.concatenate(
            [
                item.observed_available_time
                if availability_basis == "measured"
                else item.replay_available_time
                for item in shards
            ]
        )
        masks = np.concatenate([item.valid_mask for item in shards])
        configs = list(chain.from_iterable(item.configuration_ids for item in shards))
        if np.any(np.diff(ends) <= np.timedelta64(0, "ns")):
            raise ValueError("Chronological bin timestamps are not unique and increasing.")
        frequency = shards[0].frequency_hz
        channel = np.flatnonzero(frequency == analysis_frequency_hz)
        edges = shards[0].range_edges_m
        lower, upper = analysis_range_m
        ranges = np.flatnonzero((edges[:-1] >= lower) & (edges[1:] <= upper))
        if (
            len(channel) != 1
            or not len(ranges)
            or lower >= upper
            or not (edges == lower).any()
            or not (edges == upper).any()
        ):
            raise ValueError("Analysis channel and exact physical band are required.")
        widths = np.diff(edges)[ranges]
        eligible_days: set[str] = set()
        split_start, split_end = split_times[part]
        for cutoff in ends[ends.astype("datetime64[m]").astype(np.int64) % 60 == 0]:
            context_start = cutoff - np.timedelta64(24, "h")
            target_end = cutoff + np.timedelta64(6, "h")
            if context_start < split_start or target_end > split_end:
                continue
            extent = np.flatnonzero((ends > context_start) & (ends <= target_end))
            if len(extent) != 120 or not np.array_equal(
                ends[extent],
                context_start + np.arange(1, 121) * np.timedelta64(15, "m"),
            ):
                continue
            if (
                None in (configs[index] for index in extent)
                or len({configs[index] for index in extent}) != 1
            ):
                continue
            context = extent[:96]
            if not np.all((~np.isnat(available[context])) & (available[context] <= cutoff)):
                continue
            if not masks[context, channel[0], :][:, ranges].any(axis=1).all():
                continue
            supported = True
            for horizon in (1, 3, 6):
                start = cutoff + np.timedelta64(horizon - 1, "h")
                target = np.flatnonzero((ends > start) & (ends <= start + np.timedelta64(1, "h")))
                if len(target) != 4:
                    supported = False
                    break
                target_mask = masks[np.ix_(target, channel, ranges)][:, 0, :]
                weights = np.broadcast_to(widths, target_mask.shape)
                if float(weights[target_mask].sum() / weights.sum()) < 0.8:
                    supported = False
                    break
            if supported:
                eligible_windows[part] += 1
                eligible_days.add(str(cutoff.astype("datetime64[D]")))
        after_qc[part] = len(eligible_days)
    total_days = sum(after_qc.values())
    eligible = (
        total_days >= REQUIRED_DAYS["overall"]
        and after_qc["calibration"] >= REQUIRED_DAYS["calibration"]
        and after_qc["test"] >= REQUIRED_DAYS["test"]
    )
    return {
        "status": "ELIGIBLE_FOR_REVIEW" if eligible else "INELIGIBLE",
        "required_days": REQUIRED_DAYS.copy(),
        "before_qc_days": before_qc,
        "after_qc_days": after_qc,
        "eligible_windows": eligible_windows,
        "protected_partition_inspection": "timestamps_masks_provenance_only",
        "test_metadata_opened": exposure_record is not None,
        "test_acoustic_values_opened": False,
        "test_outcomes_opened": False,
        "availability_basis": availability_basis,
        "availability_claim": (
            "observed_only" if availability_basis == "measured" else "simulation_only"
        ),
        "promotion_status": PROMOTION_STATUS,
        "exposure_record": str(exposure_record) if exposure_record is not None else None,
        "split_sha256": store.identity.split_sha256,
        "processing_manifest_sha256": _sha256_file(store.manifest_path),
    }
