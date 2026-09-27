"""Stream only hash-registered candidate-2 TRAIN development native day shards."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from marine_echo.training.v2_native import NativeObservationSlot


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _valid_digest(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


class NativeDevelopmentStream:
    """One verified day in memory at a time; no held-out payload access."""

    def __init__(
        self,
        root: Path,
        index: Path,
        *,
        index_sha256: str,
        start_day: str | None = None,
        end_day_exclusive: str | None = None,
        fixture_only: bool = False,
    ) -> None:
        self.root = root.resolve(strict=True)
        self.index_path = index.resolve(strict=True)
        if (
            not self.index_path.is_relative_to(self.root)
            or _sha256(self.index_path) != index_sha256
        ):
            raise ValueError("Native TRAIN index digest or location differs.")
        document = json.loads(self.index_path.read_text(encoding="utf-8"))
        if document.get("data_kind") != "REAL" or document.get("study_id") != "v2_candidate2":
            raise ValueError("Only the declared native real-data candidate is supported.")
        days = document.get("days")
        if not isinstance(days, dict) or not days:
            raise ValueError("Native TRAIN index has no registered day shards.")
        self.bindings = document.get("bindings")
        if not isinstance(self.bindings, dict) or not self.bindings:
            raise ValueError("Native processing bindings are absent.")
        if not fixture_only and (
            start_day is None
            or end_day_exclusive is None
            or not "2020-02-17" <= start_day < end_day_exclusive <= "2020-04-15"
        ):
            raise ValueError("Only the fixed TRAIN development date range is permitted.")
        self.days: dict[str, dict[str, Any]] = {}
        for day, entry in days.items():
            if day != str(np.datetime64(day, "D")):
                raise ValueError("Native index day is malformed.")
            if not fixture_only and not "2020-02-17" <= day < "2020-04-15":
                raise ValueError("Native index contains a non-TRAIN development day.")
            if (start_day is None or day >= start_day) and (
                end_day_exclusive is None or day < end_day_exclusive
            ):
                if not isinstance(entry, dict):
                    raise ValueError("Native index day entry is malformed.")
                self.days[day] = entry
        if not self.days:
            raise ValueError("Selected TRAIN development range has no native shards.")
        self.fixture_only = fixture_only

    def _registered_path(self, relative: str, digest: object) -> Path:
        if not isinstance(relative, str) or not _valid_digest(digest):
            raise ValueError("Native registered artifact path or digest is malformed.")
        path = (self.root / relative).resolve(strict=True)
        if not path.is_relative_to(self.root) or _sha256(path) != digest:
            raise ValueError("Native registered artifact digest or location differs.")
        return path

    def _historical_sources(self, day: str, manifest: dict[str, Any]) -> None:
        if self.fixture_only:
            return
        compact = day.replace("-", "")
        census_path = self.root / "evidence/continuation/train_census_v2_execution.json"
        census = json.loads(census_path.read_text(encoding="utf-8"))
        prior = (
            self.root
            / f"evidence/continuation/train-candidate-{compact}-census-v2/processing_manifest.json"
        )
        expected = census["days"][day]["manifest_sha256"]
        if _sha256(prior) != expected or manifest.get("historical_manifest_sha256") != expected:
            raise ValueError("Native sources differ from preserved TRAIN provenance digest.")
        old = json.loads(prior.read_text(encoding="utf-8"))
        keys = (
            "source",
            "source_sha256",
            "configuration_sha256",
            "pings_in_day",
            "first_utc",
            "last_utc",
        )
        expected_sources = sorted(tuple(item[key] for key in keys) for item in old["source_hours"])
        actual_sources = sorted(
            tuple(item[key] for key in keys) for item in manifest["source_hours"]
        )
        if actual_sources != expected_sources:
            raise ValueError("Native source list differs from preserved TRAIN source hours.")

    def __iter__(self) -> Iterator[NativeObservationSlot]:
        for day, entry in sorted(self.days.items()):
            expected_manifest = f"evidence/v2/native-development/{day}.json"
            if Path(entry.get("manifest_path", "")).as_posix() != expected_manifest:
                raise ValueError("Native manifest path differs from the fixed namespace.")
            manifest_path = self._registered_path(
                entry["manifest_path"], entry.get("manifest_sha256")
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("data_kind") != "REAL"
                or manifest.get("study_id") != "v2_candidate2"
                or manifest.get("status")
                != "NATIVE_TRAIN_DEVELOPMENT_PROCESSED_NOT_APPROVED_FOR_FITTING"
                or manifest.get("day") != day
                or manifest.get("bindings") != self.bindings
            ):
                raise ValueError("Native TRAIN manifest identity or processing bindings differ.")
            self._historical_sources(day, manifest)
            shard_relative = f"data/processed/v2-native-development/{day}.npz"
            if Path(manifest.get("shard_path", "")).as_posix() != shard_relative:
                raise ValueError("Native shard path differs from the fixed namespace.")
            if entry.get("shard_sha256") != manifest.get("shard_sha256"):
                raise ValueError("Native shard digest differs between index and manifest.")
            shard = self._registered_path(manifest["shard_path"], manifest["shard_sha256"])
            source_hours = manifest.get("source_hours")
            if not isinstance(source_hours, list) or not source_hours:
                raise ValueError("Native TRAIN source hours are missing.")
            sources = []
            for source in source_hours:
                if not isinstance(source, dict) or not all(
                    _valid_digest(source.get(key))
                    for key in ("source_sha256", "configuration_sha256")
                ):
                    raise ValueError("Native source SHA-256 or configuration hash is invalid.")
                first = np.datetime64(source["first_utc"], "ns")
                last = np.datetime64(source["last_utc"], "ns")
                if np.isnat(first) or np.isnat(last) or first > last:
                    raise ValueError("Native source acquisition times are invalid.")
                sources.append(
                    (first, last, source["source_sha256"], source["configuration_sha256"])
                )
            with np.load(shard, allow_pickle=False) as data:
                required = {
                    "bin_start",
                    "bin_end",
                    "linear_sv",
                    "detected_range_ping_m",
                    "observed_ping_count",
                    "expected_ping_count",
                    "raw_ping_time",
                    "frequency_hz",
                    "range_edges_m",
                    "configuration_id",
                    "native_detected_sample_count",
                }
                if set(data.files) != required:
                    raise ValueError("Native TRAIN shard fields differ from the frozen schema.")
                arrays = {key: data[key] for key in required}
            starts = arrays["bin_start"]
            ends = arrays["bin_end"]
            linear = arrays["linear_sv"]
            weight = arrays["detected_range_ping_m"]
            observed = arrays["observed_ping_count"]
            expected = arrays["expected_ping_count"]
            raw_times = arrays["raw_ping_time"]
            configs = arrays["configuration_id"]
            sample_count = arrays["native_detected_sample_count"]
            first_day = np.datetime64(day, "ns")
            if (
                starts.shape != (96,)
                or ends.shape != (96,)
                or linear.shape != (96, 4, 64)
                or weight.shape != linear.shape
                or observed.shape != (96,)
                or expected.shape != (96,)
                or configs.shape != (96,)
                or sample_count.shape != (96,)
                or not np.array_equal(arrays["frequency_hz"], [38000, 125000, 200000, 455000])
                or not np.array_equal(arrays["range_edges_m"], np.arange(65) * 2.0)
                or not np.array_equal(starts, first_day + np.arange(96) * np.timedelta64(15, "m"))
                or not np.array_equal(ends, starts + np.timedelta64(15, "m"))
                or (observed < 0).any()
                or (expected != 60).any()
                or (sample_count < 0).any()
                or raw_times.dtype.kind != "M"
                or np.isnat(raw_times).any()
                or (np.diff(raw_times) <= np.timedelta64(0, "ns")).any()
                or (raw_times < first_day).any()
                or (raw_times >= first_day + np.timedelta64(1, "D")).any()
                or len(raw_times) != int(observed.sum())
                or manifest.get("observed_pings") != len(raw_times)
            ):
                raise ValueError("Native TRAIN shard time, geometry or counts are invalid.")
            ping_bins = ((raw_times - first_day) / np.timedelta64(15, "m")).astype(int)
            if not np.array_equal(np.bincount(ping_bins, minlength=96), observed):
                raise ValueError("Native scheduled-bin acquisition counts differ from raw pings.")
            for index in range(96):
                config = str(configs[index]) or None
                matched = [
                    (digest, source_config)
                    for first, last, digest, source_config in sources
                    if first < ends[index] and last >= starts[index]
                ]
                if observed[index] and (
                    config is None or not matched or {value for _, value in matched} != {config}
                ):
                    raise ValueError("Native observed bin configuration or source lineage differs.")
                slot = NativeObservationSlot(
                    start=starts[index],
                    end=ends[index],
                    linear_sv=linear[index],
                    detected_range_ping_m=weight[index],
                    observed_pings=int(observed[index]),
                    expected_pings=int(expected[index]),
                    configuration_id=config,
                    source_sha256=tuple(sorted({digest for digest, _ in matched})),
                    processed_sha256=entry["shard_sha256"],
                )
                slot.validate()
                yield slot
