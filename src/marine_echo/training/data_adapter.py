"""Bounded fixture-only canonical windows for train and validation executors."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from numpy.typing import NDArray

from marine_echo.data.canonical_store import (
    PROMOTION_STATUS,
    CanonicalStore,
    split_bounds_sha256,
)
from marine_echo.data.preprocessing import BinnedAcoustics, build_window


@dataclass(frozen=True)
class WindowBatch:
    partition: str
    context_db: NDArray[np.float64]
    context_mask: NDArray[np.bool_]
    target_db: NDArray[np.float64]
    target_support: NDArray[np.float64]
    cutoffs: NDArray[np.datetime64]
    row_ids: tuple[str, ...]
    source_file_ids: tuple[frozenset[str], ...]
    provenance: dict[str, str]


class CanonicalWindowAdapter:
    """Materialize finite windows; never materialize protected partitions."""

    def __init__(
        self,
        store: CanonicalStore,
        bounds: dict[str, tuple[str, str]],
        *,
        fixture_only: bool,
        availability_basis: str = "zero_latency_replay",
    ) -> None:
        manifest = store._manifest()
        if not fixture_only or manifest.get("promotion_status") != PROMOTION_STATUS:
            raise ValueError(
                "R0/R1 promotion gate is absent; only synthetic fixtures are accepted."
            )
        if split_bounds_sha256(bounds) != store.identity.split_sha256:
            raise ValueError("Adapter split differs from canonical store identity.")
        if availability_basis not in ("measured", "zero_latency_replay"):
            raise ValueError("Named availability basis is required.")
        seen_sources: dict[str, str] = {}
        for entry in manifest["shards"].values():
            partition = entry["partition"]
            for source in entry["source_file_ids"]:
                previous = seen_sources.setdefault(source, partition)
                if previous != partition:
                    raise ValueError("A raw source is registered across partitions.")
        self.store = store
        self.bounds = bounds
        self.availability_basis = availability_basis
        self.cache: OrderedDict[str, BinnedAcoustics] = OrderedDict()
        self.peak_cached_days = 0

    def _day(self, day: str) -> BinnedAcoustics:
        if day in self.cache:
            self.cache.move_to_end(day)
            return self.cache[day]
        series = self.store.load_day(day)
        if not np.array_equal(
            series.frequency_hz, np.array([38000, 125000, 200000, 455000])
        ) or not np.array_equal(series.range_edges_m, np.arange(65, dtype=float) * 2):
            raise ValueError(
                "Canonical channel order or range edges differ from fixed baseline grid."
            )
        self.cache[day] = series
        if len(self.cache) > 3:
            self.cache.popitem(last=False)
        self.peak_cached_days = max(self.peak_cached_days, len(self.cache))
        return series

    @staticmethod
    def _join(days: list[BinnedAcoustics]) -> BinnedAcoustics:
        if not days:
            raise ValueError("Window requires at least one canonical day.")
        first = days[0]
        for day in days[1:]:
            if (
                not np.array_equal(day.frequency_hz, first.frequency_hz)
                or not np.array_equal(day.range_edges_m, first.range_edges_m)
                or not np.array_equal(day.supported_range, first.supported_range)
                or any(
                    getattr(day, key) != getattr(first, key)
                    for key in (
                        "dataset_id",
                        "deployment_id",
                        "instrument_id",
                        "calibration_report_sha256",
                        "config_sha256",
                        "split_sha256",
                        "processing_sha256",
                    )
                )
            ):
                raise ValueError("Canonical day grids or provenance differ across a window.")
        arrays = {
            name: np.concatenate([getattr(day, name) for day in days])
            for name in (
                "bin_start",
                "bin_end",
                "observed_available_time",
                "replay_available_time",
                "sv_linear",
                "sv_db",
                "valid_mask",
                "ping_count",
                "configuration_boundary",
                "raw_ping_times",
            )
        }
        tuples = {
            name: tuple(value for day in days for value in getattr(day, name))
            for name in ("configuration_ids", "source_file_ids")
        }
        values: dict[str, Any] = {field.name: getattr(first, field.name) for field in fields(first)}
        values.update(arrays)
        values.update(tuples)
        return BinnedAcoustics(**values)

    def materialize(self, partition: str, *, max_windows: int) -> WindowBatch:
        if partition not in ("train", "validation"):
            raise ValueError("Calibration and test are protected from this adapter.")
        if max_windows <= 0 or max_windows > 1024:
            raise ValueError("A bounded window limit from 1 to 1024 is required.")
        manifest = self.store._manifest()
        registered = {
            day: entry
            for day, entry in manifest["shards"].items()
            if entry["partition"] == partition
        }
        allowed = {source for entry in registered.values() for source in entry["source_file_ids"]}
        start_text, end_text = self.bounds[partition]
        start = np.datetime64(start_text[:-1], "ns")
        end = np.datetime64(end_text[:-1], "ns")
        first_cutoff = start + np.timedelta64(24, "h")
        last_cutoff = end - np.timedelta64(6, "h")
        windows = []
        for cutoff in np.arange(
            first_cutoff, last_cutoff + np.timedelta64(1, "h"), np.timedelta64(1, "h")
        ):
            first_ns = int(cutoff.astype("datetime64[ns]").astype("int64")) - 86_400_000_000_000
            first_day = np.datetime64(first_ns, "ns").astype("datetime64[D]")
            last_ns = int(cutoff.astype("datetime64[ns]").astype("int64")) + 21_600_000_000_000 - 1
            last_day = np.datetime64(last_ns, "ns").astype("datetime64[D]")
            required = [
                str(day)
                for day in np.arange(
                    first_day, last_day + np.timedelta64(1, "D"), np.timedelta64(1, "D")
                )
            ]
            if any(day not in registered for day in required):
                continue
            days = []
            for day in required:
                loaded = self._day(day)
                actual_sources = set().union(*loaded.source_file_ids)
                if actual_sources != set(registered[day]["source_file_ids"]):
                    raise ValueError("Canonical day source provenance differs from the manifest.")
                days.append(loaded)
            series = self._join(days)
            try:
                window = build_window(
                    series,
                    cutoff=cutoff,
                    split_start=start,
                    split_end=end,
                    partition=partition,
                    allowed_source_file_ids=allowed,
                    expected_dataset_id=self.store.identity.dataset_id,
                    expected_deployment_id=self.store.identity.deployment_id,
                    expected_instrument_id=self.store.identity.instrument_id,
                    expected_calibration_report_sha256=self.store.identity.calibration_report_sha256,
                    expected_config_sha256=self.store.identity.config_sha256,
                    expected_split_sha256=self.store.identity.split_sha256,
                    expected_processing_sha256=self.store.identity.processing_sha256,
                    analysis_frequency_hz=38000,
                    analysis_range_m=(10.0, 100.0),
                    availability_basis=self.availability_basis,
                )
            except ValueError as error:
                if str(error) not in (
                    "Window crosses a configuration boundary or missing segment.",
                    "A primary context bin is missing or unavailable at issue time.",
                    "Target support is below the frozen threshold.",
                ):
                    raise
                continue
            windows.append(window)
            if len(windows) > max_windows:
                raise ValueError("Window materialization exceeded the bounded limit.")
        if not windows:
            raise ValueError("No eligible canonical windows in the declared partition.")
        context = np.stack([window.context_linear for window in windows])
        mask = np.stack([window.context_mask for window in windows])
        context_db = np.full_like(context, np.nan)
        context_db[mask] = 10.0 * np.log10(np.maximum(context[mask], 1e-18))
        identity = self.store.identity
        return WindowBatch(
            partition=partition,
            context_db=context_db,
            context_mask=mask,
            target_db=np.stack([window.target_index_db for window in windows]),
            target_support=np.stack([window.target_support for window in windows]),
            cutoffs=np.array([window.cutoff for window in windows], dtype="datetime64[ns]"),
            row_ids=tuple(window.row_id for window in windows),
            source_file_ids=tuple(window.source_file_ids for window in windows),
            provenance={
                "dataset_id": identity.dataset_id,
                "deployment_id": identity.deployment_id,
                "instrument_id": identity.instrument_id,
                "calibration_report_sha256": identity.calibration_report_sha256,
                "config_sha256": identity.config_sha256,
                "split_sha256": identity.split_sha256,
                "processing_sha256": identity.processing_sha256,
                "availability_basis": self.availability_basis,
                "promotion_status": PROMOTION_STATUS,
            },
        )


class TrainOnlyScaler:
    """Masked per-cell normalization with immutable train-only provenance."""

    def __init__(self) -> None:
        self.mean: NDArray[np.float64] | None = None
        self.scale: NDArray[np.float64] | None = None
        self.provenance: dict[str, str] = {}

    def fit(self, batch: WindowBatch) -> TrainOnlyScaler:
        if batch.partition != "train":
            raise ValueError("Scaler statistics require train windows only.")
        values = batch.context_db
        valid = batch.context_mask
        count = valid.sum(axis=(0, 1))
        total = np.where(valid, values, 0).sum(axis=(0, 1))
        mean = np.divide(total, count, out=np.zeros(values.shape[2:]), where=count > 0)
        variance = np.divide(
            np.where(valid, (values - mean) ** 2, 0).sum(axis=(0, 1)),
            count,
            out=np.zeros(values.shape[2:]),
            where=count > 0,
        )
        self.mean = mean
        self.scale = np.maximum(np.sqrt(variance), 1e-6)
        self.provenance = {**batch.provenance, "partition": "train"}
        return self

    def transform(self, batch: WindowBatch) -> NDArray[np.float64]:
        if self.mean is None or self.scale is None:
            raise RuntimeError("Fit scaler on train before transforming windows.")
        if batch.provenance != {
            key: value for key, value in self.provenance.items() if key != "partition"
        }:
            raise ValueError("Scaler and windows have different canonical provenance.")
        return np.where(batch.context_mask, (batch.context_db - self.mean) / self.scale, np.nan)
