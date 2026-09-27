"""Hash-verified TRAIN candidate streaming and scheduled causal observation windows.

This reads the preserved, unpromoted TRAIN census. A window is a development
record, not an approved benchmark row. The target plan must be frozen separately.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _valid_digest(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


@dataclass(frozen=True)
class ObservationSlot:
    start: np.datetime64
    end: np.datetime64
    linear_sv: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    valid_ping_count: NDArray[np.int64]
    frequency_hz: NDArray[np.int64]
    range_edges_m: NDArray[np.float64]
    source_sha256: tuple[str, ...]
    processed_sha256: str
    configuration_id: str | None
    configuration_boundary: bool
    expected_pings: int
    observed_pings: int

    def with_linear_sv(self, values: NDArray[np.float64]) -> ObservationSlot:
        """Return a changed value array for isolated software tests."""
        if values.shape != self.linear_sv.shape:
            raise ValueError("Observation shape differs.")
        mask = np.isfinite(values) & (values > 0)
        return replace(
            self,
            linear_sv=values,
            valid_mask=mask,
            valid_ping_count=np.where(mask, self.valid_ping_count, 0),
        )


@dataclass(frozen=True)
class WindowPlan:
    context_slots: int
    horizon_offsets: tuple[int, ...]
    target_frequency_hz: int
    range_bins: tuple[int, int]
    minimum_target_support: float = 0.0

    def __post_init__(self) -> None:
        if (
            self.context_slots < 1
            or not self.horizon_offsets
            or any(offset < 1 for offset in self.horizon_offsets)
            or tuple(sorted(set(self.horizon_offsets))) != self.horizon_offsets
            or self.range_bins[0] < 0
            or self.range_bins[1] <= self.range_bins[0]
            or not 0 <= self.minimum_target_support <= 1
        ):
            raise ValueError("Invalid frozen observation window plan.")


@dataclass(frozen=True)
class ObservationWindow:
    row_id: str
    cutoff: np.datetime64
    target_time: NDArray[np.datetime64]
    context: NDArray[np.float64]
    context_mask: NDArray[np.bool_]
    context_age_minutes: NDArray[np.float64]
    target_db: NDArray[np.float64]
    target_mask: NDArray[np.bool_]
    target_support: NDArray[np.float64]
    target_profile_db: NDArray[np.float64]
    target_profile_mask: NDArray[np.bool_]
    future_train_db: NDArray[np.float64]
    future_train_mask: NDArray[np.bool_]
    source_sha256: tuple[str, ...]


@dataclass(frozen=True)
class HourlyWindow:
    """Development row for the proposed fixed sampled detected-cell estimand."""

    row_id: str
    partition: str
    cutoff: np.datetime64
    context: NDArray[np.float64]
    context_mask: NDArray[np.bool_]
    context_acquisition_fraction: NDArray[np.float64]
    context_detection_fraction: NDArray[np.float64]
    context_age_minutes: NDArray[np.float64]
    target_interval_start: NDArray[np.datetime64]
    target_interval_end: NDArray[np.datetime64]
    target_db: NDArray[np.float64]
    target_mask: NDArray[np.bool_]
    target_detection_fraction: NDArray[np.float64]
    target_detection_mask: NDArray[np.bool_]
    target_acquisition_fraction: NDArray[np.float64]
    future_train_db: NDArray[np.float64]
    future_train_mask: NDArray[np.bool_]
    source_sha256: tuple[str, ...]


class CandidateTrainStream:
    """Read one TRAIN day at a time using the immutable census hash register."""

    def __init__(self, root: Path, census: Path, *, fixture_only: bool = False) -> None:
        self.root = root.resolve(strict=True)
        self.census = census.resolve(strict=True)
        document = json.loads(self.census.read_text(encoding="utf-8"))
        if document.get("status") != "COMPLETE_TRAIN_CENSUS_NOT_BENCHMARK":
            raise ValueError("A complete TRAIN census is required.")
        calendar = document.get("identity", {}).get("calendar")
        artifacts = document.get("artifact_sha256")
        if artifacts is None and isinstance(document.get("days"), dict):
            artifacts = {}
            for day, record in document["days"].items():
                compact = day.replace("-", "")
                artifacts[
                    f"evidence/continuation/train-candidate-{compact}-census-v2/processing_manifest.json"
                ] = record.get("manifest_sha256")
                artifacts[
                    f"evidence/continuation/train-candidate-{compact}-census-v2/configuration_boundaries.json"
                ] = record.get("configuration_sha256")
                artifacts[f"data/processed/candidate-train-{compact}-census-v2/{day}.npz"] = (
                    record.get("shard_sha256")
                )
        if not isinstance(calendar, list) or not calendar or not isinstance(artifacts, dict):
            raise ValueError("TRAIN census calendar or digest register is missing.")
        if len(set(calendar)) != len(calendar) or calendar != sorted(calendar):
            raise ValueError("TRAIN calendar must be unique and ordered.")
        if set(document.get("days", calendar)) != set(calendar):
            raise ValueError("TRAIN execution days differ from the frozen calendar.")
        self.calendar: tuple[str, ...] = tuple(calendar)
        self.artifacts: dict[str, str] = artifacts
        self.fixture_only = fixture_only

    def _registered(self, suffix: str) -> Path:
        matches = [name for name in self.artifacts if name.endswith(suffix)]
        if not matches and self.fixture_only:
            matches = [name for name in self.artifacts if name == Path(suffix).name]
        if len(matches) != 1:
            raise ValueError("Expected one registered TRAIN artifact for the day.")
        name = matches[0]
        path = (self.root / name).resolve(strict=True)
        if not path.is_relative_to(self.root):
            raise ValueError("Registered TRAIN artifact escapes the corpus root.")
        digest = self.artifacts[name]
        if not _valid_digest(digest) or _sha256(path) != digest:
            raise ValueError("Registered TRAIN artifact digest differs.")
        return path

    def __iter__(self) -> Iterator[ObservationSlot]:
        previous_end: np.datetime64 | None = None
        for day in self.calendar:
            if len(day) != 10 or day != str(np.datetime64(day, "D")):
                raise ValueError("Invalid TRAIN calendar date.")
            compact = day.replace("-", "")
            report_path = self._registered(
                f"train-candidate-{compact}-census-v2/processing_manifest.json"
            )
            # A small artificial fixture can use the same content contract.
            if self.fixture_only and not report_path.exists():
                raise ValueError("Fixture report is missing.")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("status") != "TRAIN_CANDIDATE_PROCESSED_REQUIRES_QC_REVIEW":
                raise ValueError("Unexpected TRAIN candidate status.")
            source_hours = report.get("source_hours")
            if not isinstance(source_hours, list) or not source_hours:
                raise ValueError("TRAIN source provenance is absent.")
            if any(not isinstance(item, dict) for item in source_hours):
                raise ValueError("TRAIN source provenance is malformed.")
            sources = tuple(sorted({str(item.get("source_sha256")) for item in source_hours}))
            if not sources or not all(_valid_digest(value) for value in sources):
                raise ValueError("A real source SHA-256 is required for every TRAIN source.")
            if self.fixture_only and not any(
                name.endswith(f"train-candidate-{compact}-census-v2/configuration_boundaries.json")
                for name in self.artifacts
            ):
                configuration_ids = None
                configuration_flags = None
            else:
                config_path = self._registered(
                    f"train-candidate-{compact}-census-v2/configuration_boundaries.json"
                )
                config_report = json.loads(config_path.read_text(encoding="utf-8"))
                configuration_ids = config_report.get("bin_configuration_ids")
                configuration_flags = config_report.get("boundary_mask")
            day_path = self._registered(f"{day}.npz")
            day_digest = _sha256(day_path)
            if report.get("daily_candidate_sha256") != day_digest:
                raise ValueError("TRAIN day digest differs from its processing report.")
            with np.load(day_path, allow_pickle=False) as data:
                required = {
                    "bin_start",
                    "bin_end",
                    "linear_sv",
                    "valid_ping_count",
                    "expected_ping_count",
                    "observed_ping_count",
                    "range_edges_m",
                    "frequency_hz",
                    "configuration_boundary",
                }
                if not required.issubset(data.files):
                    raise ValueError("TRAIN day lacks required acoustic arrays.")
                starts = data["bin_start"]
                ends = data["bin_end"]
                linear = data["linear_sv"]
                counts = data["valid_ping_count"]
                expected = data["expected_ping_count"]
                observed = data["observed_ping_count"]
                ranges = data["range_edges_m"]
                frequencies = data["frequency_hz"]
                boundaries = data["configuration_boundary"]
            n = len(starts)
            if (
                starts.dtype.kind != "M"
                or ends.dtype.kind != "M"
                or np.isnat(starts).any()
                or np.isnat(ends).any()
                or linear.shape != (n, 4, 64)
                or counts.shape != linear.shape
                or expected.shape != (n,)
                or observed.shape != (n,)
                or boundaries.shape != (n,)
                or frequencies.shape != (4,)
                or ranges.shape != (65,)
                or not np.all(np.diff(ranges) > 0)
                or not np.array_equal(frequencies, [38000, 125000, 200000, 455000])
                or np.any(ends - starts != np.timedelta64(15, "m"))
                or np.any(starts[1:] != ends[:-1])
                or np.any(expected < 0)
                or np.any(observed < 0)
                or np.any(counts < 0)
                or np.any(counts > observed[:, None, None])
                or np.any(np.isfinite(linear) & (linear <= 0))
                or (configuration_ids is not None and len(configuration_ids) != n)
                or (configuration_flags is not None and configuration_flags != boundaries.tolist())
            ):
                raise ValueError("TRAIN day has invalid shape, time, grid or acoustic values.")
            if previous_end is not None and starts[0] < previous_end:
                raise ValueError("TRAIN days overlap or are out of order.")
            previous_end = ends[-1]
            for index in range(n):
                if configuration_ids is None:
                    config_id = "artificial-fixture"
                else:
                    config_id = configuration_ids[index]
                    if config_id is not None and not _valid_digest(config_id):
                        raise ValueError("TRAIN bin has an invalid configuration hash.")
                bin_sources = []
                for item in source_hours:
                    if "first_utc" not in item or "last_utc" not in item:
                        if self.fixture_only:
                            bin_sources.append(item["source_sha256"])
                        else:
                            raise ValueError("TRAIN source acquisition bounds are missing.")
                    elif item["first_utc"] is not None and item["last_utc"] is not None:
                        first = np.datetime64(item["first_utc"], "ns")
                        last = np.datetime64(item["last_utc"], "ns")
                        if first < ends[index] and last >= starts[index]:
                            bin_sources.append(item["source_sha256"])
                yield ObservationSlot(
                    start=starts[index],
                    end=ends[index],
                    linear_sv=linear[index],
                    valid_mask=(counts[index] > 0) & np.isfinite(linear[index]),
                    valid_ping_count=counts[index],
                    frequency_hz=frequencies,
                    range_edges_m=ranges,
                    source_sha256=tuple(sorted(set(bin_sources))),
                    processed_sha256=day_digest,
                    configuration_id=config_id,
                    configuration_boundary=bool(boundaries[index]),
                    expected_pings=int(expected[index]),
                    observed_pings=int(observed[index]),
                )


def iter_windows(slots: Iterable[ObservationSlot], plan: WindowPlan) -> Iterator[ObservationWindow]:
    """Use fixed scheduled offsets; never search for the next valid target."""
    window_size = plan.context_slots + max(plan.horizon_offsets)
    buffer: deque[ObservationSlot] = deque(maxlen=window_size)
    band = slice(*plan.range_bins)
    for slot in slots:
        buffer.append(slot)
        if len(buffer) < window_size:
            continue
        relevant = list(buffer)
        frequency = np.flatnonzero(relevant[0].frequency_hz == plan.target_frequency_hz)
        if len(frequency) != 1 or plan.range_bins[1] > relevant[0].linear_sv.shape[1]:
            raise ValueError("Frozen target frequency/range is unavailable.")
        channel = int(frequency[0])
        if any(item.configuration_boundary for item in relevant[1:]):
            continue
        if any(left.end != right.start for left, right in pairwise(relevant)):
            continue
        context_slots = relevant[: plan.context_slots]
        targets = [relevant[plan.context_slots - 1 + offset] for offset in plan.horizon_offsets]
        cutoff = context_slots[-1].end
        context_linear = np.stack([item.linear_sv for item in context_slots])
        context_mask = np.stack([item.valid_mask for item in context_slots])
        context = np.full(context_linear.shape, np.nan)
        context[context_mask] = 10 * np.log10(context_linear[context_mask])
        future_linear = np.stack([item.linear_sv for item in targets])
        future_mask = np.stack([item.valid_mask for item in targets])
        future_db = np.full(future_linear.shape, np.nan)
        future_db[future_mask] = 10 * np.log10(future_linear[future_mask])
        support = future_mask[:, channel, band].mean(axis=1)
        target_mask = (support > 0) & (support >= plan.minimum_target_support)
        target_db = np.full(len(targets), np.nan)
        for horizon, present in enumerate(target_mask):
            if present:
                values = future_linear[horizon, channel, band]
                valid = future_mask[horizon, channel, band]
                target_db[horizon] = 10 * np.log10(values[valid].mean())
        source_ids = tuple(sorted({source for slot in relevant for source in slot.source_sha256}))
        row_id = hashlib.sha256((str(cutoff) + ":" + repr(plan)).encode("utf-8")).hexdigest()
        yield ObservationWindow(
            row_id=row_id,
            cutoff=cutoff,
            target_time=np.array([item.end for item in targets]),
            context=context,
            context_mask=context_mask,
            context_age_minutes=np.array(
                [(cutoff - item.end) / np.timedelta64(1, "m") for item in context_slots]
            ),
            target_db=target_db,
            target_mask=target_mask,
            target_support=support,
            target_profile_db=future_db,
            target_profile_mask=future_mask,
            future_train_db=future_db,
            future_train_mask=future_mask,
            source_sha256=source_ids,
        )


def _one_known_configuration(slots: list[ObservationSlot]) -> bool:
    observed = [item for item in slots if item.observed_pings > 0]
    return (
        bool(observed)
        and all(item.configuration_id is not None for item in observed)
        and len({item.configuration_id for item in observed}) == 1
    )


def iter_hourly_windows(
    slots: Iterable[ObservationSlot], *, partition: str
) -> Iterator[HourlyWindow]:
    """Stream fixed hourly issues and +1/+3/+6 hour sampled future targets.

    The only supported input partition is TRAIN while the proposed protocol and
    physical target await review. Future observations occur only in target fields.
    """
    if partition != "train":
        raise ValueError("Only approved-scope TRAIN development may read candidate values.")
    buffer: deque[ObservationSlot] = deque(maxlen=120)
    for slot in slots:
        buffer.append(slot)
        if len(buffer) < 120:
            continue
        sequence = list(buffer)
        context_slots = sequence[:96]
        cutoff = context_slots[-1].end
        if cutoff.astype("datetime64[m]").astype(int) % 60:
            continue
        if any(left.end != right.start for left, right in pairwise(sequence)):
            continue
        if not _one_known_configuration(context_slots):
            continue
        observed_context = np.array([item.observed_pings for item in context_slots])
        if (
            observed_context[-4:].sum() < 120
            or np.count_nonzero(observed_context.reshape(24, 4).sum(axis=1)) < 12
        ):
            continue
        input_counts = np.stack([item.valid_ping_count for item in context_slots])
        if input_counts[-4:, 0, 5:50].sum() == 0:
            continue
        linear = np.stack([item.linear_sv for item in context_slots])
        valid = np.stack([item.valid_mask for item in context_slots])
        context_mask = np.zeros_like(valid)
        context_mask[:, 0, 5:50] = valid[:, 0, 5:50]
        context = np.full(linear.shape, np.nan)
        context[context_mask] = 10 * np.log10(linear[context_mask])
        expected_context = np.array([item.expected_pings for item in context_slots])
        acquisition = np.divide(
            observed_context,
            expected_context,
            out=np.full(96, np.nan),
            where=expected_context > 0,
        )
        detection = np.divide(
            input_counts[:, 0, 5:50].sum(axis=1),
            45 * observed_context,
            out=np.full(96, np.nan),
            where=observed_context > 0,
        )
        target_groups = [sequence[first : first + 4] for first in (96, 104, 116)]
        future_linear = np.stack([[item.linear_sv for item in group] for group in target_groups])
        future_count = np.stack(
            [[item.valid_ping_count for item in group] for group in target_groups]
        )
        future_mask = np.zeros_like(future_linear, dtype=bool)
        future_mask[:, :, 0, 5:50] = future_count[:, :, 0, 5:50] > 0
        future_db = np.full(future_linear.shape, np.nan)
        future_db[future_mask] = 10 * np.log10(future_linear[future_mask])
        target_db = np.full(3, np.nan)
        target_mask = np.zeros(3, dtype=bool)
        target_detection = np.full(3, np.nan)
        target_detection_mask = np.zeros(3, dtype=bool)
        target_acquisition = np.full(3, np.nan)
        for horizon, group in enumerate(target_groups):
            observed = sum(item.observed_pings for item in group)
            expected = sum(item.expected_pings for item in group)
            detected = future_count[horizon, :, 0, 5:50].sum()
            target_acquisition[horizon] = observed / expected if expected else np.nan
            if observed >= 120 and _one_known_configuration(group):
                target_detection_mask[horizon] = True
                target_detection[horizon] = detected / (45 * observed)
                if detected and target_detection[horizon] >= 0.1:
                    values = future_linear[horizon, :, 0, 5:50]
                    weights = future_count[horizon, :, 0, 5:50]
                    target_db[horizon] = 10 * np.log10(
                        np.sum(values * weights, where=weights > 0) / detected
                    )
                    target_mask[horizon] = True
        sources = tuple(sorted({digest for item in sequence for digest in item.source_sha256}))
        row_id = hashlib.sha256((str(cutoff) + ":sampled-detected-v2").encode()).hexdigest()
        yield HourlyWindow(
            row_id=row_id,
            partition=partition,
            cutoff=cutoff,
            context=context,
            context_mask=context_mask,
            context_acquisition_fraction=acquisition,
            context_detection_fraction=detection,
            context_age_minutes=np.array(
                [(cutoff - item.end) / np.timedelta64(1, "m") for item in context_slots]
            ),
            target_interval_start=np.array([group[0].start for group in target_groups]),
            target_interval_end=np.array([group[-1].end for group in target_groups]),
            target_db=target_db,
            target_mask=target_mask,
            target_detection_fraction=target_detection,
            target_detection_mask=target_detection_mask,
            target_acquisition_fraction=target_acquisition,
            future_train_db=future_db,
            future_train_mask=future_mask,
            source_sha256=sources,
        )
