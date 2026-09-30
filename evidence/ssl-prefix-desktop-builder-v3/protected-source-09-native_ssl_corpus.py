"""Native integrated-product observations for the separately reviewed SSL study.

This reader deliberately does not amend the historical 0â€“200 m readers.
Source values are conditioned published products, with unknown absolute censoring
and clock timezone where no such information is supplied.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path

import numpy as np

from marine_echo.training.aeon_corpus import FREQUENCIES_HZ, SPECIAL_SV

MEMBER = re.compile(
    r"(?:[^/]+/)?AEON\d+_\d+_(038|125|200|455)_\d{4}_\d{2}[a-z]?_60minFullDepth\.csv"
)
METADATA_FIELDS = (
    "frequency_hz",
    "interval_seconds",
    "geometry_kind",
    "upper_m",
    "lower_m",
    "orientation_code_zero_unknown",
    "processing_known",
    "instrument_known",
    "clock_known",
    "relative_offset_intervals",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class NativeSlot:
    interval: int
    timestamp: np.datetime64
    values: np.ndarray
    observed: np.ndarray
    bounds: np.ndarray
    processing: tuple[str, ...]
    ping_counts: tuple[int, ...]
    deployment: str
    archive_sha256: str
    qc: tuple[str, ...]


def check_numeric_review(split_path: Path, review_path: Path, role: str) -> dict:
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("status") != "APPROVED_NATIVE_NUMERIC_ACCESS":
        raise ValueError("Distinct native numeric access approval required.")
    if not review.get("reviewer_session_id") or review.get("reviewer_session_id") == review.get(
        "implementer_session_id"
    ):
        raise ValueError("Numeric reader requires a distinct reviewer session.")
    if role not in review.get("allowed_roles", []):
        raise ValueError("Corpus role is not approved for numeric access.")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    protocol = split_path.resolve().parents[1] / split["protocol_path"]
    required = (
        split_path,
        protocol,
        Path(__file__),
        Path(__file__).parents[1] / "training/aeon_corpus.py",
    )
    bindings = review.get("bindings", {})
    for path in required:
        if bindings.get(str(path.resolve())) != sha256(path):
            raise ValueError(f"Missing or stale numeric review binding: {path}")
    if sha256(protocol) != split["protocol_sha256"]:
        raise ValueError("Split protocol identity differs.")
    return split


def _timestamp(row: dict) -> np.datetime64:
    raw = row["Date_M"] + row["Time_M"].strip()
    for fmt in ("%Y%m%d%H:%M:%S.%f", "%Y%m%d%H:%M:%S"):
        try:
            return np.datetime64(datetime.strptime(raw, fmt), "us")  # noqa: DTZ007 -- source timezone unknown
        except ValueError:
            pass
    raise ValueError("Unrecognized source clock format; no timezone inferred.")


def read_source(source: dict) -> list[NativeSlot]:
    """Read an already hash-reviewed source; caller must pass the access gate."""
    path = Path(source["path"])
    if sha256(path) != source["archive_sha256"]:
        raise ValueError("Native source archive identity differs.")
    records: dict[int, dict[int, list[dict]]] = {}
    with zipfile.ZipFile(path) as archive:
        for member in sorted(archive.namelist()):
            match = MEMBER.fullmatch(member)
            if match is None:
                continue
            frequency = int(match.group(1)) * 1000
            with archive.open(member) as stream:
                reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig"))
                required = {
                    "Interval",
                    "Date_M",
                    "Time_M",
                    "Sv_mean",
                    "Layer",
                    "Layer_depth_min",
                    "Layer_depth_max",
                    "Ping_S",
                    "Ping_E",
                    "Process_ID",
                }
                if not required.issubset(reader.fieldnames or []):
                    raise ValueError("Native source header lacks contract fields.")
                for row in reader:
                    if not source["start_date"] <= row["Date_M"] <= source["end_date_inclusive"]:
                        continue  # Never parse excluded historical outcomes.
                    records.setdefault(int(row["Interval"]), {}).setdefault(frequency, []).append(
                        row
                    )
    slots = []
    for interval, channels in sorted(records.items()):
        if len(channels.get(38000, [])) != 1:
            continue
        reference = channels[38000][0]
        timestamp = _timestamp(reference)
        bounds = np.tile(
            [float(reference["Layer_depth_min"]), float(reference["Layer_depth_max"])], (4, 1)
        )
        values, observed = np.zeros(4), np.zeros(4, dtype=bool)
        processing, pings, qc = [], [], []
        for index, frequency in enumerate(FREQUENCIES_HZ):
            rows = channels.get(frequency, [])
            status = "MISSING_CHANNEL" if not rows else "DUPLICATE_ROW"
            process, count = "UNKNOWN", 0
            if len(rows) == 1:
                row = rows[0]
                bounds[index] = [float(row["Layer_depth_min"]), float(row["Layer_depth_max"])]
                process = row["Process_ID"].strip() or "UNKNOWN"
                count = int(row["Ping_E"]) - int(row["Ping_S"]) + 1
                status = "OBSERVED_CENSORING_UNKNOWN"
                if abs(_timestamp(row) - timestamp) > np.timedelta64(5, "m"):
                    status = "CLOCK_MISMATCH"
                elif (
                    int(row["Layer"]) != 1
                    or not np.isfinite(bounds[index]).all()
                    or bounds[index, 1] <= bounds[index, 0]
                ):
                    status = "INVALID_NATIVE_GEOMETRY"
                elif count not in source["complete_ping_counts"]:
                    status = "PARTIAL_SOURCE_INTERVAL"
                else:
                    value = float(row["Sv_mean"]) if row["Sv_mean"].strip() else np.nan
                    if not np.isfinite(value) or value in SPECIAL_SV:
                        status = "INVALID_OR_SENTINEL"
                    else:
                        values[index], observed[index] = value, True
            processing.append(process)
            pings.append(count)
            qc.append(status)
        slots.append(
            NativeSlot(
                interval,
                timestamp,
                values,
                observed,
                bounds,
                tuple(processing),
                tuple(pings),
                source["deployment"],
                source["archive_sha256"],
                tuple(qc),
            )
        )
    return slots


def _same_configuration(left: NativeSlot, right: NativeSlot) -> bool:
    # Primary complete-source configuration is always structural. Missing or
    # partial secondary observations are masked, not a new instrument mode.
    for index in range(4):
        missing = {"MISSING_CHANNEL", "PARTIAL_SOURCE_INTERVAL", "DUPLICATE_ROW"}
        if index and (left.qc[index] in missing or right.qc[index] in missing):
            continue
        if (
            not np.array_equal(left.bounds[index], right.bounds[index])
            or left.processing[index] != right.processing[index]
            or left.ping_counts[index] != right.ping_counts[index]
        ):
            return False
    return True


def _consecutive(left: NativeSlot, right: NativeSlot) -> bool:
    return bool(
        right.interval == left.interval + 1
        and np.timedelta64(55, "m") <= right.timestamp - left.timestamp <= np.timedelta64(65, "m")
        and _same_configuration(left, right)
    )


def _metadata(slot: NativeSlot) -> np.ndarray:
    meta = np.zeros((4, 10), dtype=np.float32)
    meta[:, 0] = np.asarray(FREQUENCIES_HZ) / 455000
    meta[:, 1:3] = 1
    meta[:, 3:5] = slot.bounds / 250
    # Process IDs establish boundaries; model sees known/unknown status, not IDs.
    meta[:, 6] = [process != "UNKNOWN" for process in slot.processing]
    return meta


def issue_windows(slots: list[NativeSlot], *, history: int = 96) -> dict[str, np.ndarray]:
    if history not in (24, 96):
        raise ValueError("Declared histories are 24 and 96.")
    if (
        len({slot.deployment for slot in slots}) > 1
        or len({slot.archive_sha256 for slot in slots}) > 1
    ):
        raise ValueError("A window source must be one deployment and archive.")
    if len({slot.interval for slot in slots}) != len(slots):
        raise ValueError("Duplicate source interval IDs.")
    by_id = {slot.interval: slot for slot in slots}
    rows = []
    for end in range(history - 1, max(history - 1, len(slots) - 9)):
        context = slots[end - history + 1 : end + 1]
        cutoff = context[-1]
        if not all(slot.observed[0] for slot in context) or not all(
            _consecutive(a, b) for a, b in pairwise(context)
        ):
            continue
        # Enforce broadcast metadata for every genuinely observed channel, also
        # across temporary absence where adjacent comparisons alone are insufficient.
        if any(not _same_configuration(cutoff, slot) for slot in context):
            continue
        meta = _metadata(cutoff)
        compatible = True
        references = [None] * 4
        for channel in range(4):
            observed_slots = [s for s in context if s.observed[channel]]
            if not observed_slots:
                continue
            reference = observed_slots[-1]
            references[channel] = reference
            meta[channel] = _metadata(reference)[channel]
            if any(
                not np.array_equal(_metadata(s)[channel], meta[channel])
                or s.processing[channel] != reference.processing[channel]
                or s.ping_counts[channel] != reference.ping_counts[channel]
                for s in observed_slots
            ):
                compatible = False
                break
        if not compatible:
            continue
        # Available archive extent is metadata; future values never decide issuance.
        future, future_observed = (
            np.zeros((3, 4, 4), dtype=np.float32),
            np.zeros((3, 4, 4), dtype=bool),
        )
        y, ym = np.zeros(3, dtype=np.float32), np.zeros(3, dtype=bool)
        dates = np.full(3, "", dtype="U10")
        future_ids = cutoff.interval + np.asarray([1, 3, 6])[:, None] + np.arange(4)
        contiguous_future = {}
        previous = cutoff
        for step in range(1, 10):
            current = by_id.get(cutoff.interval + step)
            if (
                current is None
                or not _consecutive(previous, current)
                or not _same_configuration(cutoff, current)
            ):
                break
            if any(
                current.observed[channel]
                and not np.array_equal(_metadata(current)[channel], meta[channel])
                for channel in range(4)
            ):
                break
            # Known flags alone cannot distinguish two processing IDs. Use the
            # last observed context configuration when the cutoff channel is missing.
            if any(
                reference is not None
                and current.qc[channel]
                not in {"MISSING_CHANNEL", "PARTIAL_SOURCE_INTERVAL", "DUPLICATE_ROW"}
                and (
                    current.processing[channel] != reference.processing[channel]
                    or current.ping_counts[channel] != reference.ping_counts[channel]
                    or not np.array_equal(current.bounds[channel], reference.bounds[channel])
                )
                for channel, reference in enumerate(references)
            ):
                break
            contiguous_future[step] = current
            previous = current
        future_metadata = np.zeros((3, 4, 4, 10), dtype=np.float32)
        future_processing = np.full((3, 4, 4), "UNKNOWN", dtype="U128")
        future_pings = np.zeros((3, 4, 4), dtype=np.int64)
        for h_index, horizon in enumerate((1, 3, 6)):
            for offset in range(4):
                item = contiguous_future.get(horizon + offset)
                if item is None:
                    continue
                future[h_index, offset] = item.values
                future_observed[h_index, offset] = item.observed
                future_metadata[h_index, offset] = _metadata(item)
                future_metadata[h_index, offset, :, -1] = horizon + offset
                future_processing[h_index, offset] = item.processing
                future_pings[h_index, offset] = item.ping_counts
                if offset == 0:
                    y[h_index], ym[h_index] = item.values[0], item.observed[0]
                    dates[h_index] = str(item.timestamp.astype("datetime64[D]"))
        query = np.tile(meta[0], (3, 1))
        query[:, -1] = [1, 3, 6]
        identity = f"native-v1:{cutoff.archive_sha256}:{cutoff.interval}:{history}:{cutoff.bounds[0].tolist()}"
        rows.append(
            {
                "x": np.asarray([s.values for s in context], dtype=np.float32),
                "observed": np.asarray([s.observed for s in context]),
                "metadata": meta,
                "future": future,
                "future_observed": future_observed,
                "y": y,
                "y_observed": ym,
                "query": query,
                "row_id": hashlib.sha256(identity.encode()).hexdigest(),
                "deployment": cutoff.deployment,
                "cutoff": cutoff.interval,
                "target_dates": dates,
                "context_ids": [s.interval for s in context],
                "future_ids": future_ids,
                "archive_sha256": cutoff.archive_sha256,
                "context_metadata": [_metadata(s) for s in context],
                "future_metadata": future_metadata,
                "processing_id_or_unknown": [s.processing for s in context],
                "future_processing_id_or_unknown": future_processing,
                "ping_counts": [s.ping_counts for s in context],
                "future_ping_counts": future_pings,
                "instrument_mode_or_unknown": "UNKNOWN",
                "orientation": "UNKNOWN",
                "frequency_hz": FREQUENCIES_HZ,
                "interval_start": [str(s.timestamp - np.timedelta64(30, "m")) for s in context],
                "interval_end": [str(s.timestamp + np.timedelta64(30, "m")) for s in context],
                "qc": [s.qc for s in context],
                "clock_status": "SOURCE_TIMEZONE_UNKNOWN",
            }
        )
    if not rows:
        return {"x": np.empty((0, history, 4), dtype=np.float32)}
    result = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    result["ssl_eligible"] = result["future_observed"][:, :, :, 0].all(axis=(1, 2))
    return result


def train_statistics(
    corpus: dict[str, np.ndarray], *, role: str, split_path: Path | None = None
) -> dict[str, np.ndarray]:
    if role != "train" or str(corpus.get("corpus_role")) != "train":
        raise ValueError("Scaler fitting requires TRAIN role.")
    if split_path is None or str(corpus.get("split_sha256")) != sha256(split_path):
        raise ValueError("Scaler fitting requires verified TRAIN split identity.")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    allowed = {
        s["deployment"]: s["archive_sha256"] for s in split["sources"] if s["role"] == "train"
    }
    if "archive_sha256" not in corpus or len(corpus["archive_sha256"]) != len(corpus["x"]):
        raise ValueError("Scaler fitting requires per-row TRAIN source provenance.")
    if any(
        allowed.get(str(deployment)) != str(archive)
        for deployment, archive in zip(corpus["deployment"], corpus["archive_sha256"], strict=True)
    ):
        raise ValueError("Scaler fitting requires TRAIN deployment and archive membership.")
    x, mask = corpus["x"], corpus["observed"]
    return {
        "channel_mean": np.asarray([x[:, :, i][mask[:, :, i]].mean() for i in range(4)]),
        "channel_std": np.asarray(
            [max(float(x[:, :, i][mask[:, :, i]].std()), 0.01) for i in range(4)]
        ),
    }


def materialize(split_path: Path, review_path: Path, role: str, output: Path) -> dict:
    split = check_numeric_review(split_path, review_path, role)
    pieces, reports = [], []
    for source in split["sources"]:
        if source["role"] != role:
            continue
        slots = read_source(source)
        arrays = issue_windows(slots, history=split["support_history"])
        reports.append(
            {
                "deployment": source["deployment"],
                "slots": len(slots),
                "issued": len(arrays["x"]),
                "native_bounds": sorted({tuple(s.bounds[0]) for s in slots}),
                "qc": {
                    str(q): sum(q in s.qc for s in slots) for q in {q for s in slots for q in s.qc}
                },
            }
        )
        if len(arrays["x"]):
            pieces.append(arrays)
    if not pieces:
        raise ValueError("No eligible native windows; no synthetic substitute.")
    joined = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
    joined["corpus_role"] = np.asarray(role)
    joined["split_sha256"] = np.asarray(sha256(split_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **joined)
    report = {
        "status": "MATERIALIZED_REAL_NATIVE_PRODUCTS",
        "role": role,
        "issued": len(joined["x"]),
        "ssl_eligible": int(joined["ssl_eligible"].sum()),
        "source_reports": reports,
        "npz_sha256": sha256(output),
        "split_sha256": sha256(split_path),
        "numeric_review_sha256": sha256(review_path),
        "reader_sha256": sha256(Path(__file__)),
        "contract_fields": [
            "value",
            "observed_mask",
            "censoring_or_qc",
            "frequency_hz",
            "interval_start",
            "interval_end",
            "clock_status",
            "geometry_kind",
            "upper_bound",
            "lower_bound",
            "orientation",
            "processing_id_or_unknown",
            "instrument_mode_or_unknown",
            "deployment_id_for_grouping",
        ],
        "interval_convention": "Source Date_M/Time_M center proxy +/-30min; exact publisher interval edges unknown",
        "censoring": "Unknown absolute censoring; sentinel/missing/partial QC explicit",
        "metadata_fields": METADATA_FIELDS,
        "metadata_encoding": "frequency/455000Hz; interval/3600s; geometry1=integrated; bounds/250m; orientation0=UNKNOWN; processing/instrument/clock known flags; relative offset in source intervals",
        "broadcast_metadata_invariant": "All observed context/future channels have matching encoded metadata; actual per-observation metadata and processing provenance retained separately",
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
