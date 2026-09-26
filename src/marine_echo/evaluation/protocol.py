"""Calendar boundaries independent of observed acoustic values."""

import hashlib
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

from marine_echo.serving.api import utc


def calendar_split(start: str, end: str) -> dict[str, Any]:
    first, last = utc(start), utc(end)
    if first.time().isoformat() != "00:00:00":
        first = first.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    last = last.replace(hour=0, minute=0, second=0, microsecond=0)
    days = (last - first).days
    if days < 1:
        raise ValueError("No complete UTC calendar days.")
    # Cumulative boundaries, rounded down once. Final partition retains the remainder.
    offsets = [0, int(days * 0.60), int(days * 0.75), int(days * 0.85), days]
    partitions = {}
    for index, name in enumerate(("train", "validation", "calibration", "test")):
        partitions[name] = {
            "start": (first + timedelta(days=offsets[index])).isoformat().replace("+00:00", "Z"),
            "end": (first + timedelta(days=offsets[index + 1])).isoformat().replace("+00:00", "Z"),
            "calendar_days": offsets[index + 1] - offsets[index],
        }
    return {
        "complete_days": days,
        "rounding": "floor_cumulative_calendar_fraction",
        "partitions": partitions,
    }


def digest_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def create_protocol(config: Path, datasets: Path, output: Path) -> dict[str, Any]:
    experiment = json.loads(config.read_text(encoding="utf-8"))
    registry = json.loads(datasets.read_text(encoding="utf-8"))
    source = next(d for d in registry["datasets"] if d["id"] == registry["active_dataset"])
    document = {
        "schema_version": "1.0",
        "status": "DRAFT_REQUIRES_R1_AND_R2",
        "dataset_id": source["id"],
        "configuration": experiment,
        "split": calendar_split(source["start"], source["end"]),
        "config_sha256": digest_file(config),
        "dataset_config_sha256": digest_file(datasets),
        "availability_assumption": "zero_latency_replay_only_not_measured_delivery",
        "test_opened": False,
        "calibration_verified": False,
    }
    serialized = json.dumps(document, indent=2, allow_nan=False) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != serialized:
        raise ValueError("Protocol already exists with different content; create a new version.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    return document


def verify_seal(protocol_path: Path, seal_path: Path) -> dict[str, Any]:
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("review_verdict") != "APPROVED" or seal.get("review_id") != "R2":
        raise ValueError("Independent R2 approval is required before test evaluation.")
    if digest_file(protocol_path) != seal["protocol_sha256"]:
        raise ValueError("Protocol changed after freeze.")
    for entry in seal["files"]:
        if digest_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("Frozen artifact changed.")
    return seal
