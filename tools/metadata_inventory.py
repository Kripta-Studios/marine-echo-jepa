"""Materialize source lineage and calendar coverage from existing extraction metadata.

This opens no acoustic payload and makes no calibrated support or day-eligibility claim.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    extracted = (
        ROOT.parent / "marine-echo-jepa-core/data/raw/pangaea/mosaic_azfp_down_2020_extracted"
    )
    manifest_path = extracted / ".extraction-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    split_path = ROOT / "reports/active/protocol.json"
    split = json.loads(split_path.read_text())["split"]["partitions"]
    rows, coverage, phases = [], {}, {}
    for record in manifest["files"]:
        name = record["path"]
        match = re.fullmatch(r"(\d{8})\.(\d{2})([A-Z])", name)
        if not match:
            continue
        stamp = datetime.strptime(match[1], "%y%m%d%H").replace(tzinfo=UTC)
        day = stamp.date().isoformat()
        coverage.setdefault(day, []).append(stamp.hour)
        phase = match[2]
        phases.setdefault(phase, []).append(stamp.isoformat())
        partition = next(
            (
                key
                for key, bound in split.items()
                if datetime.fromisoformat(bound["start"])
                <= stamp
                < datetime.fromisoformat(bound["end"])
            ),
            "outside_complete_calendar_days",
        )
        rows.append(
            {
                "source_file": name,
                "source_file_sha256": record["sha256"],
                "source_archive_sha256": manifest["archive_sha256"],
                "dataset_id": "mosaic_azfp_down_2020",
                "instrument_id": "AZFP55170",
                "filename_hour_utc": stamp.isoformat(),
                "sampling_phase_token": phase,
                "chunk_suffix": match[3],
                "calendar_partition_from_filename_only": partition,
                "calibration_status": "UNVERIFIED",
                "raw_timestamps_decoded": False,
            }
        )
    out = ROOT / "evidence/continuation/metadata"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out / "raw_inventory.parquet")
    start = datetime.fromisoformat(split["train"]["start"])
    end = datetime.fromisoformat(split["test"]["end"])
    with (out / "coverage_by_day.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "date_utc",
                "raw_chunks",
                "distinct_filename_hours",
                "duplicate_hour_chunks",
                "calibrated_usable_day",
            ],
        )
        writer.writeheader()
        stamp = start
        while stamp < end:
            hours = coverage.get(stamp.date().isoformat(), [])
            writer.writerow(
                {
                    "date_utc": stamp.date().isoformat(),
                    "raw_chunks": len(hours),
                    "distinct_filename_hours": len(set(hours)),
                    "duplicate_hour_chunks": len(hours) - len(set(hours)),
                    "calibrated_usable_day": "NOT_EVALUATED",
                }
            )
            stamp += timedelta(days=1)
    result = {
        "status": "METADATA_ONLY_NOT_CALIBRATED_PREPROCESSING",
        "acoustic_payloads_opened": 0,
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "raw_chunks": len(rows),
        "partition_chunk_counts": dict(
            Counter(r["calendar_partition_from_filename_only"] for r in rows)
        ),
        "sampling_phase_extents_not_verified_configuration_boundaries": {
            key: {"first_filename_hour": min(times), "last_filename_hour": max(times)}
            for key, times in phases.items()
        },
        "limitations": [
            "Filename hours are not verified raw timestamps; no scientific split assignments are approved.",
            "Repeated hourly suffixes are consecutive chunks, not duplicate independent observations.",
            "Presence of an hourly chunk does not establish temporal, range, target or daily support.",
        ],
    }
    (out / "inventory_report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
