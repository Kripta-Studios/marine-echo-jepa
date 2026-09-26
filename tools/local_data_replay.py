"""Create a bounded real AZFP raw-count replay for the offline UI."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import echopype as ep

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from marine_echo.data.raw_replay import aggregate_raw_count_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a one-day uncalibrated raw-count replay."
    )
    parser.add_argument("--extracted", type=Path, required=True)
    parser.add_argument("--day", type=date.fromisoformat, required=True)
    parser.add_argument("--chunk-suffix", default="01B")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(
        (args.extracted / ".extraction-manifest.json").read_text(encoding="utf-8")
    )
    known = {item["path"]: item for item in manifest["files"]}
    xml = args.extracted / "20021600.XML"
    if not xml.is_file():
        raise FileNotFoundError(xml)
    rows_by_start: dict[str, dict[str, object]] = {}
    sources = []
    channels = [38000.0, 125000.0, 200000.0, 455000.0]
    for hour in range(24):
        filename = f"{args.day:%y%m%d}{hour:02d}.{args.chunk_suffix}"
        if filename not in known:
            raise FileNotFoundError(f"Required same-mode chunk is absent: {filename}")
        path = args.extracted / filename
        ed = ep.open_raw(path, sonar_model="AZFP", xml_path=xml)
        beam = ed["Sonar/Beam_group1"]
        if beam["frequency_nominal"].values.tolist() != channels:
            raise ValueError(f"Unexpected channel map in {filename}")
        raw = beam["backscatter_r"]
        if raw.attrs.get("units") != "count":
            raise ValueError(f"Unexpected units in {filename}")
        file_rows = aggregate_raw_count_file(beam["ping_time"].values, raw.values)
        for row in file_rows:
            key = row["bin_start_utc"]
            if key in rows_by_start:
                raise ValueError(f"Overlapping decoded ping bins: {key}")
            rows_by_start[key] = row
        sources.append(
            {
                "name": filename,
                "sha256": known[filename]["sha256"],
                "pings": int(beam.sizes["ping_time"]),
                "first_ping_utc": str(beam["ping_time"].values[0]) + "Z",
                "last_ping_utc": str(beam["ping_time"].values[-1]) + "Z",
            }
        )
        del ed
    start = datetime.combine(args.day, datetime.min.time(), tzinfo=timezone.utc)
    rows = []
    for index in range(96):
        stamp = start + timedelta(minutes=15 * index)
        key = stamp.isoformat(timespec="seconds").replace("+00:00", "Z")
        row = rows_by_start.get(key)
        if row is None:
            row = {
                "bin_start_utc": key,
                "bin_end_utc": (stamp + timedelta(minutes=15))
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
                "ping_count": 0,
                "counts": None,
                "status": "MISSING",
            }
        else:
            row["status"] = "OBSERVED"
        row["event_time_utc"] = row["bin_end_utc"]
        rows.append(row)
    result = {
        "schema_version": "1.0",
        "dataset_id": "mosaic_azfp_down_2020",
        "dataset_version": manifest["archive_sha256"],
        "calibration_status": "RAW_COUNTS_ONLY",
        "calibrated": False,
        "data_kind": "public_real",
        "unit": "AZFP raw digitizer count",
        "units": "raw_counts",
        "range_axis": "sample_index_not_meters",
        "range_convention": "sample_index",
        "sample_bins": 64,
        "sample_bin_rule": "64 equal-width partitions of the 999 sample indices; means of finite counts across pings and sample indices",
        "frequency_hz": channels,
        "time_bin_rule": "UTC half-open [start,end), 15 minutes",
        "included_chunk_suffix": args.chunk_suffix,
        "excluded_same_hour_chunks_note": "Other suffixes are separate acquisition chunks and are retained in the source inventory; this diagnostic uses one consistent suffix only.",
        "sources": sources,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=args.output.parent,
        prefix=args.output.name + ".stage.",
        delete=False,
    ) as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
        staged = Path(stream.name)
    os.replace(staged, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": len(rows),
                "observed_rows": sum(row["status"] == "OBSERVED" for row in rows),
                "sources": len(sources),
                "raw_pings": sum(item["pings"] for item in sources),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
