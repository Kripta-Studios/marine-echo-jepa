"""Compare fixed environmental matches using both preserved acoustic-buoy trackers.

Corrects the earlier Iridium-only coverage audit; no acoustic outcome is read.
Nearest time, never nearest spatial distance, selects a trajectory observation.
"""

import csv
import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import netCDF4
from continuation_records import write_report

from marine_echo.data.environment import distance_km, parse_iridium_position

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    iridium = ROOT / "data/raw/pangaea/949811/iridium-msgs.txt"
    xeos = ROOT / "data/raw/pangaea/949811/xeos-rover-mosaicdown-transponder-16feb-02aug2020.csv"
    profiles = ROOT / "data/raw/environment/mosaic_core/MOSAiC_daily_profiles.nc"
    inventory = json.loads(
        (ROOT / "data/manifests/mosaic_azfp_down_2020/source_inventory.json").read_text()
    )
    for path in (iridium, xeos):
        expected = next(item["sha256"] for item in inventory["files"] if item["name"] == path.name)
        if sha(path) != expected:
            raise ValueError("Tracker differs from registered source bytes.")
    env_inventory = json.loads(
        (ROOT / "evidence/calibration/environment_inventory.json").read_text()
    )
    if sha(profiles) != next(
        item["sha256"] for item in env_inventory if item["name"] == profiles.name
    ):
        raise ValueError("Environmental source differs from inventory.")
    points = []
    for line in iridium.read_text().splitlines():
        try:
            t, lat, lon = parse_iridium_position(line)
            points.append((t, lat, lon, "Iridium_code1"))
        except (ValueError, IndexError):
            pass
    with xeos.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["Device"] != "MOSAICdown":
                continue
            match = re.search(
                r"Timestamp: ([^,]+).*Latitude: ([\d.-]+), Longitude: ([\d.-]+)", row["Message"]
            )
            if match:
                text, lat, lon = match.groups()
                stamp = datetime.fromisoformat(text)
                if stamp.tzinfo is None:
                    raise ValueError("XEOS UTC observation time required.")
                distance_km(float(lat), float(lon), float(lat), float(lon))
                points.append((stamp, float(lat), float(lon), "XEOS_MOSAICdown"))
    protocol_path = ROOT / "reports/active/protocol.json"
    splits = json.loads(protocol_path.read_text())["split"]["partitions"]
    environmental = []
    with netCDF4.Dataset("memory", memory=profiles.read_bytes()) as ds:
        for index, t in enumerate(ds["time"][:]):
            stamp = (datetime.fromordinal(int(t)) + timedelta(days=float(t) % 1 - 366)).replace(
                tzinfo=UTC
            )
            if not "2020-02-17" <= stamp.date().isoformat() < "2020-08-02":
                continue
            environmental.append(
                {
                    "source": "daily_composite",
                    "day": stamp.date().isoformat(),
                    "time": stamp,
                    "latitude": float(ds["latitude"][index]),
                    "longitude": float(ds["longitude"][index]),
                }
            )
    sit_path = ROOT / "evidence/continuation/sit_applicability.json"
    for row in json.loads(sit_path.read_text())["profiles"]:
        environmental.append(
            {
                "source": row["buoy"],
                "day": row["day"],
                "time": datetime.fromisoformat(row["observation_time"]),
                "latitude": row["latitude"],
                "longitude": row["longitude"],
            }
        )
    rows = []
    policies = {
        "Iridium_only": [p for p in points if p[3] == "Iridium_code1"],
        "XEOS_only": [p for p in points if p[3] == "XEOS_MOSAICdown"],
        "nearest_time_both": points,
    }
    for profile in environmental:
        stamp = profile["time"]
        partition = next(
            part
            for part, bounds in splits.items()
            if bounds["start"][:10] <= profile["day"] < bounds["end"][:10]
        )
        for policy, candidates in policies.items():
            # Predetermined tie ordering by source name; no spatial/performance selection.
            nearest = min(
                candidates, key=lambda p: (abs((p[0] - stamp).total_seconds()), p[0], p[3])
            )
            gap = abs((nearest[0] - stamp).total_seconds()) / 3600
            distance = distance_km(
                profile["latitude"], profile["longitude"], nearest[1], nearest[2]
            )
            rows.append(
                {
                    "environmental_source": profile["source"],
                    "day": profile["day"],
                    "observation_time": stamp.isoformat(),
                    "partition": partition,
                    "trajectory_policy": policy,
                    "trajectory_source": nearest[3],
                    "trajectory_time": nearest[0].isoformat(),
                    "gap_hours": gap,
                    "distance_km": distance,
                    "illustrative_5km_24h": gap <= 24 and distance <= 5,
                }
            )
    summaries = []
    for source in sorted({row["environmental_source"] for row in rows}):
        for policy in policies:
            selected = [
                row
                for row in rows
                if row["environmental_source"] == source and row["trajectory_policy"] == policy
            ]
            summaries.append(
                {
                    "environmental_source": source,
                    "trajectory_policy": policy,
                    "matched_days": {
                        part: sum(
                            row["illustrative_5km_24h"]
                            for row in selected
                            if row["partition"] == part
                        )
                        for part in splits
                    },
                    "profile_days": len(selected),
                }
            )
    report = {
        "status": "DESCRIPTIVE_MATCHING_ONLY_NOT_CALIBRATION_APPROVAL",
        "benchmark_eligible": False,
        "checked_at": datetime.now(UTC).isoformat(),
        "code_sha256": sha(Path(__file__)),
        "input_hashes": {
            str(p.relative_to(ROOT)): sha(p)
            for p in (iridium, xeos, profiles, sit_path, protocol_path)
        },
        "tracker_records": {
            name: sum(p[3] == name for p in points) for name in ("Iridium_code1", "XEOS_MOSAICdown")
        },
        "correction": "The earlier primary numerical assay described environmental matching using Iridium alone. It omitted nearer-in-time independent XEOS fixes during Iridium gaps. Preserve old result as limited Iridium-only evidence, not overall coverage.",
        "policy": "Fixed5km/24h illustrative screen retained; nearest absolute time with deterministic ties, never nearest spatial distance; no interpolation or threshold tuning; no prospective predictor availability assumed.",
        "summaries": summaries,
        "matches": rows,
        "limitations": [
            "Tracker QC and source fix semantics still need review",
            "Near-time position does not prove water-column applicability",
            "Missing environmental days and ranges remain unsupported",
            "Acoustic test outcomes remain unopened",
        ],
    }
    write_report(ROOT / "evidence/continuation/trajectory_matches.json", report)
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
