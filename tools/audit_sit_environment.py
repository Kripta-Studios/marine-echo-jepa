"""Metadata and hydrographic-QC audit of original MOSAiC SIT environmental records."""

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from continuation_records import write_report

from marine_echo.data.environment import distance_km, parse_iridium_position

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    started = time.monotonic()
    inventory = json.loads((ROOT / "evidence/continuation/sit_inventory.json").read_text())
    protocol = json.loads((ROOT / "reports/active/protocol.json").read_text())
    gps = []
    for line in (ROOT / "data/raw/pangaea/949811/iridium-msgs.txt").read_text().splitlines():
        try:
            gps.append(parse_iridium_position(line))
        except (ValueError, IndexError):
            pass
    if not gps:
        raise ValueError("No accepted acoustic-buoy trajectory records.")
    rows, summaries = [], []
    for source in inventory["files"]:
        path = ROOT / source["path"]
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != source["sha256"]:
                raise ValueError("SIT environmental source changed.")
        with path.open(encoding="utf-8") as stream:
            header_lines = 0
            for line in stream:
                header_lines += 1
                if line.strip() == "*/":
                    break
                if header_lines > 200:
                    raise ValueError("Unexpected PANGAEA metadata header.")
        selections = []
        columns = [
            "Date/Time",
            "Latitude",
            "Longitude",
            "Gear ID",
            "Temp [°C]",
            "Press [dbar]",
            "Depth water [m]",
            "Sal",
            "QF water temp",
            "QF water press",
            "QF water depth",
            "QF sal",
            "Flag buoy",
        ]
        for chunk in pd.read_csv(
            path, sep="\t", skiprows=header_lines, usecols=columns, chunksize=100_000
        ):
            stamp = pd.to_datetime(chunk["Date/Time"], utc=True)
            midnight = stamp.dt.round("D")
            offset = (stamp - midnight).dt.total_seconds().abs()
            keep = (midnight >= "2020-02-17") & (midnight < "2020-08-02") & (offset <= 120)
            if keep.any():
                selected = chunk.loc[keep].copy()
                selected["day"] = midnight[keep].dt.strftime("%Y-%m-%d")
                selected["offset_seconds"] = offset[keep]
                selections.append(selected)
        selected_rows = pd.concat(selections, ignore_index=True)
        for day, group in selected_rows.groupby("day", sort=True):
            best_offset = group["offset_seconds"].min()
            group = group[group["offset_seconds"] == best_offset]
            # Resolve a symmetric tie by earlier observation, not by water values or distance.
            selected_stamp = group["Date/Time"].min()
            group = group[group["Date/Time"] == selected_stamp].sort_values("Gear ID")
            if group["Gear ID"].duplicated().any():
                raise ValueError("Duplicate gear/time in environmental profile.")
            stamp = datetime.fromisoformat(selected_stamp).replace(tzinfo=UTC)
            nearest = min(gps, key=lambda item: abs((item[0] - stamp).total_seconds()))
            lat, lon = float(group["Latitude"].iloc[0]), float(group["Longitude"].iloc[0])
            if group["Latitude"].nunique() != 1 or group["Longitude"].nunique() != 1:
                raise ValueError("SIT profile has inconsistent coordinates.")
            separation = distance_km(lat, lon, nearest[1], nearest[2])
            hours = abs((nearest[0] - stamp).total_seconds()) / 3600
            partition = next(
                part
                for part, bounds in protocol["split"]["partitions"].items()
                if bounds["start"][:10] <= day < bounds["end"][:10]
            )
            quality = (
                group[["QF water temp", "QF water press", "QF water depth", "QF sal"]]
                .isin([1, 2])
                .all(axis=1)
            )
            finite = np.isfinite(
                group[["Temp [°C]", "Press [dbar]", "Depth water [m]", "Sal"]].to_numpy()
            ).all(axis=1)
            row = {
                "buoy": source["buoy"],
                "day": day,
                "partition": partition,
                "observation_time": stamp.isoformat(),
                "latitude": lat,
                "longitude": lon,
                "distance_km": separation,
                "nearest_azfp_position_hours": hours,
                "matched_illustrative_5km_24h": separation <= 5 and hours <= 24,
                "good_or_modified_gears": int((quality & finite).sum()),
                "measured_depth_extent_m": [
                    -float(group["Depth water [m]"].max()),
                    -float(group["Depth water [m]"].min()),
                ],
                "flag_buoy_values": sorted(int(v) for v in group["Flag buoy"].unique()),
                "available_at": None,
                "benchmark_eligible": False,
            }
            rows.append(row)
        own = [row for row in rows if row["buoy"] == source["buoy"]]
        counts = {
            part: sum(r["matched_illustrative_5km_24h"] for r in own if r["partition"] == part)
            for part in protocol["split"]["partitions"]
        }
        test = [r for r in own if r["partition"] == "test"]
        summary = {
            "buoy": source["buoy"],
            "profile_days": len(own),
            "matched_days_by_partition": counts,
            "test_distance_km_min_median_max": [
                float(v) for v in np.quantile([r["distance_km"] for r in test], [0, 0.5, 1])
            ],
            "test_days_good_at_all_five_gears": sum(r["good_or_modified_gears"] == 5 for r in test),
        }
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    report = {
        "status": "ENVIRONMENTAL_APPLICABILITY_AUDIT_NOT_CALIBRATION_APPROVAL",
        "benchmark_eligible": False,
        "checked_at": datetime.now(UTC).isoformat(),
        "sources": inventory["files"],
        "summaries": summaries,
        "profiles": rows,
        "selection_rule": "All four recovered SIT buoys; nearest observation within120s of each fixed calendar midnight, earlier timestamp on ties; no acoustic-value selection",
        "matching_rule": "Same unapproved illustrative <=5km and <=24h to nearest accepted Iridium position; no interpolation; not scientific eligibility",
        "variables": "In-situ ITS-90 temperature; PSS-78 Practical Salinity; pressure dbar; negative-up depth from publisher",
        "qc_rule": "Record count of finite T/S/P/depth with flags1 good or2 modified; no claim of full-column coverage or interpolation approval",
        "availability": "Flag1 records transmitted observations but not shore receipt. Flag0 is internally recorded. Processed QC includes retrospective filters/corrections; neither is approved as issue-time input.",
        "limitations": [
            "AZFP Iridium fix-code semantics/gaps/duplicates remain under review",
            "SIT GPS may be interpolated during dataset processing",
            "Five sensors do not resolve every water layer",
            "No AZFP acoustic outcomes were opened",
        ],
        "elapsed_seconds": time.monotonic() - started,
        "final_process_rss_gib_not_peak": psutil.Process().memory_info().rss / 1024**3,
    }
    write_report(ROOT / "evidence/continuation/sit_applicability.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
