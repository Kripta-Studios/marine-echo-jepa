"""Bounded metadata-only check for genuine layer exports in acquired archives."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    split = json.loads((ROOT / "configs/native_ssl_split_v1.json").read_text(encoding="utf-8"))
    reports = []
    for source in split["sources"]:
        with zipfile.ZipFile(source["path"]) as archive:
            names = sorted(archive.namelist())
            layer_names = [name for name in names if "_038_" in name and "60minPartition" in name]
            daily_integrated = [
                name
                for name in names
                if "_038_" in name and "FullDepth.csv" in name and "60min" not in name
            ]
            report = {
                "file_id": source["file_id"],
                "role": source["role"],
                "source_archive_sha256": source["archive_sha256"],
                "layer_export_members": len(layer_names),
                "daily_integrated_named_members": len(daily_integrated),
                "numeric_acoustic_values_interpreted": False,
                "profile_training_status": "NOT_RUN_INTEGRATED_CAMPAIGN_INDEPENDENT",
            }
            # Final-test metadata inventory names suffice; no new member payload access.
            if layer_names and source["role"] != "final_test":
                coordinates = set()
                intervals = set()
                with archive.open(layer_names[0]) as stream:
                    reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig"))
                    for index, row in enumerate(reader):
                        if index >= 5000:
                            break
                        coordinates.add(
                            (row["Layer"], row["Layer_depth_min"], row["Layer_depth_max"])
                        )
                        intervals.add(row["Interval"])
                report.update(
                    sample_member=layer_names[0],
                    sample_rows=min(index + 1, 5000),
                    observed_layer_coordinates=sorted(coordinates),
                    sample_unique_intervals=len(intervals),
                    geometry_evidence="SOURCE_PARTITION_METADATA_ONLY_NO_PROFILE_VALUES_PARSED",
                )
            reports.append(report)
    output = ROOT / "evidence/ssl-research-v1/profile-inventory.json"
    output.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            [
                {key: value for key, value in report.items() if key != "observed_layer_coordinates"}
                for report in reports
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
