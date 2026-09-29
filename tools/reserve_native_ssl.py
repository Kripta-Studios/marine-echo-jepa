"""Reserve source-native study roles using existing metadata, never acoustic values."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    scout_path = ROOT / "evidence/aeon_scale/metadata_scout_20260928.json"
    scout = json.loads(scout_path.read_text(encoding="utf-8"))
    sources = []
    roles = {61937263: "train", 61937272: "train", 61937266: "development", 61937278: "final_test"}
    for item in scout["files"]:
        path = Path("E:/marine-echo-jepa-scale/data") / item["name"]
        sha = digest(path)
        if sha != item["local_sha256"]:
            raise ValueError(f"Source hash mismatch: {path}")
        site = "AEON2_ECS" if item["file_id"] == 61937278 else "AEON4_JOB"
        sources.append(
            {
                "file_id": item["file_id"],
                "path": str(path.resolve()),
                "archive_sha256": sha,
                "site": site,
                "deployment": f"{site}:{item['file_id']}",
                "role": roles[item["file_id"]],
                "start_date": item["first_date_field_38khz"],
                "end_date_inclusive": item["last_date_field_38khz"],
                "geometry": item["geometry_histogram_38khz"],
                "complete_ping_counts": [150, 180]
                if site == "AEON2_ECS"
                else [item["modal_complete_ping_count"]],
                "historical_exposure": "METADATA_ONLY_RECORDED_SCOUT_NO_FITTED_ANCESTOR",
                "numeric_access": "RESERVED_PENDING_DISTINCT_PREFIT",
            }
        )
    additions = [
        (
            61937275,
            "data/prospective/aeon-external-20260928/AEON3_GEB_Feb2023-Feb2024_AZFP_Sv.zip",
            "AEON3_GEB",
            "train",
            "20230201",
            "20240229",
            "b6d8380ed986c91d565ea7d669761ff8f66da8f2b5cd3e794fbf71c040bf1d37",
            "FITTED_EXPANDED_ANCESTOR_TRAIN_AND_HISTORICAL_OUTCOMES",
        ),
        (
            61937281,
            "data/prospective/aeon-azfp-20260927/AEON3_GEB_Mar2024-Mar2025_AZFP_Sv.zip",
            "AEON3_GEB",
            "train",
            "20240306",
            "20241007",
            "4e72dd4dbec707b6bf15168e51f380cbe9145a78b595ef886d78cc3806c0ecde",
            "FITTED_ANCESTOR_TRAIN_ONLY_ALLOWED_DATES_OTHER_PARTITIONS_EXCLUDED",
        ),
        (
            61937269,
            "data/prospective/aeon-external-20260928/AEON2_ECS_Apr2024-Apr2025_AZFP_Sv.zip",
            "AEON2_ECS",
            "final_test",
            "20240401",
            "20250430",
            "d0c57bcf73a09ad3fad1f6be3704d25263ce2a1842080dfe1ba680b57527364b",
            "HISTORICAL_EXTERNAL_COMPATIBILITY_INVESTIGATION_NOT_UNIVERSALLY_SEALED",
        ),
    ]
    for fid, relative, site, role, start, end, expected, exposure in additions:
        path = ROOT / relative
        sha = digest(path)
        if sha != expected:
            raise ValueError(f"Source hash mismatch: {path}")
        sources.append(
            {
                "file_id": fid,
                "path": str(path.resolve()),
                "archive_sha256": sha,
                "site": site,
                "deployment": f"{site}:{fid}",
                "role": role,
                "start_date": start,
                "end_date_inclusive": end,
                "geometry": "NATIVE_SOURCE_FIELDS_REQUIRED",
                "complete_ping_counts": [150],
                "historical_exposure": exposure,
                "numeric_access": "RESERVED_PENDING_DISTINCT_PREFIT",
            }
        )
    for source in sources:
        with zipfile.ZipFile(source["path"]) as archive:
            members = sorted((i.filename, i.file_size, i.CRC) for i in archive.infolist())
        source["zip_inventory_sha256"] = hashlib.sha256(
            json.dumps(members, separators=(",", ":")).encode()
        ).hexdigest()
        source["member_count"] = len(members)
        source["profile_inventory_names"] = [name for name, _, _ in members if "Partition" in name]
    result = {
        "schema_version": "native_acoustic_ssl_v1",
        "status": "RESERVED_BEFORE_NEW_NUMERICAL_SELECTION",
        "owner_authorization": "2026-09-29 model-first mission and explicit native contract authorization",
        "policy": "AEON2_WHOLE_SITE_TEST_AEON4_LATEST_DEPLOYMENT_DEV_EARLIER_TRAIN",
        "new_weight_ancestors": "NONE_RANDOM_INITIALIZATION_TRAIN_ONLY_NEW_SCALERS",
        "scout_sha256": digest(scout_path),
        "protocol_path": "docs/adr/0015-native-acoustic-ssl-research.md",
        "protocol_sha256": digest(ROOT / "docs/adr/0015-native-acoustic-ssl-research.md"),
        "sources": sources,
        "history_intervals": [24, 96],
        "support_history": 96,
        "horizons": [1, 3, 6],
        "future_block_intervals": 4,
        "frequency_hz": [38000, 125000, 200000, 455000],
        "numeric_values_opened_by_reservation": False,
        "test_access": "PROHIBITED_UNTIL_FROZEN_FINALISTS_AND_INDEPENDENT_REVIEW",
    }
    output = ROOT / "configs/native_ssl_split_v1.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "split_path": str(output),
                "split_sha256": digest(output),
                "sources": len(sources),
                "numeric_values_opened": False,
            }
        )
    )


if __name__ == "__main__":
    main()
