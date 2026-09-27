"""Reconcile local evidence without loading or printing protected acoustic outcomes."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from continuation_records import write_report

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    output = ROOT / "evidence/continuation/state_audit.json"
    protocol = json.loads((ROOT / "reports/active/protocol.json").read_text())
    inventory = json.loads(
        (ROOT / "data/manifests/mosaic_azfp_down_2020/source_inventory.json").read_text()
    )
    source = ROOT / "data/raw/pangaea/949811"
    sources = []
    for row in inventory["files"]:
        path = source / row["name"]
        sources.append(
            {
                "name": path.name,
                "present": path.is_file(),
                "actual_bytes": path.stat().st_size if path.is_file() else None,
                "registered": row,
                "hash_rechecked": False,
                "size_matches_registered": path.is_file() and path.stat().st_size == row["bytes"],
            }
        )
    replay = []
    # Read only timestamps/identities from already exposed artifacts, never return count arrays.
    for folder in (
        "evidence/data",
        "release/meeting-20260926-r2/demo/artifacts",
        "release/demo/artifacts",
    ):
        for path in (ROOT / folder).glob("*.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or "rows" not in value:
                continue
            times = [r.get("event_time_utc", r.get("bin_end_utc")) for r in value["rows"]]
            times = sorted(t for t in times if t)
            if times:
                test = protocol["split"]["partitions"]["test"]
                replay.append(
                    {
                        "path": path.relative_to(ROOT).as_posix(),
                        "sha256": digest(path),
                        "bytes": path.stat().st_size,
                        "rows": len(times),
                        "first_bin_end": times[0],
                        "last_bin_end": times[-1],
                        "overlaps_candidate_test": times[-1] > test["start"]
                        and times[0] <= test["end"],
                    }
                )
    registries = []
    for name in ("orchestration/run_ledger.json", "reports/active/training_registry.json"):
        value = json.loads((ROOT / name).read_text())
        statuses: dict[str, int] = {}
        for run in value["runs"]:
            statuses[run["status"]] = statuses.get(run["status"], 0) + 1
        registries.append({"path": name, "sha256": digest(ROOT / name), "statuses": statuses})
    package = ROOT / "release/meeting-20260926-r2.zip"
    with zipfile.ZipFile(package) as archive:
        package_members = len(archive.infolist())
    census_exposure = []
    for name in ("train_census_execution.json", "train_census_v2_execution.json"):
        path = ROOT / "evidence/continuation" / name
        if path.exists():
            census = json.loads(path.read_text(encoding="utf-8"))
            census_exposure.append(
                {
                    "execution_report": path.relative_to(ROOT).as_posix(),
                    "sha256": digest(path),
                    "status": census["status"],
                    "attempted_train_dates": list(census["days"]),
                    "completed_train_dates": [
                        day for day, entry in census["days"].items() if entry["exit_code"] == 0
                    ],
                    "failed_dates_preserved_as_partial_access": [
                        day for day, entry in census["days"].items() if entry["exit_code"] != 0
                    ],
                    "held_out_acoustic_payloads_processed_recorded": census[
                        "held_out_acoustic_payloads_processed"
                    ],
                    "exposure_kind": "Automated TRAIN parsing/calibration/QC; counts/manifests inspected, no full-census echograms visualized",
                }
            )
    report = {
        "checked_at": datetime.now(UTC).isoformat(),
        "source_verification_scope": "existence/size against preserved prior hashes; no repeat multi-GB hashing",
        "sources": sources,
        "registries": registries,
        "candidate_split": protocol["split"],
        "replay_exposure": replay,
        "train_census_exposure": census_exposure,
        "other_known_exposure": [
            {
                "period": "2020-02-16T14:00:00Z/2020-02-16T15:00:00Z",
                "kind": "partial deployment diagnostic probe outside complete-day split",
            },
            {
                "period": "2020-03-03T00:00:00Z/2020-03-03T01:00:00Z",
                "kind": "training acoustic numerical assay in this continuation",
            },
            {
                "period": "2020-03-03T00:00:00Z/2020-03-04T00:00:00Z",
                "kind": "Complete TRAIN day processed and visualized for independent candidate QC review; failed v1 and corrected v2 retained.",
                "evidence": "evidence/continuation/train-candidate-v2/training-review.html",
            },
            {
                "periods": ["2020-02-17", "2020-03-01", "2020-04-01", "2020-05-01"],
                "kind": "Four complete outcome-independent monthly TRAIN days calibrated and visualized, plus the separate March3 day; coordinator and independent reviewer inspected all five.",
                "evidence": "evidence/continuation/monthly_train_execution.json",
            },
            {
                "dataset": "ooi_ce04osps_ek60_2017",
                "source": "OOI-D20170821-T000000.raw",
                "kind": "Historical one-file acoustic calibration diagnostic already inspected; never treat this period as sealed if a D2 protocol is later created.",
                "evidence": "evidence/data/ooi_calibration.json",
            },
            {
                "dataset": "ooi_ce04osps_ek60_2017",
                "period": "2017-08-20",
                "kind": "Continuation FullNC HTTP/HDF5 metadata and coordinates only; no Sv array values indexed/read by reviewer. Bounded range transport can contain uninterpreted adjacent bytes.",
                "evidence": "evidence/continuation/ooi_fallback_disposition.json",
            },
            {
                "period": "2020-02-17/2020-08-02",
                "kind": "Environmental measurements, trajectories and regional physical envelopes inspected over the full calendar, including candidate test dates. This is environmental-data exposure, not acoustic outcome exposure; all sources remain retrospective and unavailable as issue-time covariates.",
                "evidence": [
                    "evidence/continuation/trajectory_matches.json",
                    "evidence/continuation/sit_applicability.json",
                    "evidence/continuation/fixed_environment_sensitivity.json",
                    "evidence/continuation/depth_environment_sensitivity.json",
                ],
            },
        ],
        "replay_serialization_lineage": {
            "tracked_vs_packaged_semantically_equal": json.loads(
                (ROOT / "evidence/data/diagnostic_raw_replay_2020-02-17.json").read_text()
            )
            == json.loads(
                (ROOT / "release/meeting-20260926-r2/demo/artifacts/observations.json").read_text()
            ),
            "note": "Historical report hash describes packaged JSON serialization; preserve both bytes and hashes.",
        },
        "exposure_disposition": "EXPOSURE_REVIEW_REQUIRED"
        if any(
            row["held_out_acoustic_payloads_processed_recorded"] is not False
            for row in census_exposure
        )
        or any(row["overlaps_candidate_test"] for row in replay)
        else "NO_KNOWN_TEST_OUTCOME_EXPOSURE_IN_AUDITED_REPLAYS_NOT_PROOF_OF_SEAL",
        "exposure_limitations": [
            "Browser human viewing history is not exhaustively logged.",
            "Prior agent context and discarded artifacts cannot be reconstructed.",
            "No R2 approval or test seal exists; do not infer one from zero runs.",
        ],
        "historical_release": {
            "path": str(package.relative_to(ROOT)),
            "sha256": digest(package),
            "bytes": package.stat().st_size,
            "members": package_members,
        },
    }
    write_report(output, report)
    print(
        json.dumps(
            {
                "report": str(output.relative_to(ROOT)),
                "replay_exposure": replay,
                "registries": registries,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
