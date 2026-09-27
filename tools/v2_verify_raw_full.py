"""Verify full raw-response processing lineage and count-only QC without scores."""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parent / "marine-echo-jepa-core/data/raw/pangaea/mosaic_azfp_down_2020_extracted"
INDEX = ROOT / "evidence/v2/raw-response-development-reviewed/index.json"
OUT = ROOT / "evidence/v2/raw_response_full_verification.json"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    index = json.loads(INDEX.read_text())
    days = [(date(2020, 2, 17) + timedelta(days=i)).isoformat() for i in range(58)]
    if (
        index.get("data_kind") != "REAL"
        or index.get("study_id") != "raw_response_development_v1"
        or index.get("processing_run_id") != "raw_response_full_v2_20260927"
        or list(index.get("days", {})) != days
    ):
        raise ValueError("Full raw-response index identity or calendar differs")
    totals = {
        "observed_pings": 0,
        "valid_target_pings": 0,
        "zero_only_pings": 0,
        "zero_affected_pings": 0,
        "zero_affected_quarter_hours": 0,
        "nonfinite_pings": 0,
        "zero_samples": 0,
        "nonfinite_samples": 0,
        "source_records_hashed": 0,
        "target_code_samples": 0,
        "profile_code_samples": 0,
    }
    worker_seconds = 0.0
    peak_rss = 0.0
    for day, entry in index["days"].items():
        manifest = ROOT / entry["manifest_path"]
        if sha(manifest) != entry["manifest_sha256"]:
            raise ValueError("Raw-response manifest hash differs")
        doc = json.loads(manifest.read_text())
        if (
            doc.get("day") != day
            or doc.get("data_kind") != "REAL"
            or doc.get("study_id") != index["study_id"]
            or doc.get("processing_run_id") != index["processing_run_id"]
            or doc.get("bindings") != index["bindings"]
            or doc.get("shard_sha256") != entry["shard_sha256"]
        ):
            raise ValueError("Raw-response manifest identity differs")
        shard = ROOT / doc["shard_path"]
        if sha(shard) != doc["shard_sha256"]:
            raise ValueError("Raw-response shard hash differs")
        old_path = ROOT / (
            f"evidence/continuation/train-candidate-{day.replace('-', '')}-census-v2/processing_manifest.json"
        )
        if sha(old_path) != doc["historical_manifest_sha256"]:
            raise ValueError("Historical source manifest hash differs")
        old = json.loads(old_path.read_text())
        keys = ("source", "source_sha256", "configuration_sha256", "pings_in_day", "first_utc", "last_utc")
        if [tuple(x[k] for k in keys) for x in old["source_hours"]] != [
            tuple(x[k] for k in keys) for x in doc["source_hours"]
        ]:
            raise ValueError("Historical source list differs")
        for source in doc["source_hours"]:
            name = source["source"]
            if not name or Path(name).name != name or sha(RAW / name) != source["source_sha256"]:
                raise ValueError("Raw-response source hash differs")
            totals["source_records_hashed"] += 1
        with np.load(shard, allow_pickle=False) as data:
            obs = data["observed_ping_count"]
            val = data["valid_target_ping_count"]
            zero = data["zero_or_undefined_ping_count"]
            affected = data["zero_affected_ping_count"]
            nonfinite = data["nonfinite_ping_count"]
            if (
                obs.shape != (96,)
                or not np.array_equal(val + zero + nonfinite, obs)
                or not np.array_equal(data["target_code_count"], 180 * val)
                or int(obs.sum()) != doc["observed_pings"]
                or len(data["raw_ping_time"]) != int(obs.sum())
                or not np.array_equal(data["expected_ping_count"], np.full(96, 60))
                or not np.array_equal(data["frequency_hz"], [38000, 125000, 200000, 455000])
                or not np.array_equal(data["averaged_bin_index"], np.arange(20, 200))
            ):
                raise ValueError("Raw-response shard count/QC geometry differs")
            for name, value in (
                ("observed_pings", obs.sum()),
                ("valid_target_pings", val.sum()),
                ("zero_only_pings", zero.sum()),
                ("zero_affected_pings", affected.sum()),
                ("zero_affected_quarter_hours", (affected > 0).sum()),
                ("nonfinite_pings", nonfinite.sum()),
                ("zero_samples", data["zero_or_undefined_sample_count"].sum()),
                ("nonfinite_samples", data["nonfinite_sample_count"].sum()),
                ("target_code_samples", data["target_code_count"].sum()),
                ("profile_code_samples", data["profile_code_count"].sum()),
            ):
                totals[name] += int(value)
        worker_seconds += doc["elapsed_seconds"]
        peak_rss = max(peak_rss, doc["peak_sampled_process_rss_gib"])
    result = {
        "study_id": index["study_id"],
        "processing_run_id": index["processing_run_id"],
        "data_kind": "REAL",
        "index_sha256": sha(INDEX),
        "fixed_days": len(days),
        "counts": totals,
        "summed_worker_seconds_not_wall_time": worker_seconds,
        "maximum_sampled_single_process_rss_gib": peak_rss,
        "model_scores_computed": False,
        "non_train_acoustic_payloads_read": False,
    }
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
