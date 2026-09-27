"""Recompute exact one-day raw-response provenance and shard invariants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parent / "marine-echo-jepa-core/data/raw/pangaea/mosaic_azfp_down_2020_extracted"
DAY = "2020-02-17"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    index_path = ROOT / "evidence/v2/raw-response-development/index.json"
    index = json.loads(index_path.read_text())
    if index["data_kind"] != "REAL" or index["study_id"] != "raw_response_development_v1":
        raise ValueError("Raw pilot index identity differs")
    if set(index["days"]) != {DAY}:
        raise ValueError("Raw pilot must have exactly the preselected day")
    entry = index["days"][DAY]
    manifest_path = ROOT / entry["manifest_path"]
    if sha(manifest_path) != entry["manifest_sha256"]:
        raise ValueError("Raw pilot manifest digest differs")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["data_kind"] != "REAL"
        or manifest["study_id"] != index["study_id"]
        or manifest["bindings"] != index["bindings"]
        or manifest["day"] != DAY
    ):
        raise ValueError("Raw pilot manifest identity differs")
    shard_path = ROOT / manifest["shard_path"]
    if sha(shard_path) != manifest["shard_sha256"] or entry["shard_sha256"] != manifest["shard_sha256"]:
        raise ValueError("Raw pilot shard digest differs")
    old_path = ROOT / "evidence/continuation/train-candidate-20200217-census-v2/processing_manifest.json"
    old = json.loads(old_path.read_text())
    if sha(old_path) != manifest["historical_manifest_sha256"]:
        raise ValueError("Historical acquisition manifest digest differs")
    source_keys = ("source", "source_sha256", "configuration_sha256", "pings_in_day", "first_utc", "last_utc")
    old_sources = [tuple(item[key] for key in source_keys) for item in old["source_hours"]]
    new_sources = [tuple(item[key] for key in source_keys) for item in manifest["source_hours"]]
    if old_sources != new_sources or len(new_sources) != 48:
        raise ValueError("Pilot source-hour list differs")
    for source in manifest["source_hours"]:
        if sha(RAW / source["source"]) != source["source_sha256"]:
            raise ValueError("Pilot raw source hash differs")
    with np.load(shard_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if (
        arrays["profile_code_sum"].shape != (96, 4, 64)
        or arrays["profile_code_count"].shape != (96, 4, 64)
        or arrays["target_code_sum"].shape != (96,)
        or arrays["target_code_count"].shape != (96,)
        or not np.array_equal(arrays["frequency_hz"], [38000, 125000, 200000, 455000])
        or not np.array_equal(arrays["averaged_bin_index"], np.arange(20, 200))
        or not np.array_equal(arrays["expected_ping_count"], np.full(96, 60))
    ):
        raise ValueError("Raw pilot shard geometry differs")
    observed = arrays["observed_ping_count"]
    valid = arrays["valid_target_ping_count"]
    zero = arrays["zero_or_undefined_ping_count"]
    nonfinite = arrays["nonfinite_ping_count"]
    if (
        observed.sum() != 5713
        or observed.sum() != manifest["observed_pings"]
        or (valid + zero + nonfinite != observed).any()
        or not np.array_equal(arrays["target_code_count"], 180 * valid)
        or (arrays["zero_affected_ping_count"] < zero).any()
        or not np.isfinite(arrays["target_code_sum"]).all()
        or not np.isfinite(arrays["profile_code_sum"]).all()
        or (arrays["profile_code_count"] < 0).any()
        or (arrays["profile_code_count"] > 3 * observed[:, None, None]).any()
    ):
        raise ValueError("Raw pilot QC or aggregation arithmetic differs")
    times = arrays["raw_ping_time"]
    with np.load(ROOT / "data/processed/v2-native-development/2020-02-17.npz", allow_pickle=False) as native:
        if not np.array_equal(times, native["raw_ping_time"]):
            raise ValueError("Raw pilot timestamps differ from independently processed native day")
    if (
        len(times) != 5713
        or np.isnat(times).any()
        or (np.diff(times) <= np.timedelta64(0, "ns")).any()
        or (times < np.datetime64(DAY)).any()
        or (times >= np.datetime64("2020-02-18")).any()
        or len(set(arrays["configuration_id"])) != 1
        or arrays["configuration_id"][0] != manifest["source_hours"][0]["configuration_sha256"]
    ):
        raise ValueError("Raw pilot timestamp or configuration identity differs")
    result = {
        "study_id": index["study_id"],
        "day": DAY,
        "index_sha256": sha(index_path),
        "manifest_sha256": sha(manifest_path),
        "shard_sha256": sha(shard_path),
        "source_hashes_recomputed": len(new_sources),
        "observed_pings": int(observed.sum()),
        "valid_target_pings": int(valid.sum()),
        "zero_only_target_pings": int(zero.sum()),
        "zero_affected_pings": int(arrays["zero_affected_ping_count"].sum()),
        "zero_affected_quarter_hours": int((arrays["zero_affected_ping_count"] > 0).sum()),
        "nonfinite_target_pings": int(nonfinite.sum()),
        "target_code_samples": int(arrays["target_code_count"].sum()),
        "profile_code_samples": int(arrays["profile_code_count"].sum()),
        "sampled_process_rss_gib": manifest["peak_sampled_process_rss_gib"],
        "worker_elapsed_seconds": manifest["elapsed_seconds"],
        "quantity": manifest["quantity"],
    }
    out = ROOT / "evidence/v2/raw_response_pilot_verification.json"
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
