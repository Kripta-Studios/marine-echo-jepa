"""Inspect preserved TRAIN acquisition/QC artifacts without rerunning calibration."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def disjoint_qc_counts(raw, corrected, noise, reasons, ranges, native_bins):
    """Count physically sampled native cells, never padded array entries."""
    population = np.isfinite(ranges[:, :native_bins]) & (
        (ranges[:, :native_bins] >= 10) & (ranges[:, :native_bins] < 100)
    )
    raw = raw[:, :native_bins]
    corrected = corrected[:, :native_bins]
    noise = noise[:, :native_bins]
    reasons = reasons[:, :native_bins]
    left = population.copy()
    result = {"sampled_native_band_cells": int(left.sum())}
    for name, selected in (
        ("invalid_measurement", ~np.isfinite(raw) | ((reasons & 1) != 0)),
        ("saturation", (reasons & 4) != 0),
        ("geometry", (reasons & (2 | 8)) != 0),
        ("noise_estimate_unavailable", ~np.isfinite(noise)),
        ("below_fixed_detection_rule", ~np.isfinite(corrected)),
        ("detected", np.isfinite(corrected)),
    ):
        result[name] = int((left & selected).sum())
        left &= ~selected
    if (
        left.any()
        or sum(v for k, v in result.items() if k != "sampled_native_band_cells")
        != result["sampled_native_band_cells"]
    ):
        raise ValueError("QC categories do not partition the native population")
    return result


def main() -> None:
    execution = json.loads(
        (ROOT / "evidence/continuation/train_census_v2_execution.json").read_text()
    )
    total_observed = total_expected = 0
    configs = Counter()
    overlapping = Counter()
    geometry = {}
    bindings = {}
    for day, record in execution["days"].items():
        folder = ROOT / f"evidence/continuation/train-candidate-{day.replace('-', '')}-census-v2"
        path = folder / "processing_manifest.json"
        if sha(path) != record["manifest_sha256"]:
            raise ValueError("Historical manifest hash differs")
        document = json.loads(path.read_text())
        bindings[str(path.relative_to(ROOT))] = sha(path)
        total_observed += document["observed_pings"]
        total_expected += document["expected_pings"]
        for source in document["source_hours"]:
            configs[source["configuration_sha256"]] += 1
            overlapping.update(source["QC_reason_counts"])
            for channel in source["range_support"]:
                frequency = str(channel["frequency_hz"])
                extent = channel["source_edges_min_max_m"]
                previous = geometry.get(frequency, extent)
                geometry[frequency] = [min(previous[0], extent[0]), max(previous[1], extent[1])]
    sample = []
    # These four dates were selected prospectively in the historical monthly audit.
    # The first UTC hour is fixed here before native QC counts are computed.
    for day in ("2020-02-17", "2020-03-01", "2020-04-01", "2020-05-01"):
        compact = day.replace("-", "")
        manifest = (
            ROOT
            / f"evidence/continuation/train-candidate-{compact}-monthly-v1/processing_manifest.json"
        )
        document = json.loads(manifest.read_text())
        for source in document["source_hours"]:
            if source["source"][:8] != compact[2:] + "00":
                continue
            path = ROOT / source["candidate_file"]
            if sha(path) != source["candidate_sha256"]:
                raise ValueError("Preserved native sample hash differs")
            bindings[str(path.relative_to(ROOT))] = sha(path)
            with np.load(path, allow_pickle=False) as data:
                for channel, frequency in enumerate((38000, 125000, 200000, 455000)):
                    counts = disjoint_qc_counts(
                        data["pre_noise_Sv"][channel],
                        data["corrected_Sv"][channel],
                        data["Sv_noise"][channel],
                        data["raw_qc_reason"][channel],
                        data["echo_range"][channel],
                        source["configuration"]["number_of_bins_per_channel"][channel],
                    )
                    sample.append(
                        {
                            "day": day,
                            "source": source["source"],
                            "frequency_hz": frequency,
                            **counts,
                        }
                    )
    report = {
        "status": "PRESERVED_TRAIN_ACQUISITION_AND_FIXED_SAMPLE_QC_AUDIT",
        "observed_pings": total_observed,
        "expected_pings": total_expected,
        "configuration_source_counts": dict(configs),
        "native_range_extents_m": geometry,
        "historical_overlapping_qc_counts_not_disjoint": dict(overlapping),
        "fixed_native_sample": sample,
        "artifact_sha256": bindings,
        "limitations": [
            "Native QC samples are four predetermined first UTC hours, not full-corpus causal attribution.",
            "No new acoustic payload processing, candidate support selection or model scores.",
            "Scheduled and observed counts alone do not identify why a scheduled ping was absent.",
        ],
    }
    output = ROOT / "evidence/v2/acquisition_qc_audit.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(output), "sha256": sha(output), "native_records": len(sample)}))


if __name__ == "__main__":
    main()
