"""Recompute eligibility from Candidate 2 native sufficient statistics, without scores."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from v2_support_audit import audit_partition, sha

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    index_path = ROOT / "evidence/v2/native-development/index.json"
    index = json.loads(index_path.read_text())
    freeze_path = ROOT / "evidence/v2/native_candidate_freeze_v3.json"
    freeze = json.loads(freeze_path.read_text())
    if (
        sha(ROOT / freeze["proposal_path"]) != freeze["proposal_sha256"]
        or sha(ROOT / freeze["inherited_contract"]["path"])
        != freeze["inherited_contract"]["sha256"]
    ):
        raise ValueError("Native frozen contract changed")
    expected_days = [(date(2020, 2, 17) + timedelta(days=i)).isoformat() for i in range(58)]
    if list(index["days"]) != expected_days:
        raise ValueError("Exact complete 58-day development calendar required")
    hours: list[dict[str, Any]] = []
    bindings = {str(index_path.relative_to(ROOT)): sha(index_path)}
    for record in index["days"].values():
        manifest = ROOT / record["manifest_path"]
        if sha(manifest) != record["manifest_sha256"]:
            raise ValueError("Native day manifest changed")
        document = json.loads(manifest.read_text())
        if document["bindings"] != index["bindings"] or document["data_kind"] != "REAL":
            raise ValueError("Native processing identity differs")
        shard = ROOT / document["shard_path"]
        if sha(shard) != document["shard_sha256"] or sha(shard) != record["shard_sha256"]:
            raise ValueError("Native shard changed")
        bindings[str(manifest.relative_to(ROOT))] = sha(manifest)
        bindings[str(shard.relative_to(ROOT))] = sha(shard)
        with np.load(shard, allow_pickle=False) as data:
            np.testing.assert_array_equal(data["frequency_hz"], [38000, 125000, 200000, 455000])
            np.testing.assert_array_equal(data["range_edges_m"], np.arange(0.0, 130.0, 2.0))
            observed = data["observed_ping_count"]
            weights = data["detected_range_ping_m"][:, 0, 5:50]
            if (
                not np.isfinite(weights).all()
                or (weights < 0).any()
                or (weights > 2 * observed[:, None] + 1e-8).any()
            ):
                raise ValueError("Impossible native detected lengths")
            for j in range(0, 96, 4):
                ids = {str(data["configuration_id"][k]) for k in range(j, j + 4) if observed[k] > 0}
                identity = next(iter(ids)) if len(ids) == 1 else ("MIXED" if ids else None)
                if identity == "":
                    identity = None
                count = int(observed[j : j + 4].sum())
                hours.append(
                    {
                        "time": str(data["bin_start"][j].astype("datetime64[h]")),
                        "observed": count,
                        "expected": int(data["expected_ping_count"][j : j + 4].sum()),
                        "detected": float(weights[j : j + 4].sum()),
                        "observed_range_ping_m": 90.0 * count,
                        "configuration": identity,
                    }
                )
    output = ROOT / "evidence/v2/native-support"
    output.mkdir(exist_ok=False)
    partitions = {}
    for name, start, end, minimum_days in (
        ("development_fit", "2020-02-17", "2020-04-01", 20),
        ("development_assessment", "2020-04-01", "2020-04-15", 5),
    ):
        result = audit_partition([x for x in hours if start <= x["time"] < end])
        rows = result.pop("rows")
        for row in rows:
            for target in row["horizons"].values():
                target["detected_range_ping_m"] = target.pop("detected_pairs")
        row_path = output / f"{name}_rows.json"
        with row_path.open("x", encoding="utf-8") as stream:
            json.dump(rows, stream, indent=2)
            stream.write("\n")
        partitions[name] = {
            "start": start,
            "end": end,
            "row_path": str(row_path.relative_to(ROOT)),
            "rows_sha256": sha(row_path),
            "minimum_target_days": minimum_days,
            "adequate_all_horizons": all(
                x["unique_index_target_days"] >= minimum_days for x in result["horizons"].values()
            ),
            **result,
        }
    result = {
        "status": "NATIVE_SUPPORT_ADEQUATE_PENDING_REVIEW"
        if all(x["adequate_all_horizons"] for x in partitions.values())
        else "NATIVE_CANDIDATE_INELIGIBLE_STOP_D1_TARGET_SEARCH",
        "partitions": partitions,
        "artifact_sha256": bindings,
        "freeze_sha256": sha(freeze_path),
        "code_sha256": sha(Path(__file__)),
        "support_code_sha256": sha(ROOT / "tools/v2_support_audit.py"),
        "model_scores_computed": False,
        "non_train_acoustic_payloads_read": False,
    }
    path = output / "eligibility_v2.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "partitions": partitions}, indent=2))


if __name__ == "__main__":
    main()
