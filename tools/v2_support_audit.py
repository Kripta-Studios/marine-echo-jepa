"""Support-only qualification of the single prospectively declared v2 TRAIN target."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit_partition(hours: list[dict[str, Any]]) -> dict[str, Any]:
    times = np.array([x["time"] for x in hours], dtype="datetime64[h]")
    if (
        len(times) == 0
        or np.isnat(times).any()
        or not np.all(np.diff(times) == np.timedelta64(1, "h"))
    ):
        raise ValueError("Strictly increasing contiguous hourly sequence required")
    rows = []
    exclusions: Counter[str] = Counter()
    for i, hour in enumerate(hours):
        row: dict[str, Any] = {"cutoff": hour["time"], "issued": False, "horizons": {}}
        if i < 24 or i + 6 > len(hours):
            reason = "partition_boundary"
        else:
            context = hours[i - 24 : i]
            ids = {x["configuration"] for x in context if x["observed"] > 0}
            if ids != {context[-1]["configuration"]} or None in ids or "MIXED" in ids:
                reason = "past_configuration"
            elif context[-1]["observed"] < 120:
                reason = "past_last_hour_acquisition"
            elif context[-1]["detected"] == 0:
                reason = "past_last_hour_zero_detection"
            elif sum(x["observed"] > 0 for x in context) < 12:
                reason = "past_context_acquisition"
            else:
                reason = None
                row["issued"] = True
                row["configuration"] = context[-1]["configuration"]
        row["issue_exclusion"] = reason
        if reason:
            exclusions[reason] += 1
        else:
            for horizon in (1, 3, 6):
                target = hours[i + horizon - 1]
                observed = target["observed"]
                denominator = target.get("observed_range_ping_m", 45 * observed)
                fraction = target["detected"] / denominator if denominator else None
                if observed < 120:
                    target_reason = "insufficient_acquisition"
                elif target["configuration"] != row["configuration"]:
                    target_reason = "target_configuration"
                elif target["detected"] == 0:
                    target_reason = "zero_detection"
                elif fraction is None or fraction < 0.1:
                    target_reason = "below_detection_fraction_floor"
                else:
                    target_reason = None
                row["horizons"][str(horizon)] = {
                    "target_start": target["time"],
                    "observed": observed,
                    "detected_pairs": target["detected"],
                    "fraction": fraction,
                    "scheduled_coverage": observed / target["expected"],
                    "excess_pings": max(0, observed - target["expected"]),
                    "index_eligible": target_reason is None,
                    "fraction_eligible": target_reason
                    not in ("insufficient_acquisition", "target_configuration"),
                    "reason": target_reason,
                }
        rows.append(row)
    summary = {}
    origin = np.datetime64(hours[0]["time"], "h")
    for horizon_key in ("1", "3", "6"):
        issued = [x for x in rows if x["issued"]]
        eligible = [x for x in issued if x["horizons"][horizon_key]["index_eligible"]]
        cutoff_days = {x["cutoff"][:10] for x in eligible}
        days = {x["horizons"][horizon_key]["target_start"][:10] for x in eligible}
        blocks = {
            (int(np.datetime64(x["cutoff"], "h").astype("int64")) - int(origin.astype("int64")))
            // 48
            for x in eligible
        }
        fractions = [
            x["horizons"][horizon_key]["fraction"]
            for x in issued
            if x["horizons"][horizon_key]["fraction_eligible"]
        ]
        coverages = [x["horizons"][horizon_key]["scheduled_coverage"] for x in issued]
        summary[horizon_key] = {
            "issued_rows": len(issued),
            "index_eligible_rows": len(eligible),
            "fraction_eligible_rows": sum(
                x["horizons"][horizon_key]["fraction_eligible"] for x in issued
            ),
            "unique_index_days": len(days),
            "unique_index_target_days": len(days),
            "unique_index_cutoff_days": len(cutoff_days),
            "fixed_48h_blocks": len(blocks),
            "score_exclusions": dict(
                Counter(
                    x["horizons"][horizon_key]["reason"]
                    for x in issued
                    if x["horizons"][horizon_key]["reason"]
                )
            ),
            "detected_fraction_quantiles_0_25_50_75_100": np.quantile(
                fractions, [0, 0.25, 0.5, 0.75, 1]
            ).tolist()
            if fractions
            else None,
            "scheduled_coverage_quantiles_0_25_50_75_100": np.quantile(
                coverages, [0, 0.25, 0.5, 0.75, 1]
            ).tolist()
            if coverages
            else None,
            "index_days": sorted(days),
        }
    return {"rows": rows, "issue_exclusions": dict(exclusions), "horizons": summary}


def main() -> None:
    ledger_path = ROOT / "evidence/continuation/train_census_v2_execution.json"
    ledger = json.loads(ledger_path.read_text())
    hours: list[dict[str, Any]] = []
    bindings = {str(ledger_path.relative_to(ROOT)): sha(ledger_path)}
    for day, entry in ledger["days"].items():
        compact = day.replace("-", "")
        shard = ROOT / f"data/processed/candidate-train-{compact}-census-v2/{day}.npz"
        config = (
            ROOT
            / f"evidence/continuation/train-candidate-{compact}-census-v2/configuration_boundaries.json"
        )
        for path, key in ((shard, "shard_sha256"), (config, "configuration_sha256")):
            digest = sha(path)
            if digest != entry[key]:
                raise ValueError("Historical artifact changed")
            bindings[str(path.relative_to(ROOT))] = digest
        ids = json.loads(config.read_text())["bin_configuration_ids"]
        with np.load(shard, allow_pickle=False) as data:
            np.testing.assert_array_equal(data["frequency_hz"], [38000, 125000, 200000, 455000])
            np.testing.assert_array_equal(data["range_edges_m"], np.arange(0, 130, 2))
            for j in range(0, 96, 4):
                observed = data["observed_ping_count"][j : j + 4]
                detected = data["valid_ping_count"][j : j + 4, 0, 5:50]
                if (detected < 0).any() or (detected > observed[:, None]).any():
                    raise ValueError("Impossible detected counts")
                configs = {ids[k] for k in range(j, j + 4) if data["observed_ping_count"][k] > 0}
                identity = (
                    next(iter(configs)) if len(configs) == 1 else ("MIXED" if configs else None)
                )
                hours.append(
                    {
                        "time": str(data["bin_start"][j].astype("datetime64[h]")),
                        "observed": int(observed.sum()),
                        "expected": int(data["expected_ping_count"][j : j + 4].sum()),
                        "detected": int(detected.sum()),
                        "configuration": identity,
                    }
                )
    partitions = {
        "development_fit": ("2020-02-17", "2020-04-01"),
        "development_assessment": ("2020-04-01", "2020-04-15"),
        "all_original_train": ("2020-02-17", "2020-05-27"),
    }
    result = {}
    output = ROOT / "evidence/v2/support"
    output.mkdir(exist_ok=False)
    for name, (start, end) in partitions.items():
        data = audit_partition([x for x in hours if start <= x["time"] < end])
        rows = data.pop("rows")
        row_path = output / f"{name}_rows.json"
        with row_path.open("x", encoding="utf-8") as stream:
            json.dump(rows, stream, indent=2)
            stream.write("\n")
        result[name] = {
            "start": start,
            "end": end,
            "row_path": str(row_path.relative_to(ROOT)),
            "rows_sha256": sha(row_path),
            **data,
        }
    report = {
        "status": "SINGLE_CANDIDATE_TRAIN_SUPPORT_ONLY",
        "candidate_freeze_sha256": sha(ROOT / "evidence/v2/candidate_freeze.json"),
        "development_freeze_sha256": sha(ROOT / "evidence/v2/development_freeze.json"),
        "code_sha256": sha(Path(__file__)),
        "partitions": result,
        "artifact_sha256": bindings,
        "non_train_payloads_read": False,
        "model_scores_computed": False,
    }
    path = output / "eligibility_v2.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v["horizons"] for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
