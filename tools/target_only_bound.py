"""Generous TRAIN target-support bound independent of context-quality policy."""

import json
from pathlib import Path

import numpy as np
from continuation_records import write_report
from train_census import paths, sha, validate_ledger, verify_completed

ROOT = Path(__file__).resolve().parents[1]


def target_only_cutoffs(counts, denominator):
    if (
        counts.shape != (len(denominator), 45)
        or counts.dtype.kind not in "iu"
        or denominator.dtype.kind not in "iu"
        or np.any(counts < 0)
        or np.any(denominator <= 0)
        or np.any(counts > denominator[:, None])
    ):
        raise ValueError("Exact primary-band integer coverage counts required")
    return [
        cutoff
        for cutoff in range(0, len(counts) - 24 + 1, 4)
        if all(
            counts[cutoff + offset : cutoff + offset + 4].sum()
            / (45 * denominator[cutoff + offset : cutoff + offset + 4].sum())
            >= 0.8
            for offset in (0, 8, 20)
        )
    ]


def main():
    contract = ROOT / "evidence/continuation/target_only_bound_contract.json"
    if sha(contract) != "774f7bcac90ca275eb486d243a5b3a20488814c1d1de2060bc8a188322dd769f":
        raise ValueError("Prospective target-only bound contract changed")
    execution_path = ROOT / "evidence/continuation/train_census_execution.json"
    strict_path = ROOT / "evidence/continuation/train_census_eligibility.json"
    execution = json.loads(execution_path.read_text())
    strict = json.loads(strict_path.read_text())
    identity = execution["identity"]
    days = np.arange("2020-02-17", "2020-05-27", dtype="datetime64[D]").astype(str).tolist()
    validate_ledger(execution, identity, days)
    if (
        execution["status"] != "COMPLETE_TRAIN_CENSUS_NOT_BENCHMARK"
        or identity["calendar"] != days
        or identity["driver_sha256"]
        != "15ff07d4a993823f5a09b6ba60b806dec0b796969977c1ebd43177237e375e20"
        or identity["bindings"]["code_sha256"]
        != "98120ac7d19701ced0d7855d0199e4b196d869eebbcfc5d6be26b86a3a73a6f9"
        or identity["bindings"]["contract_sha256"]
        != "117ed3b17f94dd2969cbf4e82f6231f4197bd7feff8e930e1bc67f5d18eeb149"
        or identity["bindings"]["protocol_sha256"] != sha(ROOT / "reports/active/protocol.json")
        or strict["execution_report_sha256"] != sha(execution_path)
        or strict["method_identity"] != identity
        or len(execution["days"]) != 100
    ):
        raise ValueError("Exact completed reviewed census required")
    counts, denominators, summaries = [], [], []
    for day in days:
        verify_completed(ROOT, day, execution["days"][day], identity["bindings"])
        _, _, shard = paths(ROOT, day)
        with np.load(shard, allow_pickle=False) as data:
            if (
                not np.array_equal(data["frequency_hz"], [38000, 125000, 200000, 455000])
                or not np.array_equal(data["range_edges_m"], np.arange(0, 130, 2))
                or not np.array_equal(
                    data["bin_start"],
                    np.datetime64(day, "ns") + np.arange(96) * np.timedelta64(15, "m"),
                )
                or not np.array_equal(
                    data["support_denominator_ping_count"],
                    np.maximum(data["expected_ping_count"], data["observed_ping_count"]),
                )
            ):
                raise ValueError("Unexpected grid, calendar or support denominator")
            c = data["valid_ping_count"][:, 0, 5:50]
            d = data["support_denominator_ping_count"]
            if c.shape != (96, 45) or d.shape != (96,):
                raise ValueError("Unexpected count shape")
            target_only_cutoffs(c, d)
            counts.append(c)
            denominators.append(d)
            summaries.append(
                {
                    "date": day,
                    "quarter_hours_at_least_80pct": int((c.sum(axis=1) / (45 * d) >= 0.8).sum()),
                    "max_quarter_hour_support": float((c.sum(axis=1) / (45 * d)).max()),
                }
            )
    cutoffs = target_only_cutoffs(np.concatenate(counts), np.concatenate(denominators))
    candidate_dates = sorted({days[index // 96] for index in cutoffs})
    maximum = len(candidate_dates) + 67
    report = {
        "status": "D1_INELIGIBLE_TARGET_SUPPORT_UPPER_BOUND"
        if maximum < 90
        else "NO_GLOBAL_INELIGIBILITY_CONCLUSION",
        "contract_sha256": sha(contract),
        "code_sha256": sha(Path(__file__)),
        "execution_report_sha256": sha(execution_path),
        "strict_context_report_sha256": sha(strict_path),
        "strict_context_report_disposition": "Its global classification is superseded by this context-independent bound; strict context rule remains subject to R1.",
        "processed_train_calendar_days": 100,
        "target_supported_train_anchor_days_upper_bound": len(candidate_dates),
        "target_supported_train_hourly_anchors_upper_bound": len(cutoffs),
        "target_supported_dates": candidate_dates,
        "unmeasured_nontrain_day_upper_bound": 67,
        "overall_eligible_day_upper_bound": maximum,
        "minimum_overall_days": 90,
        "interval_calibration_days": "NOT_EVALUATED",
        "test_days": "NOT_EVALUATED",
        "held_out_acoustic_payloads_processed": False,
        "completed_benchmark_runs": 0,
        "benchmark_eligible": False,
        "daily_support": summaries,
        "limitations": "Fixed reviewed noise/QC method; necessary target support only. Ignores all context/config restrictions. Not a JEPA outcome or real-corpus approval.",
    }
    write_report(ROOT / "evidence/continuation/target_only_bound.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "daily_support"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
