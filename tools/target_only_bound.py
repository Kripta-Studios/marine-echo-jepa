"""Generous TRAIN target-support bound independent of context-quality policy."""

import json
from pathlib import Path

import numpy as np
from continuation_records import write_report
from train_census import paths as old_paths
from train_census import verify_completed as verify_old_completed
from train_census_v2 import (
    assert_shards_equal,
    paths,
    previous_execution,
    sha,
    validate_ledger,
    verify_completed,
)

ROOT = Path(__file__).resolve().parents[1]


def validate_lineage(identity, strict, prior_digest, method_digest):
    if (
        identity.get("prior_failed_execution_sha256") != prior_digest
        or identity.get("method_review_sha256") != method_digest
        or strict.get("status") != "TRAIN_CENSUS_COMPLETE_STRICT_CONTEXT_CANDIDATE_ONLY"
        or strict.get("global_ineligibility_conclusion")
        != "NOT_ESTABLISHED_TARGET_ONLY_BOUND_REQUIRED"
        or strict.get("held_out_acoustic_payloads_processed") is not False
        or type(strict.get("completed_benchmark_runs")) is not int
        or strict["completed_benchmark_runs"] != 0
        or strict.get("benchmark_eligible") is not False
    ):
        raise ValueError("Repaired census lineage or strict report disposition differs")


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
    input_contract = ROOT / "evidence/continuation/target_only_bound_v2_input_contract.json"
    if sha(input_contract) != "1b27db9b08eedf3f937f363fbabcb522633d73175c7836e926f7248c9434c4ef":
        raise ValueError("Prospective repaired-census input binding changed")
    execution_path = ROOT / "evidence/continuation/train_census_v2_execution.json"
    strict_path = ROOT / "evidence/continuation/train_census_v2_eligibility.json"
    execution = json.loads(execution_path.read_text())
    strict = json.loads(strict_path.read_text())
    identity = execution["identity"]
    previous = previous_execution()
    validate_lineage(
        identity,
        strict,
        sha(ROOT / "evidence/continuation/train_census_execution.json"),
        sha(ROOT / "orchestration/reviews/TRAIN_CENSUS_METHOD_20260927.json"),
    )
    days = np.arange("2020-02-17", "2020-05-27", dtype="datetime64[D]").astype(str).tolist()
    validate_ledger(execution, identity, days)
    if (
        execution["status"] != "COMPLETE_TRAIN_CENSUS_NOT_BENCHMARK"
        or identity["calendar"] != days
        or identity["driver_sha256"]
        != "a5f4194989f2a8051b10881c21666c46c85b76ac79a614d42010d2afeafa3265"
        or identity["bindings"]["code_sha256"]
        != "a0ddb50f64a4a92fcf45e657aeb385aa27ddbd32cdd30af50571cd79b6bd0e4d"
        or identity["bindings"]["contract_sha256"]
        != "581141caa866b3cf3756fecdda49e379e38df7ea89e19f156c9074b3f5961e0c"
        or identity["bindings"].get("compat_code_sha256")
        != "4a60f0287f538f674f947ecc3f2f20f302e1b70bbe66360adf419f607e22bbf5"
        or identity.get("singleton_review_sha256")
        != sha(ROOT / "orchestration/reviews/AZFP_SINGLETON_COMPAT_20260927.json")
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
        previous_entry = previous["days"].get(day)
        if previous_entry is not None and previous_entry["exit_code"] == 0:
            verify_old_completed(ROOT, day, previous_entry, previous["identity"]["bindings"])
            assert_shards_equal(old_paths(ROOT, day)[2], shard)
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
        "input_contract_sha256": sha(input_contract),
        "census_generation": "census-v2",
        "code_sha256": sha(Path(__file__)),
        "execution_report_sha256": sha(execution_path),
        "strict_context_report_sha256": sha(strict_path),
        "strict_context_report_disposition": "Strict context rule remains subject to R1; only this context-independent target bound can establish global ineligibility.",
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
