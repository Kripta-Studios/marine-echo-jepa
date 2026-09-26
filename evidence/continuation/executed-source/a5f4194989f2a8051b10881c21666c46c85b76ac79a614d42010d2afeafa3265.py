"""Resumable full TRAIN coverage census; never opens held-out acoustic payloads."""

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import psutil
from continuation_records import write_report

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def eligible_cutoffs(valid, denominator, configs):
    if valid.shape != (len(denominator), 45) or len(configs) != len(denominator):
        raise ValueError("Exact primary45-bin grid and aligned metadata required")
    if np.any(valid < 0) or np.any(denominator <= 0) or np.any(valid > denominator[:, None]):
        raise ValueError("Corrupt coverage counts")
    context_support = valid.sum(axis=1) / (45 * denominator)
    result = []
    for cutoff in range(96, len(valid) - 24 + 1, 4):
        scope = configs[cutoff - 96 : cutoff + 24]
        if (
            None in scope
            or len(set(scope)) != 1
            or not np.all(context_support[cutoff - 96 : cutoff] >= 0.8)
        ):
            continue
        if all(
            valid[cutoff + offset : cutoff + offset + 4].sum()
            / (45 * denominator[cutoff + offset : cutoff + offset + 4].sum())
            >= 0.8
            for offset in (0, 8, 20)
        ):
            result.append(cutoff)
    return result


def paths(root, day):
    compact = day.replace("-", "")
    evidence = root / f"evidence/continuation/train-candidate-{compact}-census-v2"
    shard = root / f"data/processed/candidate-train-{compact}-census-v2/{day}.npz"
    return evidence / "processing_manifest.json", evidence / "configuration_boundaries.json", shard


def verify_completed(root, day, entry, bindings):
    manifest, configuration, shard = paths(root, day)
    if any(path.is_symlink() for path in (manifest, configuration, shard)):
        raise ValueError("Linked census artifacts are not accepted")
    if any(
        sha(path) != entry[key]
        for path, key in (
            (manifest, "manifest_sha256"),
            (configuration, "configuration_sha256"),
            (shard, "shard_sha256"),
        )
    ):
        raise ValueError("Completed census artifact differs from journal")
    document = json.loads(manifest.read_text())
    if (
        document["bindings"] != bindings
        or document["status"] != "TRAIN_CANDIDATE_PROCESSED_REQUIRES_QC_REVIEW"
        or document["daily_candidate_sha256"] != sha(shard)
    ):
        raise ValueError("Census source/code/contract bindings differ")
    return document


def validate_ledger(ledger, identity, days):
    if set(ledger) != {
        "status",
        "identity",
        "days",
        "held_out_acoustic_payloads_processed",
        "completed_benchmark_runs",
    }:
        raise ValueError("Unexpected census journal schema")
    if ledger["held_out_acoustic_payloads_processed"] is not False:
        raise ValueError("Exposure record differs; preserve it and stop")
    if (
        type(ledger["completed_benchmark_runs"]) is not int
        or ledger["completed_benchmark_runs"] != 0
    ):
        raise ValueError("Census cannot contain benchmark executions")
    if ledger["identity"] != identity or not isinstance(ledger["days"], dict):
        raise ValueError("Cannot resume a different census")
    if list(ledger["days"]) != days[: len(ledger["days"])]:
        raise ValueError("Journal must contain an exact chronological calendar prefix")
    base = {
        "exit_code",
        "resource_stop",
        "elapsed_seconds",
        "peak_sampled_process_tree_rss_gib",
        "log",
    }
    successful = {
        "manifest_sha256",
        "configuration_sha256",
        "shard_sha256",
        "primary_bins_at_least_80pct",
    }
    failed = False
    for day, entry in ledger["days"].items():
        if (
            not isinstance(entry, dict)
            or type(entry.get("exit_code")) is not int
            or type(entry.get("resource_stop")) is not bool
        ):
            raise ValueError("Invalid execution entry")
        success = entry["exit_code"] == 0 and not entry["resource_stop"]
        if set(entry) != base | (successful if success else set()):
            raise ValueError("Unexpected execution entry fields")
        if failed:
            raise ValueError("Census continued after a failed day")
        failed = not success
        for field in ("elapsed_seconds", "peak_sampled_process_tree_rss_gib"):
            if (
                type(entry[field]) not in (int, float)
                or not math.isfinite(entry[field])
                or entry[field] < 0
            ):
                raise ValueError("Invalid execution resource measurement")
        if entry["log"] != f"evidence/continuation/census-v2-{day}.log":
            raise ValueError("Unexpected execution log path")
        if success:
            if (
                type(entry["primary_bins_at_least_80pct"]) is not int
                or not 0 <= entry["primary_bins_at_least_80pct"] <= 96
            ):
                raise ValueError("Invalid primary support count")
            for field in successful - {"primary_bins_at_least_80pct"}:
                digest = entry[field]
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                ):
                    raise ValueError("Invalid artifact digest")
    allowed = {"BLOCKED_FAILED_TRAIN_DAY"} if failed else {"RUNNING_TRAIN_ONLY_CENSUS"}
    if not failed and len(ledger["days"]) == len(days):
        allowed.add("COMPLETE_TRAIN_CENSUS_NOT_BENCHMARK")
    if ledger["status"] not in allowed:
        raise ValueError("Census status contradicts recorded executions")


def finalize_reports(execution_path, ledger, result_path, result):
    ledger["status"] = "COMPLETE_TRAIN_CENSUS_NOT_BENCHMARK"
    write_report(execution_path, ledger)
    result["execution_report_sha256"] = sha(execution_path)
    write_report(result_path, result)


def previous_execution():
    from train_census import validate_ledger as validate_previous

    path = ROOT / "evidence/continuation/train_census_execution.json"
    contract = json.loads(
        (ROOT / "evidence/continuation/train_census_v2_contract.json").read_text()
    )
    if sha(path) != contract["prior_failure"]["sha256"]:
        raise ValueError("Frozen prior census execution differs")
    previous = json.loads(path.read_text())
    days = np.arange("2020-02-17", "2020-05-27", dtype="datetime64[D]").astype(str).tolist()
    validate_previous(previous, previous["identity"], days)
    if (
        previous["status"] != "BLOCKED_FAILED_TRAIN_DAY"
        or list(previous["days"]) != days[:17]
        or any(previous["days"][day]["exit_code"] != 0 for day in days[:16])
        or previous["days"]["2020-03-04"]["exit_code"] != 1
        or previous["identity"]["driver_sha256"]
        != "15ff07d4a993823f5a09b6ba60b806dec0b796969977c1ebd43177237e375e20"
        or previous["identity"]["bindings"]["code_sha256"]
        != "98120ac7d19701ced0d7855d0199e4b196d869eebbcfc5d6be26b86a3a73a6f9"
    ):
        raise ValueError("Unexpected preserved prior census record")
    return previous


def assert_shards_equal(old_shard, new_shard):
    with (
        np.load(old_shard, allow_pickle=False) as old,
        np.load(new_shard, allow_pickle=False) as new,
    ):
        if set(old.files) != set(new.files):
            raise ValueError("Census generation array fields differ")
        for field in old.files:
            if old[field].dtype != new[field].dtype:
                raise ValueError("Census generation array dtype differs")
            np.testing.assert_array_equal(old[field], new[field], err_msg=f"Changed {field}")
        fields = old.files
    return fields


def compare_previous_day(day, new_shard):
    """Require exact arrays for every successfully completed v1 day."""
    from train_census import paths as previous_paths
    from train_census import verify_completed as verify_previous

    previous = previous_execution()
    entry = previous["days"].get(day)
    if entry is None or entry["exit_code"] != 0:
        return
    verify_previous(ROOT, day, entry, previous["identity"]["bindings"])
    _, _, old_shard = previous_paths(ROOT, day)
    fields = assert_shards_equal(old_shard, new_shard)
    write_report(
        ROOT / f"evidence/continuation/census-v2-equivalence/{day}.json",
        {
            "status": "EXACT_ARRAY_EQUIVALENCE",
            "date": day,
            "previous_shard_sha256": sha(old_shard),
            "new_shard_sha256": sha(new_shard),
            "fields": fields,
            "scope": "All arrays including count/time/config masks and linear Sv, TRAIN only; no tolerance or outcome selection",
        },
    )


def main():
    previous_execution()
    protocol = json.loads((ROOT / "reports/active/protocol.json").read_text())
    bounds = protocol["split"]["partitions"]["train"]
    start = date.fromisoformat(bounds["start"][:10])
    stop = date.fromisoformat(bounds["end"][:10])
    days = [(start + timedelta(days=i)).isoformat() for i in range((stop - start).days)]
    if len(days) != 100 or days[0] != "2020-02-17" or days[-1] != "2020-05-26":
        raise ValueError("Original fixed TRAIN calendar required")
    code = ROOT / "tools/preprocess_train_candidate.py"
    contract = ROOT / "evidence/continuation/train_census_v2_contract.json"
    review = ROOT / "orchestration/reviews/TRAIN_CENSUS_METHOD_20260927.json"
    if json.loads(review.read_text())["disposition"] != "APPROVE_TRAIN_ONLY_CENSUS_METHOD":
        raise ValueError("Independent TRAIN method decision required")
    bindings = json.loads(
        (
            ROOT
            / "evidence/continuation/train-candidate-20200217-monthly-v1/processing_manifest.json"
        ).read_text()
    )["bindings"]
    bindings.update(
        code_sha256=sha(code),
        contract_sha256=sha(contract),
        compat_code_sha256=sha(ROOT / "src/marine_echo/data/azfp_compat.py"),
    )
    if bindings["qc_code_sha256"] != sha(ROOT / "src/marine_echo/data/candidate_qc.py") or bindings[
        "regrid_code_sha256"
    ] != sha(ROOT / "src/marine_echo/data/range_grid.py"):
        raise ValueError("Reviewed QC/regrid method changed")
    ledger_path = ROOT / "evidence/continuation/train_census_v2_execution.json"
    identity = {
        "driver_sha256": sha(Path(__file__)),
        "bindings": bindings,
        "method_review_sha256": sha(review),
        "singleton_review_sha256": sha(
            ROOT / "orchestration/reviews/AZFP_SINGLETON_COMPAT_20260927.json"
        ),
        "prior_failed_execution_sha256": sha(
            ROOT / "evidence/continuation/train_census_execution.json"
        ),
        "calendar": days,
    }
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        validate_ledger(ledger, identity, days)
    else:
        ledger = {
            "status": "RUNNING_TRAIN_ONLY_CENSUS",
            "identity": identity,
            "days": {},
            "held_out_acoustic_payloads_processed": False,
            "completed_benchmark_runs": 0,
        }
        write_report(ledger_path, ledger)
    for day in days:
        if day in ledger["days"]:
            entry = ledger["days"][day]
            if entry["exit_code"] != 0:
                raise ValueError("Prior failed day needs explicit preserved-version repair")
            verify_completed(ROOT, day, entry, bindings)
            compare_previous_day(day, paths(ROOT, day)[2])
            continue
        manifest, configuration, shard = paths(ROOT, day)
        if manifest.parent.exists() or shard.parent.exists():
            raise ValueError("Unjournaled candidate artifacts require investigation; no overwrite")
        if sha(code) != bindings["code_sha256"] or sha(contract) != bindings["contract_sha256"]:
            raise ValueError("Frozen census code/contract changed during execution")
        command = [sys.executable, str(code), "--date", day, "--variant", "census-v2"]
        log = ROOT / f"evidence/continuation/census-v2-{day}.log"
        env = os.environ.copy()
        env.update(PYTHONPATH=str(ROOT / "src"), PYTHONUTF8="1")
        started, peak, stopped = time.monotonic(), 0, False
        with log.open("x", encoding="utf-8") as stream:
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
            while process.poll() is None:
                try:
                    parent = psutil.Process(process.pid)
                    peak = max(
                        peak,
                        sum(
                            p.memory_info().rss for p in [parent, *parent.children(recursive=True)]
                        ),
                    )
                    if peak > 22 * 1024**3 or time.monotonic() - started > 600:
                        stopped = True
                        for child in reversed(parent.children(recursive=True)):
                            child.terminate()
                        parent.terminate()
                        process.wait(timeout=15)
                except psutil.Error:
                    pass
                time.sleep(0.2)
        entry = {
            "exit_code": process.returncode,
            "resource_stop": stopped,
            "elapsed_seconds": time.monotonic() - started,
            "peak_sampled_process_tree_rss_gib": peak / 1024**3,
            "log": log.relative_to(ROOT).as_posix(),
        }
        if process.returncode == 0 and not stopped:
            entry.update(
                manifest_sha256=sha(manifest),
                configuration_sha256=sha(configuration),
                shard_sha256=sha(shard),
            )
            document = verify_completed(ROOT, day, entry, bindings)
            compare_previous_day(day, shard)
            entry["primary_bins_at_least_80pct"] = document["primary_bins_at_least_80pct"]
        ledger["days"][day] = entry
        if process.returncode != 0 or stopped:
            ledger["status"] = "BLOCKED_FAILED_TRAIN_DAY"
        write_report(ledger_path, ledger)
        print(json.dumps({"date": day, **entry}), flush=True)
        if process.returncode != 0 or stopped:
            ledger["status"] = "BLOCKED_FAILED_TRAIN_DAY"
            write_report(ledger_path, ledger)
            return 2
    valid, denominators, configs = [], [], []
    previous_last = None
    for day in days:
        verify_completed(ROOT, day, ledger["days"][day], bindings)
        manifest, configuration, shard = paths(ROOT, day)
        with np.load(shard, allow_pickle=False) as data:
            counts = data["valid_ping_count"][:, 0, 5:50]
            den = data["support_denominator_ping_count"]
            if (
                counts.shape != (96, 45)
                or den.shape != (96,)
                or not np.array_equal(
                    den, np.maximum(data["expected_ping_count"], data["observed_ping_count"])
                )
            ):
                raise ValueError("Unexpected day/count schema")
            stamps = data["raw_ping_time"]
            if len(stamps):
                if (np.diff(stamps).astype("int64") <= 0).any() or (
                    previous_last is not None and stamps[0] <= previous_last
                ):
                    raise ValueError("Raw timestamp overlap/order error across TRAIN days")
                previous_last = stamps[-1]
            valid.append(counts)
            denominators.append(den)
        configs.extend(json.loads(configuration.read_text())["bin_configuration_ids"])
    cutoffs = eligible_cutoffs(np.concatenate(valid), np.concatenate(denominators), configs)
    eligible_days = sorted({days[index // 96] for index in cutoffs})
    result = {
        "status": "TRAIN_CENSUS_COMPLETE_STRICT_CONTEXT_CANDIDATE_ONLY",
        "global_ineligibility_conclusion": "NOT_ESTABLISHED_TARGET_ONLY_BOUND_REQUIRED",
        "method_identity": identity,
        "processed_train_calendar_days": 100,
        "eligible_train_cutoff_days": len(eligible_days),
        "eligible_dates": eligible_days,
        "eligible_hourly_anchors": len(cutoffs),
        "unresolved_train_days": 0,
        "unmeasured_nontrain_day_upper_bound": 67,
        "strict_context_policy_day_upper_bound": len(eligible_days) + 67,
        "minimum_overall_days": 90,
        "interval_calibration_day_minimum": "NOT_EVALUATED",
        "test_day_minimum": "NOT_EVALUATED",
        "held_out_acoustic_payloads_processed": False,
        "benchmark_eligible": False,
        "completed_benchmark_runs": 0,
        "availability_assumption": "zero_latency_replay_at_bin_end",
        "scope": "Necessary physical/QC support eligibility only; not real-corpus promotion or a JEPA result",
    }
    finalize_reports(
        ledger_path, ledger, ROOT / "evidence/continuation/train_census_v2_eligibility.json", result
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "method_identity"}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
