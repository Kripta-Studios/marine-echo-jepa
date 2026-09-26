"""Synthetic count-only tests for the complete TRAIN native-grid audit."""

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _module():
    path = Path(__file__).resolve().parents[2] / "tools/train_support_map.py"
    spec = importlib.util.spec_from_file_location("train_support_map", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _counts():
    valid = np.zeros((100, 96, 4, 64), dtype=np.int16)
    effective = np.full((100, 96), 60, dtype=np.int16)
    days = np.arange("2020-02-17", "2020-05-27", dtype="datetime64[D]").astype(str).tolist()
    return valid, effective, days


def test_full_grid_count_map_preserves_boundary_and_daily_variation() -> None:
    module = _module()
    valid, effective, days = _counts()
    valid[0, 0, 0, 0] = 47
    valid[0, 1, 0, 0] = 48
    valid[1, 0, 0, 0] = 60
    effective[2, 0] = 61
    valid[2, 0, 0, 0] = 48
    valid[3, 0, 0, 0] = 61
    effective[3, 0] = 61
    result = module.summarize_support(valid, effective, days)
    assert result["processed_train_calendar_days"] == 100
    assert len(result["cells"]) == 4 * 64
    first = result["cells"][0]
    assert first["frequency_hz"] == 38000
    assert first["range_start_m"] == 0
    assert first["range_end_m"] == 2
    assert first["valid_ping_count"] == 47 + 48 + 60 + 48 + 61
    assert first["effective_ping_count"] == 100 * 96 * 60 + 2
    assert first["quarter_hours_at_least_80_pct"] == 3
    assert first["utc_days_with_any_quarter_hour_at_least_80_pct"] == 3
    assert first["median_daily_support_fraction"] == 0
    assert first["maximum_daily_support_fraction"] == (47 + 48) / (96 * 60)
    assert result["daily"][0]["valid_ping_count"][0][0] == 95
    assert result["daily"][3]["effective_ping_count"][0][0] == 96 * 60 + 1
    assert result["cells"][-1]["frequency_hz"] == 455000
    assert result["cells"][-1]["range_end_m"] == 128
    assert result["cells"][-1]["valid_ping_count"] == 0


@pytest.mark.parametrize("corrupt", ["float", "negative", "exceeds_denominator", "bad_shape"])
def test_count_map_rejects_invalid_count_arrays(corrupt: str) -> None:
    module = _module()
    valid, effective, days = _counts()
    if corrupt == "float":
        valid = valid.astype(float)
    elif corrupt == "negative":
        valid[0, 0, 0, 0] = -1
    elif corrupt == "exceeds_denominator":
        valid[0, 0, 0, 0] = 61
    else:
        valid = valid[:, :, :, :-1]
    with pytest.raises(ValueError):
        module.summarize_support(valid, effective, days)


def test_audit_refuses_missing_complete_v2_evidence_before_npz_access(tmp_path: Path) -> None:
    module = _module()
    with pytest.raises((ValueError, FileNotFoundError)):
        module.audit_support_map(tmp_path)


def _write_json(root: Path, relative: str, document: dict) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complete_synthetic_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _module()
    days = np.arange("2020-02-17", "2020-05-27", dtype="datetime64[D]").astype(str).tolist()
    contract_source = Path(__file__).resolve().parents[2] / module._CONTRACT
    contract = tmp_path / module._CONTRACT
    contract.parent.mkdir(parents=True, exist_ok=True)
    contract.write_bytes(contract_source.read_bytes())
    _write_json(
        tmp_path,
        module._REVIEW,
        {
            "reviewer_session": "/root/continuation_review",
            "disposition": "APPROVE_TRAIN_SUPPORT_MAP_METHOD",
            "reviewed_contract_sha256": _sha(contract),
            "reviewed_code_sha256": _sha(Path(module.__file__)),
        },
    )
    source_files = {
        "driver_sha256": "tools/train_census_v2.py",
        "code_sha256": "tools/preprocess_train_candidate.py",
        "contract_sha256": "evidence/continuation/train_census_v2_contract.json",
        "compat_code_sha256": "src/marine_echo/data/azfp_compat.py",
        "qc_code_sha256": "src/marine_echo/data/candidate_qc.py",
        "regrid_code_sha256": "src/marine_echo/data/range_grid.py",
        "protocol_sha256": "reports/active/protocol.json",
        "method_review_sha256": "orchestration/reviews/TRAIN_CENSUS_METHOD_20260927.json",
        "singleton_review_sha256": "orchestration/reviews/AZFP_SINGLETON_COMPAT_20260927.json",
        "prior_failed_execution_sha256": "evidence/continuation/train_census_execution.json",
    }
    digests = {}
    for field, relative in source_files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(field, encoding="utf-8")
        digests[field] = _sha(path)
    bindings = {
        key: value
        for key, value in digests.items()
        if key
        not in (
            "driver_sha256",
            "method_review_sha256",
            "singleton_review_sha256",
            "prior_failed_execution_sha256",
        )
    }
    identity = {
        "calendar": days,
        "bindings": bindings,
        **{
            key: digests[key]
            for key in (
                "driver_sha256",
                "method_review_sha256",
                "singleton_review_sha256",
                "prior_failed_execution_sha256",
            )
        },
    }
    _write_json(
        tmp_path,
        module._TARGET_INPUT,
        {
            "execution_path": module._EXECUTION,
            "strict_report_path": module._STRICT,
            "driver_sha256": digests["driver_sha256"],
            "preprocessor_sha256": digests["code_sha256"],
            "census_contract_sha256": digests["contract_sha256"],
            "compat_code_sha256": digests["compat_code_sha256"],
            "method_contract_sha256": _sha(
                _write_json(tmp_path, module._TARGET_METHOD, {"id": "synthetic-method"})
            ),
        },
    )
    target_code = tmp_path / "tools/target_only_bound.py"
    target_code.write_text("# synthetic target verifier\n", encoding="utf-8")

    def paths(root, day):
        prefix = f"synthetic/{day}"
        return tuple(
            root / f"{prefix}/{name}" for name in ("manifest.json", "config.json", "counts.npz")
        )

    entries = {}
    for day in days:
        manifest, configuration, shard = paths(tmp_path, day)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}", encoding="utf-8")
        configuration.write_text("{}", encoding="utf-8")
        np.savez_compressed(
            shard,
            valid_ping_count=np.zeros((96, 4, 64), dtype=np.int16),
            expected_ping_count=np.full(96, 60, dtype=np.int16),
            observed_ping_count=np.full(96, 60, dtype=np.int16),
            support_denominator_ping_count=np.full(96, 60, dtype=np.int16),
            bin_start=np.datetime64(day, "ns") + np.arange(96) * np.timedelta64(15, "m"),
            frequency_hz=np.array([38000, 125000, 200000, 455000]),
            range_edges_m=np.arange(0, 130, 2),
        )
        entries[day] = {
            "manifest_sha256": _sha(manifest),
            "configuration_sha256": _sha(configuration),
            "shard_sha256": _sha(shard),
        }
    execution = {
        "status": "COMPLETE_TRAIN_CENSUS_NOT_BENCHMARK",
        "identity": identity,
        "days": entries,
        "held_out_acoustic_payloads_processed": False,
        "completed_benchmark_runs": 0,
    }
    execution_path = _write_json(tmp_path, module._EXECUTION, execution)
    strict_path = _write_json(
        tmp_path,
        module._STRICT,
        {
            "status": "TRAIN_CENSUS_COMPLETE_STRICT_CONTEXT_CANDIDATE_ONLY",
            "global_ineligibility_conclusion": "NOT_ESTABLISHED_TARGET_ONLY_BOUND_REQUIRED",
            "execution_report_sha256": _sha(execution_path),
            "method_identity": identity,
            "processed_train_calendar_days": 100,
            "unresolved_train_days": 0,
            "held_out_acoustic_payloads_processed": False,
            "completed_benchmark_runs": 0,
            "benchmark_eligible": False,
        },
    )
    _write_json(
        tmp_path,
        module._TARGET,
        {
            "status": "D1_INELIGIBLE_TARGET_SUPPORT_UPPER_BOUND",
            "census_generation": "census-v2",
            "contract_sha256": _sha(tmp_path / module._TARGET_METHOD),
            "input_contract_sha256": _sha(tmp_path / module._TARGET_INPUT),
            "code_sha256": _sha(target_code),
            "execution_report_sha256": _sha(execution_path),
            "strict_context_report_sha256": _sha(strict_path),
            "held_out_acoustic_payloads_processed": False,
            "completed_benchmark_runs": 0,
            "benchmark_eligible": False,
            "processed_train_calendar_days": 100,
            "target_supported_train_anchor_days_upper_bound": 0,
            "target_supported_train_hourly_anchors_upper_bound": 0,
            "target_supported_dates": [],
            "overall_eligible_day_upper_bound": 67,
            "unmeasured_nontrain_day_upper_bound": 67,
            "minimum_overall_days": 90,
        },
    )
    v2 = types.ModuleType("train_census_v2")
    v2.paths = paths
    v2.validate_ledger = lambda ledger, method, calendar: (
        None if ledger["identity"] == method and calendar == days else ValueError("ledger")
    )
    v2.previous_execution = lambda: {"days": {}}

    def verify_completed(root, day, entry, method):
        assert method == bindings
        for path, key in zip(
            paths(root, day), ("manifest_sha256", "configuration_sha256", "shard_sha256")
        ):
            if _sha(path) != entry[key]:
                raise ValueError("Tampered shard")

    v2.verify_completed = verify_completed
    old = types.ModuleType("train_census")
    old.paths = paths
    old.verify_completed = verify_completed
    target_module = types.ModuleType("target_only_bound")
    target_module.validate_lineage = lambda *args: None
    target_module.target_only_cutoffs = lambda counts, denominator: (
        [] if counts.shape == (9600, 45) and denominator.shape == (9600,) else None
    )
    for name, dependency in (
        ("train_census_v2", v2),
        ("train_census", old),
        ("target_only_bound", target_module),
    ):
        monkeypatch.setitem(sys.modules, name, dependency)
    return module, days, paths


def test_complete_synthetic_audit_reads_only_counts_and_detects_shard_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, days, paths = _complete_synthetic_audit(tmp_path, monkeypatch)
    result = module.audit_support_map(tmp_path)
    assert result["status"] == "TRAIN_QC_ONLY_NOT_BENCHMARK"
    assert result["processed_train_calendar_days"] == 100
    assert len(result["cells"]) == 256
    assert len(result["daily"]) == 100
    assert result["cells"][0]["valid_ping_count"] == 0
    assert result["completed_benchmark_runs"] == 0
    shard = paths(tmp_path, days[0])[2]
    shard.write_bytes(shard.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="Tampered shard"):
        module.audit_support_map(tmp_path)


def test_complete_audit_rejects_v1_target_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, _, _ = _complete_synthetic_audit(tmp_path, monkeypatch)
    target = tmp_path / module._TARGET
    document = json.loads(target.read_text(encoding="utf-8"))
    document["census_generation"] = "census-v1"
    target.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="target-only v2"):
        module.audit_support_map(tmp_path)


def test_complete_audit_rejects_partial_v2_before_count_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, _, _ = _complete_synthetic_audit(tmp_path, monkeypatch)
    execution = tmp_path / module._EXECUTION
    document = json.loads(execution.read_text(encoding="utf-8"))
    document["status"] = "RUNNING_TRAIN_ONLY_CENSUS"
    execution.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("Count NPZ was opened"))
    with pytest.raises(ValueError, match="Complete v2 TRAIN census required"):
        module.audit_support_map(tmp_path)


def test_main_preserves_existing_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    output = tmp_path / module._OUTPUT
    output.parent.mkdir(parents=True)
    output.write_bytes(b"prior report sentinel")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "audit_support_map", lambda root: {"status": "SYNTHETIC"})
    with pytest.raises(FileExistsError):
        module.main()
    assert output.read_bytes() == b"prior report sentinel"


def test_count_audit_rejects_reordered_frequency_grid_even_with_valid_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, days, paths = _complete_synthetic_audit(tmp_path, monkeypatch)
    shard = paths(tmp_path, days[0])[2]
    with np.load(shard, allow_pickle=False) as data:
        fields = {name: data[name] for name in data.files}
    fields["frequency_hz"] = fields["frequency_hz"][::-1]
    np.savez_compressed(shard, **fields)
    monkeypatch.setattr(sys.modules["train_census_v2"], "verify_completed", lambda *args: None)
    with pytest.raises(ValueError, match="grid schema"):
        module.audit_support_map(tmp_path)
