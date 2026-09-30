import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "evidence/ssl-band-replication-builder-v2")
)
import replication_test_support  # noqa: F401

"""SYNTHETIC_CORRECTNESS_ONLY CPU policy checks; no actual child or ledger."""

import importlib.util
import sys
from pathlib import Path

import pytest

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
SPEC = importlib.util.spec_from_file_location(
    "band_operational_job_under_test", BUILDER / "tools/execute_native_band_operational_job_v4.py"
)
job = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = job
SPEC.loader.exec_module(job)


def test_band_failed_and_resumed_attempts_count_full_owned_time():
    ledger = {
        "gpu_limit_hours": 96,
        "gpu_hours_spent_owned_scientific_jobs": 90,
        "runs": [
            {
                "budget_family": "native_band_v1",
                "status": "FAILED_REAL_CUDA_ATTEMPT",
                "elapsed_owned_seconds": 5 * 3600,
            },
            {
                "budget_family": "native_band_v1",
                "status": "COMPLETED_REAL_CUDA",
                "elapsed_owned_seconds": 6 * 3600,
                "resume": "synthetic.pt",
            },
        ],
    }
    assert job.budget_allowance(ledger) == 3600


@pytest.mark.parametrize("seed", [13, 23])
def test_mixed_v1_v2_failed_attempts_use_one_family_and_charge_both_limits(case, seed):
    c = case
    c.config["seed"] = seed
    c.ledger["gpu_hours_spent_owned_scientific_jobs"] = 95
    c.ledger["runs"] = [
        {
            "budget_family": "native_band_v1",
            "status": "FAILED_REAL_BAND_CUDA_ATTEMPT",
            "elapsed_owned_seconds": 5 * 3600,
            "implementation_version": 1,
        },
        {
            "budget_family": "native_band_v1",
            "status": "COMPLETED_REAL_BAND_CUDA",
            "elapsed_owned_seconds": 6 * 3600,
            "implementation_version": 2,
        },
    ]
    c.exit_code, c.write_report = 9, False
    c.seal()
    plan = invoke(c)
    assert plan.band_hours == 11 and plan.aggregate_hours == 95
    assert plan.deadline_seconds == 3600
    code = job._execute_for_test(
        c.request,
        c.root,
        popen=c.popen,
        supervisor=c.supervisor,
        config_validator=c.validator,
        clock=c.clock,
        writer=c.writer,
    )
    assert code == 9
    assert c.journals[-1]["native_band_gpu_hours_spent_full_owned"] == pytest.approx(11 + 60 / 3600)
    assert c.journals[-1]["gpu_hours_spent_owned_scientific_jobs"] == pytest.approx(95 + 60 / 3600)
    assert c.journals[-1]["runs"][-1]["budget_family"] == "native_band_v1"
    assert c.journals[-1]["runs"][-1]["verified_completion"] is False


def test_unknown_band_job_blocks_without_launch_or_mutation(case):
    c = case
    c.ledger["runs"] = [
        {
            "budget_family": "native_band_v1",
            "status": "UNKNOWN_PENDING_JOB",
            "elapsed_owned_seconds": 0,
        }
    ]
    c.seal()
    with pytest.raises(ValueError, match="Unknown"):
        invoke(c)
    assert c.fs.mutations == [] and c.calls == []


import copy
import inspect
import json
from types import SimpleNamespace

SUPPORT_SPEC = importlib.util.spec_from_file_location(
    "band_policy_support",
    BUILDER / "evidence/ssl-band-replication-builder-v2/replication_policy_support.py",
)
support = importlib.util.module_from_spec(SUPPORT_SPEC)
sys.modules[SUPPORT_SPEC.name] = support
SUPPORT_SPEC.loader.exec_module(support)
generator = support.load_generator(BUILDER)


LEGACY_SPEC = importlib.util.spec_from_file_location(
    "legacy_policy_fixture_job", BUILDER / "tools/execute_native_band_replication_job.py"
)
legacy_job = importlib.util.module_from_spec(LEGACY_SPEC)
sys.modules[LEGACY_SPEC.name] = legacy_job
LEGACY_SPEC.loader.exec_module(legacy_job)


def operational_case(monkeypatch, **kwargs):
    c = support.policy_case(legacy_job, generator, monkeypatch, BUILDER, MAIN, **kwargs)
    required = job.source_paths(MAIN)
    for path in required:
        c.fs.files[c.root / path.relative_to(MAIN)] = path.read_bytes()
    original_seal = c.seal

    def seal():
        original_seal()
        c.review['bindings'].update({str(c.root / path.relative_to(MAIN)):
                                    job.digest(c.root / path.relative_to(MAIN))
                                    for path in required})
        c.fs.files[c.request.review] = support.encode(c.review)

    c.seal = seal
    c.seal()
    return c


@pytest.fixture
def case(monkeypatch):
    return operational_case(monkeypatch)


def invoke(c):
    # This helper does not inject a production root: tests call the private runner
    # directly at each call site so its inspected caller really is a test.
    return job._admit(c.request, c.root, config_validator=c.validator)


def test_actual_catalog_exact_three_seeds_fixed_shared_and_direct():
    from marine_echo.training import native_band_replication_downstream as downstream
    from marine_echo.training import native_band_replication_ssl as ssl

    screens, prospective, unsupported = generator._catalog(ssl.Config, downstream.DownstreamConfig)
    assert len(screens) == 6 and len(prospective) == 6 and unsupported == []
    assert {c["seed"] for c in screens.values()} == {7, 13, 23}
    for c in screens.values():
        if "mode" in c:
            assert (
                c
                == downstream.DownstreamConfig(
                    method="direct", mode="direct_end_to_end", seed=c["seed"]
                ).to_dict()
            )
            assert (c["updates"], c["cadence"]) == (3000, 750)
        else:
            assert c == ssl.Config(seed=c["seed"]).to_dict()
            assert (
                c["pretrain_updates"],
                c["pretrain_cadence"],
                c["readout_updates"],
                c["readout_cadence"],
            ) == (6000, 1500, 500, 250)


def test_config_exclusive_generator_and_collision_no_partial_overwrite(case):
    c = case
    output = c.root / "configs/SYNTHETIC_CORRECTNESS_ONLY-generated"
    result = generator._generate(output, c.root, c.definitions)
    assert len(result["screens"]) == 4
    assert result["prospective_downstream"] == []
    assert result["scientific_approval"] is False
    saved = {p: raw for p, raw in c.fs.files.items() if p.is_relative_to(output)}
    count = len(c.fs.mutations)
    with pytest.raises(FileExistsError):
        generator._generate(output, c.root, c.definitions)
    assert saved == {p: raw for p, raw in c.fs.files.items() if p.is_relative_to(output)}
    assert len(c.fs.mutations) == count


def test_private_root_not_available_as_cli_or_helper_bypass(case):
    assert "root" not in inspect.signature(job.execute).parameters
    assert "root" not in inspect.signature(generator.generate).parameters
    with pytest.raises(ValueError, match="Private"):
        generator._generate_for_test(case.root / "configs/fake", MAIN, case.definitions)


def test_admission_before_mutation_or_launch_and_exact_command(case):
    plan = invoke(case)
    assert case.fs.mutations == []
    assert case.calls == []
    assert plan.command[3] == "marine_echo.training.native_band_operational_ssl"
    assert plan.command[-1] == "cuda:0"
    assert str(case.root / "data/processed/native_ssl_v1/train.npz") in plan.command
    assert not any("final_test" in v for v in plan.command)
    assert plan.deadline_seconds == 12 * 3600


@pytest.mark.parametrize(
    "key",
    [
        "tools/execute_native_band_operational_job_v4.py",
        "tools/prepare_native_band_replication_configs.py",
        "tools/native_reference_supervisor.py",
        "src/marine_echo/training/native_band_ssl.py",
        "src/marine_echo/models/native_band_temporal.py",
        "configs/config.json",
        "orchestration/native_band_budget_owner_resolution_v1.json",
        "data/processed/native_ssl_v1/train.npz",
        "data/processed/native_ssl_v1/development.npz",
        "configs/native_ssl_split_v1.json",
        "uv.lock",
    ],
)
def test_stale_binding_fails_before_mutation(case, key):
    c = case
    c.fs.files[c.root / key] += b"changed"
    with pytest.raises(ValueError):
        invoke(c)
    assert c.fs.mutations == [] and c.calls == []


@pytest.mark.parametrize(
    "required",
    [
        "tools/execute_native_band_operational_job_v4.py",
        "tools/native_reference_supervisor.py",
        "src/marine_echo/models/native_temporal.py",
        "configs/config.json",
        "orchestration/native_band_budget_owner_resolution_v1.json",
        ".venv/Scripts/python.exe",
        "docs/adr/0018-native-acoustic-downstream-transfer.md",
    ],
)
def test_missing_binding_fails_before_mutation(case, required):
    c = case
    c.review["bindings"].pop(str(c.root / required))
    c.fs.files[c.request.review] = support.encode(c.review)
    with pytest.raises(ValueError, match="binding"):
        invoke(c)
    assert c.fs.mutations == []


def test_every_declared_binding_verified_not_just_required_subset(case):
    c = case
    path = c.root / "orchestration/SYNTHETIC_CORRECTNESS_ONLY-extra-source.py"
    c.fs.files[path] = b"old"
    c.review["bindings"][str(path)] = job.digest(path)
    c.fs.files[path] = b"changed"
    c.fs.files[c.request.review] = support.encode(c.review)
    with pytest.raises(ValueError, match="Stale"):
        invoke(c)
    assert c.fs.mutations == []


@pytest.mark.parametrize(
    "bad",
    [
        "self",
        "root",
        "missing_identity",
        "wrong_status",
        "architecture",
        "evidence",
        "roles",
        "method",
        "seed",
        "runtime",
        "approved_config",
        "budget_status",
        "budget_path",
    ],
)
def test_review_identity_scope_and_path_gates(case, bad):
    c = case
    if bad == "self":
        c.review["reviewer_session_id"] = job.IMPLEMENTER.upper()
    elif bad == "root":
        c.review["reviewer_session_id"] = c.review["root_coordinator_session_id"]
    elif bad == "missing_identity":
        del c.review["root_coordinator_session_id"]
    elif bad == "wrong_status":
        c.review["status"] = "APPROVED_FINAL_ASSESSMENT"
    elif bad == "architecture":
        c.review["architecture"] = "legacy"
    elif bad == "evidence":
        c.review["evidence_kind"] = "SYNTHETIC_CORRECTNESS_ONLY"
    elif bad == "roles":
        c.review["allowed_roles"].append("final_test")
    elif bad == "method":
        c.review["allowed_methods"] = ["cf_jepa"]
    elif bad == "seed":
        c.review["allowed_seeds"] = [17]
    elif bad == "runtime":
        c.review["runtime_arguments"]["output"] = "another-output"
    elif bad == "approved_config":
        c.review["approved_config"] = {}
    elif bad == "budget_status":
        c.review["band_budget_status"] = "PENDING"
    else:
        c.review["budget_resolution_path"] = "different-receipt.json"
    c.fs.files[c.request.review] = support.encode(c.review)
    with pytest.raises(ValueError):
        invoke(c)
    assert c.fs.mutations == [] and c.calls == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 17),
        ("seed", True),
        ("history", 24),
        ("method", "cf_jepa"),
        ("method", "direct"),
        ("architecture", "legacy"),
        ("latent", 128),
        ("lr", 0.003),
    ],
)
def test_config_wrong_family_seed_or_recipe_denied(case, field, value):
    c = case
    c.config[field] = value
    c.seal()
    with pytest.raises(ValueError):
        invoke(c)
    assert c.fs.mutations == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed7_recipe_limit_including_controls", 12),
        ("revision_gpu_hours_limit_full_owned", 13),
        ("aggregate_gpu_hours_limit_full_owned", 97),
        ("revision_gpu_hours_limit_full_owned", True),
        ("single_revision_recipes", ["extra_cf_control"]),
        ("status", "PENDING"),
    ],
)
def test_owner_budget_exact_limits_and_five_recipes(case, field, value):
    c = case
    c.budget[field] = value
    c.seal()
    with pytest.raises(ValueError):
        invoke(c)
    assert c.fs.mutations == []


@pytest.mark.parametrize(
    "bad",
    [
        "unaccounted",
        "nan",
        "negative",
        "missing_elapsed",
        "conflict",
        "aggregate_underreported",
        "band_exhausted",
        "aggregate_exhausted",
    ],
)
def test_invalid_historical_accounting_denied(case, bad):
    c = case
    record = {
        "budget_family": job.FAMILY,
        "status": "FAILED_REAL_CUDA_ATTEMPT",
        "elapsed_owned_seconds": 1.0,
    }
    c.ledger["runs"] = [record]
    c.ledger["gpu_hours_spent_owned_scientific_jobs"] = 1
    if bad == "unaccounted":
        record.pop("budget_family")
        record["architecture"] = job.ARCHITECTURE
    elif bad == "nan":
        record["elapsed_owned_seconds"] = float("nan")
        c.fs.files[c.root / "orchestration/native_ssl_run_ledger_v1.json"] = json.dumps(
            c.ledger
        ).encode()
    elif bad == "negative":
        record["elapsed_owned_seconds"] = -1
    elif bad == "missing_elapsed":
        record.pop("elapsed_owned_seconds")
    elif bad == "conflict":
        record["resources_full_attempt"] = {"elapsed_full_attempt_seconds": 2}
    elif bad == "aggregate_underreported":
        c.ledger["gpu_hours_spent_owned_scientific_jobs"] = 0
    elif bad == "band_exhausted":
        record["elapsed_owned_seconds"] = 12 * 3600
        c.ledger["gpu_hours_spent_owned_scientific_jobs"] = 12
    else:
        c.ledger["gpu_hours_spent_owned_scientific_jobs"] = 96
    if bad != "nan":
        c.seal()
    with pytest.raises(ValueError):
        invoke(c)
    assert c.fs.mutations == []


@pytest.mark.parametrize("active", ["RUNNING_CUDA", "RUNNING_CPU_FIT"])
def test_active_ledger_preserved_before_launch(case, active):
    c = case
    c.ledger["runs"] = [{"status": active}]
    c.seal()
    with pytest.raises(ValueError, match="active"):
        invoke(c)
    assert c.fs.mutations == []


def test_existing_gpu_owner_lock_unknown_preserved(case):
    c = case
    lock = c.root / "evidence/ssl-builder-v1/gpu-owner.lock"
    c.fs.files[lock] = b"unknown owner"
    with pytest.raises(ValueError, match="lock"):
        invoke(c)
    assert c.fs.files[lock] == b"unknown owner"
    assert c.fs.mutations == []


@pytest.mark.parametrize("collision", ["output", "receipt", "completed_resume"])
def test_existing_paths_and_completed_resume_protected(case, collision):
    c = case
    if collision == "output":
        c.fs.dirs.add(c.request.output)
    elif collision == "receipt":
        c.fs.dirs.add(c.request.receipt)
    else:
        c.fs.dirs.add(c.request.output)
        c.fs.files[c.request.output / "run.json"] = b"completed"
        c.fs.files[c.request.output / "latest.pt"] = b"synthetic unparsed checkpoint"
        c.request = job.Job(**{**vars(c.request), "resume": c.request.output / "latest.pt"})
    with pytest.raises((FileExistsError, ValueError)):
        invoke(c)
    assert c.fs.mutations == []


def test_unapproved_final_npz_binding_never_hashed(case, monkeypatch):
    c = case
    path = c.root / "data/final_test.npz"
    c.review["bindings"][str(path)] = "a" * 64
    c.fs.files[c.request.review] = support.encode(c.review)
    original = job.digest

    def checked(p):
        assert Path(p) != path, "Final numerical artifact must remain closed."
        return original(p)

    monkeypatch.setattr(job, "digest", checked)
    with pytest.raises(ValueError, match="NPZ"):
        invoke(c)
    assert c.fs.mutations == []


def test_valid_success_charges_full_elapsed_and_commits_verified_receipt(case):
    c = case
    code = job._execute_for_test(
        c.request,
        c.root,
        popen=c.popen,
        supervisor=c.supervisor,
        config_validator=c.validator,
        clock=c.clock,
        writer=c.writer,
    )
    assert code == 0 and len(c.calls) == 1
    assert c.journals[0]["runs"][-1]["status"] == "RUNNING_CUDA"
    final = c.journals[-1]
    record = final["runs"][-1]
    assert record["verified_completion"] is True
    assert record["owned_tree_cleanup_verified"] is True and record["gpu_lock_absent"] is True
    assert final["gpu_hours_spent_owned_scientific_jobs"] == pytest.approx(60 / 3600)
    assert final["native_band_gpu_hours_spent_full_owned"] == pytest.approx(60 / 3600)
    assert final["gpu_hours_completed_scientific_training"] == pytest.approx(30 / 3600)
    assert record["budget_family"] == job.FAMILY
    assert record["pid"] == 77
    assert c.supervisor_args["rss_limit_bytes"] == 22 * 2**30
    assert c.supervisor_args["deadline_seconds"] == 12 * 3600
    receipt = json.loads(c.fs.files[c.request.receipt / "attempt.json"])
    assert receipt["run_sha256"] == job.digest(c.request.output / "run.json")


@pytest.mark.parametrize("exit_code", [1, 7, -9])
def test_failed_child_exit_preserved_and_full_time_charged(case, exit_code):
    c = case
    c.exit_code = exit_code
    code = job._execute_for_test(
        c.request,
        c.root,
        popen=c.popen,
        supervisor=c.supervisor,
        config_validator=c.validator,
        clock=c.clock,
        writer=c.writer,
    )
    assert code == exit_code
    final = c.journals[-1]
    assert final["runs"][-1]["status"] == "FAILED_REAL_BAND_CUDA_ATTEMPT"
    assert final["runs"][-1]["exit_code"] == exit_code
    assert final["scientific_fits_completed"] == 0
    assert final["native_band_gpu_hours_spent_full_owned"] == pytest.approx(60 / 3600)


@pytest.mark.parametrize(
    "bad",
    [
        "missing",
        "status",
        "evidence",
        "architecture",
        "mode",
        "seed",
        "artifact_hash",
        "report_json",
    ],
)
def test_no_success_from_missing_or_invalid_report(case, bad):
    c = case
    original = c.supervisor
    if bad == "missing":
        c.write_report = False

    def supervise(child, **kw):
        result = original(child, **kw)
        path = c.request.output / "run.json"
        if bad == "missing":
            return result
        if bad == "report_json":
            c.fs.files[path] = b"invalid json"
            return result
        report = json.loads(c.fs.files[path])
        if bad == "status":
            report["status"] = "FAILED"
        elif bad == "evidence":
            report["evidence_kind"] = "SYNTHETIC_CORRECTNESS_ONLY"
        elif bad == "architecture":
            report["architecture"] = "legacy"
        elif bad == "mode":
            report["config"]["mode"] = "wrong"
        elif bad == "seed":
            report["config"]["seed"] = 17
        else:
            report["inference_sha256"] = "a" * 64
        c.fs.files[path] = support.encode(report)
        return result

    if bad == "report_json":
        with pytest.raises(ValueError):
            job._execute_for_test(
                c.request,
                c.root,
                popen=c.popen,
                supervisor=supervise,
                config_validator=c.validator,
                clock=c.clock,
                writer=c.writer,
            )
        assert c.journals[-1]["runs"][-1]["status"] == "RUNNING_CUDA"
        assert c.journals[-1]["runs"][-1]["requires_reconciliation"]
    else:
        assert (
            job._execute_for_test(
                c.request,
                c.root,
                popen=c.popen,
                supervisor=supervise,
                config_validator=c.validator,
                clock=c.clock,
                writer=c.writer,
            )
            == 1
        )
        assert c.journals[-1]["runs"][-1]["verified_completion"] is False


@pytest.mark.parametrize("matching", [True, False])
def test_timeout_cleanup_only_matching_terminated_owned_tree(case, matching):
    c = case
    c.reason = "FULL_OWNED_ATTEMPT_DEADLINE"
    lock = c.root / "evidence/ssl-builder-v1/gpu-owner.lock"
    original = c.supervisor

    def supervise(child, **kw):
        r = original(child, **kw)
        c.fs.files[lock] = support.encode(
            {"pid": 78 if matching else 999, "output": str(c.request.output)}
        )
        return r

    code = job._execute_for_test(
        c.request,
        c.root,
        popen=c.popen,
        supervisor=supervise,
        config_validator=c.validator,
        clock=c.clock,
        writer=c.writer,
    )
    assert code == 1
    assert lock.exists() != matching
    record = c.journals[-1]["runs"][-1]
    assert record.get("owned_terminated_tree_lock_cleanup", False) == matching
    assert record.get("unknown_lock_preserved", False) != matching


def test_normal_child0_with_remaining_lock_is_failure_without_deleting_lock(case):
    c = case
    lock = c.root / "evidence/ssl-builder-v1/gpu-owner.lock"
    original = c.supervisor

    def supervise(child, **kw):
        result = original(child, **kw)
        c.fs.files[lock] = support.encode({"pid": 77, "output": str(c.request.output)})
        return result

    assert (
        job._execute_for_test(
            c.request,
            c.root,
            popen=c.popen,
            supervisor=supervise,
            config_validator=c.validator,
            clock=c.clock,
            writer=c.writer,
        )
        == 1
    )
    assert lock.exists()
    assert c.journals[-1]["runs"][-1]["requires_reconciliation"]


def test_supervisor_exception_retains_active_reconciliation_and_charges_known_time(case):
    c = case

    def supervise(child, **kw):
        c.now += 7
        raise RuntimeError("SYNTHETIC_CORRECTNESS_ONLY supervisor failure")

    with pytest.raises(RuntimeError):
        job._execute_for_test(
            c.request,
            c.root,
            popen=c.popen,
            supervisor=supervise,
            config_validator=c.validator,
            clock=c.clock,
            writer=c.writer,
        )
    record = c.journals[-1]["runs"][-1]
    assert record["status"] == "RUNNING_CUDA" and record["requires_reconciliation"]
    assert record["verified_completion"] is False
    assert "owned_tree_cleanup_verified" not in record
    assert c.journals[-1]["native_band_gpu_hours_spent_full_owned"] == pytest.approx(7 / 3600)
    assert not (c.request.receipt / "attempt.json").exists()


def test_no_owned_cleanup_proof_never_marked_completed(case):
    c = case
    original = c.supervisor

    def supervise(child, **kw):
        result = original(child, **kw)
        result["owned_tree_cleanup_verified"] = False
        return result

    with pytest.raises(ValueError, match="cleanup"):
        job._execute_for_test(
            c.request,
            c.root,
            popen=c.popen,
            supervisor=supervise,
            config_validator=c.validator,
            clock=c.clock,
            writer=c.writer,
        )
    assert c.journals[-1]["runs"][-1]["status"] == "RUNNING_CUDA"


def test_downstream_direct_actual_entrypoint_no_ancestor(monkeypatch):
    c = operational_case(monkeypatch, kind="downstream")
    plan = invoke(c)
    assert plan.command[3] == "marine_echo.training.native_band_operational_downstream"
    assert "--encoder" not in plan.command
    assert "--adr0016" in plan.command
    assert str(c.root / "docs/adr/0018-native-acoustic-downstream-transfer.md") in plan.command
    assert (
        job._execute_for_test(
            c.request,
            c.root,
            popen=c.popen,
            supervisor=c.supervisor,
            config_validator=c.validator,
            clock=c.clock,
            writer=c.writer,
        )
        == 0
    )


def test_downstream_missing_ancestor_args_denied(monkeypatch):
    c = operational_case(
        monkeypatch,
        kind="downstream",
        mode="frozen_readout",
        method="shared_ssl",
    )
    c.request = job.Job(**{**vars(c.request), "ancestor_review": None})
    with pytest.raises(ValueError, match="ancestor"):
        invoke(c)
    assert c.fs.mutations == []


def test_original_supervisor_signature_inspected_without_process_execution():
    path = MAIN / "tools/native_reference_supervisor.py"
    spec = importlib.util.spec_from_file_location("immutable_band_supervisor_inspected", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parameters = inspect.signature(module.supervise_owned).parameters
    assert {"child", "started", "deadline_seconds", "rss_limit_bytes"} <= set(parameters)
    assert parameters["poll_seconds"].default == 0.1


def test_report_verification_elapsed_deadline_never_returns_success(case, monkeypatch):
    c = case
    c.ledger["gpu_hours_spent_owned_scientific_jobs"] = 96 - 61 / 3600
    c.seal()
    old = job._completion

    def verify(plan):
        value = old(plan)
        c.now += 2
        return value

    monkeypatch.setattr(job, "_completion", verify)
    assert (
        job._execute_for_test(
            c.request,
            c.root,
            popen=c.popen,
            supervisor=c.supervisor,
            config_validator=c.validator,
            clock=c.clock,
            writer=c.writer,
        )
        == 1
    )
    assert c.journals[-1]["runs"][-1]["verified_completion"] is False


def test_actual_immutable_supervisor_terminates_only_captured_fake_owned_tree():
    path = MAIN / "tools/native_reference_supervisor.py"
    spec = importlib.util.spec_from_file_location("immutable_supervisor_fake_tree", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    terminated = []

    class Member:
        def __init__(self, pid, rss):
            self.pid, self.rss, self.alive = pid, rss, True

        def children(self, recursive):
            assert recursive is True
            return [members[78]]

        def memory_info(self):
            return SimpleNamespace(rss=self.rss, peak_wset=self.rss)

        def is_running(self):
            return self.alive

        def terminate(self):
            assert self.pid in (77, 78), "Foreign identity must remain untouched."
            terminated.append(self.pid)
            self.alive = False

        def kill(self):
            pytest.fail("Already terminated fake tree should not need kill.")

    members = {77: Member(77, 100), 78: Member(78, 100), 999: Member(999, 1)}
    module.psutil = SimpleNamespace(
        Process=lambda pid: members[pid],
        NoSuchProcess=LookupError,
        AccessDenied=PermissionError,
        wait_procs=lambda values, timeout: (values, []),
    )
    module.time = SimpleNamespace(
        monotonic=lambda: 30.0, sleep=lambda _: pytest.fail("Immediate RAM stop")
    )
    child = SimpleNamespace(pid=77, poll=lambda: None if members[77].alive else 7, wait=lambda: 7)
    result = module.supervise_owned(child, started=0, deadline_seconds=100, rss_limit_bytes=150)
    assert result["stopped_for"] == "PROCESS_RAM_LIMIT"
    assert sorted(terminated) == [77, 78] and members[999].alive
    assert result["owned_tree_cleanup_verified"] is True
    assert result["owned_process_pids"] == [77, 78]


def test_resume_failed_attempt_uses_original_trainer_review_and_charges_both(case):
    c = case
    c.exit_code, c.write_report = 7, False
    assert (
        job._execute_for_test(
            c.request,
            c.root,
            popen=c.popen,
            supervisor=c.supervisor,
            config_validator=c.validator,
            clock=c.clock,
            writer=c.writer,
        )
        == 7
    )
    original = c.root / "reviews/original-prefit.json"
    c.fs.files[original] = c.fs.files[c.request.review]
    c.fs.files[c.request.output / "latest.pt"] = b"SYNTHETIC_CORRECTNESS_ONLY unparsed resume bytes"
    c.ledger = copy.deepcopy(c.journals[-1])
    c.request = job.Job(
        **{
            **vars(c.request),
            "resume": c.request.output / "latest.pt",
            "receipt": c.root / "evidence/SYNTHETIC_CORRECTNESS_ONLY-resume-receipt",
        }
    )
    c.seal()
    c.review["trainer_review_path"] = str(original)
    c.review["runtime_arguments"]["trainer-review"] = str(original)
    c.fs.files[c.request.review] = support.encode(c.review)
    c.exit_code = 9
    code = job._execute_for_test(
        c.request,
        c.root,
        popen=c.popen,
        supervisor=c.supervisor,
        config_validator=c.validator,
        clock=c.clock,
        writer=c.writer,
    )
    assert code == 9
    assert c.calls[-1][c.calls[-1].index("--review") + 1] == str(original)
    assert c.journals[-1]["native_band_gpu_hours_spent_full_owned"] == pytest.approx(120 / 3600)
    assert c.journals[-1]["gpu_hours_spent_owned_scientific_jobs"] == pytest.approx(120 / 3600)
    assert len(c.journals[-1]["runs"]) == 2
    assert c.journals[-1]["runs"][-1]["trainer_review_sha256"] == job.digest(original)


def test_resume_without_exact_original_prefit_refused(case):
    c = case
    c.fs.dirs.add(c.request.output)
    c.fs.files[c.request.output / "latest.pt"] = b"SYNTHETIC_CORRECTNESS_ONLY no tensor decoding"
    c.request = job.Job(**{**vars(c.request), "resume": c.request.output / "latest.pt"})
    c.seal()
    with pytest.raises(ValueError, match="original trainer review"):
        invoke(c)
    assert c.fs.mutations == []


def test_changed_source_after_admission_denied_before_receipt_creation(case):
    c = case
    plan = invoke(c)
    c.fs.files[c.root / "tools/native_reference_supervisor.py"] += b"tamper"
    with pytest.raises(ValueError, match="changed"):
        job._execute(
            plan, popen=c.popen, supervisor=c.supervisor, clock=c.clock, write_ledger=c.writer
        )
    assert c.fs.mutations == [] and c.calls == []


def test_popen_oserror_charges_known_time_but_keeps_active_reconciliation(case):
    c = case

    def popen(*args, **kwargs):
        c.now += 3
        raise OSError("SYNTHETIC_CORRECTNESS_ONLY launch failure, no actual OS denial")

    with pytest.raises(OSError):
        job._execute_for_test(
            c.request,
            c.root,
            popen=popen,
            supervisor=c.supervisor,
            config_validator=c.validator,
            clock=c.clock,
            writer=c.writer,
        )
    record = c.journals[-1]["runs"][-1]
    assert record["status"] == "RUNNING_CUDA" and record["pid"] is None
    assert record["requires_reconciliation"]
    assert c.journals[-1]["native_band_gpu_hours_spent_full_owned"] == pytest.approx(3 / 3600)


def test_journal_write_failure_not_retried_and_no_fake_completion(case):
    c = case
    attempts = []

    def writer(root, ledger):
        attempts.append(1)
        if len(attempts) == 2:
            raise PermissionError("Synthetic explicit denial; not an OS operation")
        c.writer(root, ledger)

    with pytest.raises(PermissionError):
        job._execute_for_test(
            c.request,
            c.root,
            popen=c.popen,
            supervisor=c.supervisor,
            config_validator=c.validator,
            clock=c.clock,
            writer=writer,
        )
    assert len(attempts) == 2
    assert c.journals[-1]["runs"][-1]["status"] == "RUNNING_CUDA"
    assert not (c.request.receipt / "attempt.json").exists()


def test_unfinished_journal_preserved_before_mutation(case):
    c = case
    path = (c.root / "orchestration/native_ssl_run_ledger_v1.json").with_suffix(".band-pending")
    c.fs.files[path] = b"unknown partial journal"
    with pytest.raises(ValueError, match="unfinished"):
        invoke(c)
    assert c.fs.files[path] == b"unknown partial journal"
    assert c.fs.mutations == []


def test_pretrained_downstream_preserves_exact_ancestor_arguments(monkeypatch):
    c = operational_case(
        monkeypatch,
        kind="downstream",
        mode="frozen_readout",
        method="shared_ssl",
    )
    plan = invoke(c)
    assert plan.command[plan.command.index("--encoder") + 1] == str(c.request.encoder)
    assert plan.command[plan.command.index("--ancestor-review") + 1] == str(
        c.request.ancestor_review
    )
    assert plan.command[plan.command.index("--ancestor-config") + 1] == str(
        c.request.ancestor_config
    )
    assert c.fs.mutations == []


def test_declared_ancestor_changed_rejected_before_launch(monkeypatch):
    c = operational_case(
        monkeypatch,
        kind="downstream",
        mode="frozen_readout",
        method="shared_ssl",
    )
    c.fs.files[c.request.encoder] += b"changed"
    with pytest.raises(ValueError, match="Stale"):
        invoke(c)
    assert c.fs.mutations == []
