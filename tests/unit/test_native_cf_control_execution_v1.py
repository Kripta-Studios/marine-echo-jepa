"""Synthetic executor admission; no CUDA, corpora or fitted weights."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import execute_native_cf_control_job_v1 as executor


def test_fixed_catalog_has_exactly_six_parent_free_recipes():
    configs = executor.catalog()
    assert len(configs) == 6
    assert {c["seed"] for c in configs.values()} == {7, 13, 23}
    assert sum(c["updates"] for c in configs.values()) == 15000
    for c in configs.values():
        assert (c["width"], c["latent"], c["blocks"], c["history"]) == (256, 128, 5, 96)
        assert (c["updates"], c["cadence"]) == (
            (2000, 500) if c["method"] == "cf_random_frozen" else (3000, 750)
        )


def test_missing_prefit_stops_before_import_spawn_or_mutation(tmp_path, monkeypatch):
    root = tmp_path / "SYNTHETIC_CORRECTNESS_ONLY"
    root.mkdir()
    review = root / "review.json"
    review.write_text(json.dumps({"status": "PROPOSED_NOT_APPROVED"}))
    job = executor.Job(root / "config.json", review, root / "output", root / "receipt")
    monkeypatch.setattr(
        executor, "source_paths", lambda *a: pytest.fail("Rejected review reached sources")
    )
    with pytest.raises(ValueError, match="prefit"):
        executor.admit(job, root)
    assert not job.output.exists() and not job.receipt.exists()


@pytest.mark.parametrize("reviewer", ["author", "root", " ROOT ", "AUTHOR"])
def test_review_identity_cannot_be_impersonated_by_case_or_whitespace(reviewer):
    record = {
        "status": "APPROVED_CF_CONTROL_PREFIT",
        "evidence_kind": "REAL_TRAIN_DEVELOPMENT_FIT",
        "reviewer_session_id": reviewer,
        "implementer_session_id": "author",
        "root_coordinator_session_id": "root",
        "allowed_roles": ["train", "development"],
    }
    with pytest.raises(ValueError):
        executor.review_identity(record, implementer="author")


def test_final_site_role_never_enters_control_review():
    record = {
        "status": "APPROVED_CF_CONTROL_PREFIT",
        "evidence_kind": "REAL_TRAIN_DEVELOPMENT_FIT",
        "reviewer_session_id": "reviewer",
        "implementer_session_id": "author",
        "root_coordinator_session_id": "root",
        "allowed_roles": ["train", "final_test"],
    }
    with pytest.raises(ValueError, match="roles"):
        executor.review_identity(record, implementer="author")


def test_owned_reservation_precedes_spawn_and_failed_spawn_remains_unreconciled(
    tmp_path, monkeypatch
):
    root = tmp_path / "SYNTHETIC_CORRECTNESS_ONLY"
    (root / "orchestration").mkdir(parents=True)
    ledger_path = root / "orchestration/native_ssl_run_ledger_v1.json"
    ledger = {
        "gpu_limit_hours": 96,
        "gpu_hours_spent_owned_scientific_jobs": 17.802355808369175,
        "runs": [],
    }
    ledger_path.write_text(json.dumps(ledger))
    job = executor.Job(root / "config", root / "review", root / "output", root / "receipt")
    plan = executor.Plan(
        job,
        root,
        next(iter(executor.catalog().values())),
        {},
        {},
        ledger,
        executor.digest(ledger_path),
        executor.limits(ledger),
        {},
    )
    monkeypatch.setattr(executor, "clock", lambda: 100)
    # Model-importing tests share this process; isolate the stdlib parent fixture.
    monkeypatch.setattr(psutil.Process, "memory_info", lambda self: SimpleNamespace(rss=1))

    def denied_spawn(*a, **k):
        active = json.loads(ledger_path.read_text())["runs"][-1]
        assert active["status"] == "RUNNING_CUDA"
        assert active["requires_reconciliation"] is True
        raise OSError("SYNTHETIC launch failure")

    with pytest.raises(OSError, match="launch failure"):
        executor.execute_plan(plan, popen=denied_spawn)
    failed = json.loads(ledger_path.read_text())["runs"][-1]
    assert failed["requires_reconciliation"] is True
    assert "owned_tree_cleanup_verified" not in failed
    assert not job.output.exists()


def test_changed_ledger_prevents_receipt_creation(tmp_path):
    root = tmp_path / "SYNTHETIC_CORRECTNESS_ONLY"
    (root / "orchestration").mkdir(parents=True)
    p = root / "orchestration/native_ssl_run_ledger_v1.json"
    ledger = {"gpu_limit_hours": 96, "gpu_hours_spent_owned_scientific_jobs": 17.8, "runs": []}
    p.write_text(json.dumps(ledger))
    job = executor.Job(root / "config", root / "review", root / "output", root / "receipt")
    plan = executor.Plan(
        job,
        root,
        next(iter(executor.catalog().values())),
        {},
        {},
        ledger,
        executor.digest(p),
        executor.limits(ledger),
        {},
    )
    p.write_text(json.dumps({**ledger, "gpu_hours_spent_owned_scientific_jobs": 18}))
    with pytest.raises(ValueError, match="ownership"):
        executor.execute_plan(plan)
    assert not job.receipt.exists()


def admitted_fixture(tmp_path, monkeypatch):
    root = tmp_path / "SYNTHETIC_CORRECTNESS_ONLY"
    identifier, config = next(iter(executor.catalog().items()))
    files = [
        "orchestration/native_cf_control_budget_owner_resolution_v1.json",
        "orchestration/native_cf_control_owner_scope_v1.json",
        "orchestration/ssl_vnext_cf_matched_controls_builder_contract.txt",
        "docs/adr/0022-cf-backbone-matched-controls.md",
        "docs/adr/0025-cf-matched-controls-continuation.md",
        "docs/adr/0016-native-ssl-selection-and-source-baseline.md",
        "data/processed/native_ssl_v1/train.npz",
        "data/processed/native_ssl_v1/development.npz",
        "data/processed/native_ssl_v1/train.json",
        "data/processed/native_ssl_v1/development.json",
        "configs/native_ssl_split_v1.json",
        "configs/control.json",
        "tools/execute_native_cf_control_job_v1.py",
        ".venv/Scripts/python.exe",
        "pyproject.toml",
        "uv.lock",
    ]
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("SYNTHETIC OPAQUE METADATA ONLY")
    budget = root / files[0]
    budget.write_text(
        json.dumps(
            {
                "status": "ROOT_RESOLVED",
                "study": "CF_MATCHED_CONTROLS",
                "fits": 6,
                "cf_gpu_hours": 12,
                "aggregate_gpu_hours": 96,
                "evaluation_reserve_hours": 12,
                "seeds": [7, 13, 23],
            }
        )
    )
    cfg = root / "configs/control.json"
    cfg.write_text(json.dumps(config))
    job = executor.Job(
        cfg,
        root / "review.json",
        root / "outputs/native_cf_matched_controls_v1" / identifier,
        root / "evidence/ssl-research-v1" / (identifier + "-attempt-01"),
    )
    inputs = {
        "train": str(root / "data/processed/native_ssl_v1/train.npz"),
        "dev": str(root / "data/processed/native_ssl_v1/development.npz"),
        "train_cohort": str(root / "data/processed/native_ssl_v1/train.json"),
        "dev_cohort": str(root / "data/processed/native_ssl_v1/development.json"),
        "split": str(root / "configs/native_ssl_split_v1.json"),
        "adr0016": str(root / "docs/adr/0016-native-ssl-selection-and-source-baseline.md"),
        "protocol": str(root / "docs/adr/0025-cf-matched-controls-continuation.md"),
        "config": str(cfg),
        "review": str(job.review),
        "executor": str(root / "tools/execute_native_cf_control_job_v1.py"),
        "budget": str(budget),
    }
    review = {
        "status": "APPROVED_CF_CONTROL_PREFIT",
        "evidence_kind": "REAL_TRAIN_DEVELOPMENT_FIT",
        "reviewer_session_id": "SYNTHETIC_REVIEWER",
        "implementer_session_id": executor.IMPLEMENTER,
        "root_coordinator_session_id": "SYNTHETIC_ROOT",
        "allowed_roles": ["train", "development"],
        "allowed_methods": [config["method"]],
        "allowed_seeds": [config["seed"]],
        "approved_config": config,
        "runtime": {
            "output": str(job.output),
            "device": "cuda:0",
            "executor": inputs["executor"],
            "budget": str(budget),
        },
        "execution_runtime": {
            **inputs,
            "trainer_review": str(job.review),
            "output": str(job.output),
            "receipt": str(job.receipt),
            "resume": None,
            "device": "cuda:0",
        },
        "bindings": {str(root / p): executor.digest(root / p) for p in files},
    }
    job.review.write_text(json.dumps(review))
    (root / "orchestration/native_ssl_run_ledger_v1.json").write_text(
        json.dumps(
            {
                "gpu_limit_hours": 96,
                "gpu_hours_spent_owned_scientific_jobs": 17.802355808369175,
                "runs": [],
            }
        )
    )
    monkeypatch.setattr(executor, "source_paths", lambda *a: [])
    return job, root, review


def test_exact_synthetic_admission_reads_metadata_only_and_keeps_destinations_absent(
    tmp_path, monkeypatch
):
    job, root, review = admitted_fixture(tmp_path, monkeypatch)
    plan = executor.admit(job, root)
    assert plan.allowance.deadline_seconds == 43200
    assert plan.config == review["approved_config"]
    assert not job.output.exists() and not job.receipt.exists()


@pytest.mark.parametrize(
    "change", ["missing_binding", "stale_source", "final_npz", "runtime", "seed_bool", "journal"]
)
def test_admission_refuses_unsafe_mutations_without_creating_receipts(
    tmp_path, monkeypatch, change
):
    job, root, review = admitted_fixture(tmp_path, monkeypatch)
    if change == "missing_binding":
        review["bindings"].pop(str(job.config))
    elif change == "stale_source":
        (root / "tools/execute_native_cf_control_job_v1.py").write_text("CHANGED")
    elif change == "final_npz":
        final = root / "data/processed/native_ssl_v1/final.npz"
        final.write_text("SYNTHETIC OPAQUE FORBIDDEN FINAL")
        review["bindings"][str(final)] = executor.digest(final)
    elif change == "runtime":
        review["execution_runtime"]["device"] = "cpu"
    elif change == "seed_bool":
        cfg = {**review["approved_config"], "seed": True}
        job.config.write_text(json.dumps(cfg))
    elif change == "journal":
        (root / "orchestration/existing.band-pending").write_text("INCOMPLETE")
    job.review.write_text(json.dumps(review))
    with pytest.raises((ValueError, FileExistsError)):
        executor.admit(job, root)
    assert not job.output.exists() and not job.receipt.exists()


def test_failed_child_is_charged_and_preserves_original_budget_families(tmp_path, monkeypatch):
    job, root, _ = admitted_fixture(tmp_path, monkeypatch)
    ledger_path = root / "orchestration/native_ssl_run_ledger_v1.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["native_band_gpu_hours_spent_full_owned"] = 8.318836388888881
    ledger_path.write_text(json.dumps(ledger))
    plan = executor.admit(job, root)
    monkeypatch.setattr(executor, "clock", lambda: 100)
    monkeypatch.setattr(psutil.Process, "memory_info", lambda self: SimpleNamespace(rss=1))

    class Child:
        pid = 123456

    def supervise(*a, **k):
        return {
            "exit_code": 2,
            "stopped_for": None,
            "owned_tree_cleanup_verified": True,
            "elapsed_full_attempt_seconds": 60,
            "peak_process_rss_bytes": 10000,
        }

    assert executor.execute_plan(plan, popen=lambda *a, **k: Child(), supervisor=supervise) == 2
    closed = json.loads(ledger_path.read_text())
    assert (
        closed["native_band_gpu_hours_spent_full_owned"]
        == ledger["native_band_gpu_hours_spent_full_owned"]
    )
    assert (
        closed["gpu_hours_spent_owned_scientific_jobs"]
        == ledger["gpu_hours_spent_owned_scientific_jobs"] + 60 / 3600
    )
    assert closed["native_cf_control_hours_spent_full_owned"] == 60 / 3600
    assert closed["runs"][-1]["status"] == "FAILED_REAL_CF_CONTROL_ATTEMPT"
    assert executor.limits(closed).extension_hours == 60 / 3600
    assert not job.output.exists()


def test_parent_ram_guard_stops_before_spawn_and_preserves_unreconciled_attempt(
    tmp_path, monkeypatch
):
    job, root, _ = admitted_fixture(tmp_path, monkeypatch)
    plan = executor.admit(job, root)
    monkeypatch.setattr(
        psutil.Process, "memory_info", lambda self: SimpleNamespace(rss=executor.PARENT_RAM_RESERVE)
    )
    with pytest.raises(RuntimeError, match="reserved RAM"):
        executor.execute_plan(plan, popen=lambda *a, **k: pytest.fail("RAM guard reached Popen"))
    record = json.loads((root / "orchestration/native_ssl_run_ledger_v1.json").read_bytes())[
        "runs"
    ][-1]
    assert record["requires_reconciliation"] is True
    assert "owned_tree_cleanup_verified" not in record
    assert not job.output.exists()
