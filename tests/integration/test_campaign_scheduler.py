"""Synthetic executor contract tests for the finite, pre-test campaign."""

import hashlib
import json
from pathlib import Path

import pytest

from marine_echo.training.campaign import CampaignIdentity, ExecutionResult, run_campaign


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "experiments.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "seeds": [7, 13, 23],
                "families": [
                    "persistence",
                    "seasonal",
                    "ridge",
                    "hist_gradient_boosting",
                    "direct",
                    "ema_jepa",
                    "shared_sigreg",
                ],
                "hybrid_candidates": ["ema_jepa_plus_raw", "shared_sigreg_plus_raw"],
                "controls": ["random_encoder", "shuffled_future"],
                "max_learned_configs_per_family": 2,
                "max_updates_per_phase": 3000,
                "max_tree_configs": 8,
            }
        ),
        encoding="utf-8",
    )
    return path


def _identity() -> CampaignIdentity:
    return CampaignIdentity(
        scope="synthetic-fixture-only",
        protocol_sha256=_sha("protocol"),
        source_inventory_sha256=_sha("inventory"),
        split_sha256=_sha("split"),
        calibration_sha256=_sha("calibration"),
        processing_sha256=_sha("processing"),
    )


def _executors(calls: list[tuple[str, tuple[str, ...]]]):
    def execute(slot, context):
        assert set(context.dependency_artifacts) == set(context.completed_dependencies)
        assert set(context.dependency_sha256) == set(context.completed_dependencies)
        for dependency in context.completed_dependencies:
            artifact_path = context.dependency_artifacts[dependency]
            assert (
                hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                == (context.dependency_sha256[dependency])
            )
        calls.append((slot.run_id, context.completed_dependencies))
        artifact = context.attempt_dir / "result.json"
        artifact.write_text(json.dumps({"fixture": slot.run_id}), encoding="utf-8")
        metric = 2.0 - (slot.configuration or 0) * 0.2 + slot.seed / 1000
        return ExecutionResult(
            artifact=artifact,
            validation_metric=metric,
            updates=10,
            reusable_seed7=slot.phase == "development" and slot.configuration == 1,
            scope="synthetic-fixture-only",
        )

    return {
        family: execute
        for family in (
            "persistence",
            "seasonal",
            "ridge",
            "hist_gradient_boosting",
            "direct",
            "ema_jepa",
            "shared_sigreg",
            "ema_jepa_plus_raw",
            "shared_sigreg_plus_raw",
        )
    }


def test_25_slot_campaign_order_selection_reuse_and_verified_resume(tmp_path: Path) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    config = _config(tmp_path)
    kwargs = {
        "root": tmp_path / "campaign",
        "trainer_lock": tmp_path / "single-trainer.lock",
        "config": config,
        "identity": _identity(),
        "executors": _executors(calls),
    }
    first = run_campaign(**kwargs)
    assert len(first["runs"]) == 25
    assert first["fixture_completed_slots"] == 25
    assert first["completed_benchmark_runs"] == 0
    assert first["test_opened"] is False
    assert first["selected_configs"] == {"direct": 1, "ema_jepa": 1, "shared_sigreg": 1}
    assert len(calls) == 22
    for family in ("direct", "ema_jepa", "shared_sigreg"):
        reused = first["runs"][f"{family}-seed7"]
        assert reused["status"] == "REUSED_FIXTURE"
        assert reused["reuse_of"] == f"{family}-development1-seed7"
    seen = set()
    for run_id, dependencies in calls:
        assert set(dependencies).issubset(
            seen | {f"{family}-seed7" for family in ("direct", "ema_jepa", "shared_sigreg")}
        )
        seen.add(run_id)
    second = run_campaign(**kwargs)
    assert len(calls) == 22
    assert second["runs"] == first["runs"]
    assert not (tmp_path / "single-trainer.lock").exists()


def test_missing_neural_executor_preserves_baselines_without_fake_completion(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    executors = _executors(calls)
    for family in ("direct", "ema_jepa", "shared_sigreg"):
        del executors[family]
    result = run_campaign(
        root=tmp_path / "campaign",
        trainer_lock=tmp_path / "trainer.lock",
        config=_config(tmp_path),
        identity=_identity(),
        executors=executors,
    )
    assert result["fixture_completed_slots"] == 4
    assert result["completed_benchmark_runs"] == 0
    assert result["runs"]["direct-development0-seed7"]["status"] == "BLOCKED_EXECUTOR"
    assert result["runs"]["direct-seed13"]["status"] == "WAITING_DEPENDENCY"
    assert len(calls) == 4


def test_failure_attempt_history_and_exact_resume(tmp_path: Path) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    executors = _executors(calls)
    original = executors["direct"]
    failed = False

    def flaky(slot, context):
        nonlocal failed
        if slot.run_id == "direct-seed13" and not failed:
            failed = True
            raise RuntimeError("fixture failure")
        return original(slot, context)

    executors["direct"] = flaky
    kwargs = {
        "root": tmp_path / "campaign",
        "trainer_lock": tmp_path / "trainer.lock",
        "config": _config(tmp_path),
        "identity": _identity(),
        "executors": executors,
    }
    first = run_campaign(**kwargs)
    assert first["runs"]["direct-seed13"]["status"] == "FAILED_FIXTURE"
    assert first["runs"]["direct-seed13"]["attempts"][0]["status"] == "FAILED_FIXTURE"
    second = run_campaign(**kwargs)
    assert second["runs"]["direct-seed13"]["status"] == "COMPLETED_FIXTURE"
    assert len(second["runs"]["direct-seed13"]["attempts"]) == 2
    assert second["fixture_completed_slots"] == 25


def test_tamper_lock_config_and_real_scope_fail_closed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    calls: list[tuple[str, tuple[str, ...]]] = []
    kwargs = {
        "root": tmp_path / "campaign",
        "trainer_lock": tmp_path / "trainer.lock",
        "config": config,
        "identity": _identity(),
        "executors": _executors(calls),
    }
    (tmp_path / "trainer.lock").write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError, match="trainer"):
        run_campaign(**kwargs)
    (tmp_path / "trainer.lock").unlink()
    result = run_campaign(**kwargs)
    artifact = Path(result["runs"]["persistence-seed7"]["artifact"])
    artifact.write_bytes(artifact.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        run_campaign(**kwargs)
    config.write_text(config.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="configuration"):
        run_campaign(**kwargs)
    with pytest.raises(ValueError, match="R0/R1"):
        run_campaign(
            **{
                **kwargs,
                "identity": CampaignIdentity(
                    scope="real",
                    protocol_sha256=_sha("protocol"),
                    source_inventory_sha256=_sha("inventory"),
                    split_sha256=_sha("split"),
                    calibration_sha256=_sha("calibration"),
                    processing_sha256=_sha("processing"),
                ),
            }
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "selection",
        "dependencies",
        "family",
        "attempt_metric",
        "reuse_flag",
        "selected_configuration",
        "completed_to_pending",
    ],
)
def test_resume_rejects_tampered_ledger_fields(tmp_path: Path, mutation: str) -> None:
    kwargs = {
        "root": tmp_path / "campaign",
        "trainer_lock": tmp_path / "trainer.lock",
        "config": _config(tmp_path),
        "identity": _identity(),
        "executors": _executors([]),
    }
    run_campaign(**kwargs)
    path = tmp_path / "campaign/campaign.json"
    ledger = json.loads(path.read_text(encoding="utf-8"))
    run = ledger["runs"]["direct-development1-seed7"]
    if mutation == "selection":
        ledger["selected_configs"]["direct"] = 0
    elif mutation == "dependencies":
        ledger["runs"]["direct-seed13"]["dependencies"] = []
    elif mutation == "family":
        run["family"] = "ema_jepa"
    elif mutation == "attempt_metric":
        run["attempts"][-1]["validation_metric"] = 999.0
    elif mutation == "reuse_flag":
        run["attempts"][-1]["reusable_seed7"] = False
    elif mutation == "selected_configuration":
        ledger["runs"]["direct-seed13"]["configuration"] = 0
    else:
        run["status"] = "PENDING"
    path.write_text(json.dumps(ledger), encoding="utf-8")
    with pytest.raises(ValueError, match="ledger|attempt|selection|reuse"):
        run_campaign(**kwargs)


def test_replaced_trainer_lock_is_preserved(tmp_path: Path) -> None:
    lock = tmp_path / "trainer.lock"
    executors = _executors([])
    original = executors["persistence"]

    def replace_lock(slot, context):
        lock.unlink()
        lock.write_text("replacement", encoding="utf-8")
        return original(slot, context)

    executors["persistence"] = replace_lock
    with pytest.raises(ValueError, match="lock ownership"):
        run_campaign(
            root=tmp_path / "campaign",
            trainer_lock=lock,
            config=_config(tmp_path),
            identity=_identity(),
            executors=executors,
        )
    assert lock.read_text(encoding="utf-8") == "replacement"


@pytest.mark.parametrize("field,value", [("validation_metric", True), ("reusable_seed7", 1)])
def test_executor_rejects_non_strict_result_types(
    tmp_path: Path, field: str, value: object
) -> None:
    from dataclasses import replace

    executors = _executors([])
    original = executors["persistence"]

    def invalid(slot, context):
        return replace(original(slot, context), **{field: value})

    executors["persistence"] = invalid
    ledger = run_campaign(
        root=tmp_path / "campaign",
        trainer_lock=tmp_path / "trainer.lock",
        config=_config(tmp_path),
        identity=_identity(),
        executors=executors,
    )
    assert ledger["runs"]["persistence-seed7"]["status"] == "FAILED_FIXTURE"


def test_partial_ledger_cannot_hide_completed_attempt_as_waiting(tmp_path: Path) -> None:
    executors = _executors([])
    del executors["direct"]
    kwargs = {
        "root": tmp_path / "campaign",
        "trainer_lock": tmp_path / "trainer.lock",
        "config": _config(tmp_path),
        "identity": _identity(),
        "executors": executors,
    }
    ledger = run_campaign(**kwargs)
    assert ledger["status"] == "PARTIAL_FIXTURE"
    path = tmp_path / "campaign/campaign.json"
    ledger["runs"]["persistence-seed7"].update(
        status="WAITING_DEPENDENCY",
        artifact=None,
        artifact_sha256=None,
        validation_metric=None,
        reuse_of=None,
    )
    path.write_text(json.dumps(ledger), encoding="utf-8")
    with pytest.raises(ValueError, match="attempt|waiting|ledger"):
        run_campaign(**kwargs)


@pytest.mark.parametrize(
    "mutation",
    ["selection_bool", "development_bool", "hybrid_bool", "benchmark_bool", "test_opened_int"],
)
def test_resume_rejects_boolean_numeric_tamper(tmp_path: Path, mutation: str) -> None:
    kwargs = {
        "root": tmp_path / "campaign",
        "trainer_lock": tmp_path / "trainer.lock",
        "config": _config(tmp_path),
        "identity": _identity(),
        "executors": _executors([]),
    }
    run_campaign(**kwargs)
    path = tmp_path / "campaign/campaign.json"
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "selection_bool":
        ledger["selected_configs"]["direct"] = True
    elif mutation == "development_bool":
        ledger["runs"]["direct-development1-seed7"]["configuration"] = True
    elif mutation == "hybrid_bool":
        ledger["runs"]["ema_jepa_plus_raw-seed7"]["configuration"] = True
    elif mutation == "benchmark_bool":
        ledger["completed_benchmark_runs"] = False
    else:
        ledger["test_opened"] = 0
    path.write_text(json.dumps(ledger), encoding="utf-8")
    with pytest.raises(ValueError, match="ledger|configuration|selection|provenance"):
        run_campaign(**kwargs)
