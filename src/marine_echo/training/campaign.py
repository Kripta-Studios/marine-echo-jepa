"""Finite pre-test campaign scheduling with durable fixture-only attempt history.

The scheduler invokes supplied executors. It cannot promote a real acoustic
benchmark until independent R0/R1 approval and real adapters exist elsewhere.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from marine_echo.data.preprocessing import valid_sha256

BASELINES = ("persistence", "seasonal", "ridge", "hist_gradient_boosting")
LEARNED = ("direct", "ema_jepa", "shared_sigreg")
HYBRIDS = ("ema_jepa_plus_raw", "shared_sigreg_plus_raw")
CONTROLS = ("random_encoder", "shuffled_future")
SEEDS = (7, 13, 23)
SUCCESS = frozenset(("COMPLETED_FIXTURE", "REUSED_FIXTURE"))
MAX_ATTEMPTS = 2


def _sha_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json_bytes(document: Any) -> bytes:
    return (
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".stage.", dir=path.parent)
    staging = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(document))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


@dataclass(frozen=True)
class CampaignIdentity:
    scope: str
    protocol_sha256: str
    source_inventory_sha256: str
    split_sha256: str
    calibration_sha256: str
    processing_sha256: str

    def validate(self) -> None:
        if self.scope != "synthetic-fixture-only":
            raise ValueError("Real campaign requires independent R0/R1 promotion and executors.")
        if not all(valid_sha256(value) for key, value in asdict(self).items() if key != "scope"):
            raise ValueError("Campaign identity requires exact SHA-256 provenance.")


@dataclass(frozen=True)
class CampaignSlot:
    run_id: str
    family: str
    phase: str
    seed: int
    configuration: int | None
    dependencies: tuple[str, ...]
    partition: str = "train_validation_only"


@dataclass(frozen=True)
class CampaignContext:
    attempt_dir: Path
    completed_dependencies: tuple[str, ...]
    dependency_artifacts: dict[str, Path]
    dependency_sha256: dict[str, str]
    selected_configuration: int | None
    scope: str = "synthetic-fixture-only"


@dataclass(frozen=True)
class ExecutionResult:
    artifact: Path
    validation_metric: float
    updates: int
    reusable_seed7: bool
    scope: str


Executor = Callable[[CampaignSlot, CampaignContext], ExecutionResult]


def _validate_config(config: Path) -> str:
    document = json.loads(config.read_text(encoding="utf-8"))
    required = {
        "schema_version": "1.0",
        "seeds": list(SEEDS),
        "families": [*BASELINES, *LEARNED],
        "hybrid_candidates": list(HYBRIDS),
        "controls": list(CONTROLS),
        "max_learned_configs_per_family": 2,
        "max_updates_per_phase": 3000,
        "max_tree_configs": 8,
    }
    if any(document.get(key) != value for key, value in required.items()):
        raise ValueError("Campaign configuration differs from the frozen 25-slot budget.")
    return _sha_file(config)


def _plan() -> tuple[CampaignSlot, ...]:
    slots: list[CampaignSlot] = []
    for family in BASELINES:
        slots.append(CampaignSlot(f"{family}-seed7", family, "baseline", 7, None, ()))
    for family in LEARNED:
        for configuration in (0, 1):
            slots.append(
                CampaignSlot(
                    f"{family}-development{configuration}-seed7",
                    family,
                    "development",
                    7,
                    configuration,
                    (),
                )
            )
    for family in LEARNED:
        dependencies = tuple(f"{family}-development{c}-seed7" for c in (0, 1))
        for seed in SEEDS:
            slots.append(
                CampaignSlot(
                    f"{family}-seed{seed}",
                    family,
                    "selected",
                    seed,
                    None,
                    dependencies,
                )
            )
    for family in HYBRIDS:
        parent = family.removesuffix("_plus_raw")
        dependencies = tuple(f"{parent}-seed{seed}" for seed in SEEDS)
        slots.append(CampaignSlot(f"{family}-seed7", family, "hybrid", 7, None, dependencies))
    for family in LEARNED[1:]:
        dependencies = tuple(f"{family}-seed{seed}" for seed in SEEDS)
        for control in CONTROLS:
            slots.append(
                CampaignSlot(
                    f"{family}-{control}-seed7",
                    family,
                    control,
                    7,
                    None,
                    dependencies,
                )
            )
    if len(slots) != 25 or len({slot.run_id for slot in slots}) != 25:
        raise AssertionError("The bounded campaign must have exactly 25 distinct slots.")
    return tuple(slots)


def _new_ledger(
    identity: CampaignIdentity, config_sha: str, slots: tuple[CampaignSlot, ...]
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "scope": identity.scope,
        "identity": asdict(identity),
        "configuration_sha256": config_sha,
        "scheduler_sha256": _sha_file(Path(__file__)),
        "plan_sha256": hashlib.sha256(_json_bytes([asdict(slot) for slot in slots])).hexdigest(),
        "runs": {
            slot.run_id: {
                "status": "PENDING",
                "family": slot.family,
                "phase": slot.phase,
                "seed": slot.seed,
                "configuration": slot.configuration,
                "dependencies": list(slot.dependencies),
                "attempts": [],
                "artifact": None,
                "artifact_sha256": None,
                "validation_metric": None,
                "reuse_of": None,
            }
            for slot in slots
        },
        "selected_configs": {},
        "fixture_completed_slots": 0,
        "completed_benchmark_runs": 0,
        "test_opened": False,
        "status": "PARTIAL_FIXTURE",
    }


def _valid_metric(value: Any) -> bool:
    return type(value) in (float, int) and bool(np.isfinite(value)) and value >= 0


def _owns_lock(path: Path, content: str) -> bool:
    return bool(
        path.exists()
        and not path.is_symlink()
        and not path.is_junction()
        and stat.S_ISREG(path.stat().st_mode)
        and path.read_text(encoding="utf-8") == content
    )


def _verify_completed(ledger: dict[str, Any], slots: tuple[CampaignSlot, ...], root: Path) -> None:
    expected_top = {
        "schema_version",
        "scope",
        "identity",
        "configuration_sha256",
        "scheduler_sha256",
        "plan_sha256",
        "runs",
        "selected_configs",
        "fixture_completed_slots",
        "completed_benchmark_runs",
        "test_opened",
        "status",
    }
    if set(ledger) != expected_top or set(ledger["runs"]) != {slot.run_id for slot in slots}:
        raise ValueError("Campaign ledger slot identity differs from the finite plan.")
    if (
        type(ledger["fixture_completed_slots"]) is not int
        or not 0 <= ledger["fixture_completed_slots"] <= 25
        or type(ledger["completed_benchmark_runs"]) is not int
        or ledger["completed_benchmark_runs"] != 0
        or type(ledger["test_opened"]) is not bool
        or ledger["test_opened"] is not False
        or ledger["status"] not in ("PARTIAL_FIXTURE", "COMPLETE_FIXTURE")
        or not isinstance(ledger["selected_configs"], dict)
        or not set(ledger["selected_configs"]).issubset(LEARNED)
        or any(
            type(choice) is not int or choice not in (0, 1)
            for choice in ledger["selected_configs"].values()
        )
    ):
        raise ValueError("Campaign ledger counters or selection schema are invalid.")
    run_keys = {
        "status",
        "family",
        "phase",
        "seed",
        "configuration",
        "dependencies",
        "attempts",
        "artifact",
        "artifact_sha256",
        "validation_metric",
        "reuse_of",
    }
    for slot in slots:
        run = ledger["runs"][slot.run_id]
        if (
            set(run) != run_keys
            or run["family"] != slot.family
            or run["phase"] != slot.phase
            or type(run["seed"]) is not int
            or run["seed"] != slot.seed
            or run["dependencies"] != list(slot.dependencies)
            or not isinstance(run["attempts"], list)
            or len(run["attempts"]) > MAX_ATTEMPTS
        ):
            raise ValueError("Campaign ledger fixed run fields differ from the plan.")
        if slot.phase == "baseline" and run["configuration"] is not None:
            raise ValueError("Campaign ledger fixed configuration differs from the plan.")
        if slot.phase == "development" and (
            type(run["configuration"]) is not int or run["configuration"] != slot.configuration
        ):
            raise ValueError("Campaign ledger fixed configuration differs from the plan.")
        if slot.phase == "hybrid" and run["configuration"] is not None:
            raise ValueError("Campaign ledger fixed configuration differs from the plan.")
        attempts = run["attempts"]
        for index, attempt in enumerate(attempts, 1):
            status = attempt.get("status")
            expected_keys = {
                "RUNNING_FIXTURE": {"number", "status"},
                "INTERRUPTED_FIXTURE": {"number", "status"},
                "FAILED_FIXTURE": {"number", "status", "reason"},
                "COMPLETED_FIXTURE": {
                    "number",
                    "status",
                    "artifact",
                    "artifact_sha256",
                    "validation_metric",
                    "updates",
                    "reusable_seed7",
                },
            }.get(status)
            if expected_keys is None or set(attempt) != expected_keys or attempt["number"] != index:
                raise ValueError("Campaign attempt schema differs from recorded execution.")
            if status == "RUNNING_FIXTURE" and index != len(attempts):
                raise ValueError("Campaign attempt history has an unfinished earlier attempt.")
            if status == "COMPLETED_FIXTURE" and index != len(attempts):
                raise ValueError("Campaign attempt history has an earlier completed attempt.")
            if status == "COMPLETED_FIXTURE":
                artifact = Path(attempt["artifact"])
                if (
                    not _valid_metric(attempt["validation_metric"])
                    or type(attempt["updates"]) is not int
                    or not 0 <= attempt["updates"] <= 3000
                    or type(attempt["reusable_seed7"]) is not bool
                    or not valid_sha256(attempt["artifact_sha256"])
                    or artifact.is_symlink()
                    or not artifact.is_file()
                    or not artifact.resolve().is_relative_to(
                        (root / "artifacts" / slot.run_id / f"attempt-{index}").resolve()
                    )
                    or _sha_file(artifact) != attempt["artifact_sha256"]
                ):
                    raise ValueError(
                        "Campaign attempt artifact checksum or metric differs on resume."
                    )
            if status == "FAILED_FIXTURE" and not isinstance(attempt["reason"], str):
                raise ValueError("Campaign attempt failure reason is invalid.")
        status = run["status"]
        if status not in {
            "PENDING",
            "WAITING_DEPENDENCY",
            "BLOCKED_EXECUTOR",
            "RUNNING_FIXTURE",
            "FAILED_FIXTURE",
            "BLOCKED_ATTEMPT_LIMIT",
            *SUCCESS,
        }:
            raise ValueError("Campaign ledger run status is invalid.")
        if status == "PENDING" and attempts:
            raise ValueError("Campaign ledger pending run has completed attempts.")
        if status == "WAITING_DEPENDENCY" and attempts:
            raise ValueError("Campaign ledger waiting run cannot retain attempts.")
        if status == "BLOCKED_EXECUTOR" and any(
            attempt["status"] not in ("FAILED_FIXTURE", "INTERRUPTED_FIXTURE")
            for attempt in attempts
        ):
            raise ValueError("Campaign ledger executor block has incompatible attempts.")
        if (
            any(attempt["status"] == "RUNNING_FIXTURE" for attempt in attempts)
            and status != "RUNNING_FIXTURE"
        ):
            raise ValueError("Campaign running attempt differs from run status.")
        if (
            any(attempt["status"] == "COMPLETED_FIXTURE" for attempt in attempts)
            and status != "COMPLETED_FIXTURE"
        ):
            raise ValueError("Campaign completed attempt differs from run status.")
        if status == "RUNNING_FIXTURE" and (
            not attempts or attempts[-1]["status"] != "RUNNING_FIXTURE"
        ):
            raise ValueError("Campaign ledger running status differs from its final attempt.")
        if status in ("FAILED_FIXTURE", "BLOCKED_ATTEMPT_LIMIT") and (
            not attempts or attempts[-1]["status"] not in ("FAILED_FIXTURE", "INTERRUPTED_FIXTURE")
        ):
            raise ValueError("Campaign ledger failure status differs from its final attempt.")
        if status == "BLOCKED_ATTEMPT_LIMIT" and len(attempts) != MAX_ATTEMPTS:
            raise ValueError("Campaign ledger attempt limit differs from the budget.")
        if status not in SUCCESS and any(
            run[key] is not None
            for key in ("artifact", "artifact_sha256", "validation_metric", "reuse_of")
        ):
            raise ValueError("Campaign incomplete run retains completed artifact fields.")
        if run["status"] not in SUCCESS:
            continue
        if not _valid_metric(run["validation_metric"]) or not valid_sha256(run["artifact_sha256"]):
            raise ValueError("Campaign completed artifact metric or checksum is invalid.")
        artifact = Path(run["artifact"])
        if (
            artifact.is_symlink()
            or not artifact.is_file()
            or _sha_file(artifact) != run["artifact_sha256"]
        ):
            raise ValueError("Campaign completed artifact checksum or path differs.")
        if status == "COMPLETED_FIXTURE" and (
            not attempts
            or attempts[-1]["status"] != "COMPLETED_FIXTURE"
            or run["reuse_of"] is not None
            or any(
                run[key] != attempts[-1][key]
                for key in ("artifact", "artifact_sha256", "validation_metric")
            )
        ):
            raise ValueError("Campaign completed run differs from its final attempt.")
        if run["status"] == "REUSED_FIXTURE":
            if attempts or slot.phase != "selected" or slot.seed != 7:
                raise ValueError("Campaign reuse status is invalid for this run.")
            source = ledger["runs"].get(run["reuse_of"])
            if (
                source is None
                or run["reuse_of"] != f"{slot.family}-development{run['configuration']}-seed7"
                or source["status"] != "COMPLETED_FIXTURE"
                or source["attempts"][-1]["reusable_seed7"] is not True
                or source["artifact"] != run["artifact"]
                or source["artifact_sha256"] != run["artifact_sha256"]
                or source["validation_metric"] != run["validation_metric"]
            ):
                raise ValueError(
                    "Campaign seed-7 reuse reference differs from its development run."
                )
    calculated = {}
    for family in LEARNED:
        development = [ledger["runs"][f"{family}-development{c}-seed7"] for c in (0, 1)]
        if all(run["status"] in SUCCESS for run in development):
            calculated[family] = min((0, 1), key=lambda c: (development[c]["validation_metric"], c))
    if any(
        calculated.get(family) != choice for family, choice in ledger["selected_configs"].items()
    ):
        raise ValueError("Campaign saved validation selection differs from completed metrics.")
    for slot in slots:
        if slot.family not in LEARNED or slot.phase == "development":
            continue
        run = ledger["runs"][slot.run_id]
        if run["status"] in SUCCESS or run["attempts"]:
            if (
                slot.family not in ledger["selected_configs"]
                or run["configuration"] != calculated.get(slot.family)
                or type(run["configuration"]) is not int
            ):
                raise ValueError("Campaign ledger selected configuration differs from validation.")
        elif run["configuration"] is not None:
            raise ValueError("Campaign ledger unattempted configuration is invalid.")
    if ledger["status"] == "COMPLETE_FIXTURE" and (
        set(ledger["selected_configs"]) != set(calculated)
        or sum(run["status"] in SUCCESS for run in ledger["runs"].values()) != 25
        or ledger["fixture_completed_slots"] != 25
    ):
        raise ValueError("Campaign complete ledger counts or selection differ from plan.")


def _selection(ledger: dict[str, Any], family: str) -> int | None:
    development = [ledger["runs"][f"{family}-development{c}-seed7"] for c in (0, 1)]
    if not all(run["status"] in SUCCESS for run in development):
        return None
    selected = min((0, 1), key=lambda c: (development[c]["validation_metric"], c))
    previous = ledger["selected_configs"].get(family)
    if previous is not None and previous != selected:
        raise ValueError("Saved validation configuration selection differs on resume.")
    ledger["selected_configs"][family] = selected
    return selected


def _reuse_seed7(ledger: dict[str, Any], slot: CampaignSlot, configuration: int) -> bool:
    if slot.phase != "selected" or slot.seed != 7:
        return False
    source_id = f"{slot.family}-development{configuration}-seed7"
    source = ledger["runs"][source_id]
    if not source["attempts"][-1].get("reusable_seed7"):
        return False
    run = ledger["runs"][slot.run_id]
    run.update(
        {
            "status": "REUSED_FIXTURE",
            "configuration": configuration,
            "artifact": source["artifact"],
            "artifact_sha256": source["artifact_sha256"],
            "validation_metric": source["validation_metric"],
            "reuse_of": source_id,
        }
    )
    return True


def _execute(
    ledger: dict[str, Any],
    slot: CampaignSlot,
    executor: Executor,
    root: Path,
    ledger_path: Path,
    selected_configuration: int | None,
) -> None:
    run = ledger["runs"][slot.run_id]
    attempt_number = len(run["attempts"]) + 1
    attempt_dir = root / "artifacts" / slot.run_id / f"attempt-{attempt_number}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    attempt: dict[str, Any] = {"number": attempt_number, "status": "RUNNING_FIXTURE"}
    run["attempts"].append(attempt)
    run["status"] = "RUNNING_FIXTURE"
    run["configuration"] = (
        slot.configuration if slot.phase == "development" else selected_configuration
    )
    _atomic_json(ledger_path, ledger)
    try:
        result = executor(
            slot,
            CampaignContext(
                attempt_dir=attempt_dir,
                completed_dependencies=slot.dependencies,
                dependency_artifacts={
                    dependency: Path(ledger["runs"][dependency]["artifact"])
                    for dependency in slot.dependencies
                },
                dependency_sha256={
                    dependency: ledger["runs"][dependency]["artifact_sha256"]
                    for dependency in slot.dependencies
                },
                selected_configuration=selected_configuration,
            ),
        )
        artifact = result.artifact
        if (
            result.scope != "synthetic-fixture-only"
            or not _valid_metric(result.validation_metric)
            or type(result.updates) is not int
            or not 0 <= result.updates <= 3000
            or type(result.reusable_seed7) is not bool
            or artifact.is_symlink()
            or not artifact.is_file()
            or not artifact.resolve().is_relative_to(attempt_dir.resolve())
        ):
            raise ValueError("Fixture executor result violates the bounded artifact contract.")
        digest = _sha_file(artifact)
        attempt.update(
            {
                "status": "COMPLETED_FIXTURE",
                "artifact": str(artifact),
                "artifact_sha256": digest,
                "validation_metric": float(result.validation_metric),
                "updates": result.updates,
                "reusable_seed7": result.reusable_seed7,
            }
        )
        run.update(
            {
                "status": "COMPLETED_FIXTURE",
                "artifact": str(artifact),
                "artifact_sha256": digest,
                "validation_metric": float(result.validation_metric),
            }
        )
    except Exception as error:  # noqa: BLE001 - record executor failures without losing attempts
        attempt.update({"status": "FAILED_FIXTURE", "reason": f"{type(error).__name__}: {error}"})
        run["status"] = "FAILED_FIXTURE"
    _atomic_json(ledger_path, ledger)


def run_campaign(
    *,
    root: Path,
    trainer_lock: Path,
    config: Path,
    identity: CampaignIdentity,
    executors: Mapping[str, Executor],
) -> dict[str, Any]:
    """Run each eligible fixture slot once per invocation; never access final test."""
    identity.validate()
    config_sha = _validate_config(config)
    slots = _plan()
    if (
        root.is_symlink()
        or root.is_junction()
        or trainer_lock.is_symlink()
        or trainer_lock.is_junction()
    ):
        raise ValueError("Linked campaign root or trainer lock is not allowed.")
    trainer_lock.parent.mkdir(parents=True, exist_ok=True)
    lock_token = uuid.uuid4().hex
    lock_content = f"pid={os.getpid()}\ntoken={lock_token}\n"
    try:
        descriptor = os.open(trainer_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise FileExistsError("One trainer already holds the campaign lock.") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(lock_content)
            stream.flush()
            os.fsync(stream.fileno())
        ledger_path = root / "campaign.json"
        if root.exists() and not ledger_path.is_file() and any(root.iterdir()):
            raise FileExistsError("Unregistered campaign output exists and cannot be overwritten.")
        root.mkdir(parents=True, exist_ok=True)
        expected = _new_ledger(identity, config_sha, slots)
        if ledger_path.exists():
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            for key in (
                "schema_version",
                "scope",
                "identity",
                "configuration_sha256",
                "scheduler_sha256",
                "plan_sha256",
                "test_opened",
            ):
                if ledger.get(key) != expected[key]:
                    raise ValueError("Campaign configuration or provenance differs on resume.")
            _verify_completed(ledger, slots, root)
        else:
            ledger = expected
            _atomic_json(ledger_path, ledger)
        for slot in slots:
            if not _owns_lock(trainer_lock, lock_content):
                raise ValueError("Campaign trainer lock ownership was lost; replacement preserved.")
            run = ledger["runs"][slot.run_id]
            if run["status"] in SUCCESS:
                continue
            if run["status"] == "RUNNING_FIXTURE":
                run["attempts"][-1]["status"] = "INTERRUPTED_FIXTURE"
                run["status"] = "FAILED_FIXTURE"
                _atomic_json(ledger_path, ledger)
            if len(run["attempts"]) >= MAX_ATTEMPTS:
                run["status"] = "BLOCKED_ATTEMPT_LIMIT"
                continue
            if any(
                ledger["runs"][dependency]["status"] not in SUCCESS
                for dependency in slot.dependencies
            ):
                run["status"] = "WAITING_DEPENDENCY"
                continue
            selected_configuration = (
                _selection(ledger, slot.family) if slot.family in LEARNED else None
            )
            if slot.phase == "selected" and selected_configuration is None:
                run["status"] = "WAITING_DEPENDENCY"
                continue
            if selected_configuration is not None and _reuse_seed7(
                ledger, slot, selected_configuration
            ):
                _atomic_json(ledger_path, ledger)
                continue
            executor = executors.get(slot.family)
            if executor is None:
                run["status"] = "BLOCKED_EXECUTOR"
                continue
            _execute(ledger, slot, executor, root, ledger_path, selected_configuration)
        ledger["fixture_completed_slots"] = sum(
            run["status"] in SUCCESS for run in ledger["runs"].values()
        )
        ledger["completed_benchmark_runs"] = 0
        ledger["status"] = (
            "COMPLETE_FIXTURE" if ledger["fixture_completed_slots"] == 25 else "PARTIAL_FIXTURE"
        )
        _atomic_json(ledger_path, ledger)
        return ledger
    finally:
        if not _owns_lock(trainer_lock, lock_content):
            raise ValueError("Campaign trainer lock ownership was lost; replacement preserved.")
        trainer_lock.unlink()
