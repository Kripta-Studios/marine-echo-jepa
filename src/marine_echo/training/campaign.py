"""Finite pre-test campaign scheduling with durable fixture-only attempt history.

The scheduler invokes supplied executors. It cannot promote a real acoustic
benchmark until independent R0/R1 approval and real adapters exist elsewhere.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
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


def _verify_completed(ledger: dict[str, Any], slots: tuple[CampaignSlot, ...]) -> None:
    if set(ledger["runs"]) != {slot.run_id for slot in slots}:
        raise ValueError("Campaign ledger slot identity differs from the finite plan.")
    for slot in slots:
        run = ledger["runs"][slot.run_id]
        if run["status"] not in SUCCESS:
            continue
        artifact = Path(run["artifact"])
        if (
            artifact.is_symlink()
            or not artifact.is_file()
            or _sha_file(artifact) != run["artifact_sha256"]
        ):
            raise ValueError("Campaign completed artifact checksum or path differs.")
        if run["status"] == "REUSED_FIXTURE":
            source = ledger["runs"].get(run["reuse_of"])
            if (
                source is None
                or source["status"] != "COMPLETED_FIXTURE"
                or source["artifact"] != run["artifact"]
                or source["artifact_sha256"] != run["artifact_sha256"]
                or source["validation_metric"] != run["validation_metric"]
            ):
                raise ValueError(
                    "Campaign seed-7 reuse reference differs from its development run."
                )


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
    run["configuration"] = selected_configuration
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
            or not isinstance(result.validation_metric, (float, int))
            or not np.isfinite(result.validation_metric)
            or result.validation_metric < 0
            or type(result.updates) is not int
            or not 0 <= result.updates <= 3000
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
                "reusable_seed7": bool(result.reusable_seed7),
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
    if root.is_symlink() or root.is_junction() or trainer_lock.is_symlink():
        raise ValueError("Linked campaign root or trainer lock is not allowed.")
    trainer_lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(trainer_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise FileExistsError("One trainer already holds the campaign lock.") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\n")
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
            _verify_completed(ledger, slots)
        else:
            ledger = expected
            _atomic_json(ledger_path, ledger)
        for slot in slots:
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
        trainer_lock.unlink(missing_ok=True)
