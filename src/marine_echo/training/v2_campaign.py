"""Finite 25-slot v2 scheduler with saved-row validation selection and review gate.

The scheduler owns slot order and accounting. A supplied executor owns each model
fit and must write validation predictions before this scheduler can score a slot.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from marine_echo.training.v2_executor import (
    V2_PROTOCOL_SHA256,
    _code_digest,
    _rows_digest,
    _sha256,
    _valid_hash,
    _verify_predictions,
)
from marine_echo.training.v2_stream import HourlyWindow

BASELINES = ("persistence", "seasonal", "ridge", "hist_gradient_boosting")
LEARNED = ("direct", "ema_jepa", "shared_sigreg")
HYBRIDS = ("ema_jepa_plus_raw", "shared_sigreg_plus_raw")
CONTROLS = ("random_encoder", "shuffled_future")
SEEDS = (7, 13, 23)


@dataclass(frozen=True)
class V2Slot:
    run_id: str
    family: str
    phase: str
    seed: int
    configuration: int | None
    dependencies: tuple[str, ...]


@dataclass(frozen=True)
class SlotResult:
    predictions: Path
    updates: int
    reusable_seed7: bool = False
    training_phases: tuple[TrainingPhase, ...] = ()


@dataclass(frozen=True)
class TrainingPhase:
    name: str
    updates: int
    checkpoints: tuple[Path, ...]
    validation_scores: tuple[float, ...]
    validation_predictions: tuple[Path, ...] = ()


SlotExecutor = Callable[[V2Slot, int | None, Path], SlotResult]


def validate_update_cadence(
    *, updates: int, checkpoint_steps: tuple[int, ...], validation_scores: tuple[float, ...]
) -> None:
    """Require 250-update checks and the frozen four-check patience after 1,000 updates."""
    if not 250 <= updates <= 3000 or updates % 250:
        raise ValueError("Campaign updates must end on a 250-update checkpoint by 3,000.")
    expected = tuple(range(250, updates + 1, 250))
    if checkpoint_steps != expected or len(validation_scores) != len(expected):
        raise ValueError("Campaign checkpoint and validation checks must occur every 250 updates.")
    best = float("inf")
    stale = 0
    first_stop = None
    for step, score in zip(expected, validation_scores):
        if not np.isfinite(score) or score < 0:
            raise ValueError("Campaign validation score must be finite and nonnegative.")
        if score < best:
            best = score
            stale = 0
        else:
            stale += 1
        if stale >= 4 and step >= 1000 and first_stop is None:
            first_stop = step
    if first_stop is not None and first_stop != updates:
        raise ValueError("Campaign ran beyond the frozen validation early stop.")
    if first_stop is None and updates != 3000:
        raise ValueError("Campaign stopped before 3,000 updates without a frozen early stop.")


def _verified_training_phases(
    slot: V2Slot,
    result: SlotResult,
    staging: Path,
    rows: list[HourlyWindow],
    protocol_sha256: str,
) -> list[dict[str, Any]]:
    if slot.family not in LEARNED or slot.phase == "hybrid":
        if result.training_phases or result.updates != 0:
            raise ValueError("Conventional and hybrid slots cannot claim neural updates.")
        return []
    expected = (
        ("supervised",)
        if slot.family == "direct"
        else (("probe",) if slot.phase == "random_encoder" else ("pretrain", "probe"))
    )
    if tuple(phase.name for phase in result.training_phases) != expected:
        raise ValueError("Learned campaign slot lacks its declared training phases.")
    if result.updates != result.training_phases[-1].updates:
        raise ValueError("Campaign endpoint update count differs from its final phase.")
    manifest = []
    for phase in result.training_phases:
        steps = tuple(range(250, phase.updates + 1, 250))
        validate_update_cadence(
            updates=phase.updates,
            checkpoint_steps=steps if len(phase.checkpoints) == len(steps) else (),
            validation_scores=phase.validation_scores,
        )
        if phase.name != "pretrain" and len(phase.validation_predictions) != len(steps):
            raise ValueError(
                "Supervised campaign checks need saved validation rows every 250 updates."
            )
        if phase.name == "pretrain" and phase.validation_predictions:
            raise ValueError("Pretraining checks use objective loss, not forecast rows.")
        checkpoints = []
        validations = []
        for index, step in enumerate(steps):
            raw_checkpoint = phase.checkpoints[index]
            checkpoint = raw_checkpoint.resolve(strict=True)
            if raw_checkpoint.is_symlink() or not checkpoint.is_relative_to(staging.resolve()):
                raise ValueError("Campaign checkpoint escapes its slot output.")
            checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if (
                checkpoint_state.get("step") != step
                or checkpoint_state.get("protocol_sha256") != protocol_sha256
                or not checkpoint_state.get("model")
                or not checkpoint_state.get("optimizer")
            ):
                raise ValueError("Campaign checkpoint step, optimizer or protocol differs.")
            checkpoints.append(
                {
                    "step": step,
                    "path": str(checkpoint.relative_to(staging)),
                    "sha256": _sha256(checkpoint),
                }
            )
            if phase.name != "pretrain":
                raw_prediction = phase.validation_predictions[index]
                prediction = raw_prediction.resolve(strict=True)
                if raw_prediction.is_symlink() or not prediction.is_relative_to(staging.resolve()):
                    raise ValueError("Campaign validation artifact escapes its slot output.")
                score = _primary(_verify_predictions(prediction, rows))
                if score != phase.validation_scores[index]:
                    raise ValueError(
                        "Campaign validation score differs from saved checkpoint rows."
                    )
                validations.append(
                    {
                        "step": step,
                        "path": str(prediction.relative_to(staging)),
                        "sha256": _sha256(prediction),
                        "metric": score,
                    }
                )
        manifest.append(
            {
                "name": phase.name,
                "updates": phase.updates,
                "validation_scores": phase.validation_scores,
                "checkpoints": checkpoints,
                "validation_predictions": validations,
            }
        )
    return manifest


def v2_plan() -> tuple[V2Slot, ...]:
    """Return the preregistered four conventional and 21 learned/control slots."""
    slots = [V2Slot(f"{family}-seed7", family, "baseline", 7, None, ()) for family in BASELINES]
    for family in LEARNED:
        for configuration in (0, 1):
            slots.append(
                V2Slot(
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
                V2Slot(f"{family}-seed{seed}", family, "selected", seed, None, dependencies)
            )
    for family in HYBRIDS:
        parent = family.removesuffix("_plus_raw")
        slots.append(
            V2Slot(
                f"{family}-seed7",
                family,
                "hybrid",
                7,
                None,
                tuple(f"{parent}-seed{seed}" for seed in SEEDS),
            )
        )
    for family in LEARNED[1:]:
        dependencies = tuple(f"{family}-seed{seed}" for seed in SEEDS)
        for control in CONTROLS:
            slots.append(
                V2Slot(f"{family}-{control}-seed7", family, control, 7, None, dependencies)
            )
    if len(slots) != 25 or len({slot.run_id for slot in slots}) != 25:
        raise AssertionError("V2 campaign must contain exactly 25 distinct slots.")
    seen: set[str] = set()
    for slot in slots:
        if not set(slot.dependencies).issubset(seen):
            raise AssertionError("V2 campaign dependencies must precede dependent slots.")
        seen.add(slot.run_id)
    return tuple(slots)


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _validation_digest(rows: list[HourlyWindow]) -> str:
    return _rows_digest(rows)


def _write_ledger(path: Path, ledger: dict[str, Any]) -> None:
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".stage.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(ledger, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _gate(
    fixture_only: bool,
    protocol_sha256: str,
    review_path: Path | None,
    review_sha256: str | None,
    source_hashes: tuple[str, ...],
    validation_sha256: str,
    support_report_path: Path | None,
    support_report_sha256: str | None,
) -> None:
    if fixture_only:
        return
    if protocol_sha256 != V2_PROTOCOL_SHA256:
        raise ValueError("Real campaign requires the reviewed v2 protocol digest.")
    if review_path is None or review_sha256 is None or not _valid_hash(review_sha256):
        raise ValueError("An independent support and campaign review is required.")
    if (
        support_report_path is None
        or support_report_sha256 is None
        or not _valid_hash(support_report_sha256)
        or _sha256(support_report_path) != support_report_sha256
    ):
        raise ValueError("An exact native support report is required for a real campaign.")
    support = json.loads(support_report_path.read_text(encoding="utf-8"))
    if support.get("status") != "NATIVE_SUPPORT_ADEQUATE_PENDING_REVIEW":
        raise ValueError("Native support is ineligible; no D1 campaign can run.")
    if _sha256(review_path) != review_sha256:
        raise ValueError("Independent campaign review digest differs.")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if (
        review.get("status") != "APPROVED_V2_CAMPAIGN"
        or review.get("candidate_id") != "v2_candidate2"
        or review.get("support_status") != "ELIGIBLE"
        or review.get("protocol_sha256") != protocol_sha256
        or review.get("code_sha256") != _code_digest()
        or review.get("validation_sha256") != validation_sha256
        or review.get("support_report_sha256") != support_report_sha256
        or review.get("validation_source_sha256") != list(source_hashes)
    ):
        raise ValueError("Independent campaign review does not authorize this source and support.")


def _primary(metrics: dict[str, Any]) -> float:
    horizons = metrics["horizons"]
    if len(horizons) != 3:
        raise ValueError("Validation result needs all three horizons.")
    scores = [horizon["daily_mean_pinball_db"] for horizon in horizons]
    if any(value is None or not np.isfinite(value) or value < 0 for value in scores):
        raise ValueError("Validation selection requires a finite score at every horizon.")
    return float(np.mean(scores))


def _verify_ledger(
    ledger: dict[str, Any],
    *,
    identity: dict[str, Any],
    slots: tuple[V2Slot, ...],
    rows: list[HourlyWindow],
    root: Path,
) -> None:
    if ledger.get("identity") != identity or set(ledger.get("runs", {})) != {
        slot.run_id for slot in slots
    }:
        raise ValueError("Campaign ledger identity or finite slots differ.")
    success = {
        "COMPLETED_SYNTHETIC_FIXTURE",
        "COMPLETED_TRAIN_VALIDATION",
        "REUSED_SYNTHETIC_FIXTURE",
        "REUSED_TRAIN_VALIDATION",
    }
    for slot in slots:
        run = ledger["runs"][slot.run_id]
        if run["status"] not in success:
            if run["status"] not in ("PENDING", "RUNNING", "FAILED"):
                raise ValueError("Campaign ledger has an unknown run status.")
            continue
        source = run
        if run["status"].startswith("REUSED"):
            if slot.phase != "selected" or slot.seed != 7:
                raise ValueError("Only selected seed-7 endpoints can be reused.")
            if run["reuse_of"] != f"{slot.family}-development{run['selected_configuration']}-seed7":
                raise ValueError("Campaign reuse points to a different family or configuration.")
            source = ledger["runs"].get(run["reuse_of"])
            if source is None or not source["status"].startswith("COMPLETED"):
                raise ValueError("Campaign reuse source is not completed.")
        path = Path(source["predictions"])
        if not path.is_file() or path.is_symlink() or _sha256(path) != source["prediction_sha256"]:
            raise ValueError("Campaign validation artifact differs on resume.")
        metric = _primary(_verify_predictions(path, rows))
        if metric != source["validation_metric"] or metric != run["validation_metric"]:
            raise ValueError("Campaign validation metric differs from saved rows.")
        artifact_root = Path(source["artifact_root"])
        owner = run["reuse_of"] if run["status"].startswith("REUSED") else slot.run_id
        if (
            artifact_root.resolve(strict=True) != (root / "runs" / owner).resolve(strict=True)
            or not path.resolve(strict=True).is_relative_to(artifact_root.resolve(strict=True))
            or any(item.is_symlink() for item in artifact_root.rglob("*"))
        ):
            raise ValueError("Campaign artifact ownership or path differs on resume.")
        actual_files = {
            path.relative_to(artifact_root).as_posix(): _sha256(path)
            for path in artifact_root.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        registered_files = {item["path"]: item["sha256"] for item in source.get("artifacts", [])}
        if actual_files != registered_files:
            raise ValueError("Campaign model or resource artifacts differ on resume.")
        for phase in source.get("training_phases", []):
            for item in phase["checkpoints"] + phase["validation_predictions"]:
                file = (artifact_root / item["path"]).resolve(strict=True)
                if (
                    not file.is_relative_to(artifact_root.resolve())
                    or _sha256(file) != item["sha256"]
                ):
                    raise ValueError("Campaign training milestone artifact differs on resume.")
                if "metric" in item and _primary(_verify_predictions(file, rows)) != item["metric"]:
                    raise ValueError("Campaign training milestone metric differs on resume.")
    completed = sum(run["status"] in success for run in ledger["runs"].values())
    suffix = (
        "SYNTHETIC_FIXTURE" if identity["scope"] == "synthetic-fixture-only" else "TRAIN_VALIDATION"
    )
    if (
        ledger.get("completed_slots") != completed
        or ledger.get("test_opened") is not False
        or ledger.get("status")
        != (f"COMPLETE_{suffix}" if completed == 25 else f"PARTIAL_{suffix}")
    ):
        raise ValueError("Campaign ledger counts or protected-test seal differ.")
    selected = {}
    for family in LEARNED:
        development = [ledger["runs"][f"{family}-development{index}-seed7"] for index in (0, 1)]
        if all(run["status"] in success for run in development):
            selected[family] = min(
                (0, 1), key=lambda index: (development[index]["validation_metric"], index)
            )
    if any(
        selected.get(family) != choice
        for family, choice in ledger.get("selected_configs", {}).items()
    ) or (completed == 25 and ledger["selected_configs"] != selected):
        raise ValueError("Campaign saved configuration differs from validation selection.")


def execute_v2_campaign(
    root: Path,
    *,
    validation_rows: list[HourlyWindow],
    executor: SlotExecutor,
    protocol_sha256: str,
    fixture_only: bool = False,
    review_path: Path | None = None,
    review_sha256: str | None = None,
    support_report_path: Path | None = None,
    support_report_sha256: str | None = None,
    max_slots: int | None = None,
) -> dict[str, Any]:
    """Run pending slots serially; select only from saved validation predictions."""
    if not _valid_hash(protocol_sha256) or max_slots is not None and max_slots < 1:
        raise ValueError("Campaign protocol digest or step budget is invalid.")
    if not validation_rows or len({row.row_id for row in validation_rows}) != len(validation_rows):
        raise ValueError("Validation rows must be nonempty with unique identities.")
    if not fixture_only and any(row.partition != "validation" for row in validation_rows):
        raise ValueError("Real campaign selection requires validation partition rows.")
    source_hashes = tuple(
        sorted({digest for row in validation_rows for digest in row.source_sha256})
    )
    if not source_hashes or not all(_valid_hash(value) for value in source_hashes):
        raise ValueError("Validation rows require exact source hashes.")
    validation_sha256 = _validation_digest(validation_rows)
    _gate(
        fixture_only,
        protocol_sha256,
        review_path,
        review_sha256,
        source_hashes,
        validation_sha256,
        support_report_path,
        support_report_sha256,
    )
    slots = v2_plan()
    identity = {
        "scope": "synthetic-fixture-only" if fixture_only else "reviewed-train-validation",
        "protocol_sha256": protocol_sha256,
        "review_sha256": None if fixture_only else review_sha256,
        "validation_sha256": validation_sha256,
        "support_report_sha256": None if fixture_only else support_report_sha256,
        "validation_source_sha256": list(source_hashes),
        "plan_sha256": _canonical_digest([asdict(slot) for slot in slots]),
    }
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / "ledger.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        _verify_ledger(ledger, identity=identity, slots=slots, rows=validation_rows, root=root)
    else:
        if any(root.iterdir()):
            raise FileExistsError("New campaign root must be empty.")
        ledger = {
            "identity": identity,
            "runs": {slot.run_id: {"status": "PENDING"} for slot in slots},
            "selected_configs": {},
            "completed_slots": 0,
            "test_opened": False,
            "status": "PARTIAL_SYNTHETIC_FIXTURE" if fixture_only else "PARTIAL_TRAIN_VALIDATION",
        }
        _write_ledger(ledger_path, ledger)
    completed_this_call = 0
    success_prefix = "SYNTHETIC_FIXTURE" if fixture_only else "TRAIN_VALIDATION"
    for slot in slots:
        run = ledger["runs"][slot.run_id]
        if run["status"].startswith(("COMPLETED", "REUSED")):
            continue
        if run["status"] != "PENDING":
            raise RuntimeError(f"Campaign slot {slot.run_id} needs interruption review.")
        if max_slots is not None and completed_this_call >= max_slots:
            break
        if any(
            not ledger["runs"][dependency]["status"].startswith(("COMPLETED", "REUSED"))
            for dependency in slot.dependencies
        ):
            raise RuntimeError(f"Campaign slot {slot.run_id} has an unfinished dependency.")
        selected: int | None = slot.configuration
        if slot.family in LEARNED and slot.phase != "development":
            development = [
                ledger["runs"][f"{slot.family}-development{index}-seed7"] for index in (0, 1)
            ]
            selected = min(
                (0, 1), key=lambda index: (development[index]["validation_metric"], index)
            )
            prior = ledger["selected_configs"].get(slot.family)
            if prior is not None and prior != selected:
                raise ValueError("Saved campaign selection differs from validation metrics.")
            ledger["selected_configs"][slot.family] = selected
        if slot.phase == "selected" and slot.seed == 7:
            source_id = f"{slot.family}-development{selected}-seed7"
            source = ledger["runs"][source_id]
            if source["reusable_seed7"]:
                ledger["runs"][slot.run_id] = {
                    **source,
                    "status": f"REUSED_{success_prefix}",
                    "reuse_of": source_id,
                    "selected_configuration": selected,
                }
                completed_this_call += 1
                ledger["completed_slots"] += 1
                _write_ledger(ledger_path, ledger)
                continue
        slot_dir = root / "runs" / slot.run_id
        if slot_dir.exists():
            raise FileExistsError("Unregistered campaign slot directory already exists.")
        slot_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=slot.run_id + ".stage.", dir=slot_dir.parent))
        run["status"] = "RUNNING"
        _write_ledger(ledger_path, ledger)
        try:
            result = executor(slot, selected, staging)
            path = result.predictions.resolve(strict=True)
            if (
                path.is_symlink()
                or not path.is_relative_to(staging.resolve())
                or not 0 <= result.updates <= 3000
                or (result.reusable_seed7 and slot.phase != "development")
            ):
                raise ValueError("Campaign executor result exceeds the finite slot contract.")
            metrics = _verify_predictions(path, validation_rows)
            score = _primary(metrics)
            phase_manifest = (
                []
                if fixture_only
                else _verified_training_phases(
                    slot, result, staging, validation_rows, protocol_sha256
                )
            )
            name = path.relative_to(staging)
            digest = _sha256(path)
            artifacts = []
            for artifact in sorted(staging.rglob("*")):
                if artifact.is_symlink():
                    raise ValueError("Campaign slot artifact cannot be a symbolic link.")
                if artifact.is_file():
                    artifacts.append(
                        {
                            "path": artifact.relative_to(staging).as_posix(),
                            "sha256": _sha256(artifact),
                        }
                    )
            os.replace(staging, slot_dir)
            ledger["runs"][slot.run_id] = {
                "status": f"COMPLETED_{success_prefix}",
                "family": slot.family,
                "phase": slot.phase,
                "seed": slot.seed,
                "selected_configuration": selected,
                "predictions": str(slot_dir / name),
                "artifact_root": str(slot_dir),
                "artifacts": artifacts,
                "prediction_sha256": digest,
                "validation_metric": score,
                "updates": result.updates,
                "reusable_seed7": result.reusable_seed7,
                "training_phases": phase_manifest,
                "reuse_of": None,
            }
            completed_this_call += 1
            ledger["completed_slots"] += 1
            _write_ledger(ledger_path, ledger)
        except BaseException:
            run["status"] = "FAILED"
            _write_ledger(ledger_path, ledger)
            raise
    ledger["status"] = (
        f"COMPLETE_{success_prefix}"
        if ledger["completed_slots"] == 25
        else f"PARTIAL_{success_prefix}"
    )
    _write_ledger(ledger_path, ledger)
    return ledger
