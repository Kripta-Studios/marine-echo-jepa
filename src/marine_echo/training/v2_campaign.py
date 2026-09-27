"""Finite 25-slot v2 scheduler with saved-row validation selection and review gate.

The scheduler owns slot order and accounting. A supplied executor owns each model
fit and must write validation predictions before this scheduler can score a slot.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
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
    validation_score_files: tuple[Path, ...] = ()


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
        if phase.name == "pretrain" and len(phase.validation_score_files) != len(steps):
            raise ValueError("Pretraining checks need saved validation losses every 250 updates.")
        if phase.name != "pretrain" and phase.validation_score_files:
            raise ValueError("Supervised checks cannot claim pretraining validation losses.")
        checkpoints = []
        validations = []
        score_files = []
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
            else:
                raw_score = phase.validation_score_files[index]
                score_path = raw_score.resolve(strict=True)
                if raw_score.is_symlink() or not score_path.is_relative_to(staging.resolve()):
                    raise ValueError("Pretraining validation artifact escapes its slot output.")
                recorded_score = json.loads(score_path.read_text(encoding="utf-8"))
                if (
                    recorded_score.get("step") != step
                    or recorded_score.get("score") != phase.validation_scores[index]
                ):
                    raise ValueError("Pretraining validation score differs from saved loss.")
                score_files.append(
                    {
                        "step": step,
                        "path": str(score_path.relative_to(staging)),
                        "sha256": _sha256(score_path),
                        "metric": phase.validation_scores[index],
                    }
                )
        manifest.append(
            {
                "name": phase.name,
                "updates": phase.updates,
                "validation_scores": phase.validation_scores,
                "checkpoints": checkpoints,
                "validation_predictions": validations,
                "validation_score_files": score_files,
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


def _slot_record(slot: V2Slot) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(slot)))


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


def _stage_files(staging: Path) -> list[dict[str, str]]:
    files = []
    resolved_staging = staging.resolve(strict=True)
    for path in sorted(staging.rglob("*")):
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(resolved_staging):
            raise ValueError("Interrupted campaign staging contains an escaping path.")
        if path.is_file():
            files.append({"path": path.relative_to(staging).as_posix(), "sha256": _sha256(path)})
    return files


def _write_active_lease(
    staging: Path, run: dict[str, Any], identity: dict[str, Any], review_sha256: str | None
) -> None:
    history = run.setdefault("lease_history", [])
    generation = len(history)
    process = psutil.Process()
    lease = {
        "generation": generation,
        "pid": process.pid,
        "process_create_time": process.create_time(),
        "attempt_sha256": run["attempt_sha256"],
        "campaign_identity_sha256": _canonical_digest(identity),
        "code_sha256": _code_digest(),
        "resume_review_sha256": review_sha256,
    }
    path = staging / f".lease-{generation}.json"
    if path.exists():
        raise FileExistsError("Campaign active-run lease already exists.")
    path.write_text(json.dumps(lease, sort_keys=True), encoding="utf-8")
    history.append({"lease": lease, "sha256": _sha256(path)})
    run["active_lease"] = lease


def _verify_lease_history(run: dict[str, Any], storage: Path, identity: dict[str, Any]) -> None:
    history = run.get("lease_history")
    reviews = run.get("resume_review_sha256s", [])
    if not isinstance(history, list) or not history or len(history) != len(reviews) + 1:
        raise ValueError("Campaign active-run lease generations differ.")
    for generation, entry in enumerate(history):
        lease = entry.get("lease", {})
        path = storage / f".lease-{generation}.json"
        if (
            path.is_symlink()
            or not path.is_file()
            or _sha256(path) != entry.get("sha256")
            or json.loads(path.read_text(encoding="utf-8")) != lease
            or lease.get("generation") != generation
            or lease.get("attempt_sha256") != run.get("attempt_sha256")
            or lease.get("campaign_identity_sha256") != _canonical_digest(identity)
            or lease.get("code_sha256") != _code_digest()
            or lease.get("resume_review_sha256")
            != (None if generation == 0 else reviews[generation - 1])
        ):
            raise ValueError("Campaign active-run lease provenance differs.")
    if run.get("active_lease") != history[-1]["lease"]:
        raise ValueError("Campaign active-run lease pointer differs.")


def _active_process_is_live(run: dict[str, Any]) -> bool:
    lease = run["active_lease"]
    try:
        return psutil.Process(lease["pid"]).create_time() == lease["process_create_time"]
    except psutil.NoSuchProcess:
        return False


def _verify_attempt(
    run: dict[str, Any], slot: V2Slot, identity: dict[str, Any], root: Path
) -> Path:
    attempt = run.get("attempt")
    if not isinstance(attempt, dict) or run.get("attempt_sha256") is None:
        raise ValueError("Interrupted campaign slot lacks registered attempt metadata.")
    staging = Path(attempt.get("staging", ""))
    if (
        staging.is_symlink()
        or not staging.is_dir()
        or staging.resolve(strict=True).parent != (root / "runs").resolve(strict=True)
        or not staging.name.startswith(slot.run_id + ".stage.")
        or attempt.get("slot") != _slot_record(slot)
        or attempt.get("campaign_identity_sha256") != _canonical_digest(identity)
        or attempt.get("code_sha256") != _code_digest()
        or attempt.get("selected_configuration") not in (0, 1)
        or attempt.get("run_id") != slot.run_id
        or (root / "runs" / slot.run_id).exists()
    ):
        raise ValueError("Interrupted campaign staging or slot identity differs.")
    record = staging / ".attempt.json"
    if (
        not record.is_file()
        or record.is_symlink()
        or _sha256(record) != run["attempt_sha256"]
        or json.loads(record.read_text(encoding="utf-8")) != attempt
    ):
        raise ValueError("Interrupted campaign attempt metadata differs.")
    _verify_lease_history(run, staging, identity)
    return staging


def interrupted_slot_review_request(root: Path, run_id: str) -> dict[str, Any]:
    """Describe exact interrupted files for a distinct reviewer; this grants no approval."""
    root = root.resolve(strict=True)
    ledger = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
    slot = next((item for item in v2_plan() if item.run_id == run_id), None)
    if slot is None or slot.family not in LEARNED:
        raise ValueError("Only interrupted learned slots can request checkpoint review.")
    run = ledger["runs"][run_id]
    if run.get("status") not in ("RUNNING", "FAILED"):
        raise ValueError("Campaign slot is not interrupted.")
    staging = _verify_attempt(run, slot, ledger["identity"], root)
    if run["status"] == "RUNNING" and _active_process_is_live(run):
        raise ValueError("Interrupted campaign process is still running.")
    staging_files = _stage_files(staging)
    phases = ("supervised",) if slot.family == "direct" else ("probe", "pretrain")
    checkpoint: Path | None = None
    phase_name = ""
    step = 0
    for phase in phases:
        candidates = []
        for path in staging.glob(f"{phase}-checkpoint-*.pt"):
            match = re.fullmatch(rf"{phase}-checkpoint-(\d+)\.pt", path.name)
            if match:
                candidates.append((int(match.group(1)), path))
        if candidates:
            step, checkpoint = max(candidates)
            phase_name = phase
            break
    if (
        checkpoint is None
        or step < 1
        or (ledger["identity"]["scope"] != "synthetic-fixture-only" and step % 250)
    ):
        raise ValueError("Interrupted learned slot has no complete frozen checkpoint.")
    if phase_name == "pretrain" and ledger["identity"]["scope"] != "synthetic-fixture-only":
        scores = []
        for prior_step in range(250, step + 1, 250):
            path = staging / f"pretrain-validation-{prior_step}.json"
            recorded = json.loads(path.read_text(encoding="utf-8"))
            if recorded.get("step") != prior_step:
                raise ValueError("Interrupted pretraining validation cadence differs.")
            scores.append(recorded["score"])
        try:
            validate_update_cadence(
                updates=step,
                checkpoint_steps=tuple(range(250, step + 1, 250)),
                validation_scores=tuple(scores),
            )
        except ValueError as error:
            if "stopped before 3,000" not in str(error):
                raise
        else:
            phase_name = "pretrain_complete"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        state.get("step") != step
        or state.get("protocol_sha256") != ledger["identity"]["protocol_sha256"]
        or not state.get("model")
        or not state.get("optimizer")
        or "rng_cpu" not in state
    ):
        raise ValueError("Interrupted checkpoint state or protocol differs.")
    return {
        "run_id": run_id,
        "slot": _slot_record(slot),
        "selected_configuration": run["attempt"]["selected_configuration"],
        "campaign_identity_sha256": _canonical_digest(ledger["identity"]),
        "attempt_sha256": run["attempt_sha256"],
        "staging": str(staging),
        "staging_files": staging_files,
        "resume_phase": phase_name,
        "resume_checkpoint": {
            "path": checkpoint.relative_to(staging).as_posix(),
            "sha256": _sha256(checkpoint),
            "step": step,
        },
        "code_sha256": _code_digest(),
    }


def _validated_restart_review(
    root: Path, run_id: str, path: Path | None, digest: str | None
) -> dict[str, Any]:
    if path is None or digest is None or not _valid_hash(digest) or _sha256(path) != digest:
        raise ValueError("Interrupted slot needs an exact independent restart review.")
    review = json.loads(path.read_text(encoding="utf-8"))
    if (
        review.get("status") != "APPROVED_V2_INTERRUPTED_SLOT_RESTART"
        or review.get("reviewer_role") != "independent_reviewer"
        or not review.get("reviewer_session")
    ):
        raise ValueError("Interrupted slot restart lacks a distinct reviewer decision.")
    expected = interrupted_slot_review_request(root, run_id)
    if {key: review.get(key) for key in expected} != expected:
        raise ValueError("Interrupted slot staging, checkpoint or review identity differs.")
    return review


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
    suffix = (
        "SYNTHETIC_FIXTURE" if identity["scope"] == "synthetic-fixture-only" else "TRAIN_VALIDATION"
    )
    success = {f"COMPLETED_{suffix}", f"REUSED_{suffix}"}
    for slot in slots:
        run = ledger["runs"][slot.run_id]
        if run["status"].startswith(("COMPLETED", "REUSED")) and run["status"] not in success:
            raise ValueError("Campaign slot completion status differs from its scope.")
        if run["status"] not in success:
            if run["status"] not in ("PENDING", "RUNNING", "FAILED"):
                raise ValueError("Campaign ledger has an unknown run status.")
            if run["status"] in ("RUNNING", "FAILED"):
                _verify_attempt(run, slot, identity, root)
            continue
        source = run
        if run["status"].startswith("REUSED"):
            if slot.phase != "selected" or slot.seed != 7:
                raise ValueError("Only selected seed-7 endpoints can be reused.")
            if (
                run.get("family") != slot.family
                or run.get("phase") != slot.phase
                or run.get("seed") != slot.seed
                or run.get("selected_configuration") not in (0, 1)
                or run.get("reusable_seed7") is not False
            ):
                raise ValueError("Reused campaign slot identity differs from its plan.")
            if run["reuse_of"] != f"{slot.family}-development{run['selected_configuration']}-seed7":
                raise ValueError("Campaign reuse points to a different family or configuration.")
            source = ledger["runs"].get(run["reuse_of"])
            if source is None or source["status"] != f"COMPLETED_{suffix}":
                raise ValueError("Campaign reuse source is not completed.")
            expected_reuse = {
                **source,
                "status": f"REUSED_{suffix}",
                "reuse_of": run["reuse_of"],
                "family": slot.family,
                "phase": slot.phase,
                "seed": slot.seed,
                "selected_configuration": run["selected_configuration"],
                "reusable_seed7": False,
            }
            if run != expected_reuse:
                raise ValueError("Reused campaign slot artifacts differ from their source.")
        elif (
            run.get("family") != slot.family
            or run.get("phase") != slot.phase
            or run.get("seed") != slot.seed
            or run.get("selected_configuration")
            != (
                slot.configuration
                if slot.phase == "development"
                else (
                    ledger.get("selected_configs", {}).get(slot.family)
                    if slot.family in LEARNED
                    else None
                )
            )
            or run.get("reuse_of") is not None
        ):
            raise ValueError("Completed campaign slot identity differs from its plan.")
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
        if not run["status"].startswith("REUSED"):
            attempt_file = artifact_root / ".attempt.json"
            if (
                run.get("attempt_sha256") != _sha256(attempt_file)
                or json.loads(attempt_file.read_text(encoding="utf-8")) != run.get("attempt")
                or run["attempt"].get("slot") != _slot_record(slot)
                or run["attempt"].get("campaign_identity_sha256") != _canonical_digest(identity)
                or run["attempt"].get("code_sha256") != _code_digest()
            ):
                raise ValueError("Completed campaign attempt identity differs.")
            _verify_lease_history(run, artifact_root, identity)
            for index, digest in enumerate(run.get("resume_review_sha256s", []), start=1):
                if _sha256(artifact_root / f"resume-review-{index}.json") != digest:
                    raise ValueError("Completed campaign restart review differs.")
        recorded_slot = json.loads((artifact_root / "slot-result.json").read_text(encoding="utf-8"))
        if recorded_slot != {
            key: source.get(key)
            for key in (
                "family",
                "phase",
                "seed",
                "selected_configuration",
                "updates",
                "reusable_seed7",
                "training_phases",
            )
        }:
            raise ValueError("Campaign slot result differs from its saved artifact.")
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
        if suffix == "TRAIN_VALIDATION" and run["status"] == f"COMPLETED_{suffix}":
            phases = tuple(
                TrainingPhase(
                    name=phase["name"],
                    updates=phase["updates"],
                    checkpoints=tuple(
                        artifact_root / item["path"] for item in phase["checkpoints"]
                    ),
                    validation_scores=tuple(phase["validation_scores"]),
                    validation_predictions=tuple(
                        artifact_root / item["path"] for item in phase["validation_predictions"]
                    ),
                    validation_score_files=tuple(
                        artifact_root / item["path"]
                        for item in phase.get("validation_score_files", [])
                    ),
                )
                for phase in source.get("training_phases", [])
            )
            verified = _verified_training_phases(
                slot,
                SlotResult(
                    path,
                    updates=source["updates"],
                    reusable_seed7=source["reusable_seed7"],
                    training_phases=phases,
                ),
                artifact_root,
                rows,
                identity["protocol_sha256"],
            )
            if verified != source["training_phases"]:
                raise ValueError("Campaign saved training phases differ from verified cadence.")
    completed = sum(run["status"] in success for run in ledger["runs"].values())
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
    restart_review_path: Path | None = None,
    restart_review_sha256: str | None = None,
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
        resuming = run["status"] in ("RUNNING", "FAILED")
        if run["status"] != "PENDING" and not resuming:
            raise RuntimeError(f"Campaign slot {slot.run_id} needs interruption review.")
        if resuming and (restart_review_path is None or restart_review_sha256 is None):
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
        if resuming and selected != run["attempt"]["selected_configuration"]:
            raise ValueError("Interrupted campaign configuration differs from frozen selection.")
        if not resuming and slot.phase == "selected" and slot.seed == 7:
            source_id = f"{slot.family}-development{selected}-seed7"
            source = ledger["runs"][source_id]
            if source["reusable_seed7"]:
                ledger["runs"][slot.run_id] = {
                    **source,
                    "status": f"REUSED_{success_prefix}",
                    "reuse_of": source_id,
                    "family": slot.family,
                    "phase": slot.phase,
                    "seed": slot.seed,
                    "selected_configuration": selected,
                    "reusable_seed7": False,
                }
                completed_this_call += 1
                ledger["completed_slots"] += 1
                _write_ledger(ledger_path, ledger)
                continue
        slot_dir = root / "runs" / slot.run_id
        if resuming:
            review = _validated_restart_review(
                root, slot.run_id, restart_review_path, restart_review_sha256
            )
            staging = _verify_attempt(run, slot, identity, root)
            reviews = run.setdefault("resume_review_sha256s", [])
            review_copy = staging / f"resume-review-{len(reviews) + 1}.json"
            if review_copy.exists():
                raise FileExistsError("Interrupted review copy already exists.")
            review_copy.write_bytes(restart_review_path.read_bytes())  # type: ignore[union-attr]
            reviews.append(restart_review_sha256)
            _write_active_lease(staging, run, identity, restart_review_sha256)
            run["status"] = "RUNNING"
            _write_ledger(ledger_path, ledger)
        else:
            if slot_dir.exists():
                raise FileExistsError("Unregistered campaign slot directory already exists.")
            slot_dir.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=slot.run_id + ".stage.", dir=slot_dir.parent))
            process = psutil.Process()
            attempt = {
                "version": 1,
                "run_id": slot.run_id,
                "slot": _slot_record(slot),
                "selected_configuration": selected,
                "campaign_identity_sha256": _canonical_digest(identity),
                "code_sha256": _code_digest(),
                "staging": str(staging.resolve()),
                "nonce": os.urandom(16).hex(),
                "pid": process.pid,
                "process_create_time": process.create_time(),
            }
            attempt_file = staging / ".attempt.json"
            attempt_file.write_text(json.dumps(attempt, sort_keys=True), encoding="utf-8")
            run.update(
                status="RUNNING",
                attempt=attempt,
                attempt_sha256=_sha256(attempt_file),
                resume_review_sha256s=[],
            )
            _write_active_lease(staging, run, identity, None)
            _write_ledger(ledger_path, ledger)
        try:
            if resuming:
                resume = getattr(executor, "resume_slot", None)
                if resume is None or slot.family not in LEARNED:
                    raise ValueError(
                        "Interrupted learned slot lacks a checkpoint restart executor."
                    )
                result = resume(slot, selected, staging, review)
            else:
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
            slot_result = {
                "family": slot.family,
                "phase": slot.phase,
                "seed": slot.seed,
                "selected_configuration": selected,
                "updates": result.updates,
                "reusable_seed7": result.reusable_seed7,
                "training_phases": phase_manifest,
            }
            (staging / "slot-result.json").write_text(
                json.dumps(slot_result, sort_keys=True, allow_nan=False), encoding="utf-8"
            )
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
                "attempt": run["attempt"],
                "attempt_sha256": run["attempt_sha256"],
                "resume_review_sha256s": run["resume_review_sha256s"],
                "lease_history": run["lease_history"],
                "active_lease": run["active_lease"],
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
