"""Reviewed native v2 cohort to finite campaign slot executors.

No real callback is constructed unless both partition support ledgers and an
independent cohort review bind the exact rows, code and source hashes.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits

from marine_echo.models.compact import ModelConfig
from marine_echo.models.v2_conventional import ConventionalFamily, NativeConventional
from marine_echo.models.v2_development import (
    JointDirectForecaster,
    JointJEPAForecaster,
    JointPrediction,
    joint_supervised_loss,
)
from marine_echo.models.v2_hybrid import NativeHybridRidge
from marine_echo.training.loop import load_checkpoint, save_checkpoint
from marine_echo.training.v2_campaign import (
    LEARNED,
    SlotResult,
    TrainingPhase,
    V2Slot,
    _primary,
    validate_update_cadence,
)
from marine_echo.training.v2_cohort import verify_support_rows
from marine_echo.training.v2_executor import (
    V2_PROTOCOL_SHA256,
    _code_digest,
    _rows_digest,
    _Scaler,
    _sha256,
    _tensors,
    _valid_hash,
    _verify_predictions,
    _write_predictions,
)
from marine_echo.training.v2_representation import (
    _future,
    _representation_features,
    _separated_shuffle,
)
from marine_echo.training.v2_stream import HourlyWindow


def _source_hashes(rows: list[HourlyWindow]) -> list[str]:
    return sorted({digest for row in rows for digest in row.source_sha256})


def _selected_milestone(phase: dict[str, Any]) -> dict[str, Any]:
    scores = phase.get("validation_scores", [])
    checkpoints = phase.get("checkpoints", [])
    if (
        not scores
        or len(scores) != len(checkpoints)
        or any(not np.isfinite(score) for score in scores)
    ):
        raise ValueError("Parent probe lacks finite checkpoint validation scores.")
    return cast(
        dict[str, Any], checkpoints[min(range(len(scores)), key=lambda index: scores[index])]
    )


def _verify_cohort_rows(fit: list[HourlyWindow], validation: list[HourlyWindow]) -> None:
    if not fit or not validation:
        raise ValueError("Campaign fit and validation cohorts must be nonempty.")
    for rows, partition in ((fit, "train"), (validation, "validation")):
        if any(row.partition != partition for row in rows):
            raise ValueError("Campaign cohort partition differs from TRAIN/validation order.")
        if len({row.row_id for row in rows}) != len(rows):
            raise ValueError("Campaign cohort repeats a row identity.")
        if any(left.cutoff >= right.cutoff for left, right in pairwise(rows)):
            raise ValueError("Campaign cohort rows must be chronological.")
        if any(row.context_index_db is None or not row.past_source_sha256 for row in rows):
            raise ValueError("Real campaign rows require native past index and provenance.")
        if not _source_hashes(rows) or not all(
            _valid_hash(value) for value in _source_hashes(rows)
        ):
            raise ValueError("Campaign rows require registered source SHA-256 values.")
    if fit[-1].cutoff + np.timedelta64(6, "h") > validation[0].cutoff - np.timedelta64(24, "h"):
        raise ValueError("Campaign TRAIN and validation raw support overlaps.")


class NativeCampaignBackend:
    """One serial callback provider for the reviewed 25-slot calibrated campaign."""

    def __init__(
        self,
        fit: list[HourlyWindow],
        validation: list[HourlyWindow],
        *,
        fixture_only: bool = False,
        cohort_review_path: Path | None = None,
        cohort_review_sha256: str | None = None,
        support_report_path: Path | None = None,
        support_report_sha256: str | None = None,
        fit_support_rows_path: Path | None = None,
        fit_support_rows_sha256: str | None = None,
        validation_support_rows_path: Path | None = None,
        validation_support_rows_sha256: str | None = None,
        native_index_sha256: str | None = None,
        protocol_sha256: str = V2_PROTOCOL_SHA256,
        model_config: ModelConfig | None = None,
        batch_size: int = 16,
        fixture_updates: int = 2,
        tree_max_iter: int = 150,
        device: str = "cpu",
    ) -> None:
        self.fit = fit
        self.validation = validation
        self.fixture_only = fixture_only
        self.protocol_sha256 = protocol_sha256
        self.model_config = model_config or ModelConfig()
        self.batch_size = batch_size
        self.fixture_updates = fixture_updates
        self.tree_max_iter = tree_max_iter
        self.device = device
        self.cohort_review_sha256 = cohort_review_sha256
        self.support_report_sha256 = support_report_sha256
        self.fit_support_rows_sha256 = fit_support_rows_sha256
        self.validation_support_rows_sha256 = validation_support_rows_sha256
        self.native_index_sha256 = native_index_sha256
        if not fixture_only:
            if (
                cohort_review_path is None
                or cohort_review_sha256 is None
                or support_report_path is None
                or support_report_sha256 is None
                or fit_support_rows_path is None
                or fit_support_rows_sha256 is None
                or validation_support_rows_path is None
                or validation_support_rows_sha256 is None
                or native_index_sha256 is None
            ):
                raise ValueError(
                    "A complete independent cohort review and support lineage is required."
                )
            if (
                protocol_sha256 != V2_PROTOCOL_SHA256
                or self.model_config != ModelConfig()
                or batch_size != 16
                or tree_max_iter != 150
                or device not in ("cpu", "cuda")
                or not all(
                    _valid_hash(value)
                    for value in (
                        cohort_review_sha256,
                        support_report_sha256,
                        fit_support_rows_sha256,
                        validation_support_rows_sha256,
                        native_index_sha256,
                    )
                )
            ):
                raise ValueError("Real campaign configuration differs from the frozen protocol.")
            if (
                _sha256(cohort_review_path) != cohort_review_sha256
                or _sha256(support_report_path) != support_report_sha256
                or _sha256(fit_support_rows_path) != fit_support_rows_sha256
                or _sha256(validation_support_rows_path) != validation_support_rows_sha256
            ):
                raise ValueError("Real cohort review or support artifact digest differs.")
            support = json.loads(support_report_path.read_text(encoding="utf-8"))
            if support.get("status") != "NATIVE_SUPPORT_ADEQUATE_PENDING_REVIEW":
                raise ValueError("Native target support is ineligible; no D1 campaign can run.")
            registered_index = support.get("artifact_sha256", {}).get(
                "evidence\\v2\\native-development\\index.json"
            )
            if registered_index != native_index_sha256:
                raise ValueError("Campaign native index differs from the support report.")
            _verify_cohort_rows(fit, validation)
            if (
                fit[0].cutoff - np.timedelta64(24, "h") < np.datetime64("2020-02-17")
                or fit[-1].cutoff + np.timedelta64(6, "h") > np.datetime64("2020-05-27")
                or validation[0].cutoff - np.timedelta64(24, "h") < np.datetime64("2020-05-27")
                or validation[-1].cutoff + np.timedelta64(6, "h") > np.datetime64("2020-06-21")
            ):
                raise ValueError("Campaign rows exceed frozen TRAIN/validation calendars.")
            fit_part = support.get("partitions", {}).get("train", {})
            validation_part = support.get("partitions", {}).get("validation", {})
            if (
                fit_part.get("rows_sha256") != fit_support_rows_sha256
                or validation_part.get("rows_sha256") != validation_support_rows_sha256
                or fit_part.get("adequate_all_horizons") is not True
                or validation_part.get("adequate_all_horizons") is not True
            ):
                raise ValueError("Full campaign support partitions are absent or inadequate.")
            verify_support_rows(fit, json.loads(fit_support_rows_path.read_text(encoding="utf-8")))
            verify_support_rows(
                validation, json.loads(validation_support_rows_path.read_text(encoding="utf-8"))
            )
            review = json.loads(cohort_review_path.read_text(encoding="utf-8"))
            if (
                review.get("status") != "APPROVED_V2_CAMPAIGN_COHORT"
                or review.get("protocol_sha256") != protocol_sha256
                or review.get("code_sha256") != _code_digest()
                or review.get("native_index_sha256") != native_index_sha256
                or review.get("support_report_sha256") != support_report_sha256
                or review.get("fit_sha256") != _rows_digest(fit)
                or review.get("validation_sha256") != _rows_digest(validation)
                or review.get("fit_source_sha256") != _source_hashes(fit)
                or review.get("validation_source_sha256") != _source_hashes(validation)
            ):
                raise ValueError("Independent campaign cohort review does not bind these rows.")
        elif (
            not 1 <= fixture_updates <= 10
            or not 1 <= batch_size <= 16
            or device not in ("cpu", "cuda")
        ):
            raise ValueError("Fixture backend requires explicit bounded model parameters.")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")

    def __call__(self, slot: V2Slot, selected_config: int | None, output: Path) -> SlotResult:
        if not output.is_dir() or {path.name for path in output.iterdir()} not in (
            set(),
            {".attempt.json"},
        ):
            raise ValueError("Campaign slot output must contain only its attempt metadata.")
        if not self.fixture_only:
            ledger_path = output.parent.parent / "ledger.json"
            if not output.name.startswith(slot.run_id + ".stage.") or not ledger_path.is_file():
                raise ValueError("Real backend requires a running reviewed campaign slot.")
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            if (
                ledger.get("identity", {}).get("protocol_sha256") != self.protocol_sha256
                or ledger.get("identity", {}).get("validation_sha256")
                != _rows_digest(self.validation)
                or ledger.get("runs", {}).get(slot.run_id, {}).get("status") != "RUNNING"
                or ledger["runs"][slot.run_id].get("attempt_sha256")
                != _sha256(output / ".attempt.json")
            ):
                raise ValueError("Real backend slot or validation cohort differs from its ledger.")
            if slot.family in LEARNED:
                frozen_choice = (
                    slot.configuration
                    if slot.phase == "development"
                    else ledger.get("selected_configs", {}).get(slot.family)
                )
                if selected_config != frozen_choice:
                    raise ValueError(
                        "Real backend configuration differs from validation selection."
                    )
        started = time.perf_counter()
        self._peak_rss = psutil.Process().memory_info().rss
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        if slot.phase == "baseline":
            result = self._conventional(slot, output)
        elif slot.phase == "hybrid":
            result = self._hybrid(slot, output)
        else:
            if slot.family not in LEARNED or selected_config not in (0, 1):
                raise ValueError("Learned campaign slot lacks a selected frozen configuration.")
            result = (
                self._direct(slot, selected_config, output)
                if slot.family == "direct"
                else self._representation(slot, selected_config, output)
            )
        self._resource_guard(psutil.Process())
        (output / "resources.json").write_text(
            json.dumps(
                {
                    "wall_seconds": time.perf_counter() - started,
                    "peak_process_rss_bytes": self._peak_rss,
                    "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved()
                    if self.device == "cuda"
                    else None,
                },
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        return result

    def resume_slot(
        self,
        slot: V2Slot,
        selected_config: int | None,
        output: Path,
        review: dict[str, Any],
    ) -> SlotResult:
        """Continue one reviewed learned phase in its original staging directory."""
        if slot.family not in LEARNED or selected_config not in (0, 1):
            raise ValueError("Only learned campaign slots can resume from checkpoints.")
        if any(
            (output / name).exists()
            for name in (
                "validation-predictions.npz",
                "endpoint.json",
                "resources.json",
                "slot-result.json",
            )
        ):
            raise ValueError(
                "Interrupted slot already has final artifacts; manual review is required."
            )
        ledger_path = output.parent.parent / "ledger.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        run = ledger.get("runs", {}).get(slot.run_id, {})
        reviews = run.get("resume_review_sha256s", [])
        review_copy = output / f"resume-review-{len(reviews)}.json"
        if (
            run.get("status") != "RUNNING"
            or not reviews
            or _sha256(review_copy) != reviews[-1]
            or json.loads(review_copy.read_text(encoding="utf-8")) != review
            or run.get("attempt_sha256") != _sha256(output / ".attempt.json")
            or ledger.get("identity", {}).get("protocol_sha256") != self.protocol_sha256
            or ledger.get("identity", {}).get("validation_sha256") != _rows_digest(self.validation)
            or run["attempt"]["selected_configuration"] != selected_config
        ):
            raise ValueError("Interrupted backend slot differs from reviewed running attempt.")
        phase = review["resume_phase"]
        checkpoint = output / review["resume_checkpoint"]["path"]
        started = time.perf_counter()
        self._peak_rss = psutil.Process().memory_info().rss
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        result = (
            self._direct(slot, selected_config, output, resume_checkpoint=checkpoint)
            if slot.family == "direct" and phase == "supervised"
            else self._representation(
                slot,
                selected_config,
                output,
                resume_checkpoint=checkpoint,
                resume_phase=phase,
            )
            if slot.family != "direct" and phase in ("pretrain", "pretrain_complete", "probe")
            else None
        )
        if result is None:
            raise ValueError("Interrupted checkpoint phase differs from learned family.")
        self._resource_guard(psutil.Process())
        (output / "resources.json").write_text(
            json.dumps(
                {
                    "wall_seconds": time.perf_counter() - started,
                    "peak_process_rss_bytes": self._peak_rss,
                    "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved()
                    if self.device == "cuda"
                    else None,
                    "resumed_from_step": review["resume_checkpoint"]["step"],
                },
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        return result

    def _conventional(self, slot: V2Slot, output: Path) -> SlotResult:
        if slot.family not in ("persistence", "seasonal", "ridge", "hist_gradient_boosting"):
            raise ValueError("Unknown conventional campaign family.")
        with threadpool_limits(limits=4, user_api="openmp"):
            model = NativeConventional(
                cast(ConventionalFamily, slot.family), tree_max_iter=self.tree_max_iter
            ).fit(self.fit)
            prediction = model.predict(self.validation)
        path = output / "validation-predictions.npz"
        _write_predictions(path, self.validation, prediction)
        joblib.dump(model, output / "model.joblib")
        return SlotResult(path, updates=0)

    def _scaler_tensors(self) -> tuple[_Scaler, Any, Any]:
        scaler = _Scaler.fit(self.fit)
        return scaler, _tensors(self.fit, scaler), _tensors(self.validation, scaler)

    def _predict_neural(
        self, model: JointDirectForecaster | JointJEPAForecaster, scaler: _Scaler, tensors: Any
    ) -> JointPrediction:
        was_training = model.training
        model.eval()
        quantiles = []
        fractions = []
        with torch.no_grad():
            for start in range(0, len(self.validation), self.batch_size):
                part = slice(start, start + self.batch_size)
                forecast = model(
                    tensors.context[part].to(self.device),
                    tensors.mask[part].to(self.device),
                    tensors.aux[part].to(self.device),
                )
                quantiles.append(
                    forecast.quantiles.cpu().numpy() * scaler.target_std + scaler.target_mean
                )
                fractions.append(forecast.detection_fraction.cpu().numpy())
        if was_training:
            model.train()
        return JointPrediction(np.concatenate(quantiles), np.concatenate(fractions))

    def _resource_guard(self, process: psutil.Process) -> None:
        info = process.memory_info()
        self._peak_rss = max(self._peak_rss, info.rss, getattr(info, "peak_wset", info.rss))
        if self._peak_rss >= 22 * 1024**3:
            raise MemoryError("Campaign process RAM reached the 22 GiB limit.")
        if self.device == "cuda" and torch.cuda.max_memory_reserved() >= 10 * 1024**3:
            raise MemoryError("Campaign GPU reserve reached the 10 GiB target.")

    def _checkpoint_identity(self, slot: V2Slot, configuration: int, phase: str) -> dict[str, Any]:
        identity = {
            "protocol_sha256": self.protocol_sha256,
            "code_sha256": _code_digest(),
            "fit_sha256": _rows_digest(self.fit),
            "validation_sha256": _rows_digest(self.validation),
            "fit_source_sha256": _source_hashes(self.fit),
            "validation_source_sha256": _source_hashes(self.validation),
            "cohort_review_sha256": self.cohort_review_sha256,
            "support_report_sha256": self.support_report_sha256,
            "fit_support_rows_sha256": self.fit_support_rows_sha256,
            "validation_support_rows_sha256": self.validation_support_rows_sha256,
            "native_index_sha256": self.native_index_sha256,
            "slot": asdict(slot),
            "configuration": configuration,
            "phase": phase,
            "model_config": asdict(self.model_config),
            "batch_size": self.batch_size,
            "device": self.device,
            "fixture_only": self.fixture_only,
        }
        return cast(dict[str, Any], json.loads(json.dumps(identity, sort_keys=True)))

    def _save_phase_checkpoint(
        self,
        path: Path,
        model: JointDirectForecaster | JointJEPAForecaster,
        optimizer: torch.optim.Optimizer,
        *,
        step: int,
        identity: dict[str, Any],
    ) -> None:
        save_checkpoint(path, model, optimizer, step=step, protocol_sha256=self.protocol_sha256)
        path.with_suffix(".identity.json").write_text(
            json.dumps({"identity": identity, "checkpoint_sha256": _sha256(path)}, sort_keys=True),
            encoding="utf-8",
        )

    def _load_phase_checkpoint(
        self,
        path: Path,
        output: Path,
        model: JointDirectForecaster | JointJEPAForecaster,
        optimizer: torch.optim.Optimizer,
        identity: dict[str, Any],
    ) -> int:
        sidecar = path.with_suffix(".identity.json")
        if (
            not path.is_file()
            or path.is_symlink()
            or not sidecar.is_file()
            or sidecar.is_symlink()
            or path.resolve(strict=True).parent != output.resolve(strict=True)
        ):
            raise ValueError("Campaign resume checkpoint escapes its slot output.")
        recorded = json.loads(sidecar.read_text(encoding="utf-8"))
        if recorded != {"identity": identity, "checkpoint_sha256": _sha256(path)}:
            raise ValueError("Campaign resume checkpoint identity or digest differs.")
        return load_checkpoint(path, model, optimizer, protocol_sha256=self.protocol_sha256)

    def _supervised_phase(
        self,
        model: JointDirectForecaster | JointJEPAForecaster,
        *,
        optimizer: torch.optim.Optimizer,
        fit_tensors: Any,
        validation_tensors: Any,
        scaler: _Scaler,
        output: Path,
        slot: V2Slot,
        configuration: int,
        seed: int,
        learning_rate: float,
        name: str,
        resume_checkpoint: Path | None = None,
    ) -> tuple[TrainingPhase | None, Path]:
        candidates = np.flatnonzero(
            (fit_tensors.target_mask | fit_tensors.fraction_mask).any(dim=1).numpy()
        )
        if not len(candidates):
            raise ValueError("Campaign supervised TRAIN phase has no labels.")
        max_updates = self.fixture_updates if self.fixture_only else 3000
        process = psutil.Process()
        checkpoints: list[Path] = []
        validation_paths: list[Path] = []
        scores: list[float] = []
        best = float("inf")
        stale = 0
        identity = self._checkpoint_identity(slot, configuration, name)
        start_step = 0
        if resume_checkpoint is not None:
            start_step = self._load_phase_checkpoint(
                resume_checkpoint, output, model, optimizer, identity
            )
            if (
                resume_checkpoint.name != f"{name}-checkpoint-{start_step}.pt"
                or not 0 < start_step < max_updates
                or (not self.fixture_only and start_step % 250)
            ):
                raise ValueError("Campaign resume step differs from its frozen phase.")
            if not self.fixture_only:
                for prior_step in range(250, start_step + 1, 250):
                    prior = output / f"{name}-checkpoint-{prior_step}.pt"
                    if (
                        self._load_phase_checkpoint(prior, output, model, optimizer, identity)
                        != prior_step
                    ):
                        raise ValueError("Campaign prior checkpoint step differs.")
                    path = output / f"{name}-validation-{prior_step}.npz"
                    score = _primary(_verify_predictions(path, self.validation))
                    checkpoints.append(prior)
                    validation_paths.append(path)
                    scores.append(score)
                    if score < best:
                        best, stale = score, 0
                    else:
                        stale += 1
                self._load_phase_checkpoint(resume_checkpoint, output, model, optimizer, identity)
                if start_step >= 1000 and stale >= 4:
                    raise ValueError("Campaign cannot resume after the frozen early stop.")
        for step in range(start_step + 1, max_updates + 1):
            rng = np.random.default_rng(np.random.SeedSequence([seed, step, 41]))
            chosen = rng.choice(
                candidates, size=self.batch_size, replace=len(candidates) < self.batch_size
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * min(1.0, step / 150)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            forecast = model(
                fit_tensors.context[chosen].to(self.device),
                fit_tensors.mask[chosen].to(self.device),
                fit_tensors.aux[chosen].to(self.device),
            )
            loss = joint_supervised_loss(
                forecast,
                fit_tensors.target[chosen].to(self.device),
                fit_tensors.target_mask[chosen].to(self.device),
                fit_tensors.fraction[chosen].to(self.device),
                fit_tensors.fraction_mask[chosen].to(self.device),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Campaign supervised TRAIN loss is non-finite.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            self._resource_guard(process)
            if not self.fixture_only and step % 250 == 0:
                checkpoint = output / f"{name}-checkpoint-{step}.pt"
                self._save_phase_checkpoint(
                    checkpoint, model, optimizer, step=step, identity=identity
                )
                checkpoints.append(checkpoint)
                path = output / f"{name}-validation-{step}.npz"
                _write_predictions(
                    path,
                    self.validation,
                    self._predict_neural(model, scaler, validation_tensors),
                )
                validation_paths.append(path)
                score = _primary(_verify_predictions(path, self.validation))
                scores.append(score)
                if score < best:
                    best, stale = score, 0
                else:
                    stale += 1
                if step >= 1000 and stale >= 4:
                    break
        if self.fixture_only:
            checkpoint = output / f"{name}-checkpoint-{step}.pt"
            self._save_phase_checkpoint(checkpoint, model, optimizer, step=step, identity=identity)
            return None, checkpoint
        selected = checkpoints[min(range(len(scores)), key=lambda index: scores[index])]
        self._load_phase_checkpoint(selected, output, model, optimizer, identity)
        return (
            TrainingPhase(name, step, tuple(checkpoints), tuple(scores), tuple(validation_paths)),
            selected,
        )

    def _direct(
        self, slot: V2Slot, config: int, output: Path, *, resume_checkpoint: Path | None = None
    ) -> SlotResult:
        torch.manual_seed(slot.seed)
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        scaler, fit_tensors, validation_tensors = self._scaler_tensors()
        model = JointDirectForecaster(self.model_config).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3e-4, weight_decay=1e-4, betas=(0.9, 0.95)
        )
        phase, checkpoint = self._supervised_phase(
            model,
            optimizer=optimizer,
            fit_tensors=fit_tensors,
            validation_tensors=validation_tensors,
            scaler=scaler,
            output=output,
            slot=slot,
            configuration=config,
            seed=slot.seed,
            learning_rate=(3e-4, 1e-3)[config],
            name="supervised",
            resume_checkpoint=resume_checkpoint,
        )
        path = output / "validation-predictions.npz"
        _write_predictions(
            path, self.validation, self._predict_neural(model, scaler, validation_tensors)
        )
        (output / "endpoint.json").write_text(
            json.dumps({"checkpoint": checkpoint.name, "configuration": config, "seed": slot.seed}),
            encoding="utf-8",
        )
        return SlotResult(
            path,
            updates=self.fixture_updates if phase is None else phase.updates,
            reusable_seed7=slot.phase == "development" and slot.seed == 7,
            training_phases=() if phase is None else (phase,),
        )

    def _pretrain_validation_loss(
        self,
        model: JointJEPAForecaster,
        validation_tensors: Any,
        future: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> float:
        model.eval()
        squared_error = 0.0
        observed_tokens = 0
        with torch.no_grad():
            for start in range(0, len(self.validation), self.batch_size):
                part = slice(start, start + self.batch_size)
                encoded, context_valid = model.base.encoder(
                    validation_tensors.context[part].to(self.device),
                    validation_tensors.mask[part].to(self.device),
                )
                predicted = model.base.predictor(encoded, context_valid)
                target, target_valid = model.base.encode_future(
                    future[part].to(self.device), future_mask[part].to(self.device)
                )
                support = target_valid & context_valid.any(dim=1)[:, None, None]
                if support.any():
                    squared_error += float((predicted - target).square()[support].sum().item())
                    observed_tokens += int(support.sum().item()) * predicted.shape[-1]
        if not observed_tokens:
            raise ValueError("Campaign pretraining validation has no observed future tokens.")
        return squared_error / observed_tokens

    def _pretrain_phase(
        self,
        model: JointJEPAForecaster,
        *,
        slot: V2Slot,
        configuration: int,
        fit_tensors: Any,
        validation_tensors: Any,
        future: torch.Tensor,
        future_mask: torch.Tensor,
        validation_future: torch.Tensor,
        validation_future_mask: torch.Tensor,
        shuffle: np.ndarray,
        output: Path,
        resume_checkpoint: Path | None = None,
    ) -> TrainingPhase | None:
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=3e-4,
            weight_decay=1e-4,
            betas=(0.9, 0.95),
        )
        candidates = np.flatnonzero(future_mask[shuffle].any(dim=(1, 2, 3, 4)).numpy())
        if not len(candidates):
            raise ValueError("Campaign pretraining has no observed TRAIN future tokens.")
        max_updates = self.fixture_updates if self.fixture_only else 3000
        process = psutil.Process()
        checkpoints: list[Path] = []
        scores: list[float] = []
        score_paths: list[Path] = []
        best = float("inf")
        stale = 0
        identity = self._checkpoint_identity(slot, configuration, "pretrain")
        start_step = 0
        if resume_checkpoint is not None:
            start_step = self._load_phase_checkpoint(
                resume_checkpoint, output, model, optimizer, identity
            )
            if (
                resume_checkpoint.name != f"pretrain-checkpoint-{start_step}.pt"
                or not 0 < start_step < max_updates
                or (not self.fixture_only and start_step % 250)
            ):
                raise ValueError("Campaign pretraining resume step differs.")
            if not self.fixture_only:
                for prior_step in range(250, start_step + 1, 250):
                    prior = output / f"pretrain-checkpoint-{prior_step}.pt"
                    if (
                        self._load_phase_checkpoint(prior, output, model, optimizer, identity)
                        != prior_step
                    ):
                        raise ValueError("Campaign prior pretraining checkpoint step differs.")
                    score = self._pretrain_validation_loss(
                        model, validation_tensors, validation_future, validation_future_mask
                    )
                    score_path = output / f"pretrain-validation-{prior_step}.json"
                    recorded_score = json.loads(score_path.read_text(encoding="utf-8"))
                    if recorded_score != {"step": prior_step, "score": score, "identity": identity}:
                        raise ValueError("Campaign pretraining validation loss differs on resume.")
                    checkpoints.append(prior)
                    scores.append(score)
                    score_paths.append(score_path)
                    if score < best:
                        best, stale = score, 0
                    else:
                        stale += 1
                self._load_phase_checkpoint(resume_checkpoint, output, model, optimizer, identity)
                if start_step >= 1000 and stale >= 4:
                    raise ValueError("Campaign cannot resume after the frozen early stop.")
        for step in range(start_step + 1, max_updates + 1):
            rng = np.random.default_rng(np.random.SeedSequence([slot.seed, step, 17]))
            chosen = rng.choice(
                candidates, size=self.batch_size, replace=len(candidates) < self.batch_size
            )
            targets = shuffle[chosen]
            for group in optimizer.param_groups:
                group["lr"] = 3e-4 * min(1.0, step / 150)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = model.pretrain_objective(
                fit_tensors.context[chosen].to(self.device),
                fit_tensors.mask[chosen].to(self.device),
                future[targets].to(self.device),
                future_mask[targets].to(self.device),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Campaign JEPA TRAIN pretraining loss is non-finite.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if slot.family == "ema_jepa":
                model.base.update_teacher(momentum=0.99)
            self._resource_guard(process)
            if not self.fixture_only and step % 250 == 0:
                checkpoint = output / f"pretrain-checkpoint-{step}.pt"
                self._save_phase_checkpoint(
                    checkpoint, model, optimizer, step=step, identity=identity
                )
                checkpoints.append(checkpoint)
                score = self._pretrain_validation_loss(
                    model, validation_tensors, validation_future, validation_future_mask
                )
                scores.append(score)
                score_path = output / f"pretrain-validation-{step}.json"
                score_path.write_text(
                    json.dumps(
                        {"step": step, "score": score, "identity": identity}, sort_keys=True
                    ),
                    encoding="utf-8",
                )
                score_paths.append(score_path)
                if score < best:
                    best, stale = score, 0
                else:
                    stale += 1
                if step >= 1000 and stale >= 4:
                    break
        if self.fixture_only:
            self._save_phase_checkpoint(
                output / f"pretrain-checkpoint-{step}.pt",
                model,
                optimizer,
                step=step,
                identity=identity,
            )
            return None
        selected = checkpoints[min(range(len(scores)), key=lambda index: scores[index])]
        self._load_phase_checkpoint(selected, output, model, optimizer, identity)
        return TrainingPhase(
            "pretrain",
            step,
            tuple(checkpoints),
            tuple(scores),
            validation_score_files=tuple(score_paths),
        )

    def _recover_pretrain_phase(
        self,
        model: JointJEPAForecaster,
        slot: V2Slot,
        config: int,
        output: Path,
        validation_tensors: Any,
        validation_future: torch.Tensor,
        validation_future_mask: torch.Tensor,
    ) -> TrainingPhase:
        identity = self._checkpoint_identity(slot, config, "pretrain")
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=3e-4,
            weight_decay=1e-4,
            betas=(0.9, 0.95),
        )
        checkpoints: list[Path] = []
        scores: list[float] = []
        score_paths: list[Path] = []
        step = 250
        while (output / f"pretrain-checkpoint-{step}.pt").exists():
            checkpoint = output / f"pretrain-checkpoint-{step}.pt"
            if self._load_phase_checkpoint(checkpoint, output, model, optimizer, identity) != step:
                raise ValueError("Recovered pretraining checkpoint step differs.")
            score = self._pretrain_validation_loss(
                model, validation_tensors, validation_future, validation_future_mask
            )
            score_path = output / f"pretrain-validation-{step}.json"
            if json.loads(score_path.read_text(encoding="utf-8")) != {
                "step": step,
                "score": score,
                "identity": identity,
            }:
                raise ValueError("Recovered pretraining loss differs from checkpoint.")
            checkpoints.append(checkpoint)
            scores.append(score)
            score_paths.append(score_path)
            step += 250
        updates = step - 250
        validate_update_cadence(
            updates=updates,
            checkpoint_steps=tuple(range(250, updates + 1, 250)),
            validation_scores=tuple(scores),
        )
        selected = checkpoints[min(range(len(scores)), key=lambda index: scores[index])]
        self._load_phase_checkpoint(selected, output, model, optimizer, identity)
        return TrainingPhase(
            "pretrain",
            updates,
            tuple(checkpoints),
            tuple(scores),
            validation_score_files=tuple(score_paths),
        )

    def _representation(
        self,
        slot: V2Slot,
        config: int,
        output: Path,
        *,
        resume_checkpoint: Path | None = None,
        resume_phase: str | None = None,
    ) -> SlotResult:
        torch.manual_seed(slot.seed)
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        scaler, fit_tensors, validation_tensors = self._scaler_tensors()
        future, future_mask = _future(self.fit, scaler)
        validation_future, validation_future_mask = _future(self.validation, scaler)
        model = JointJEPAForecaster(
            self.model_config,
            mode="ema" if slot.family == "ema_jepa" else "shared_sigreg",
            ema_regularizer_weight=(0.03, 0.1)[config],
            sigreg_weight=(0.04, 0.1)[config],
        ).to(self.device)
        control = slot.phase if slot.phase in ("random_encoder", "shuffled_future") else None
        pretrain_phase = None
        if control != "random_encoder":
            if resume_phase in ("probe", "pretrain_complete"):
                if not self.fixture_only:
                    pretrain_phase = self._recover_pretrain_phase(
                        model,
                        slot,
                        config,
                        output,
                        validation_tensors,
                        validation_future,
                        validation_future_mask,
                    )
            else:
                shuffle = (
                    _separated_shuffle(self.fit)
                    if control == "shuffled_future"
                    else np.arange(len(self.fit))
                )
                pretrain_phase = self._pretrain_phase(
                    model,
                    slot=slot,
                    configuration=config,
                    fit_tensors=fit_tensors,
                    validation_tensors=validation_tensors,
                    future=future,
                    future_mask=future_mask,
                    validation_future=validation_future,
                    validation_future_mask=validation_future_mask,
                    shuffle=shuffle,
                    output=output,
                    resume_checkpoint=resume_checkpoint if resume_phase == "pretrain" else None,
                )
        model.freeze_encoder()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=3e-4,
            weight_decay=1e-4,
            betas=(0.9, 0.95),
        )
        probe_phase, checkpoint = self._supervised_phase(
            model,
            optimizer=optimizer,
            fit_tensors=fit_tensors,
            validation_tensors=validation_tensors,
            scaler=scaler,
            output=output,
            slot=slot,
            configuration=config,
            seed=slot.seed,
            learning_rate=3e-4,
            name="probe",
            resume_checkpoint=resume_checkpoint if resume_phase == "probe" else None,
        )
        path = output / "validation-predictions.npz"
        _write_predictions(
            path, self.validation, self._predict_neural(model, scaler, validation_tensors)
        )
        (output / "endpoint.json").write_text(
            json.dumps({"checkpoint": checkpoint.name, "configuration": config, "seed": slot.seed}),
            encoding="utf-8",
        )
        phases = (
            ()
            if probe_phase is None
            else ((probe_phase,) if pretrain_phase is None else (pretrain_phase, probe_phase))
        )
        return SlotResult(
            path,
            updates=self.fixture_updates if probe_phase is None else probe_phase.updates,
            reusable_seed7=slot.phase == "development" and slot.seed == 7,
            training_phases=phases,
        )

    def _hybrid(self, slot: V2Slot, output: Path) -> SlotResult:
        if slot.family not in ("ema_jepa_plus_raw", "shared_sigreg_plus_raw"):
            raise ValueError("Unknown frozen hybrid campaign family.")
        parent = slot.family.removesuffix("_plus_raw")
        root = output.parent.parent
        ledger_path = root / "ledger.json"
        if not ledger_path.is_file():
            raise ValueError("Hybrid needs completed reviewed parent campaign ledger.")
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        run = ledger["runs"].get(f"{parent}-seed7", {})
        config = ledger.get("selected_configs", {}).get(parent)
        if not run.get("status", "").startswith(("COMPLETED", "REUSED")) or config not in (0, 1):
            raise ValueError("Hybrid needs selected seed-7 parent endpoint.")
        artifact_root = Path(run["artifact_root"]).resolve(strict=True)
        endpoint = json.loads((artifact_root / "endpoint.json").read_text(encoding="utf-8"))
        checkpoint = (artifact_root / endpoint["checkpoint"]).resolve(strict=True)
        if not checkpoint.is_relative_to(artifact_root):
            raise ValueError("Hybrid parent checkpoint escapes its reviewed artifact.")
        if not self.fixture_only:
            phases = run.get("training_phases", [])
            if not phases or phases[-1].get("name") != "probe":
                raise ValueError("Hybrid parent lacks a supervised probe phase.")
            selected_milestone = _selected_milestone(phases[-1])
            if (
                selected_milestone["sha256"] != _sha256(checkpoint)
                or (artifact_root / selected_milestone["path"]).resolve(strict=True) != checkpoint
            ):
                raise ValueError(
                    "Hybrid parent checkpoint differs from best reviewed probe milestone."
                )
        model = JointJEPAForecaster(
            self.model_config,
            mode="ema" if parent == "ema_jepa" else "shared_sigreg",
            ema_regularizer_weight=(0.03, 0.1)[config],
            sigreg_weight=(0.04, 0.1)[config],
        ).to(self.device)
        state = torch.load(checkpoint, map_location=self.device, weights_only=True)
        if state.get("protocol_sha256") != self.protocol_sha256:
            raise ValueError("Hybrid parent checkpoint protocol differs.")
        model.load_state_dict(state["model"])
        model.freeze_encoder()
        _, fit_tensors, validation_tensors = self._scaler_tensors()
        xfit = _representation_features(
            model, self.fit, fit_tensors, batch_size=self.batch_size, device=self.device
        )
        xvalidation = _representation_features(
            model,
            self.validation,
            validation_tensors,
            batch_size=self.batch_size,
            device=self.device,
        )
        hybrid = NativeHybridRidge().fit(
            xfit,
            np.stack([row.target_db for row in self.fit]),
            np.stack([row.target_mask for row in self.fit]),
            np.stack([row.target_detection_fraction for row in self.fit]),
            np.stack([row.target_detection_mask for row in self.fit]),
            partition="train",
        )
        path = output / "validation-predictions.npz"
        _write_predictions(path, self.validation, hybrid.predict(xvalidation))
        hybrid.save(output / "model.npz")
        return SlotResult(path, updates=0)
