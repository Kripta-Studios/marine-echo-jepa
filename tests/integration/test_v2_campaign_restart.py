"""Reviewed interrupted-slot restart uses the original staging directory and state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from marine_echo.models.compact import ModelConfig
from marine_echo.models.v2_development import (
    JointDirectForecaster,
    JointJEPAForecaster,
    JointPrediction,
)
from marine_echo.training.loop import load_checkpoint, save_checkpoint
from marine_echo.training.v2_campaign import (
    SlotResult,
    execute_v2_campaign,
    interrupted_slot_review_request,
    v2_plan,
)
from marine_echo.training.v2_campaign_backend import NativeCampaignBackend
from marine_echo.training.v2_executor import _write_predictions
from marine_echo.training.v2_representation import _future
from marine_echo.training.v2_stream import HourlyWindow


def _validation_row() -> HourlyWindow:
    cutoff = np.datetime64("2020-06-01T00:00")
    return HourlyWindow(
        row_id="fixture-validation-1",
        partition="validation",
        cutoff=cutoff,
        context=np.full((96, 4, 64), np.nan),
        context_mask=np.zeros((96, 4, 64), dtype=bool),
        context_acquisition_fraction=np.ones(96),
        context_detection_fraction=np.full(96, 0.2),
        context_age_minutes=np.arange(96, dtype=float),
        target_interval_start=np.array([cutoff + np.timedelta64(h, "h") for h in (0, 2, 5)]),
        target_interval_end=np.array([cutoff + np.timedelta64(h, "h") for h in (1, 3, 6)]),
        target_db=np.full(3, -90.0),
        target_mask=np.ones(3, dtype=bool),
        target_detection_fraction=np.full(3, 0.2),
        target_detection_mask=np.ones(3, dtype=bool),
        target_acquisition_fraction=np.ones(3),
        future_train_db=np.full((3, 4, 4, 64), np.nan),
        future_train_mask=np.zeros((3, 4, 4, 64), dtype=bool),
        source_sha256=("a" * 64,),
    )


def _native_rows(first_day: str, partition: str) -> list[HourlyWindow]:
    rows = []
    for index in range(8):
        cutoff = np.datetime64(first_day) + index * np.timedelta64(2, "D")
        context = np.full((96, 4, 64), np.nan)
        context[:, 0, 5:50] = -90 + index
        future = np.full((3, 4, 4, 64), np.nan)
        future[:, :, 0, 5:50] = -88 + index
        rows.append(
            HourlyWindow(
                row_id=f"{first_day}-{index}",
                partition=partition,
                cutoff=cutoff,
                context=context,
                context_mask=np.isfinite(context),
                context_acquisition_fraction=np.full(96, 0.7),
                context_detection_fraction=np.full(96, 0.4),
                context_age_minutes=np.arange(95, -1, -1) * 15.0,
                target_interval_start=np.array(
                    [cutoff + np.timedelta64(h, "h") for h in (0, 2, 5)]
                ),
                target_interval_end=np.array([cutoff + np.timedelta64(h, "h") for h in (1, 3, 6)]),
                target_db=np.full(3, -88 + index, dtype=float),
                target_mask=np.ones(3, dtype=bool),
                target_detection_fraction=np.full(3, 0.4),
                target_detection_mask=np.ones(3, dtype=bool),
                target_acquisition_fraction=np.full(3, 0.7),
                future_train_db=future,
                future_train_mask=np.isfinite(future),
                source_sha256=("a" * 64,),
                context_index_db=np.full(96, -90 + index, dtype=float),
                past_source_sha256=("a" * 64,),
                target_source_sha256=("a" * 64,),
            )
        )
    return rows


class PlannedInterruption(Exception):
    pass


class InterruptedLearnedFixture:
    def __init__(self, rows: list, protocol_sha256: str) -> None:
        self.rows = rows
        self.protocol_sha256 = protocol_sha256
        self.started = 0
        self.resumed = 0
        self.final_weights: dict[str, torch.Tensor] | None = None

    @staticmethod
    def _model() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
        torch.manual_seed(7)
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 4), torch.nn.Dropout(0.3), torch.nn.Linear(4, 1)
        )
        return model, torch.optim.AdamW(model.parameters(), lr=3e-4)

    @staticmethod
    def _update(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        optimizer.zero_grad(set_to_none=True)
        loss = (model(torch.ones(4, 2)) - 0.25).square().mean()
        loss.backward()
        optimizer.step()

    def __call__(self, slot, selected_config, output):  # type: ignore[no-untyped-def]
        if slot.run_id == "direct-development0-seed7":
            self.started += 1
            model, optimizer = self._model()
            for _ in range(250):
                self._update(model, optimizer)
            save_checkpoint(
                output / "supervised-checkpoint-250.pt",
                model,
                optimizer,
                step=250,
                protocol_sha256=self.protocol_sha256,
            )
            raise PlannedInterruption
        path = output / "validation-predictions.npz"
        _write_predictions(
            path,
            self.rows,
            JointPrediction(np.full((1, 3, 5), -90.0), np.full((1, 3), 0.2)),
        )
        return SlotResult(path, updates=0)

    def resume_slot(self, slot, selected_config, output, review):  # type: ignore[no-untyped-def]
        self.resumed += 1
        assert slot.run_id == "direct-development0-seed7"
        assert selected_config == 0
        model, optimizer = self._model()
        checkpoint = output / review["resume_checkpoint"]["path"]
        assert (
            load_checkpoint(checkpoint, model, optimizer, protocol_sha256=self.protocol_sha256)
            == 250
        )
        self._update(model, optimizer)
        self.final_weights = {name: value.clone() for name, value in model.state_dict().items()}
        path = output / "validation-predictions.npz"
        _write_predictions(
            path,
            self.rows,
            JointPrediction(np.full((1, 3, 5), -90.0), np.full((1, 3), 0.2)),
        )
        return SlotResult(path, updates=251)


@pytest.mark.parametrize("tamper", [False, True])
def test_reviewed_interrupted_learned_slot_resumes_once_from_250(
    tmp_path: Path, tamper: bool
) -> None:
    torch.set_num_threads(1)
    rows = [_validation_row()]
    protocol = "b" * 64
    backend = InterruptedLearnedFixture(rows, protocol)
    root = tmp_path / "campaign"
    with pytest.raises(PlannedInterruption):
        execute_v2_campaign(
            root,
            validation_rows=rows,
            executor=backend,
            protocol_sha256=protocol,
            fixture_only=True,
            max_slots=5,
        )
    request = interrupted_slot_review_request(root, "direct-development0-seed7")
    review = {
        **request,
        "status": "APPROVED_V2_INTERRUPTED_SLOT_RESTART",
        "reviewer_role": "independent_reviewer",
        "reviewer_session": "distinct-reviewer-fixture-session",
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review, sort_keys=True), encoding="utf-8")
    review_sha256 = hashlib.sha256(review_path.read_bytes()).hexdigest()
    if tamper:
        checkpoint = Path(request["staging"]) / request["resume_checkpoint"]["path"]
        with checkpoint.open("ab") as stream:
            stream.write(b"tampered")
        with pytest.raises(ValueError, match="staging|checkpoint|review"):
            execute_v2_campaign(
                root,
                validation_rows=rows,
                executor=backend,
                protocol_sha256=protocol,
                fixture_only=True,
                max_slots=1,
                restart_review_path=review_path,
                restart_review_sha256=review_sha256,
            )
        assert backend.resumed == 0
        return
    ledger_path = root / "ledger.json"
    running = json.loads(ledger_path.read_text(encoding="utf-8"))
    running["runs"]["direct-development0-seed7"]["status"] = "RUNNING"
    ledger_path.write_text(json.dumps(running), encoding="utf-8")
    with pytest.raises(ValueError, match="still running"):
        execute_v2_campaign(
            root,
            validation_rows=rows,
            executor=backend,
            protocol_sha256=protocol,
            fixture_only=True,
            max_slots=1,
            restart_review_path=review_path,
            restart_review_sha256=review_sha256,
        )
    running["runs"]["direct-development0-seed7"]["status"] = "FAILED"
    ledger_path.write_text(json.dumps(running), encoding="utf-8")
    ledger = execute_v2_campaign(
        root,
        validation_rows=rows,
        executor=backend,
        protocol_sha256=protocol,
        fixture_only=True,
        max_slots=1,
        restart_review_path=review_path,
        restart_review_sha256=review_sha256,
    )
    assert backend.started == backend.resumed == 1
    assert ledger["completed_slots"] == 5
    assert ledger["runs"]["direct-development0-seed7"]["status"] == "COMPLETED_SYNTHETIC_FIXTURE"
    assert (root / "runs/direct-development0-seed7/supervised-checkpoint-250.pt").is_file()
    assert not list((root / "runs").glob("direct-development0-seed7.stage.*"))
    reference, reference_optimizer = backend._model()
    for _ in range(251):
        backend._update(reference, reference_optimizer)
    assert backend.final_weights is not None
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(value, backend.final_weights[name], rtol=0, atol=0)


def test_native_direct_callback_finishes_reviewed_fixture_interruption(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    fit = _native_rows("2020-02-18", "train")
    validation = _native_rows("2020-05-28", "validation")
    protocol = "b" * 64

    class InterruptedNativeDirect(NativeCampaignBackend):
        def __call__(self, slot, selected_config, output):  # type: ignore[no-untyped-def]
            if slot.run_id != "direct-development0-seed7":
                return super().__call__(slot, selected_config, output)
            self.fixture_updates = 1
            self._peak_rss = 0
            torch.manual_seed(slot.seed)
            scaler, fit_tensors, validation_tensors = self._scaler_tensors()
            model = JointDirectForecaster(self.model_config)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=3e-4, weight_decay=1e-4, betas=(0.9, 0.95)
            )
            self._supervised_phase(
                model,
                optimizer=optimizer,
                fit_tensors=fit_tensors,
                validation_tensors=validation_tensors,
                scaler=scaler,
                output=output,
                slot=slot,
                configuration=0,
                seed=7,
                learning_rate=3e-4,
                name="supervised",
            )
            raise PlannedInterruption

    backend = InterruptedNativeDirect(
        fit,
        validation,
        fixture_only=True,
        protocol_sha256=protocol,
        fixture_updates=1,
        batch_size=4,
        tree_max_iter=2,
        model_config=ModelConfig(width=16, layers=1, heads=4),
    )
    root = tmp_path / "native-campaign"
    with pytest.raises(PlannedInterruption):
        execute_v2_campaign(
            root,
            validation_rows=validation,
            executor=backend,
            protocol_sha256=protocol,
            fixture_only=True,
            max_slots=5,
        )
    request = interrupted_slot_review_request(root, "direct-development0-seed7")
    assert request["resume_checkpoint"]["step"] == 1
    review = {
        **request,
        "status": "APPROVED_V2_INTERRUPTED_SLOT_RESTART",
        "reviewer_role": "independent_reviewer",
        "reviewer_session": "fixture-reviewer",
    }
    review_path = tmp_path / "native-review.json"
    review_path.write_text(json.dumps(review, sort_keys=True), encoding="utf-8")
    backend.fixture_updates = 2
    ledger = execute_v2_campaign(
        root,
        validation_rows=validation,
        executor=backend,
        protocol_sha256=protocol,
        fixture_only=True,
        max_slots=1,
        restart_review_path=review_path,
        restart_review_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest(),
    )
    assert ledger["runs"]["direct-development0-seed7"]["status"] == "COMPLETED_SYNTHETIC_FIXTURE"
    checkpoint = root / "runs/direct-development0-seed7/supervised-checkpoint-2.pt"
    resumed_state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    reference = NativeCampaignBackend(
        fit,
        validation,
        fixture_only=True,
        protocol_sha256=protocol,
        fixture_updates=2,
        batch_size=4,
        tree_max_iter=2,
        model_config=ModelConfig(width=16, layers=1, heads=4),
    )
    slot = next(item for item in v2_plan() if item.run_id == "direct-development0-seed7")
    reference_output = tmp_path / "uninterrupted"
    reference_output.mkdir()
    reference(slot, 0, reference_output)
    reference_state = torch.load(
        reference_output / "supervised-checkpoint-2.pt", map_location="cpu", weights_only=True
    )["model"]
    for name, value in reference_state.items():
        torch.testing.assert_close(value, resumed_state[name], rtol=0, atol=0)


def test_native_ema_pretraining_finishes_reviewed_fixture_interruption(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    fit = _native_rows("2020-02-18", "train")
    validation = _native_rows("2020-05-28", "validation")
    protocol = "b" * 64

    class InterruptedNativeEMA(NativeCampaignBackend):
        def __call__(self, slot, selected_config, output):  # type: ignore[no-untyped-def]
            if slot.run_id != "ema_jepa-development0-seed7":
                return super().__call__(slot, selected_config, output)
            self.fixture_updates = 1
            self._peak_rss = 0
            torch.manual_seed(slot.seed)
            scaler, fit_tensors, validation_tensors = self._scaler_tensors()
            future, future_mask = _future(self.fit, scaler)
            validation_future, validation_future_mask = _future(self.validation, scaler)
            model = JointJEPAForecaster(
                self.model_config,
                mode="ema",
                ema_regularizer_weight=0.03,
                sigreg_weight=0.04,
            )
            self._pretrain_phase(
                model,
                slot=slot,
                configuration=0,
                fit_tensors=fit_tensors,
                validation_tensors=validation_tensors,
                future=future,
                future_mask=future_mask,
                validation_future=validation_future,
                validation_future_mask=validation_future_mask,
                shuffle=np.arange(len(self.fit)),
                output=output,
            )
            raise PlannedInterruption

    config = ModelConfig(width=16, layers=1, heads=4)
    backend = InterruptedNativeEMA(
        fit,
        validation,
        fixture_only=True,
        protocol_sha256=protocol,
        fixture_updates=1,
        batch_size=4,
        tree_max_iter=2,
        model_config=config,
    )
    root = tmp_path / "ema-campaign"
    with pytest.raises(PlannedInterruption):
        execute_v2_campaign(
            root,
            validation_rows=validation,
            executor=backend,
            protocol_sha256=protocol,
            fixture_only=True,
            max_slots=7,
        )
    request = interrupted_slot_review_request(root, "ema_jepa-development0-seed7")
    assert request["resume_phase"] == "pretrain"
    review = {
        **request,
        "status": "APPROVED_V2_INTERRUPTED_SLOT_RESTART",
        "reviewer_role": "independent_reviewer",
        "reviewer_session": "fixture-reviewer",
    }
    review_path = tmp_path / "ema-review.json"
    review_path.write_text(json.dumps(review, sort_keys=True), encoding="utf-8")
    backend.fixture_updates = 2
    ledger = execute_v2_campaign(
        root,
        validation_rows=validation,
        executor=backend,
        protocol_sha256=protocol,
        fixture_only=True,
        max_slots=1,
        restart_review_path=review_path,
        restart_review_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest(),
    )
    assert ledger["runs"]["ema_jepa-development0-seed7"]["status"] == "COMPLETED_SYNTHETIC_FIXTURE"
    reference = NativeCampaignBackend(
        fit,
        validation,
        fixture_only=True,
        protocol_sha256=protocol,
        fixture_updates=2,
        batch_size=4,
        tree_max_iter=2,
        model_config=config,
    )
    slot = next(item for item in v2_plan() if item.run_id == "ema_jepa-development0-seed7")
    reference_output = tmp_path / "ema-uninterrupted"
    reference_output.mkdir()
    reference(slot, 0, reference_output)
    for name in ("pretrain-checkpoint-2.pt", "probe-checkpoint-2.pt"):
        resumed_state = torch.load(
            root / "runs/ema_jepa-development0-seed7" / name,
            map_location="cpu",
            weights_only=True,
        )["model"]
        reference_state = torch.load(
            reference_output / name, map_location="cpu", weights_only=True
        )["model"]
        for key, value in reference_state.items():
            torch.testing.assert_close(value, resumed_state[key], rtol=0, atol=0)
