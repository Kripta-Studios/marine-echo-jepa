"""Campaign callbacks perform real updates on fixtures and reject unreviewed D1 cohorts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from marine_echo.models.compact import ModelConfig
from marine_echo.models.v2_development import JointDirectForecaster, JointJEPAForecaster
from marine_echo.training import v2_campaign_backend as backend_module
from marine_echo.training.v2_campaign import execute_v2_campaign, v2_plan
from marine_echo.training.v2_campaign_backend import NativeCampaignBackend
from marine_echo.training.v2_representation import _future
from marine_echo.training.v2_stream import HourlyWindow


def _rows(first_day: str, partition: str) -> list[HourlyWindow]:
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


@pytest.mark.parametrize("family", ["persistence", "seasonal", "ridge", "hist_gradient_boosting"])
def test_backend_executes_all_conventional_slots_on_fixture(family: str, tmp_path: Path) -> None:
    fit = _rows("2020-02-18", "train")
    validation = _rows("2020-05-28", "validation")
    backend = NativeCampaignBackend(fit, validation, fixture_only=True, tree_max_iter=2)
    slot = next(slot for slot in v2_plan() if slot.family == family)
    run_dir = tmp_path / family
    run_dir.mkdir()
    result = backend(slot, None, run_dir)
    assert result.updates == 0
    assert json.loads((run_dir / "resources.json").read_text())["peak_process_rss_bytes"] > 0
    with np.load(result.predictions, allow_pickle=False) as saved:
        assert saved["row_ids"].shape == (8,)
        assert saved["quantiles_db"].shape == (8, 3, 5)


@pytest.mark.parametrize("family", ["direct", "ema_jepa", "shared_sigreg"])
def test_backend_updates_neural_family_on_fixture(family: str, tmp_path: Path) -> None:
    torch.set_num_threads(1)
    fit = _rows("2020-02-18", "train")
    validation = _rows("2020-05-28", "validation")
    backend = NativeCampaignBackend(
        fit,
        validation,
        fixture_only=True,
        fixture_updates=2,
        batch_size=4,
        model_config=ModelConfig(width=16, layers=1, heads=4),
    )
    slot = next(slot for slot in v2_plan() if slot.run_id == f"{family}-development0-seed7")
    run_dir = tmp_path / family
    run_dir.mkdir()
    result = backend(slot, 0, run_dir)
    assert result.updates == 2
    assert result.predictions.is_file()
    assert json.loads((run_dir / "resources.json").read_text())["wall_seconds"] > 0
    with np.load(result.predictions, allow_pickle=False) as saved:
        assert saved["row_ids"].shape == (8,)


def test_backend_rejects_real_cohort_without_review() -> None:
    fit = _rows("2020-02-18", "train")
    validation = _rows("2020-05-28", "validation")
    with pytest.raises(ValueError, match="review"):
        NativeCampaignBackend(fit, validation)


def test_backend_rejects_failed_native_support_before_any_fit(tmp_path: Path) -> None:
    fit = _rows("2020-02-18", "train")
    validation = _rows("2020-05-28", "validation")
    support = tmp_path / "support.json"
    support.write_text(
        json.dumps({"status": "NATIVE_CANDIDATE_INELIGIBLE_STOP_D1_TARGET_SEARCH"}),
        encoding="utf-8",
    )
    review = tmp_path / "review.json"
    review.write_text("{}", encoding="utf-8")
    fit_rows = tmp_path / "fit-rows.json"
    fit_rows.write_text("[]", encoding="utf-8")
    validation_rows = tmp_path / "validation-rows.json"
    validation_rows.write_text("[]", encoding="utf-8")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="ineligible"):
        NativeCampaignBackend(
            fit,
            validation,
            cohort_review_path=review,
            cohort_review_sha256=digest(review),
            support_report_path=support,
            support_report_sha256=digest(support),
            fit_support_rows_path=fit_rows,
            fit_support_rows_sha256=digest(fit_rows),
            validation_support_rows_path=validation_rows,
            validation_support_rows_sha256=digest(validation_rows),
            native_index_sha256="a" * 64,
        )


def test_backend_runs_all_25_fixture_slots_including_controls_and_hybrids(
    tmp_path: Path,
) -> None:
    torch.set_num_threads(1)
    fit = _rows("2020-02-18", "train")
    validation = _rows("2020-05-28", "validation")
    backend = NativeCampaignBackend(
        fit,
        validation,
        fixture_only=True,
        fixture_updates=1,
        batch_size=4,
        model_config=ModelConfig(width=16, layers=1, heads=4),
        tree_max_iter=2,
    )
    ledger = execute_v2_campaign(
        tmp_path / "campaign",
        validation_rows=validation,
        executor=backend,
        protocol_sha256="b" * 64,
        fixture_only=True,
    )
    assert ledger["completed_slots"] == 25
    assert ledger["status"] == "COMPLETE_SYNTHETIC_FIXTURE"
    assert ledger["runs"]["ema_jepa_plus_raw-seed7"]["status"] == "COMPLETED_SYNTHETIC_FIXTURE"
    assert ledger["runs"]["shared_sigreg_plus_raw-seed7"]["status"] == "COMPLETED_SYNTHETIC_FIXTURE"
    model_path = tmp_path / "campaign/runs/persistence-seed7/model.joblib"
    with model_path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="artifacts differ"):
        execute_v2_campaign(
            tmp_path / "campaign",
            validation_rows=validation,
            executor=backend,
            protocol_sha256="b" * 64,
            fixture_only=True,
        )


def test_backend_direct_prediction_ignores_validation_future_truth(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    fit = _rows("2020-02-18", "train")
    validation = _rows("2020-05-28", "validation")
    changed = [
        replace(
            row,
            target_db=np.full(3, 100.0),
            future_train_db=np.full_like(row.future_train_db, 100.0),
        )
        for row in validation
    ]
    slot = next(slot for slot in v2_plan() if slot.run_id == "direct-development0-seed7")
    outputs = []
    for index, rows in enumerate((validation, changed)):
        backend = NativeCampaignBackend(
            fit,
            rows,
            fixture_only=True,
            fixture_updates=1,
            batch_size=4,
            model_config=ModelConfig(width=16, layers=1, heads=4),
        )
        output = tmp_path / str(index)
        output.mkdir()
        result = backend(slot, 0, output)
        with np.load(result.predictions, allow_pickle=False) as saved:
            outputs.append((saved["quantiles_db"], saved["detection_fraction"]))
    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])


@pytest.mark.parametrize("family", ["direct", "ema_jepa", "shared_sigreg"])
def test_campaign_phase_resume_matches_uninterrupted_updates(family: str, tmp_path: Path) -> None:
    torch.set_num_threads(1)
    backend = NativeCampaignBackend(
        _rows("2020-02-18", "train"),
        _rows("2020-05-28", "validation"),
        fixture_only=True,
        fixture_updates=2,
        batch_size=4,
        model_config=ModelConfig(width=16, layers=1, heads=4),
    )
    backend._peak_rss = 0
    slot = next(slot for slot in v2_plan() if slot.run_id == f"{family}-development0-seed7")
    scaler, fit_tensors, validation_tensors = backend._scaler_tensors()
    uninterrupted_dir = tmp_path / "uninterrupted"
    resumed_dir = tmp_path / "resumed"
    uninterrupted_dir.mkdir()
    resumed_dir.mkdir()

    def make_model() -> tuple[JointDirectForecaster | JointJEPAForecaster, torch.optim.Optimizer]:
        torch.manual_seed(7)
        if family == "direct":
            model = JointDirectForecaster(backend.model_config)
        else:
            model = JointJEPAForecaster(
                backend.model_config,
                mode="ema" if family == "ema_jepa" else "shared_sigreg",
            )
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        return model, optimizer

    def phase(
        model: JointDirectForecaster | JointJEPAForecaster,
        optimizer: torch.optim.Optimizer,
        output: Path,
        resume_checkpoint: Path | None = None,
    ) -> Path:
        _, checkpoint = backend._supervised_phase(
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
            resume_checkpoint=resume_checkpoint,
        )
        return checkpoint

    uninterrupted, uninterrupted_optimizer = make_model()
    phase(uninterrupted, uninterrupted_optimizer, uninterrupted_dir)
    first, first_optimizer = make_model()
    backend.fixture_updates = 1
    midpoint = phase(first, first_optimizer, resumed_dir)
    backend.fixture_updates = 2
    resumed, resumed_optimizer = make_model()
    phase(resumed, resumed_optimizer, resumed_dir, midpoint)
    for name, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)


def test_campaign_resume_rejects_wrong_slot_identity(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    backend = NativeCampaignBackend(
        _rows("2020-02-18", "train"),
        _rows("2020-05-28", "validation"),
        fixture_only=True,
        fixture_updates=1,
        batch_size=4,
        model_config=ModelConfig(width=16, layers=1, heads=4),
    )
    backend._peak_rss = 0
    scaler, fit_tensors, validation_tensors = backend._scaler_tensors()
    slot = next(slot for slot in v2_plan() if slot.run_id == "direct-development0-seed7")
    output = tmp_path / "run"
    output.mkdir()
    torch.manual_seed(7)
    model = JointDirectForecaster(backend.model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    _, checkpoint = backend._supervised_phase(
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
    backend.fixture_updates = 2
    wrong_slot = replace(slot, run_id="direct-development1-seed7", configuration=1)
    with pytest.raises(ValueError, match="identity"):
        backend._supervised_phase(
            model,
            optimizer=optimizer,
            fit_tensors=fit_tensors,
            validation_tensors=validation_tensors,
            scaler=scaler,
            output=output,
            slot=wrong_slot,
            configuration=1,
            seed=7,
            learning_rate=3e-4,
            name="supervised",
            resume_checkpoint=checkpoint,
        )


@pytest.mark.parametrize("family", ["ema_jepa", "shared_sigreg"])
def test_campaign_pretraining_resume_matches_uninterrupted_updates(
    family: str, tmp_path: Path
) -> None:
    torch.set_num_threads(1)
    backend = NativeCampaignBackend(
        _rows("2020-02-18", "train"),
        _rows("2020-05-28", "validation"),
        fixture_only=True,
        fixture_updates=2,
        batch_size=4,
        model_config=ModelConfig(width=16, layers=1, heads=4),
    )
    backend._peak_rss = 0
    slot = next(slot for slot in v2_plan() if slot.run_id == f"{family}-development0-seed7")
    scaler, fit_tensors, validation_tensors = backend._scaler_tensors()
    future, future_mask = _future(backend.fit, scaler)
    validation_future, validation_future_mask = _future(backend.validation, scaler)
    uninterrupted_dir = tmp_path / "uninterrupted"
    resumed_dir = tmp_path / "resumed"
    uninterrupted_dir.mkdir()
    resumed_dir.mkdir()

    def make_model() -> JointJEPAForecaster:
        torch.manual_seed(7)
        return JointJEPAForecaster(
            backend.model_config,
            mode="ema" if family == "ema_jepa" else "shared_sigreg",
        )

    def phase(
        model: JointJEPAForecaster, output: Path, resume_checkpoint: Path | None = None
    ) -> None:
        backend._pretrain_phase(
            model,
            slot=slot,
            configuration=0,
            fit_tensors=fit_tensors,
            validation_tensors=validation_tensors,
            future=future,
            future_mask=future_mask,
            validation_future=validation_future,
            validation_future_mask=validation_future_mask,
            shuffle=np.arange(len(backend.fit)),
            output=output,
            resume_checkpoint=resume_checkpoint,
        )

    uninterrupted = make_model()
    phase(uninterrupted, uninterrupted_dir)
    first = make_model()
    backend.fixture_updates = 1
    phase(first, resumed_dir)
    backend.fixture_updates = 2
    resumed = make_model()
    phase(resumed, resumed_dir, resumed_dir / "pretrain-checkpoint-1.pt")
    for name, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)


def test_synthetic_campaign_reloads_real_250_update_milestone(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    backend = NativeCampaignBackend(
        _rows("2020-02-18", "train"),
        _rows("2020-05-28", "validation"),
        fixture_only=True,
        batch_size=4,
        model_config=ModelConfig(width=16, layers=1, heads=4),
    )
    # Exercise the production 250-step cadence only on synthetic rows.
    backend.fixture_only = False
    backend._peak_rss = 0
    slot = next(slot for slot in v2_plan() if slot.run_id == "direct-development0-seed7")
    scaler, fit_tensors, validation_tensors = backend._scaler_tensors()
    output = tmp_path / "interrupted"
    output.mkdir()

    def make_model() -> tuple[JointDirectForecaster, torch.optim.Optimizer]:
        torch.manual_seed(7)
        model = JointDirectForecaster(backend.model_config)
        return model, torch.optim.AdamW(model.parameters(), lr=3e-4)

    def phase(
        model: JointDirectForecaster,
        optimizer: torch.optim.Optimizer,
        resume_checkpoint: Path | None = None,
    ) -> None:
        backend._supervised_phase(
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
            resume_checkpoint=resume_checkpoint,
        )

    class PlannedInterruption(Exception):
        pass

    calls = 0

    def interrupt_after_251(_: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 251:
            raise PlannedInterruption

    backend._resource_guard = interrupt_after_251  # type: ignore[method-assign]
    uninterrupted, optimizer = make_model()
    with pytest.raises(PlannedInterruption):
        phase(uninterrupted, optimizer)
    checkpoint = output / "supervised-checkpoint-250.pt"
    assert checkpoint.is_file()
    assert (output / "supervised-validation-250.npz").is_file()
    calls = 250
    resumed, resumed_optimizer = make_model()
    with pytest.raises(PlannedInterruption):
        phase(resumed, resumed_optimizer, checkpoint)
    for name, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("family", ["ema_jepa", "shared_sigreg"])
def test_synthetic_pretraining_reloads_real_250_update_milestone(
    family: str, tmp_path: Path
) -> None:
    torch.set_num_threads(1)
    backend = NativeCampaignBackend(
        _rows("2020-02-18", "train"),
        _rows("2020-05-28", "validation"),
        fixture_only=True,
        batch_size=4,
        model_config=ModelConfig(width=16, layers=1, heads=4),
    )
    backend.fixture_only = False
    backend._peak_rss = 0
    slot = next(slot for slot in v2_plan() if slot.run_id == f"{family}-development0-seed7")
    scaler, fit_tensors, validation_tensors = backend._scaler_tensors()
    future, future_mask = _future(backend.fit, scaler)
    validation_future, validation_future_mask = _future(backend.validation, scaler)
    output = tmp_path / "interrupted"
    output.mkdir()

    def make_model() -> JointJEPAForecaster:
        torch.manual_seed(7)
        return JointJEPAForecaster(
            backend.model_config,
            mode="ema" if family == "ema_jepa" else "shared_sigreg",
        )

    def phase(model: JointJEPAForecaster, resume_checkpoint: Path | None = None) -> None:
        backend._pretrain_phase(
            model,
            slot=slot,
            configuration=0,
            fit_tensors=fit_tensors,
            validation_tensors=validation_tensors,
            future=future,
            future_mask=future_mask,
            validation_future=validation_future,
            validation_future_mask=validation_future_mask,
            shuffle=np.arange(len(backend.fit)),
            output=output,
            resume_checkpoint=resume_checkpoint,
        )

    class PlannedInterruption(Exception):
        pass

    calls = 0

    def interrupt_after_251(_: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 251:
            raise PlannedInterruption

    backend._resource_guard = interrupt_after_251  # type: ignore[method-assign]
    uninterrupted = make_model()
    with pytest.raises(PlannedInterruption):
        phase(uninterrupted)
    checkpoint = output / "pretrain-checkpoint-250.pt"
    assert checkpoint.is_file()
    assert (output / "pretrain-validation-250.json").is_file()
    calls = 250
    resumed = make_model()
    with pytest.raises(PlannedInterruption):
        phase(resumed, checkpoint)
    for name, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)


def test_synthetic_supervised_early_stop_restores_best_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.set_num_threads(1)
    backend = NativeCampaignBackend(
        _rows("2020-02-18", "train"),
        _rows("2020-05-28", "validation"),
        fixture_only=True,
        batch_size=4,
        model_config=ModelConfig(width=16, layers=1, heads=4),
    )
    backend.fixture_only = False
    backend._peak_rss = 0
    scores = iter((0.1, 0.2, 0.3, 0.4, 0.5))
    monkeypatch.setattr(backend_module, "_primary", lambda _: next(scores))
    scaler, fit_tensors, validation_tensors = backend._scaler_tensors()
    slot = next(slot for slot in v2_plan() if slot.run_id == "direct-development0-seed7")
    output = tmp_path / "direct"
    output.mkdir()
    torch.manual_seed(7)
    model = JointDirectForecaster(backend.model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    phase, selected = backend._supervised_phase(
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
    assert phase is not None and phase.updates == 1250
    assert selected.name == "supervised-checkpoint-250.pt"
    best_state = torch.load(selected, map_location="cpu", weights_only=True)["model"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, best_state[name], rtol=0, atol=0)


def test_synthetic_pretraining_early_stop_restores_best_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.set_num_threads(1)
    backend = NativeCampaignBackend(
        _rows("2020-02-18", "train"),
        _rows("2020-05-28", "validation"),
        fixture_only=True,
        batch_size=4,
        model_config=ModelConfig(width=16, layers=1, heads=4),
    )
    backend.fixture_only = False
    backend._peak_rss = 0
    scores = iter((0.1, 0.2, 0.3, 0.4, 0.5))
    monkeypatch.setattr(backend, "_pretrain_validation_loss", lambda *_: next(scores))
    scaler, fit_tensors, validation_tensors = backend._scaler_tensors()
    future, future_mask = _future(backend.fit, scaler)
    validation_future, validation_future_mask = _future(backend.validation, scaler)
    slot = next(slot for slot in v2_plan() if slot.run_id == "ema_jepa-development0-seed7")
    output = tmp_path / "pretrain"
    output.mkdir()
    torch.manual_seed(7)
    model = JointJEPAForecaster(backend.model_config, mode="ema")
    phase = backend._pretrain_phase(
        model,
        slot=slot,
        configuration=0,
        fit_tensors=fit_tensors,
        validation_tensors=validation_tensors,
        future=future,
        future_mask=future_mask,
        validation_future=validation_future,
        validation_future_mask=validation_future_mask,
        shuffle=np.arange(len(backend.fit)),
        output=output,
    )
    assert phase is not None and phase.updates == 1250
    selected = output / "pretrain-checkpoint-250.pt"
    best_state = torch.load(selected, map_location="cpu", weights_only=True)["model"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, best_state[name], rtol=0, atol=0)
