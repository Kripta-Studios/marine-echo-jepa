"""Fail-closed and causal checks for the real raw-response development path."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from marine_echo.models.compact import ModelConfig
from marine_echo.training import raw_response_development as development
from marine_echo.training.raw_response_cohort import (
    FROZEN_RUN_CONFIG,
    INDEX_SHA256,
    ROW_SHA256,
    SUPPORT_SHA256,
    RawCohort,
    file_sha256,
    load_raw_cohorts,
)
from marine_echo.training.raw_response_development import (
    _checked_batch_label_count,
    _features,
    _load_checkpoint,
    _save_checkpoint,
    _score,
)


def _cohort() -> RawCohort:
    context = np.ones((2, 96, 4, 64), dtype=np.float32)
    mask = np.ones_like(context, dtype=np.bool_)
    targets = np.array([[2.0, 3.0, 4.0], [3.0, np.nan, 5.0]])
    target_mask = np.array([[True, True, True], [True, False, True]])
    times = np.array(
        [
            ["2020-04-02T00", "2020-04-02T02", "2020-04-02T05"],
            ["2020-04-03T00", "2020-04-03T02", "2020-04-03T05"],
        ],
        dtype="datetime64[h]",
    )
    return RawCohort(
        "assessment",
        np.array(["2020-04-02", "2020-04-03"], dtype="datetime64[D]"),
        context,
        mask,
        targets,
        target_mask,
        times,
        "a" * 64,
        "b" * 64,
    )


def test_features_never_read_future_truth() -> None:
    cohort = _cohort()
    baseline = _features(cohort)
    changed = RawCohort(
        cohort.partition,
        cohort.cutoffs,
        cohort.context,
        cohort.context_mask,
        cohort.targets + 1000,
        cohort.target_mask,
        cohort.target_times,
        cohort.row_sha256,
        cohort.index_sha256,
    )
    np.testing.assert_array_equal(_features(changed), baseline)


def test_score_uses_only_reviewed_eligible_target_days() -> None:
    cohort = _cohort()
    predictions = np.broadcast_to(np.array([1.0, 1.5, 2.0, 2.5, 3.0]), (2, 3, 5)).copy()
    scored = _score(cohort, predictions)
    assert scored["horizons"]["3"]["eligible_rows"] == 1
    assert scored["horizons"]["3"]["issued_but_unscored"] == 1
    assert scored["horizons"]["3"]["target_days"] == 1
    assert scored["horizons"]["1"]["target_days"] == 2


def test_reader_rejects_missing_independent_prefit_review(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_cohorts(tmp_path, review_path=tmp_path / "no-review.json")


def test_prefit_gate_rejects_changed_window_before_shard_read(tmp_path) -> None:
    root = Path(__file__).resolve().parents[2]
    files = {
        "window_code_sha256": "src/marine_echo/training/raw_response_windows.py",
        "cohort_code_sha256": "src/marine_echo/training/raw_response_cohort.py",
        "executor_code_sha256": "src/marine_echo/training/raw_response_development.py",
        "runner_code_sha256": "tools/v2_run_raw_development.py",
        "model_code_sha256": "src/marine_echo/models/compact.py",
    }
    hashes = {}
    for key, relative in files.items():
        copied = tmp_path / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes((root / relative).read_bytes())
        hashes[key] = file_sha256(copied)
    review = {
        "status": "APPROVED_RAW_RESPONSE_PREFIT",
        "study_id": "raw_response_development_v1",
        "reviewer_session": "/root/v2_reviewer",
        "index_sha256": INDEX_SHA256,
        "support_sha256": SUPPORT_SHA256,
        "row_sha256": ROW_SHA256,
        "run_config": FROZEN_RUN_CONFIG,
        **hashes,
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    window_path = tmp_path / files["window_code_sha256"]
    window_path.write_bytes(window_path.read_bytes() + b"\n# unreviewed change\n")
    with pytest.raises(ValueError, match="pre-fit approval"):
        load_raw_cohorts(tmp_path, review_path=review_path)


def test_empty_label_batch_fails_before_optimizer_step() -> None:
    with pytest.raises(ValueError, match="all-unlabelled"):
        _checked_batch_label_count(torch.zeros((8, 3, 1), dtype=torch.bool))


def test_checkpoint_restores_optimizer_and_matches_uninterrupted_updates(tmp_path) -> None:
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[2.0], [4.0]])

    def update(current, current_optimizer, step):
        current_optimizer.zero_grad(set_to_none=True)
        loss = (current(x[step % 2 : step % 2 + 1]) - y[step % 2 : step % 2 + 1]).square().mean()
        loss.backward()
        current_optimizer.step()

    for step in range(2):
        update(model, optimizer, step)
    identity = {"study_id": "fixture", "full_graph_sha256": "a" * 64}
    path = tmp_path / "midpoint.pt"
    _save_checkpoint(path, model, optimizer, step=2, identity=identity)
    for step in range(2, 4):
        update(model, optimizer, step)
    torch.manual_seed(99)
    resumed = torch.nn.Linear(2, 1)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.001)
    with pytest.raises(ValueError, match="provenance"):
        _load_checkpoint(
            path, resumed, resumed_optimizer, step=2, identity={"study_id": "changed"}, device="cpu"
        )
    _load_checkpoint(path, resumed, resumed_optimizer, step=2, identity=identity, device="cpu")
    for step in range(2, 4):
        update(resumed, resumed_optimizer, step)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)


def test_run_request_is_single_namespace_and_device(tmp_path, monkeypatch) -> None:
    config = {**development.FROZEN_RUN_CONFIG, "output_path": "runs/first", "device": "cpu"}
    monkeypatch.setattr(development, "FROZEN_RUN_CONFIG", config)
    expected = tmp_path / "runs/first"
    development._validate_run_request(tmp_path, expected, "cpu")
    with pytest.raises(ValueError, match="frozen"):
        development._validate_run_request(tmp_path, tmp_path / "runs/second", "cpu")
    with pytest.raises(ValueError, match="frozen"):
        development._validate_run_request(tmp_path, expected, "cuda")

    cohort = _cohort()
    monkeypatch.setattr(development, "load_raw_cohorts", lambda root, review_path: (cohort, cohort))
    monkeypatch.setattr(
        development,
        "_direct",
        lambda fit, assess, output, root, device, resume: (
            np.broadcast_to(np.array([1.0, 1.5, 2.0, 2.5, 3.0]), (2, 3, 5)).copy(),
            {"resume_equivalent": True},
        ),
    )
    review_path = tmp_path / "review.json"
    review_path.write_text("{}", encoding="utf-8")
    development.execute(tmp_path, expected, review_path, device="cpu")
    with pytest.raises(FileExistsError):
        development.execute(tmp_path, expected, review_path, device="cpu")


def test_operational_midpoint_resume_and_final_checkpoint_recovery(tmp_path, monkeypatch) -> None:
    device = "cuda" if os.getenv("MARINE_RAW_GPU_FIXTURE") == "1" else "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA fixture requested but unavailable")
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    config = {
        **development.FROZEN_RUN_CONFIG,
        "updates": 4,
        "checkpoints": [2, 4],
        "batch_size": 2,
        "direct_width": 16,
        "direct_layers": 1,
        "device": device,
    }
    monkeypatch.setattr(development, "FROZEN_RUN_CONFIG", config)
    monkeypatch.setattr(development, "MODEL_CONFIG", ModelConfig(width=16, layers=1, heads=4))
    monkeypatch.setattr(development, "UPDATES", 4)
    monkeypatch.setattr(development, "MIDPOINT", 2)
    monkeypatch.setattr(development, "BATCH", 2)
    fit = _cohort()
    fit = RawCohort(
        "fit",
        np.arange(4).astype("datetime64[h]"),
        np.repeat(fit.context, 2, axis=0),
        np.repeat(fit.context_mask, 2, axis=0),
        np.array([[2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0], [5.0, 6.0, 7.0]]),
        np.ones((4, 3), dtype=np.bool_),
        np.repeat(fit.target_times, 2, axis=0),
        "a" * 64,
        "b" * 64,
    )
    root = Path(__file__).resolve().parents[2]
    uninterrupted = tmp_path / "uninterrupted"
    interrupted = tmp_path / "interrupted"
    uninterrupted.mkdir()
    interrupted.mkdir()
    direct, fresh = development._direct(
        fit, _cohort(), uninterrupted, root, device=device, resume=False
    )
    assert fresh["resume_equivalent"] is True
    shutil.copyfile(
        uninterrupted / "direct-checkpoint-2.pt", interrupted / "direct-checkpoint-2.pt"
    )
    resumed, recovery = development._direct(
        fit, _cohort(), interrupted, root, device=device, resume=True
    )
    assert recovery["resumed_from_step"] == 2
    assert recovery["resume_equivalent"] is True
    np.testing.assert_allclose(resumed, direct, rtol=1e-6, atol=1e-6)
    final_recovered, final = development._direct(
        fit, _cohort(), uninterrupted, root, device=device, resume=True
    )
    assert final["resumed_from_step"] == 4
    assert final["resume_equivalent"] is True
    np.testing.assert_allclose(final_recovered, direct, rtol=1e-6, atol=1e-6)
