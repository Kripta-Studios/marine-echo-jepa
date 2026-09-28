"""The 30k app supplement must bind exact independently reviewed bytes."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from marine_echo.serving.aeon_development_supplements import (
    load_reviewed_expanded,
    load_reviewed_scale,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path("E:/marine-echo-jepa-scale/aeon30k_stage1")
EXPANDED_OUTPUT = Path("E:/marine-echo-jepa-scale/aeon_expanded_3k")


@pytest.mark.skipif(not OUTPUT.is_dir(), reason="Local 30k artifact is unavailable")
def test_reviewed_scale_summary_is_negative_development_only() -> None:
    result = load_reviewed_scale(ROOT, OUTPUT)
    assert result["status"] == "INDEPENDENTLY_REVIEWED_30K_STAGE1_NEGATIVE"
    assert result["final_evaluation"] is False
    assert result["stage_2"] == "NOT_RUN_BY_PREDECLARED_ONE_PERCENT_GATE"
    assert result["validation_rows"] == 1219
    assert result["slots"]["direct_seed7"]["final_pinball_db"] == pytest.approx(1.178379078764592)
    assert result["slots"]["ema_jepa_seed7"]["original_3k_pinball_db"] == pytest.approx(0.6488046342025475)
    assert result["outcome_review_sha256"] == "6d55cdff9801619f3be1acc8fb18380bf4946f966163f4a7b4b2a269ad11252a"


@pytest.mark.skipif(not OUTPUT.is_dir(), reason="Local 30k artifact is unavailable")
@pytest.mark.parametrize("target", ["manifest.json", "direct_seed7/slot.json", "direct_seed7/validation-predictions.npz", "direct_seed7/checkpoint-supervised-30000.pt"])
def test_reviewed_scale_rejects_tampered_evidence(tmp_path: Path, target: str) -> None:
    for name in ("manifest.json",):
        shutil.copy2(OUTPUT / name, tmp_path / name)
    for slot in ("direct_seed7", "ema_jepa_seed7"):
        source = OUTPUT / slot
        destination = tmp_path / slot
        destination.mkdir()
        record = json.loads((source / "slot.json").read_text(encoding="utf-8"))
        for name in ("slot.json", record["prediction_path"], record["model_path"]):
            shutil.copy2(source / name, destination / name)
    path = tmp_path / target
    with path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="digest|review|manifest|slot"):
        load_reviewed_scale(ROOT, tmp_path)


@pytest.mark.skipif(not EXPANDED_OUTPUT.is_dir(), reason="Local expanded artifact is unavailable")
def test_reviewed_expanded_summary_preserves_post_hoc_scope() -> None:
    result = load_reviewed_expanded(ROOT, EXPANDED_OUTPUT)
    assert result["status"] == "INDEPENDENTLY_REVIEWED_EXPANDED_TRAIN_3K_DEVELOPMENT"
    assert result["final_evaluation"] is False
    assert result["joint_train_windows"] == 13472
    assert result["prior_train_windows"] == 8507
    assert result["current_train_windows"] == 4965
    assert result["validation_rows"] == 1219
    assert result["slots"]["direct_seed7"]["final_pinball_db"] == pytest.approx(0.640614632904252)
    assert result["slots"]["ema_jepa_seed7"]["final_pinball_db"] == pytest.approx(0.6548551285493684)
    assert result["outcome_review_sha256"] == "3ba0452b63496c448ab51799a76d296d4b363abd4771953a1a776d925799e9f0"


@pytest.mark.skipif(not EXPANDED_OUTPUT.is_dir(), reason="Local expanded artifact is unavailable")
@pytest.mark.parametrize("target", ["manifest.json", "cohort.json", "direct_seed7/slot.json", "direct_seed7/validation-predictions.npz", "direct_seed7/checkpoint-supervised-3000.pt"])
def test_reviewed_expanded_rejects_tampered_evidence(tmp_path: Path, target: str) -> None:
    for name in ("manifest.json", "cohort.json"):
        shutil.copy2(EXPANDED_OUTPUT / name, tmp_path / name)
    for slot in ("direct_seed7", "ema_jepa_seed7"):
        source = EXPANDED_OUTPUT / slot
        destination = tmp_path / slot
        destination.mkdir()
        record = json.loads((source / "slot.json").read_text(encoding="utf-8"))
        for name in ("slot.json", record["prediction_path"], record["model_path"]):
            shutil.copy2(source / name, destination / name)
    with (tmp_path / target).open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="digest|review|manifest|slot|cohort"):
        load_reviewed_expanded(ROOT, tmp_path)


@pytest.mark.skipif(
    not OUTPUT.is_dir() or not EXPANDED_OUTPUT.is_dir(),
    reason="Local reviewed development artifacts are unavailable",
)
@pytest.mark.parametrize(
    ("review_name", "loader", "output"),
    [
        ("AEON_SCALE_30K_OUTCOME_REVIEW_20260928.json", load_reviewed_scale, OUTPUT),
        ("AEON_EXPANDED_3K_OUTCOME_REVIEW_20260928.json", load_reviewed_expanded, EXPANDED_OUTPUT),
    ],
)
def test_unreviewed_status_cannot_enter_app(
    tmp_path: Path, review_name: str, loader: Callable[[Path, Path], dict[str, Any]], output: Path,
) -> None:
    review_path = tmp_path / "orchestration/reviews" / review_name
    review_path.parent.mkdir(parents=True)
    review = json.loads((ROOT / "orchestration/reviews" / review_name).read_text(encoding="utf-8"))
    review["status"] = "PENDING_REVIEW"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    config = "aeon_scale_30k.json" if output == OUTPUT else "aeon_expanded_3k.json"
    protocol = "0013-aeon-scaling-development.md" if output == OUTPUT else "0014-aeon-expanded-data-development.md"
    (tmp_path / "configs").mkdir()
    (tmp_path / "docs/adr").mkdir(parents=True)
    shutil.copy2(ROOT / "configs" / config, tmp_path / "configs" / config)
    shutil.copy2(ROOT / "docs/adr" / protocol, tmp_path / "docs/adr" / protocol)
    with pytest.raises(ValueError, match="review digest"):
        loader(tmp_path, output)
