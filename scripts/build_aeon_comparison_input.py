"""Write the reviewable five-source AEON validation comparison inventory.

This script hashes immutable artifacts without decoding forecast arrays or
opening calibration/test acoustic values. The comparator still requires a
distinct approval binding the produced manifest before it reads predictions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from marine_echo.training.aeon_validation_compare import CORE_SLOTS, FORWARD_SLOTS, HYBRID_SLOTS


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs/aeon3_geb_2024_hourly_sv_v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spec(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": digest(path)}


def source(
    report_path: Path, review_path: Path, predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {"report": spec(report_path), "review": spec(review_path), "predictions": predictions}


def checked_prediction(path: Path, expected: str) -> dict[str, Any]:
    result = spec(path)
    if result["sha256"] != expected:
        raise ValueError(f"Saved prediction digest differs: {path}")
    return result


def build() -> dict[str, Any]:
    reviews = ROOT / "orchestration/reviews"
    core_root = OUTPUTS / "core_campaign"
    core_rescore = OUTPUTS / "validation_rescore/rescore.json"
    core_report = json.loads(core_rescore.read_text(encoding="utf-8"))
    core_predictions = {}
    for slot in CORE_SLOTS:
        slot_dir = core_root / slot
        slot_report = json.loads((slot_dir / "slot.json").read_text(encoding="utf-8"))
        core_predictions[slot] = checked_prediction(
            slot_dir / slot_report["prediction_path"],
            core_report["slots"][slot]["prediction_sha256"],
        )

    hybrid_root = OUTPUTS / "hybrid_development"
    hybrid_path = hybrid_root / "hybrid-report.json"
    hybrid_report = json.loads(hybrid_path.read_text(encoding="utf-8"))
    hybrid_predictions = {
        slot: checked_prediction(
            hybrid_root / hybrid_report["slots"][slot]["prediction_path"],
            hybrid_report["slots"][slot]["prediction_sha256"],
        )
        for slot in HYBRID_SLOTS
    }

    lightgbm_root = OUTPUTS / "sota_supervised_lightgbm"
    lightgbm_path = lightgbm_root / "report.json"
    lightgbm_report = json.loads(lightgbm_path.read_text(encoding="utf-8"))

    forward_root = OUTPUTS / "forward_ema_development"
    forward_path = forward_root / "manifest.json"
    forward_report = json.loads(forward_path.read_text(encoding="utf-8"))
    forward_predictions = {}
    for slot in FORWARD_SLOTS:
        slot_dir = forward_root / slot
        slot_path = slot_dir / "slot.json"
        slot_report = json.loads(slot_path.read_text(encoding="utf-8"))
        prediction = checked_prediction(
            slot_dir / slot_report["prediction_path"],
            forward_report["slots"][slot]["prediction_sha256"],
        )
        prediction["slot_report"] = spec(slot_path)
        if prediction["slot_report"]["sha256"] != forward_report["slots"][slot]["slot_sha256"]:
            raise ValueError(f"Forward slot digest differs: {slot}")
        forward_predictions[slot] = prediction

    chronos_root = OUTPUTS / "chronos2_zero_shot"
    chronos_path = chronos_root / "manifest.json"
    chronos_report = json.loads(chronos_path.read_text(encoding="utf-8"))
    return {
        "schema_version": "1.0",
        "study_id": "aeon3_geb_2024_hourly_sv_v1",
        "phase": "validation_comparison_only",
        "status": "PROPOSED_FOR_INDEPENDENT_COMPARISON_REVIEW",
        "calibration_access": "PROHIBITED",
        "test_access": "PROHIBITED",
        "assessment_partition": "validation",
        "validation_source_date_bounds_inclusive": ["2024-10-08", "2024-11-30"],
        "validation_issued_rows": 1219,
        "bootstrap": {"block_hours": 48, "draws": 2000, "seed": 20260926},
        "cohort_sha256": "5e475f6af798025f187c70ae638710d2b4362f688db2ad43d60ec436dda3e25d",
        "validation_row_sha256": "9ce5ed6d60c4082efd3f342b3ee09ddadc5cf6286be6d6a3e953ab0f6166a99f",
        "source_archive_sha256": "4e72dd4dbec707b6bf15168e51f380cbe9145a78b595ef886d78cc3806c0ecde",
        "sources": {
            "core": source(core_rescore, reviews / "AEON_VALIDATION_RESCORE_OUTCOME_REVIEW_20260927.json", core_predictions),
            "hybrid": source(hybrid_path, reviews / "AEON_HYBRID_OUTCOME_REVIEW_20260927.json", hybrid_predictions),
            "lightgbm": source(
                lightgbm_path, reviews / "AEON_SOTA_SUPERVISED_OUTCOME_REVIEW_20260927.json",
                {"post_hoc_lightgbm": checked_prediction(
                    lightgbm_root / lightgbm_report["prediction_path"],
                    lightgbm_report["prediction_sha256"],
                )},
            ),
            "forward": source(forward_path, reviews / "AEON_FORWARD_OUTCOME_REVIEW_20260927.json", forward_predictions),
            "chronos": source(
                chronos_path, reviews / "AEON_CHRONOS2_OUTCOME_REVIEW_20260927.json",
                {"post_hoc_chronos2": checked_prediction(
                    chronos_root / chronos_report["prediction_path"],
                    chronos_report["prediction_sha256"],
                )},
            ),
        },
    }


def main() -> None:
    output = ROOT / "configs/aeon_validation_comparison_input.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(build(), stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"{output}: {digest(output)}")


if __name__ == "__main__":
    main()
