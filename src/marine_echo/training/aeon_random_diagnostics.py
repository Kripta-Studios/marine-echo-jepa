"""Post-run, TRAIN-only representation diagnostics for frozen-random AEON controls."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

import torch

from marine_echo.models.aeon_ssl import AeonTemporalSSL
from marine_echo.training.aeon_campaign import (
    _config_gate,
    _representation_diagnostics,
    campaign_slots,
)
from marine_echo.training.aeon_corpus import AeonDevelopmentReader
from marine_echo.training.aeon_development import _context_tensors, _normalizer, _sha256
from marine_echo.training.aeon_rescore import _artifact
from marine_echo.training.aeon_windows import AeonHourlyWindow, AeonWindowPlan, iter_aeon_windows


def _checkpoint_diagnostic(
    checkpoint_path: Path,
    family: str,
    train_rows: list[AeonHourlyWindow],
    config: dict[str, Any],
    *,
    expected_source: str,
    expected_protocol: str,
) -> dict[str, Any]:
    """Inspect immutable random weights and final head without an optimizer step."""
    if family not in ("random_encoder_ema", "random_encoder_shared_sigreg"):
        raise ValueError("Only the two frozen-random AEON controls have this diagnostic.")
    if len(train_rows) < 2 or any(
        row.partition != "train" or row.source_archive_sha256 != expected_source
        for row in train_rows
    ):
        raise ValueError("AEON random diagnostic requires reviewed TRAIN source rows.")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(saved, dict)
        or saved.get("phase") != "supervised"
        or saved.get("step") != 1500
        or saved.get("family") != family
        or saved.get("seed") != 7
        or saved.get("source_sha256") != expected_source
        or saved.get("protocol_sha256") != expected_protocol
    ):
        raise ValueError("AEON random checkpoint lineage or endpoint differs.")
    scaler = _normalizer(train_rows)
    if tuple(saved.get("scaler_fit_only", ())) != scaler:
        raise ValueError("AEON random checkpoint TRAIN scaler differs from source rows.")
    mode: Literal["ema", "shared_sigreg"] = (
        "ema" if family == "random_encoder_ema" else "shared_sigreg"
    )
    neural = config["neural"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = AeonTemporalSSL(
            mode=mode,
            width=neural["encoder_width"],
            layers=neural["encoder_layers"],
            regularizer_weight=neural[f"{mode}_sigreg_weight"],
        )
    initial = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if name.startswith(("encoder.", "predictor.", "teacher."))
    }
    model.load_state_dict(saved["model_state_dict"], strict=True)
    if any(not torch.equal(model.state_dict()[name], value) for name, value in initial.items()):
        raise ValueError("AEON frozen-random representation weights changed from seed 7.")
    values, mask = _context_tensors(train_rows, scaler)
    return _representation_diagnostics(model, train_rows, values, mask, device="cpu")


def analyze_random_controls(
    archive: Path,
    split_review: Path,
    campaign_output: Path,
    config_path: Path,
    rescore_report: Path,
    output: Path,
) -> dict[str, Any]:
    """Append comparable evidence without touching training or validation outcomes."""
    config_sha256 = _sha256(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _config_gate(config)
    campaign_output = campaign_output.resolve(strict=True)
    manifest_path = campaign_output / "manifest.json"
    manifest_sha256 = _sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rescore_sha256 = _sha256(rescore_report)
    rescore = json.loads(rescore_report.read_text(encoding="utf-8"))
    slots = campaign_slots(config)
    entries = manifest.get("slots")
    if (
        manifest.get("status") != "COMPLETED_TRAIN_VALIDATION_CAMPAIGN_NOT_FINAL_EVALUATION"
        or manifest.get("config_sha256") != config_sha256
        or not isinstance(entries, dict)
        or set(entries) != {slot.run_id for slot in slots}
        or any(entry.get("status") != "DONE" for entry in entries.values())
        or rescore.get("status")
        != "PROTOCOL_VALIDATION_RESCORE_PENDING_INDEPENDENT_SELECTION_REVIEW"
        or rescore.get("campaign_manifest_sha256") != manifest_sha256
        or rescore.get("campaign_config_sha256") != config_sha256
        or rescore.get("test_access") != "PROHIBITED"
        or set(rescore.get("slots", {})) != set(entries)
    ):
        raise ValueError("AEON random diagnostic needs the completed, independently rescored campaign.")
    split_sha256 = _sha256(split_review)
    reader = AeonDevelopmentReader(archive, review_path=split_review, review_sha256=split_sha256)
    train_rows = list(iter_aeon_windows(
        reader.iter_partition("train"), plan=AeonWindowPlan(), partition="train",
        partition_start="2024-03-06", partition_end_exclusive="2024-10-08",
    ))
    if not train_rows or reader.archive_sha256 != config["source_sha256"]:
        raise ValueError("AEON random diagnostic TRAIN source differs from campaign.")
    report: dict[str, Any] = {
        "status": "POST_RUN_TRAIN_ONLY_RANDOM_REPRESENTATION_DIAGNOSTIC",
        "classification": "DIAGNOSTIC_NOT_MODEL_SELECTION_OR_FINAL_EVALUATION",
        "test_access": "PROHIBITED",
        "campaign_manifest_sha256": manifest_sha256,
        "campaign_config_sha256": config_sha256,
        "rescore_report_sha256": rescore_sha256,
        "source_archive_sha256": reader.archive_sha256,
        "split_review_sha256": split_sha256,
        "diagnostic_code_sha256": _sha256(Path(__file__)),
        "campaign_code_sha256": manifest["code_sha256"],
        "train_rows": len(train_rows),
        "controls": {},
    }
    for mode in ("ema", "shared_sigreg"):
        random_id = f"random_encoder_{mode}_seed7"
        reference_id = f"{mode if mode == 'shared_sigreg' else 'ema_jepa'}_seed7"
        random_dir = campaign_output / random_id
        reference_dir = campaign_output / reference_id
        random_record = json.loads(_artifact(
            random_dir, "slot.json", entries[random_id].get("slot_sha256")
        ).read_text(encoding="utf-8"))
        reference_record = json.loads(_artifact(
            reference_dir, "slot.json", entries[reference_id].get("slot_sha256")
        ).read_text(encoding="utf-8"))
        if (
            random_record.get("run_id") != random_id
            or random_record.get("family") != f"random_encoder_{mode}"
            or reference_record.get("run_id") != reference_id
            or random_record.get("pretrain_updates") != 0
            or random_record.get("supervised_updates") != 1500
        ):
            raise ValueError("AEON random/reference slot identities or budgets differ.")
        checkpoint = _artifact(
            random_dir, random_record.get("model_path"), random_record.get("model_sha256")
        )
        diagnostic = _checkpoint_diagnostic(
            checkpoint, f"random_encoder_{mode}", train_rows, config,
            expected_source=config["source_sha256"], expected_protocol=config["protocol_sha256"],
        )
        reference = reference_record.get("final_pretrain_train_representation_diagnostics")
        if (
            not isinstance(reference, dict)
            or reference.get("source_partition") != "train"
            or diagnostic["row_ids"] != reference.get("row_ids")
            or diagnostic["cutoff_interval_ids"] != reference.get("cutoff_interval_ids")
        ):
            raise ValueError("AEON random and SSL diagnostics differ in TRAIN row support.")
        report["controls"][random_id] = {
            "checkpoint_sha256": random_record["model_sha256"],
            "slot_sha256": entries[random_id]["slot_sha256"],
            "reference_run_id": reference_id,
            "reference_slot_sha256": entries[reference_id]["slot_sha256"],
            "reference_final_pretrain_diagnostic": reference,
            "post_run_random_diagnostic": diagnostic,
        }
    output = output.resolve()
    if output.exists():
        prior_path = output / "random-diagnostics.json"
        if not prior_path.is_file() or json.loads(prior_path.read_text(encoding="utf-8")) != report:
            raise ValueError("AEON random diagnostic output differs from immutable inputs.")
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".stage.", dir=output.parent))
    try:
        (stage / "random-diagnostics.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.replace(stage, output)
    except BaseException:
        if stage.exists() and stage.parent == output.parent:
            (stage / "random-diagnostics.json").unlink(missing_ok=True)
            stage.rmdir()
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--split-review", type=Path, required=True)
    parser.add_argument("--campaign-output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rescore-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = analyze_random_controls(
        arguments.archive, arguments.split_review, arguments.campaign_output,
        arguments.config, arguments.rescore_report, arguments.output,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
