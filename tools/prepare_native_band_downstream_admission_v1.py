"""Prepare six parent-specific Band transfer proposals after real screens close."""

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    destination = ROOT / "orchestration/native_band_downstream_admission_v1.json"
    if destination.exists():
        raise FileExistsError("Preserve previous admission proposal")
    template = load(ROOT / "evidence/ssl-research-v1/band-direct-end-to-end-prefit-review-final.json")
    proposals = []
    for method, modes in (
        ("shared_ssl", ("frozen_readout", "full_finetune")),
        ("masked_ssl", ("frozen_readout", "full_finetune")),
        ("permuted_ssl", ("frozen_readout",)),
        ("random_frozen", ("frozen_readout",)),
    ):
        parent = ROOT / "outputs/native_acoustic_ssl_v1" / f"band_{method}_seed7_h96_reviewed"
        parent_report = load(parent / "run.json")
        if (parent_report.get("status") != "COMPLETED"
                or parent_report.get("evidence_kind") != "REAL_TRAIN_DEVELOPMENT_FIT"
                or parent_report.get("architecture") != "nonlinear_frequency_conditioned_v1"
                or parent_report["config"]["method"] != method
                or parent_report["config"]["seed"] != 7
                or parent_report.get("inference_sha256") != digest(parent / "inference.pt")):
            raise ValueError("Only immutable completed real seed7 parents are eligible")
        ancestor_review = ROOT / "evidence/ssl-research-v1" / f"band-{method.replace('_', '-')}-prefit-review-final.json"
        ancestor = load(ancestor_review)
        ancestor_config = Path(ancestor["runtime_arguments"]["config"])
        if ancestor["approved_config"] != parent_report["config"]:
            raise ValueError("Completed original parent config differs from prefit")
        for name, expected in ancestor["bindings"].items():
            if digest(name) != expected:
                raise ValueError(f"Original parent binding changed: {name}")
        for mode in modes:
            config_path = ROOT / "configs/native_band_execution_v1" / f"native_band_{method}_{mode}_v1.json"
            config = load(config_path)
            name = f"band_{method}_{mode}_seed7_h96"
            review_path = ROOT / "evidence/ssl-research-v1" / f"{name}-prefit-review-final.json"
            record = copy.deepcopy(template)
            record.update(
                status="PROPOSED_DOWNSTREAM_PREFIT_NOT_APPROVAL",
                approved_config=config, allowed_methods=[method], allowed_modes=[mode],
                approval_scope="PROPOSED: exact completed-parent frozen/full endpoint only; independent review required",
            )
            record.pop("runtime_evidence", None)
            record.pop("reviewer_session_id", None)
            bindings = {name: digest(name) for name in template["bindings"]}
            for path in [config_path, ancestor_config, ancestor_review, *[parent / p for p in (
                "run.json", "inference.pt", "selected_encoder.pt", "membership.json")],
                Path(__file__).resolve()]:
                bindings[str(path)] = digest(path)
            record["bindings"] = bindings
            runtime = record["runtime_arguments"]
            runtime.update({
                "config": str(config_path), "review": str(review_path),
                "trainer-review": str(review_path),
                "output": str(ROOT / "outputs/native_acoustic_ssl_v1" / name),
                "receipt": str(ROOT / "evidence/ssl-research-v1" / (name + "-attempt-01")),
                "encoder": str(parent / "selected_encoder.pt"),
                "ancestor-review": str(ancestor_review), "ancestor-config": str(ancestor_config),
            })
            record["actual_completed_parent"] = {
                "run_sha256": digest(parent / "run.json"),
                "selected_encoder_sha256": digest(parent / "selected_encoder.pt"),
                "inference_sha256": digest(parent / "inference.pt"),
                "membership_sha256": digest(parent / "membership.json"),
                "development_score": parent_report["selected_daily_dev_pinball"],
                "not_an_ssl_selection_probe_endpoint": True,
            }
            proposals.append({"id": name, "review_path": str(review_path), "approval": record})
    with destination.open("x", encoding="utf-8") as stream:
        json.dump({"status": "SIX_PROPOSED_PARENT_SPECIFIC_BAND_ENDPOINTS", "jobs": proposals},
                  stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "PROPOSED_NOT_REVIEWED_OR_EXECUTED", "jobs": len(proposals)}))


if __name__ == "__main__":
    main()
