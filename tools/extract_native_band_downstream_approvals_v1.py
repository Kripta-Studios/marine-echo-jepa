"""Verify six actual-parent approvals before copying their unchanged JSON fields."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEWER = "01a0ef27-876b-7692-917e-3975afc6893d"
EXPECTED = {
    ("shared_ssl", "frozen_readout"),
    ("shared_ssl", "full_finetune"),
    ("masked_ssl", "frozen_readout"),
    ("masked_ssl", "full_finetune"),
    ("permuted_ssl", "frozen_readout"),
    ("random_frozen", "frozen_readout"),
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parent = ROOT / "evidence/ssl-research-v1/band-downstream-prefit-review-final-v1.json"
    proposal_path = ROOT / "orchestration/native_band_downstream_admission_v1.json"
    combined = json.loads(parent.read_bytes())
    proposal = json.loads(proposal_path.read_bytes())
    if combined.get("status") != "APPROVED_BAND_DOWNSTREAM_PREFIT":
        raise ValueError("All six explicit independent approvals required")
    leaves = []

    def visit(value):
        if isinstance(value, dict) and "approved_config" in value:
            leaves.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(combined.get("approvals"))
    approvals = {(r["approved_config"]["method"], r["approved_config"]["mode"]): r for r in leaves}
    proposed = {
        (j["approval"]["approved_config"]["method"], j["approval"]["approved_config"]["mode"]): j
        for j in proposal["jobs"]
    }
    if len(leaves) != 6 or set(approvals) != EXPECTED or set(proposed) != EXPECTED:
        raise ValueError("Exact six distinct method/mode endpoints required")
    for name, expected in combined.get("proof_bindings", {}).items():
        if digest(name) != expected:
            raise ValueError(f"Changed combined proof: {name}")
    outputs = {}
    for key, review in approvals.items():
        method, mode = key
        original = proposed[key]["approval"]
        runtime = review["runtime_arguments"]
        if (
            review.get("status") != "APPROVED_DOWNSTREAM_PREFIT"
            or review.get("reviewer_session_id") != REVIEWER
            or review.get("architecture") != "nonlinear_frequency_conditioned_v1"
            or review.get("evidence_kind") != "REAL_TRAIN_DEVELOPMENT_FIT"
            or set(review.get("allowed_roles", [])) != {"train", "development"}
            or review.get("allowed_seeds") != [7]
            or review.get("allowed_methods") != [method]
            or review.get("allowed_modes") != [mode]
            or review.get("band_budget_status") != "ROOT_RESOLVED"
            or review["approved_config"] != original["approved_config"]
            or runtime != original["runtime_arguments"]
            or runtime.get("kind") != "downstream"
        ):
            raise ValueError("Wrong identity, scientific config, budget or runtime scope")
        if any(review["bindings"].get(p) != sha for p, sha in original["bindings"].items()):
            raise ValueError("Proposal source and actual-parent bindings must be retained")
        for name, expected in review["bindings"].items():
            if digest(name) != expected:
                raise ValueError(f"Changed independently reviewed binding: {name}")
        destination = Path(runtime["review"]).resolve()
        output, receipt = Path(runtime["output"]).resolve(), Path(runtime["receipt"]).resolve()
        if (
            destination.parent != ROOT / "evidence/ssl-research-v1"
            or str(destination) != proposed[key]["review_path"]
            or runtime.get("trainer-review") != str(destination)
            or not output.is_relative_to(ROOT / "outputs")
            or not receipt.is_relative_to(ROOT / "evidence")
            or destination.exists()
            or output.exists()
            or receipt.exists()
        ):
            raise ValueError("Exact fresh root-owned runtime paths required")
        outputs[key] = destination
    target = ROOT / "evidence/ssl-research-v1/band-downstream-approval-extraction-v1.json"
    if target.exists():
        raise FileExistsError("Preserve earlier extraction receipt")
    derived = {}
    for key, destination in outputs.items():
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(approvals[key], stream, indent=2, sort_keys=True)
            stream.write("\n")
        if json.loads(destination.read_bytes()) != approvals[key]:
            raise ValueError("Independent scientific fields changed during extraction")
        derived["/".join(key)] = {
            "path": str(destination),
            "sha256": digest(destination),
            "binding_count": len(approvals[key]["bindings"]),
        }
    receipt = {
        "status": "SIX_EXACT_BAND_DOWNSTREAM_APPROVALS_VERIFIED_AND_EXTRACTED",
        "parent_review_sha256": digest(parent),
        "proposal_sha256": digest(proposal_path),
        "scientific_fields_unchanged": True,
        "derived": derived,
        "training_execution": "NOT_RUN",
        "final_numeric_access": "NOT_AUTHORIZED",
    }
    with target.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "jobs": len(derived)}))


if __name__ == "__main__":
    main()
