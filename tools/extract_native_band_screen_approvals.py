"""Verify and preserve five exact per-job approvals without changing fields."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEWER = "01a0ef27-876b-7692-917e-3975afc6893d"
ARCHITECTURE = "nonlinear_frequency_conditioned_v1"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parent = ROOT / "evidence/ssl-research-v1/band-seed7-prefit-review-final.json"
    combined = json.loads(parent.read_text(encoding="utf-8"))
    if combined.get("status") in {"REQUEST_CHANGES", "BLOCKED", "REJECTED"}:
        raise ValueError("Unapproved combined review cannot fit")
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
    approvals = {review["approved_config"].get("method"): review for review in leaves}
    recipe_methods = {"shared_ssl", "masked_ssl", "permuted_ssl", "random_frozen", "direct"}
    if len(leaves) != 5 or set(approvals) != recipe_methods:
        raise ValueError("Five distinct exact per-job approvals required")
    for path, expected_sha in combined.get("proof_bindings", {}).items():
        if digest(Path(path)) != expected_sha:
            raise ValueError(f"Changed combined engineering proof: {path}")
    outputs = {}
    for name, review in approvals.items():
        status = "APPROVED_DOWNSTREAM_PREFIT" if name == "direct" else "APPROVED_PREFIT"
        if (
            review.get("status") != status
            or review.get("reviewer_session_id") != REVIEWER
            or review.get("architecture") != ARCHITECTURE
            or review.get("evidence_kind") != "REAL_TRAIN_DEVELOPMENT_FIT"
            or set(review.get("allowed_roles", [])) != {"train", "development"}
            or review.get("allowed_seeds") != [7]
            or name not in review.get("allowed_methods", [])
            or not set(review.get("allowed_methods", [])).issubset(recipe_methods)
            or review.get("band_budget_status") != "ROOT_RESOLVED"
        ):
            raise ValueError(
                "Wrong status/identity/architecture/roles/recipe/evidence/budget scope"
            )
        if name == "direct" and review.get("allowed_modes") != ["direct_end_to_end"]:
            raise ValueError("Only scratch supervised direct endpoint may be approved")
        for path, expected_sha in review["bindings"].items():
            if digest(Path(path)) != expected_sha:
                raise ValueError(f"Changed prefit source binding: {path}")
        destination = Path(review["runtime_arguments"]["review"]).resolve()
        if (
            destination.parent != ROOT / "evidence/ssl-research-v1"
            or destination.suffix != ".json"
            or review["runtime_arguments"].get("trainer-review") != str(destination)
        ):
            raise ValueError("Exact new root-owned same-review runtime paths required")
        if destination.exists():
            raise FileExistsError("Preserve old derived approvals")
        outputs[name] = destination
    for name, destination in outputs.items():
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(approvals[name], stream, indent=2, sort_keys=True)
            stream.write("\n")
        if json.loads(destination.read_text(encoding="utf-8")) != approvals[name]:
            raise AssertionError("Scientific approval fields changed during extraction")
    receipt = {
        "status": "FIVE_EXACT_BAND_SCREEN_APPROVALS_VERIFIED_AND_EXTRACTED",
        "parent_review_sha256": digest(parent),
        "scientific_fields_unchanged": True,
        "derived": {
            name: {
                "path": str(path),
                "sha256": digest(path),
                "binding_count": len(approvals[name]["bindings"]),
            }
            for name, path in outputs.items()
        },
        "training_execution": "NOT_RUN",
        "final_numeric_access": "NOT_AUTHORIZED",
    }
    with (ROOT / "evidence/ssl-research-v1/band-screen-approval-extraction-v1.json").open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
