"""Verify the completed distinct review without recomputing or selecting models."""

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    folder = ROOT / "evidence/ssl-research-v1"
    path = folder / "development-comparison-v3-numeric-review-final-v2.json"
    review = json.loads(path.read_bytes())
    witness = json.loads((folder / "development-comparison-v3-numeric-review-exit-witness-v2.json").read_bytes())
    if (witness.get("actual_cli_exit_code") != 0 or witness.get("session_id") != 41506
            or review.get("status") != "VERIFIED_SAVED_DEVELOPMENT_RECONSTRUCTION"
            or review.get("reviewer_session_id") != "01a0ef27-876b-7692-917e-3975afc6893d"
            or review.get("allowed_roles") != ["development"]
            or review.get("unresolved_defects") != []
            or review["common_support"]["methods"] != 28
            or review["common_support"]["total_daily_loss_records"] != 26544
            or len(review.get("bindings", {})) != 228):
        raise ValueError("Actual closed distinct numerical review required")
    for name, expected in review["bindings"].items():
        if digest(name) != expected:
            raise ValueError(f"Reviewed binding changed: {name}")
    for key in ("max_point_score_discrepancy_db", "max_daily_loss_discrepancy_db",
                "max_paired_point_difference_discrepancy_db", "max_interval_endpoint_discrepancy_db"):
        value = review["independent_reconstruction"][key]
        if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value < 1e-12:
            raise ValueError("Independent numerical disagreement remains")
    receipt = {"status": "ALL_228_CLOSED_DISTINCT_NUMERICAL_REVIEW_BINDINGS_VERIFIED",
               "actual_exit_witness": witness, "review_sha256": digest(path),
               "binding_count": len(review["bindings"]),
               "independent_errors": {k: v for k, v in review["independent_reconstruction"].items()
                                      if k.startswith("max_")},
               "seed_summaries": review["seed_summaries"],
               "scope": "Saved development artifacts; one development deployment; no final-site claim",
               "capacity_failure_preserved": True, "recovery_redecoded_predictions": False,
               "new_fitting_or_selection": False, "final_numeric_access": "NOT_RUN"}
    target = folder / "development-comparison-v3-numeric-review-closeout-v2.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "bindings": len(review["bindings"]),
                      "seed_summaries": receipt["seed_summaries"]}))


if __name__ == "__main__":
    main()
