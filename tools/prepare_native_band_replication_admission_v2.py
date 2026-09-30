"""Propose four fixed-seed replications with the repaired exact source closure."""

import copy
import hashlib
import json
from pathlib import Path

from execute_native_band_replication_job import source_paths

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    destination = ROOT / "orchestration/native_band_replication_admission_v2.json"
    if destination.exists():
        raise FileExistsError("Preserve earlier replication proposals")
    root_checks = (
        ROOT / "evidence/ssl-band-replication-builder-v2/root-checks-closeout-repaired-v2.json"
    )
    checks = json.loads(root_checks.read_bytes())
    if checks.get("status") != "REPAIRED_REPLICATION_ROOT_233_CHECKS_AND_DURABLE_CPU_REPLAY_PASSED":
        raise ValueError("Actual repaired root checks required")
    for name, expected in checks["current_authored_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Closed repaired replication source changed")
    jobs = []
    for seed in (13, 23):
        for method in ("shared_ssl", "direct"):
            kind = "ssl" if method == "shared_ssl" else "downstream"
            stem = "shared_ssl" if kind == "ssl" else "direct_end_to_end"
            config_path = (
                ROOT
                / "configs/native_band_replication_v2"
                / f"native_band_{stem}_seed{seed}_v2.json"
            )
            config = json.loads(config_path.read_bytes())
            template_path = (
                ROOT
                / "evidence/ssl-research-v1"
                / (
                    "band-shared-ssl-prefit-review-final.json"
                    if kind == "ssl"
                    else "band-direct-end-to-end-prefit-review-final.json"
                )
            )
            proposal = copy.deepcopy(json.loads(template_path.read_bytes()))
            name = f"band_{stem}_seed{seed}_h96_replication_v2"
            review_path = ROOT / "evidence/ssl-research-v1" / f"{name}-prefit-review-final.json"
            proposal.update(
                status="PROPOSED_REPLICATION_PREFIT_NOT_APPROVAL",
                approved_config=config,
                allowed_methods=[method],
                allowed_seeds=[seed],
                approval_scope="PROPOSED: exact fixed seed replication; independent review required",
            )
            if kind == "downstream":
                proposal["allowed_modes"] = ["direct_end_to_end"]
            proposal.pop("reviewer_session_id", None)
            proposal.pop("runtime_evidence", None)
            runtime = proposal["runtime_arguments"]
            runtime.update(
                kind=kind,
                config=str(config_path),
                review=str(review_path),
                output=str(ROOT / "outputs/native_acoustic_ssl_v1" / name),
                receipt=str(ROOT / "evidence/ssl-research-v1" / (name + "-attempt-01")),
            )
            runtime["trainer-review"] = str(review_path)
            if any(
                runtime.get(key) is not None
                for key in ("encoder", "ancestor-review", "ancestor-config", "resume")
            ):
                raise ValueError(
                    "Replicated SSL and matched fresh direct start without fitted parents"
                )
            required = [
                *source_paths(ROOT),
                config_path,
                root_checks,
                Path(__file__).resolve(),
                ROOT / "tests/unit/test_native_band_replication_review_metadata.py",
                ROOT / "evidence/ssl-band-replication-builder-v2/handoff-v2.json",
                ROOT / "evidence/ssl-band-replication-builder-v2/source-proof-final-v2b.json",
                ROOT / "evidence/ssl-band-replication-builder-v2/root-checks-failure-v2.json",
            ]
            bindings = {name: digest(name) for name in proposal["bindings"]}
            bindings.update({str(path): digest(path) for path in required})
            proposal["bindings"] = bindings
            proposal["replication_scope"] = {
                "version": 2,
                "fixed_seed": seed,
                "same_model_recipe_as_seed7": True,
                "new_screening_recipe": False,
                "new_architecture": False,
                "source_metadata_repair": checks["repair"],
                "budget_family": "native_band_v1",
                "band_full_owned_gpu_limit_hours": 12,
                "aggregate_full_owned_gpu_limit_hours": 96,
            }
            jobs.append(
                {"id": name, "kind": kind, "review_path": str(review_path), "approval": proposal}
            )
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(
            {"status": "FOUR_FIXED_REPLICATION_PROPOSALS_NOT_APPROVED", "jobs": jobs},
            stream,
            indent=2,
        )
        stream.write("\n")
    print(json.dumps({"status": "PROPOSED_NOT_APPROVED_OR_FITTED", "jobs": len(jobs)}))


if __name__ == "__main__":
    main()
