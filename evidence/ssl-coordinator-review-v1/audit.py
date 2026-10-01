"""Read-only admission audit. This is coordinator evidence, never prefit approval."""

import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import execute_native_band_replication_job as wrapper
from marine_echo.training import native_band_replication_downstream as downstream
from marine_echo.training import native_band_replication_ssl as core


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def main():
    direct_path = ROOT / "orchestration/native_band_direct13_ownership_retry_admission_v2.json"
    transfer_path = ROOT / "orchestration/native_band_replication_downstream_seed23_v3.json"
    refs_path = ROOT / "orchestration/native_band_downstream_references_seed23_v3.json"
    direct = wrapper.read_json(direct_path)
    transfer = wrapper.read_json(transfer_path)
    refs = wrapper.read_json(refs_path)
    assert refs["proposal_sha256"] == wrapper.digest(transfer_path)
    jobs = [direct, *transfer["jobs"]]
    references = {j["job_id"]: j for j in refs["jobs"]}
    checked = []
    union = {}
    for job in jobs:
        proposal = job["approval"]
        assert proposal["status"].startswith("PROPOSED")
        assert "reviewer_session_id" not in proposal
        assert set(proposal["allowed_roles"]) == {"train", "development"}
        runtime = proposal["runtime_arguments"]
        config = wrapper.read_json(runtime["config"])
        assert config == proposal["approved_config"]
        downstream.DownstreamConfig(**config).validate()
        wrapper._check_config(config, "downstream", ROOT)
        inputs = downstream.RunInputs(**{
            k: Path(runtime[k.replace("_", "-")])
            for k in ("train", "dev", "train_cohort", "dev_cohort", "split", "adr0016", "protocol", "config", "review")
        }, encoder=Path(runtime["encoder"]) if runtime["encoder"] else None,
            ancestor_review=Path(runtime["ancestor-review"]) if runtime["ancestor-review"] else None,
            ancestor_config=Path(runtime["ancestor-config"]) if runtime["ancestor-config"] else None)
        backbone = downstream._core_config(inputs, downstream.DownstreamConfig(**config),
            correctness_smoke=False, smoke_core_config=None)
        required = [*wrapper.source_paths(ROOT), *downstream.required_paths(inputs, backbone)]
        verified = wrapper._verify_bindings(proposal, required, {inputs.train, inputs.dev})
        union.update(verified)
        roles = downstream._split_roles(inputs)
        downstream._cohort(inputs, "train")
        downstream._cohort(inputs, "development")
        absent = {k: not Path(runtime[k]).exists() for k in ("output", "receipt", "review", "trainer-review")}
        assert all(absent.values())
        assert runtime["resume"] is None
        assert runtime["device"] == "cuda:0"
        assert Path(proposal["required_entrypoint"]).resolve() == Path(wrapper.__file__).resolve()
        if job["id"] in references:
            ref = references[job["id"]]
            for key, value in (("approved_config", config), ("runtime_arguments", runtime)):
                text = canonical(value)
                assert text == ref[key + "_canonical_json"]
                assert hashlib.sha256(text.encode()).hexdigest() == ref[key + "_sha256"]
        parent_checks = None
        if inputs.encoder:
            parent = wrapper.read_json(inputs.encoder.parent / "run.json")
            ancestor = wrapper.read_json(inputs.ancestor_review)
            wrapper._review_identity(ancestor, "APPROVED_PREFIT")
            parent_bindings = wrapper._verify_bindings(ancestor, [], {inputs.train, inputs.dev})
            union.update(parent_bindings)
            assert parent["status"] == "COMPLETED"
            assert parent["evidence_kind"] == wrapper.EVIDENCE
            assert parent["config"] == backbone.to_dict() == ancestor["approved_config"]
            assert parent["review_sha256"] == wrapper.digest(inputs.ancestor_review)
            for name in ("inference", "membership"):
                assert parent[name + "_sha256"] == wrapper.digest(inputs.encoder.parent / (name + (".pt" if name == "inference" else ".json")))
            membership = wrapper.read_json(inputs.encoder.parent / "membership.json")
            assert len(membership["train_row_ids"]) == 18593
            assert all(roles["train"].get(d) == a for d, a in zip(membership["train_deployments"], membership["train_archive_sha256"], strict=True))
            scalers = wrapper.read_json(inputs.encoder.parent / "scalers.json")
            core.Scalers.from_dict(scalers)
            attempt_path = ROOT / "evidence/ssl-research-v1" / (inputs.encoder.parent.name + "-attempt-01") / "attempt.json"
            attempt = wrapper.read_json(attempt_path)
            assert attempt["resources_full_attempt"]["exit_code"] == 0
            assert attempt["resources_full_attempt"]["owned_tree_cleanup_verified"] is True
            parent_checks = {"run_sha256": wrapper.digest(inputs.encoder.parent / "run.json"), "train_rows": 18593, "source_bindings_verified": len(parent_bindings), "receipt_cleanup_verified": True}
        else:
            assert config["method"] == "direct" and config["mode"] == "direct_end_to_end"
            assert runtime["ancestor-config"] is runtime["ancestor-review"] is None
        self_record = copy.deepcopy(proposal)
        self_record["reviewer_session_id"] = proposal["root_coordinator_session_id"]
        try:
            wrapper._review_identity(self_record, self_record["status"])
        except ValueError as error:
            assert "distinct" in str(error)
            rejection = str(error)
        else:
            raise AssertionError("Coordinator self-review was admitted")
        checked.append({"id": job["id"], "verified_proposal_bindings": len(verified), "config": config,
            "fresh_destinations": absent, "parent": parent_checks, "self_review_rejected": rejection,
            "independent_prefit_present": False})
    budget = wrapper.read_json(ROOT / "orchestration/native_band_budget_owner_resolution_v1.json")
    wrapper.validate_budget(budget)
    ledger_path = ROOT / "orchestration/native_ssl_run_ledger_v1.json"
    aggregate, band, remaining = wrapper.budget_totals(wrapper.read_json(ledger_path))
    assert not ledger_path.with_suffix(".band-pending").exists()
    assert not (ROOT / "evidence/ssl-builder-v1/gpu-owner.lock").exists()
    report = {"status": "COORDINATOR_REVIEW_COMPLETED_NOT_INDEPENDENT_APPROVAL", "independent": False,
        "created_at": datetime.now(timezone.utc).isoformat(), "reviewed_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "jobs": checked, "unique_verified_bindings": len(union), "verified_bindings": union,
        "proposal_hashes": {str(p): wrapper.digest(p) for p in (direct_path, transfer_path, refs_path)},
        "budget": {"aggregate_hours": aggregate, "band_hours": band, "remaining_band_seconds": remaining, "ledger_sha256": wrapper.digest(ledger_path)},
        "scope": "Metadata, raw file hashes, source admission and CPU synthetic correctness; no acoustic arrays or fitted tensor payloads decoded by this audit.",
        "final_numeric_access": "NOT_RUN", "scientific_fitting": "NOT_RUN", "approval_files_written": [],
        "findings": [{"id": "R1", "severity": "medium", "description": "Four preserved v2 admission proposals omit native_latent.py required by the current trainer. Default source-closure regression fails four checks. Current v3 source-closure checks pass; do not reuse or edit historical v2 approvals."},
            {"id": "R2", "severity": "blocking", "description": "All three prospective jobs lack distinct prefit approval. Coordinator review cannot satisfy ADR0015/0018 or the production reviewer identity gate."}]}
    destination = Path(__file__).parent / "audit.json"
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "jobs": len(checked), "unique_verified_bindings": len(union), "budget": report["budget"]}))


if __name__ == "__main__":
    main()
