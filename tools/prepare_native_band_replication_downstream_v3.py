"""Propose two matched endpoints only after a fixed replication parent actually completes."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from execute_native_band_replication_job import source_paths

from marine_echo.training import native_band_replication_ssl as core
from marine_echo.training.native_band_replication_downstream import DownstreamConfig

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(13, 23), required=True)
    seed = parser.parse_args().seed
    folder = ROOT / "evidence/ssl-research-v1"
    parent = ROOT / "outputs/native_acoustic_ssl_v1" / f"band_shared_ssl_seed{seed}_h96_replication_v2"
    ancestor_review = folder / f"band_shared_ssl_seed{seed}_h96_replication_v2-prefit-review-final.json"
    report = json.loads((parent / "run.json").read_bytes())
    ancestor = json.loads(ancestor_review.read_bytes())
    attempt_path = folder / (parent.name + "-attempt-01") / "attempt.json"
    attempt = json.loads(attempt_path.read_bytes())
    if (report.get("status") != "COMPLETED" or report.get("evidence_kind") != "REAL_TRAIN_DEVELOPMENT_FIT"
            or report.get("architecture") != "nonlinear_frequency_conditioned_v1"
            or report["config"]["seed"] != seed or report["config"]["method"] != "shared_ssl"
            or report.get("inference_sha256") != digest(parent / "inference.pt")
            or report["config"] != ancestor["approved_config"] or ancestor.get("status") != "APPROVED_PREFIT"
            or attempt.get("resources_full_attempt", {}).get("exit_code") != 0
            or attempt["resources_full_attempt"].get("owned_tree_cleanup_verified") is not True
            or attempt.get("budget_family") != "native_band_v1"):
        raise ValueError("Actual completed reviewed V2 SSL parent and owned cleanup required")
    for name, expected in ancestor["bindings"].items():
        if digest(name) != expected:
            raise ValueError("Completed parent's reviewed source closure changed")
    template_path = folder / f"band_direct_end_to_end_seed{seed}_h96_replication_v2-prefit-review-final.json"
    template = json.loads(template_path.read_bytes())
    destination = ROOT / "orchestration" / f"native_band_replication_downstream_seed{seed}_v3.json"
    config_dir = ROOT / "configs/native_band_replication_downstream_v3"
    configs, jobs = {}, []
    ancestor_config = Path(ancestor["runtime_arguments"]["config"])
    for mode in ("frozen_readout", "full_finetune"):
        config = DownstreamConfig(method="shared_ssl", mode=mode, seed=seed)
        config.validate()
        config_path = config_dir / f"native_band_shared_ssl_{mode}_seed{seed}_v3.json"
        identifier = f"band_shared_ssl_{mode}_seed{seed}_h96_replication_v3"
        review_path = folder / (identifier + "-prefit-review-final.json")
        record = copy.deepcopy(template)
        record.update(status="PROPOSED_COMPLETED_PARENT_DOWNSTREAM_NOT_APPROVAL", approved_config=config.to_dict(),
                      allowed_methods=["shared_ssl"], allowed_modes=[mode], allowed_seeds=[seed],
                      required_entrypoint="tools/execute_native_band_replication_job.py",
                      approval_scope="PROPOSED: two exact completed-parent matched endpoints; distinct review required")
        record.pop("reviewer_session_id", None)
        record.pop("referential_review", None)
        record.pop("replication_scope", None)
        record["runtime_arguments"].update({"kind": "downstream", "config": str(config_path), "review": str(review_path),
            "trainer-review": str(review_path), "output": str(ROOT / "outputs/native_acoustic_ssl_v1" / identifier),
            "receipt": str(folder / (identifier + "-attempt-01")), "encoder": str(parent / "selected_encoder.pt"),
            "ancestor-review": str(ancestor_review), "ancestor-config": str(ancestor_config), "resume": None})
        required = [*source_paths(ROOT), *core.required_sources(core.Config(seed=seed)), ancestor_review,
                    ancestor_config, attempt_path, template_path, Path(__file__).resolve(),
                    *[parent / name for name in ("run.json", "selected_encoder.pt", "inference.pt", "membership.json", "scalers.json")]]
        record["bindings"].update({str(path): digest(path) for path in required})
        record["actual_completed_parent"] = {"run_sha256": digest(parent / "run.json"),
            "selected_encoder_sha256": digest(parent / "selected_encoder.pt"), "inference_sha256": digest(parent / "inference.pt"),
            "membership_sha256": digest(parent / "membership.json"), "scalers_sha256": digest(parent / "scalers.json"),
            "parent_probe_score_is_not_this_endpoint_result": True}
        if any(Path(record["runtime_arguments"][key]).exists() for key in ("output", "receipt", "review")) or config_path.exists():
            raise FileExistsError("Original absent prospective endpoint destinations required")
        configs[config_path] = config.to_dict()
        jobs.append({"id": identifier, "review_path": str(review_path), "approval": record})
    if destination.exists():
        raise FileExistsError("Preserve earlier downstream proposals")
    config_dir.mkdir(parents=True, exist_ok=True)
    for path, config in configs.items():
        with path.open("x", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    for job in jobs:
        path = Path(job["approval"]["runtime_arguments"]["config"])
        job["approval"]["bindings"][str(path)] = digest(path)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump({"status": "TWO_COMPLETED_PARENT_PROPOSALS_NOT_APPROVED_OR_FITTED", "seed": seed, "jobs": jobs}, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "TWO_COMPLETED_PARENT_PROPOSALS_NOT_APPROVED_OR_FITTED", "seed": seed, "jobs": 2}))


if __name__ == "__main__":
    main()
