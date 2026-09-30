"""Derive immutable local ancestry from closed real runs, without corpus decoding.

This inventory is not final selection, assessment admission or numeric access.
Original reports, embedded scalers and all recursively fitted parents are retained.
"""

import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "orchestration/native_assessment_seed7_metadata_v1"
EVIDENCE = "REVIEWED_FROZEN_ASSESSMENT"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return str(path)


def identities(sources):
    return [{"archive_id": str(s["file_id"]), "deployment_id": s["deployment"],
             "site_id": s["site"], "source_ids": [s["archive_sha256"]]}
            for s in sources]


def main():
    if DESTINATION.exists():
        raise FileExistsError("Preserve all earlier ancestry inventories")
    split_path = ROOT / "configs/native_ssl_split_v1.json"
    split = load(split_path)
    train_npz = ROOT / "data/processed/native_ssl_v1/train.npz"
    train_report_path = train_npz.with_suffix(".json")
    train_report = load(train_report_path)
    train_sha, split_sha = digest(train_npz), digest(split_path)
    if (train_report["role"] != "train" or train_report["npz_sha256"] != train_sha
            or train_report["split_sha256"] != split_sha):
        raise ValueError("Original real TRAIN corpus identity changed")
    matrix = load(ROOT / "orchestration/native_development_comparison_v2.json")
    neural = {}
    for name, prediction in matrix["methods"].items():
        directory = Path(prediction).parent
        if not (directory / "run.json").exists():
            continue
        run, member = load(directory / "run.json"), load(directory / "membership.json")
        if run.get("status") != "COMPLETED" or run.get("evidence_kind") != "REAL_TRAIN_DEVELOPMENT_FIT":
            raise ValueError("Only actual completed original neural fits are eligible")
        artifact_path = directory / "inference.pt"
        if run.get("inference_sha256") != digest(artifact_path):
            raise ValueError("Frozen inference artifact differs from completed report")
        artifact = torch.load(artifact_path, weights_only=True, map_location="cpu")
        if artifact.get("kind") != "native_ssl_weights_only_inference_v1":
            raise ValueError("Unsupported weight-only artifact")
        if artifact["config"] != run.get("core_config", run["config"]):
            raise ValueError("Embedded actual config differs from run")
        if run.get("membership_sha256") != digest(directory / "membership.json"):
            raise ValueError("Actual training membership changed")
        expected_train = {s["deployment"]: s["archive_sha256"]
                          for s in split["sources"] if s["role"] == "train"}
        rows = list(zip(member["train_deployments"], member["train_row_ids"], strict=True))
        if (len(rows) != train_report["issued"] or len(set(rows)) != len(rows)
                or any(expected_train.get(dep) != sha for dep, sha in
                       zip(member["train_deployments"], member["train_archive_sha256"], strict=True))):
            raise ValueError("Exact whole TRAIN membership is incomplete or outside split")
        neural[name] = {"directory": directory, "run": run, "artifact": artifact,
                        "run_sha": digest(directory / "run.json"), "rows": rows}
    if len(neural) != 16 or len({tuple(v["rows"]) for v in neural.values()}) != 1:
        raise ValueError("All sixteen closed neural endpoints must share exact TRAIN issuance")
    run_index = {item["run_sha"]: name for name, item in neural.items()}
    for item in neural.values():
        ancestry = item["artifact"].get("supervised_ancestry")
        if ancestry and ancestry.get("ancestor_run_sha256") is not None:
            parent = neural[run_index[ancestry["ancestor_run_sha256"]]]
            if digest(parent["directory"] / "selected_encoder.pt") != ancestry["ancestor_encoder_sha256"]:
                raise ValueError("Selected fitted encoder parent differs")
    DESTINATION.mkdir()
    cohort_path = DESTINATION / "train_cohort.json"
    write(cohort_path, {"role": "train", "evidence_kind": EVIDENCE,
                       "input_npz_sha256": train_sha, "split_sha256": split_sha,
                       "rows": next(iter(neural.values()))["rows"],
                       "identities": identities([s for s in split["sources"] if s["role"] == "train"]),
                       "derived_from_original_report": str(train_report_path),
                       "original_report_sha256": digest(train_report_path),
                       "derivation": "All sixteen actual fitted membership files agree; corpus numerical values not decoded"})
    split_receipt = DESTINATION / "whole_deployment_split.json"
    write(split_receipt, {"train": identities([s for s in split["sources"] if s["role"] == "train"]),
                          "reserved_test": identities([s for s in split["sources"] if s["role"] == "final_test"]),
                          "original_split_path": str(split_path), "original_split_sha256": split_sha,
                          "status": "DERIVED_IDENTITY_RECEIPT_NOT_A_NEW_PARTITION"})
    inventory = {}
    for name, item in neural.items():
        directory, run, artifact = item["directory"], item["run"], item["artifact"]
        config_path = DESTINATION / f"{name}.config.json"
        stats_path = DESTINATION / f"{name}.statistics.json"
        write(config_path, artifact["config"])
        write(stats_path, artifact["scalers"])
        weights = directory / "inference.pt"
        manifest_path = DESTINATION / f"{name}.fit_input.json"
        write(manifest_path, {"role": "train", "input_sha256": train_sha,
                              "cohort_sha256": digest(cohort_path), "statistics_sha256": digest(stats_path),
                              "config_sha256": digest(config_path), "split_sha256": split_sha,
                              "original_split_sha256": split_sha,
                              "actual_run_path": str(directory / "run.json"), "actual_run_sha256": item["run_sha"],
                              "actual_membership_path": str(directory / "membership.json"),
                              "actual_membership_sha256": digest(directory / "membership.json"),
                              "original_embedded_bindings": artifact["bindings"]})
        parent_info = artifact.get("supervised_ancestry")
        parent_names = [] if not parent_info or parent_info.get("ancestor_run_sha256") is None else [
            run_index[parent_info["ancestor_run_sha256"]]]
        artifacts = [{"path": str(directory / p), "sha256": digest(directory / p)}
                     for p in ("inference.pt", "selected_encoder.pt", "run.json", "membership.json")]
        ancestry_path = DESTINATION / f"{name}.ancestry.json"
        write(ancestry_path, {"kind": "native_assessment_ancestry_v1", "locally_fitted": True,
                              "historical_initial_weights": False, "completeness": "COMPLETE_LOCAL_ANCESTRY",
                              "parents": [str(DESTINATION / f"{p}.ancestry.json") for p in parent_names],
                              "artifacts": artifacts, "config_sha256": digest(config_path),
                              "statistics_sha256": digest(stats_path),
                              "fit_inputs": [{"input_path": str(train_npz), "cohort_path": str(cohort_path),
                                              "manifest_path": str(manifest_path), "statistics_path": str(stats_path),
                                              "config_path": str(config_path), "split_path": str(split_path)}],
                              "original_run_kind": run.get("mode", "core_frozen_readout"),
                              "scientific_quality": "NOT_ESTABLISHED"})
        selection_path = DESTINATION / f"{name}.selection.json"
        write(selection_path, {"selection_role": "development", "frozen_before_numeric_access": True,
                               "model_sha256": {str(weights): digest(weights)},
                               "config_sha256": digest(config_path), "statistics_sha256": digest(stats_path),
                               "actual_run_path": str(directory / "run.json"), "actual_run_sha256": item["run_sha"],
                               "is_finalist_selection": False})
        inventory[name] = {"model_paths": [str(weights)], "config_path": str(config_path),
                           "statistics_path": str(stats_path), "ancestry_path": str(ancestry_path),
                           "selection_path": str(selection_path), "parent_names": parent_names,
                           "method": artifact["config"]["method"], "seed": artifact["config"]["seed"],
                           "mode": run.get("mode", "core_frozen_readout"),
                           "kind": artifact["kind"], "loader": "builtin"}
    write(DESTINATION / "inventory.json", {"status": "DERIVED_CLOSED_SEED7_ANCESTRY_NOT_FINAL_SELECTION",
                                           "models": inventory, "numeric_corpus_decoded": False,
                                           "independent_ancestry_review": "NOT_RUN",
                                           "final_numeric_access": "NOT_RUN"})
    print(json.dumps({"status": "METADATA_DERIVED_NOT_ASSESSMENT_ADMITTED", "neural_models": len(neural),
                      "train_rows": len(next(iter(neural.values()))["rows"]), "output": str(DESTINATION)}))


if __name__ == "__main__":
    main()
