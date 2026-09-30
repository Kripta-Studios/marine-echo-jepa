"""Bind all 35 currently completed neural endpoints, without selecting or decoding data."""

import hashlib
import json
from pathlib import Path

from marine_echo.evaluation.native_ancestry_inventory import (
    required_sources,
    resolve_original_source_binding,
)

ROOT = Path(__file__).resolve().parents[1]
KIND = "native_ssl_weights_only_inference_v1"
BAND_KIND = "native_band_ssl_weights_only_inference_v1"
BAND_V2_KIND = "native_band_replication_ssl_weights_only_inference_v2"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    matrix_path = ROOT / "orchestration/native_development_comparison_v3.json"
    ledger_path = ROOT / "orchestration/native_ssl_run_ledger_v1.json"
    matrix, ledger = (json.loads(path.read_bytes()) for path in (matrix_path, ledger_path))
    original = {name: (Path(path).parent, KIND) for name, path in matrix["methods"].items()
                if name not in {"persistence", "seasonal24", "lightgbm", "chronos2_zero_shot"}}
    if len(original) != 24:
        raise ValueError("Exactly24 original neural endpoints required")
    base = ROOT / "outputs/native_acoustic_ssl_v1"
    band = {"band_" + method + "_short_probe": (base / f"band_{method}_seed7_h96_reviewed", BAND_KIND)
            for method in ("shared_ssl", "masked_ssl", "permuted_ssl", "random_frozen")}
    band["band_direct_end_to_end"] = (base / "band_direct_end_to_end_seed7_h96_reviewed", BAND_KIND)
    for method, modes in (("shared_ssl", ("frozen_readout", "full_finetune")),
                          ("masked_ssl", ("frozen_readout", "full_finetune")),
                          ("permuted_ssl", ("frozen_readout",))):
        for mode in modes:
            name = f"band_{method}_{mode}"
            band[name] = (base / f"{name}_seed7_h96", BAND_KIND)
    band["band_shared_ssl_short_probe_seed13"] = (base / "band_shared_ssl_seed13_h96_replication_v2", BAND_V2_KIND)
    destinations = {**original, **band}
    if len(destinations) != 35:
        raise ValueError("Exactly35 actual currently completed neural endpoints required")
    output = ROOT / "evidence/native-completed-inventory-v3"
    target = ROOT / "orchestration/native_completed_inventory_v3.json"
    if target.exists() or output.exists():
        raise FileExistsError("Preserve every earlier inventory or attempted manifest")
    bindings = {str(path): digest(path) for path in (*required_sources(), matrix_path, ledger_path, Path(__file__).resolve())}
    archive_index_path = ROOT / "evidence/ssl-research-v1/original-source-archives-v3/index.json"
    archive_index = json.loads(archive_index_path.read_bytes())
    if archive_index.get("runs_inspected") != 35 or archive_index.get("scientific_approval") is not False:
        raise ValueError("Exact non-approving source recovery index required")
    bindings[str(archive_index_path)] = digest(archive_index_path)
    archive_catalogs = [Path(path) for path in archive_index["catalogs"]]
    archive_records = [json.loads(path.read_bytes()) for path in archive_catalogs]
    for catalog, record in zip(archive_catalogs, archive_records, strict=True):
        bindings.update({str(path): digest(path) for path in (catalog, Path(record["path"]))})
    reports, endpoints = {}, {}
    for name, (folder, kind) in destinations.items():
        folder = folder.resolve()
        report_path = folder / "run.json"
        report = json.loads(report_path.read_bytes())
        if report.get("status") != "COMPLETED" or report.get("evidence_kind") != "REAL_TRAIN_DEVELOPMENT_FIT":
            raise ValueError("All explicitly named runs must be real and complete")
        reviews = {(ROOT / record["review"]).resolve() for record in ledger["runs"]
                   if record.get("output") and record.get("review")
                   and (ROOT / record["output"]).resolve() == folder}
        reviews = {path for path in reviews if digest(path) == report["review_sha256"]}
        configs = [Path(path) for path in report["bindings"]
                   if Path(path).is_relative_to(ROOT / "configs") and Path(path).suffix == ".json"
                   and json.loads(Path(path).read_bytes()) == report["config"]]
        if len(reviews) != 1 or len(configs) != 1:
            raise ValueError(f"One exact original review/config must resolve: {name}")
        review_path = next(iter(reviews))
        review = json.loads(review_path.read_bytes())
        method = report.get("core_config", report["config"])["method"]
        for group in (report["bindings"], review["bindings"]):
            for path, sha in group.items():
                try:
                    resolved = resolve_original_source_binding(path, sha, method, archive_records)
                except ValueError as error:
                    raise ValueError(f"Historical source resolution failed for {name}: {path}, recorded SHA256 {sha}") from error
                bindings[str(resolved)] = sha
                bindings[path] = digest(path)
        scalers = folder / "scalers.json"
        if not scalers.exists():
            # Older downstream artifacts embed the unchanged TRAIN scalers without a duplicate JSON.
            # The safe decoder must compare both selected/inference tensors' scaler state to this exact file.
            scalers = base / "shared_ssl_seed7_h96_cuda0/scalers.json"
        for path in (report_path, review_path, configs[0], scalers, *(folder / leaf for leaf in
                     ("membership.json", "inference.pt", "selected_encoder.pt"))):
            bindings[str(path)] = digest(path)
        config = report.get("core_config", report["config"])
        endpoints[name] = {"directory": str(folder), "kind": kind, "method": config["method"], "seed": config["seed"],
                           "mode": report.get("mode", "core_frozen_readout"), "parent": None,
                           "config_path": str(configs[0]), "review_path": str(review_path), "scalers_path": str(scalers)}
        reports[name] = report
    by_run_hash = {digest(Path(entry["directory"]) / "run.json"): name for name, entry in endpoints.items()}
    if len(by_run_hash) != len(endpoints):
        raise ValueError("No alias or duplicate endpoint selection")
    for name, report in reports.items():
        ancestry = report.get("supervised_ancestry")
        if ancestry is not None and ancestry.get("ancestor_run_sha256") is not None:
            parent = by_run_hash.get(ancestry["ancestor_run_sha256"])
            if parent is None:
                raise ValueError("Every actual fitted parent must be explicitly present")
            endpoints[name]["parent"] = parent
    train = {"input_path": str(ROOT / "data/processed/native_ssl_v1/train.npz"),
             "report_path": str(ROOT / "data/processed/native_ssl_v1/train.json"),
             "identity_cohort_path": str(ROOT / "orchestration/native_assessment_seed7_metadata_v1/train_cohort.json")}
    split_path = ROOT / "configs/native_ssl_split_v1.json"
    bindings.update({str(path): digest(path) for path in (split_path, *(Path(value) for value in train.values()))})
    manifest = {"kind": "native_research_inventory_manifest_v1", "purpose": "LOCAL_METADATA_DERIVATION_ONLY",
                "execution": "ROOT_LOCAL_COMPLETED_METADATA_AUDIT", "evidence_kind": "REAL_TRAIN_DEVELOPMENT_FIT",
                "device": "cpu", "owner_session_id": "01a0ef1a-b166-7f83-91b7-2a2aff7c1b10",
                "output_path": str(output), "split_path": str(split_path), "train": train,
                "endpoints": endpoints, "references": {}, "bindings": bindings,
                "source_archives": [str(path) for path in archive_catalogs]}
    with target.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": "ALL35_CURRENT_COMPLETED_NEURAL_ENDPOINTS_BOUND", "bindings": len(bindings),
                      "fit_authority": False, "final_selection": False, "numerical_corpus_decoded": False}))


if __name__ == "__main__":
    main()
