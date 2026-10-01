"""Private SYNTHETIC_CORRECTNESS_ONLY helpers; immutable MAIN imports."""
import importlib
import importlib.util
import io
import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER if BUILDER.name == "marine-echo-jepa" else BUILDER.parent / "marine-echo-jepa"
sys.path.insert(0, str(MAIN / "src"))
import marine_echo

marine_echo.__path__ = [str(MAIN / "src/marine_echo")]
for folder in ("training", "evaluation", "inference", "models"):
    package = importlib.import_module("marine_echo." + folder)
    package.__path__ = [str(MAIN / "src/marine_echo" / folder)]
from marine_echo.training import native_prefix_desktop_transfer_v3 as legacy
from marine_echo.training import native_cf_controls as controls
from marine_echo.evaluation import native_assessment_replication, native_comparison, native_product
from marine_echo.training import native_references
for folder in ("training", "evaluation"):
    package = importlib.import_module("marine_echo." + folder)
    if BUILDER != MAIN:
        package.__path__ = [str(BUILDER / "src/marine_echo" / folder), *package.__path__]


def prefix_module():
    try:
        return importlib.import_module("marine_echo.training.native_prefix_matched_transfer_v4")
    except ModuleNotFoundError:
        return legacy  # Baseline behavior red only, never a production fallback.


spec = importlib.util.spec_from_file_location("root_prefix_metadata_fixture_source", MAIN / "evidence/ssl-prefix-native-config-builder-v3/prefix_test_support.py")
metadata_helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metadata_helper)


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")


def write_npz(path, data):
    with path.open("xb") as stream:
        np.savez_compressed(stream, **data)


def physical_case():
    """Actual private disk codecs, synthetic metadata. No optimizer or fitting."""
    p = prefix_module()
    folder = BUILDER / "evidence/ssl-transfer-integration-builder-v1" / ("SYNTHETIC_CORRECTNESS_ONLY-" + uuid.uuid4().hex)
    folder.mkdir()
    registry = metadata_helper.metadata(cutoffs=tuple(range(18)) + tuple(range(888, 906)) + tuple(range(912, 930)))
    part = p.partitions(registry, 1)
    cfg = p.PrefixConfig(method="direct", mode="scratch_direct", correctness_smoke=True, updates=4, cadence=1, batch_size=8)
    backbone = p.core.Config(method="direct", width=8, latent=4, blocks=1, heads=2).to_dict()
    stats = {"channel_mean": [-50.] * 4, "channel_std": [5.] * 4, "target_mean": [-50.] * 3, "target_std": [5.] * 3}
    members = [{"site": "synthetic-TRAIN-site", "deployment": "synthetic-TRAIN-deployment", "archive": "synthetic-TRAIN-archive"}]
    prefix_rows = [r for r in registry["rows"] if (r["deployment"], r["row_id"]) in set(part["fit_rows"])]
    suffix_rows = [r for r in registry["rows"] if (r["deployment"], r["row_id"]) in set(part["suffix_rows"])]
    arrays = metadata_helper.arrays(prefix_rows, registry)
    dev = metadata_helper.arrays([], registry, "development", count=18)
    suffix = metadata_helper.arrays(suffix_rows, registry)
    suffix["corpus_role"] = np.array("suffix")
    suffix["cutoff"] = np.array([registry["intervals"][r["context_ids"][-1][0]]["source_interval_index"] for r in suffix_rows])
    write_npz(folder / "prefix.npz", arrays)
    write_npz(folder / "dev.npz", dev)
    write_npz(folder / "suffix.npz", suffix)
    (folder / "train.npz").write_bytes(b"SYNTHETIC_CORRECTNESS_ONLY opaque TRAIN ancestor; never decoded")
    documents = {"config.json": cfg.to_dict(), "backbone.json": backbone, "raw.json": registry,
        "prefix-cohort.json": {"complete": True, "role": "prefix", "rows": part["fit_rows"]},
        "dev-cohort.json": {"complete": True, "role": "development", "native_bounds_m": [0, 225], "rows": list(zip(dev["deployment"].tolist(), dev["row_id"].tolist()))},
        "suffix-cohort.json": {"complete": True, "role": "suffix", "rows": sorted(part["suffix_rows"])},
        "train-cohort.json": {"role": "train", "complete": True, "members": members}, "stats.json": stats,
        "sources.json": {"sources": registry["sources"], "quarantined_pings": [165]},
        "split.json": {"sources": [{"deployment": "synthetic-site", "archive_sha256": "synthetic-site-archive", "role": "test"}, {"deployment": "synthetic-original-dev", "archive_sha256": "synthetic-dev-archive", "role": "development"}, {"deployment": "synthetic-TRAIN-deployment", "archive_sha256": "synthetic-TRAIN-archive", "role": "train"}]}}
    for name, value in documents.items():
        write_json(folder / name, value)
    for name in ("protocol.txt", "lock.txt", "supervisor.txt", "executor.txt"):
        (folder / name).write_text("SYNTHETIC_CORRECTNESS_ONLY metadata authority fixture; not approval")
    node = {"kind": "native_prefix_train_ancestry_v1", "complete": True, "role": "train", "historical_initial_weights": False, "parents": [],
        "inputs": [{"cohort": "train-cohort.json", "npz": "train.npz", "raw_intervals": "train-intervals.json", "members": members}],
        "artifacts": [{"path": "stats.json", "sha256": p.sha((folder / "stats.json").read_bytes())}]}
    write_json(folder / "ancestry.json", node)
    write_json(folder / "train-intervals.json", {"kind": "native_prefix_train_interval_receipt_v1", "role": "train", "complete": True, "members": members, "source_npz_sha256": p.sha((folder / "train.npz").read_bytes()), "cohort_sha256": p.sha((folder / "train-cohort.json").read_bytes()), "split_sha256": p.sha((folder / "split.json").read_bytes())})
    write_json(folder / "selection.json", {"role": "original_development", "cohort_sha256": p.sha((folder / "dev-cohort.json").read_bytes()), "opportunities": [1, 2, 3, 4], "floor": 18, "aggregation": "equal_target_source_date_then_horizon_then_deployment"})
    manifest = {"kind": "native_prefix_matched_transfer_manifest_v4", "role": "prefix_transfer", "evidence_kind": p.EVIDENCE, "fixture_identity": p.EVIDENCE,
        "implementer_session_id": p.IMPLEMENTER_SESSION_ID, "coordinator_session_id": "SYNTHETIC-coordinator", "config": cfg.to_dict(),
        **{k: v for k, v in [("config_path", "config.json"), ("backbone_config", "backbone.json"), ("protocol", "protocol.txt"), ("split", "split.json"), ("selection", "selection.json"), ("dependency_lock", "lock.txt"), ("supervisor", "supervisor.txt"), ("source_manifest", "sources.json"), ("raw_intervals", "raw.json"), ("prefix_cohort", "prefix-cohort.json"), ("dev_cohort", "dev-cohort.json"), ("scalers", "stats.json"), ("scaler_ancestry", "ancestry.json"), ("prefix_npz", "prefix.npz"), ("dev_npz", "dev.npz"), ("numeric_access_review", "numeric.json")]}, "encoder": None}
    write_json(folder / "manifest.json", manifest)
    bindings = {str(path): p.sha(path.read_bytes()) for path in p.required_sources()}
    bindings.update({str(path): p.sha(path.read_bytes()) for path in folder.iterdir() if path.is_file()})
    common = {"role": "prefix_transfer", "evidence_kind": p.EVIDENCE, "implementer_session_id": p.IMPLEMENTER_SESSION_ID, "coordinator_session_id": "SYNTHETIC-coordinator", "reviewer_session_id": "SYNTHETIC-fixture-not-signature", "config": cfg.to_dict()}
    numeric = {**common, "status": "APPROVED_PREFIX_TRANSFER_NUMERIC_ACCESS", "scope": "prefix_only_numeric_fit", "bindings": bindings.copy(), "allowed_uses": ["prefix_fit", "original_development_selection", "suffix_metadata_only"]}
    write_json(folder / "numeric.json", numeric)
    bindings[str(folder / "numeric.json")] = p.sha((folder / "numeric.json").read_bytes())
    write_json(folder / "review.json", {**common, "status": "APPROVED_PREFIX_TRANSFER_PREFIT", "scope": "native_prefix_transfer_fit", "bindings": bindings, "allowed_cells": [{k: cfg.to_dict()[k] for k in ("method", "mode", "seed", "prefix_days", "family")}]})
    return SimpleNamespace(folder=folder, manifest=manifest, config=cfg, backbone=backbone, statistics=stats, registry=registry, prefix=arrays, dev=dev, suffix=suffix, partition=part)


def suffix_case():
    from marine_echo.evaluation import native_prefix_suffix_assessment_v1 as evaluator
    p, case = prefix_module(), physical_case()
    f = case.folder
    admitted = p.admit(f / "manifest.json", f / "review.json", f / "not-fitted")
    model = p.prepare_model(case.config, case.backbone, None, case.statistics)
    feature = model.feature_ancestor
    # Synthetic typed snapshot for actual Torch replay, never a fitted result.
    artifact = {"kind": evaluator.NEURAL_KIND, "config": case.config.to_dict(), "backbone_config": case.backbone, "architecture": p.ARCHITECTURES[case.config.family],
        "scalers": case.statistics, "model": p.core.cpu_state(model), "selected_step": 1, "identities": admitted.identities, "feature_ancestor": feature, "training_kind": "scratch_supervised_prefix_encoder", "zero_shot": False}
    (f / "synthetic-model.pt").write_bytes(p.encode_checkpoint(artifact))
    receipt = {"kind": "native_prefix_matched_transfer_completion_v4", "status": "COMPLETED", "evidence_kind": p.EVIDENCE, "config": case.config.to_dict(), "partition": case.partition, "bindings": admitted.identities,
        "zero_shot": False, "suffix_numerical_access": False, "label_counts": case.prefix["target_observed"].sum(0).tolist(), "unique_prefix_rows": len(case.prefix["x"]), "selected_step": 1, "feature_ancestor": feature}
    write_json(f / "synthetic-completion.json", receipt)
    cells = [{"name": "scratch", "deployment": "synthetic-site", "kind": evaluator.NEURAL_KIND, "zero_shot": False,
        "model": str(f / "synthetic-model.pt"), "completion": str(f / "synthetic-completion.json"), "fit_manifest": str(f / "manifest.json"), "fit_review": str(f / "review.json"),
        **{k: getattr(case.config, k) for k in ("method", "seed", "mode", "prefix_days")}}, {"name": "persistence", "kind": "persistence", "zero_shot": True, "model": None, "deployment": "synthetic-site"}]
    manifest = {"kind": "native_prefix_suffix_assessment_manifest_v1", "role": evaluator.ROLE, "evidence_kind": p.EVIDENCE, "fixture_identity": p.EVIDENCE,
        "implementer_session_id": p.IMPLEMENTER_SESSION_ID, "coordinator_session_id": "SYNTHETIC-coordinator", "device": "cpu", "recipe": evaluator.RECIPE, "cells": cells, "reference_name": "persistence",
        **{k: str(f / v) for k, v in [("protocol", "protocol.txt"), ("split", "split.json"), ("selection_freeze", "freeze.json"), ("dependency_lock", "lock.txt"), ("executor", "executor.txt"), ("cohort", "suffix-cohort.json"), ("raw_intervals", "raw.json"), ("input_npz", "suffix.npz"), ("software_review", "suffix-software.json"), ("numeric_access_review", "suffix-access.json")]}}
    model_bindings = {str(f / name): p.sha((f / name).read_bytes()) for name in ("synthetic-model.pt", "synthetic-completion.json", "manifest.json", "config.json", "backbone.json", "stats.json")}
    write_json(f / "freeze.json", {"kind": "native_prefix_suffix_selection_freeze_v1", "cells": cells, "selection": "original_development_only", "suffix_selection": False, "model_bindings": model_bindings, "frozen_at": "2020-01-01T00:00:00+00:00"})
    write_json(f / "suffix-manifest.json", manifest)
    bindings = {str(path): p.sha(path.read_bytes()) for path in evaluator.required_sources()}
    bindings.update({str(path): p.sha(path.read_bytes()) for path in f.iterdir() if path.is_file()})
    common = {"allowed_roles": [evaluator.ROLE], "evidence_kind": p.EVIDENCE, "implementer_session_id": p.IMPLEMENTER_SESSION_ID, "coordinator_session_id": "SYNTHETIC-coordinator", "reviewer_session_id": "SYNTHETIC-fixture-not-a-real-review"}
    write_json(f / "suffix-software.json", {**common, "status": "APPROVED_PREFIX_SUFFIX_SOFTWARE", "scope": "prefix_suffix_assessment_software", "bindings": bindings.copy()})
    bindings[str(f / "suffix-software.json")] = p.sha((f / "suffix-software.json").read_bytes())
    write_json(f / "suffix-access.json", {**common, "status": "APPROVED_PREFIX_SUFFIX_NUMERIC_ACCESS", "scope": "prefix_suffix_numeric_assessment", "bindings": bindings.copy(), "allowed_uses": ["frozen_adapted_suffix_assessment"], "selection_freeze_sha256": p.sha((f / "freeze.json").read_bytes()), "issued_at": "2020-01-02T00:00:00+00:00"})
    bindings[str(f / "suffix-access.json")] = p.sha((f / "suffix-access.json").read_bytes())
    write_json(f / "suffix-review.json", {**common, "status": "APPROVED_PREFIX_SUFFIX_ASSESSMENT_EXECUTION", "scope": "prefix_suffix_assessment_execution", "bindings": bindings, "runtime": {"device": "cpu", "output": str(f / "suffix-output")}, "allowed_cells": cells})
    return case


def rebind_suffix_fixture(case):
    """Re-bind malicious toy metadata to test semantic gates, not stale hashes.

    Private synthetic identities only; never a genuine reviewer signature.
    """
    from marine_echo.evaluation import native_prefix_suffix_assessment_v1 as evaluator
    f, p = case.folder, prefix_module()
    freeze = json.loads((f / "freeze.json").read_text())
    freeze["model_bindings"] = {name: p.sha(Path(name).read_bytes()) for name in freeze["model_bindings"]}
    write_json(f / "freeze.json", freeze)
    excluded = {"suffix-software.json", "suffix-access.json", "suffix-review.json"}
    bindings = {str(path): p.sha(path.read_bytes()) for path in evaluator.required_sources()}
    bindings.update({str(path): p.sha(path.read_bytes()) for path in f.iterdir() if path.is_file() and path.name not in excluded})
    for name in ("suffix-software.json", "suffix-access.json", "suffix-review.json"):
        record = json.loads((f / name).read_text())
        record["bindings"] = bindings.copy()
        if name == "suffix-access.json":
            record["selection_freeze_sha256"] = p.sha((f / "freeze.json").read_bytes())
        write_json(f / name, record)
        bindings[str(f / name)] = p.sha((f / name).read_bytes())


def control_parent_metadata():
    """Actual control report schema; opaque private weights never decoded here."""
    p, case = prefix_module(), physical_case()
    f = case.folder
    cfg = controls.Config(width=8, latent=8, blocks=1, updates=4, cadence=1, batch_size=4)
    backbone = cfg.to_dict()
    inputs = controls.RunInputs(**{key: f / name for key, name in {
        "train": "train.npz", "dev": "dev.npz", "train_cohort": "train-cohort.json",
        "dev_cohort": "dev-cohort.json", "split": "split.json", "adr0016": "protocol.txt",
        "protocol": "protocol.txt", "config": "control-config.json", "review": "control-review.json",
        "executor": "executor.txt", "budget": "control-budget.json"}.items()})
    write_json(inputs.config, backbone)
    write_json(inputs.budget, {"evidence_kind": p.EVIDENCE, "fixture_identity": f.name})
    bindings = {str(path): p.sha(path.read_bytes()) for path in controls.required_paths(inputs, cfg)}
    review = {"status": p.EVIDENCE, "evidence_kind": p.EVIDENCE,
        "reviewer_session_id": "SYNTHETIC-parent-review-not-signature",
        "implementer_session_id": controls.IMPLEMENTER_SESSION_ID,
        "root_coordinator_session_id": "SYNTHETIC-coordinator",
        "allowed_roles": ["train", "development"], "allowed_methods": [cfg.method],
        "allowed_modes": [cfg.mode], "allowed_seeds": [cfg.seed], "bindings": bindings,
        **{name: p.sha(path.read_bytes()) for name, path in [("train_npz_sha256", inputs.train),
            ("dev_npz_sha256", inputs.dev), ("split_sha256", inputs.split)]}}
    write_json(inputs.review, review)
    write_json(f / "control-inputs.json", {k: str(v) if v is not None else None for k, v in vars(inputs).items()})
    rows = ["synthetic-train-" + str(i) for i in range(18)]
    sequence = []
    for step in range(4):
        ids = controls.supervised_indices(list(range(18)), cfg, step)
        sequence.append({"step": step, "indices": ids.tolist(), "row_ids": [rows[i] for i in ids], "sha256": p.core.sequence_hash(ids)})
    membership = {"train_row_ids": rows, "train_deployments": ["synthetic-TRAIN-deployment"] * 18,
        "train_archive_sha256": ["synthetic-TRAIN-archive"] * 18, "ssl_updates": 0,
        "ssl_eligible_indices": [], "supervised_indices": list(range(18)), "sequence": sequence,
        "sequence_sha256": p.sha(json.dumps(sequence, sort_keys=True, separators=(",", ":")).encode())}
    ancestry = {"mode": cfg.mode, "ssl_only": False, "ssl_updates": 0,
        "ancestor_encoder_sha256": None, "ancestor_run_sha256": None,
        "initialization_seed": 7, "readout_initialization_seed": 100007,
        "supervised_updates": 4, "selected_supervised_step": 1, "encoder_supervised_updates": 0}
    encoder_raw = b"SYNTHETIC_CORRECTNESS_ONLY opaque parent encoder for metadata gate"
    parent = {"config": backbone, "core_config": controls.core_config(cfg).to_dict(), "status": "COMPLETED",
        "evidence_kind": p.EVIDENCE, "test_access": "NOT_RUN", "ssl_updates": 0, "fitted_weight_ancestors": [],
        "training_kind": controls.TRAINING_KINDS[cfg.method], "mode": cfg.mode,
        "review_sha256": p.sha(inputs.review.read_bytes()), "reviewer_session_id": review["reviewer_session_id"],
        "selected_encoder_sha256": p.sha(encoder_raw), "supervised_ancestry": ancestry,
        "supervised_updates": 4, "selected_supervised_step": 1, "bindings": bindings}
    manifest = {**case.manifest, "parent_config": str(inputs.config), "parent_review": str(inputs.review),
        "parent_inputs": str(f / "control-inputs.json"), "dev_npz": str(inputs.dev)}
    from dataclasses import replace
    config = replace(case.config, family="cf", method=cfg.method, mode="frozen_readout")
    def bound(path):
        return Path(path).read_bytes()
    def doc(path):
        path = p._path(path, f)
        return path, p._json(bound(path))
    return SimpleNamespace(config=config, manifest=manifest, parent=parent, backbone=backbone,
        encoder_raw=encoder_raw, membership=membership, inputs=inputs, base=f, bound=bound, doc=doc)
