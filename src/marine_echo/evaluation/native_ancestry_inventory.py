"""Derive exact local fitted ancestry without corpus decoding or model construction.

The explicit endpoint manifest is an input inventory, never a selection or access
approval. ROOT audits real safe artifacts after integration. Builder fixtures are
private SYNTHETIC_CORRECTNESS_ONLY. No provenance path is executed or imported.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import stat
import struct
import time
from pathlib import Path

ARCHITECTURE = "nonlinear_frequency_conditioned_v1"
REAL = "REAL_TRAIN_DEVELOPMENT_FIT"
SYNTHETIC = "SYNTHETIC_CORRECTNESS_ONLY"
ASSESSMENT_EVIDENCE = "REVIEWED_FROZEN_ASSESSMENT"
RAM_LIMIT = 22 * 1024**3
KINDS = {
    "native_ssl_weights_only_inference_v1": (
        "native_ssl_selected_encoder_v1",
        "native_downstream_supervised_encoder_v1",
        "native_ssl",
        "native_downstream",
    ),
    "native_band_ssl_weights_only_inference_v1": (
        "native_band_ssl_selected_encoder_v1",
        "native_band_downstream_supervised_encoder_v1",
        "native_band_ssl",
        "native_band_downstream",
    ),
    "native_band_replication_ssl_weights_only_inference_v2": (
        "native_band_replication_ssl_selected_encoder_v2",
        "native_band_replication_downstream_supervised_encoder_v2",
        "native_band_replication_ssl",
        "native_band_replication_downstream",
    ),
}
METHODS = {"shared_ssl", "masked_ssl", "permuted_ssl", "random_frozen", "direct", "cf_jepa"}
LABELS = {
    "shared_ssl": "SSL pretrained features",
    "masked_ssl": "Non-JEPA masked SSL features",
    "cf_jepa": "CF SSL EMA features",
    "permuted_ssl": "Permuted pairing SSL control",
    "random_frozen": "Untrained random feature control",
    "direct": "Supervised features",
}
ENDPOINT_FIELDS = {
    "directory",
    "kind",
    "method",
    "seed",
    "mode",
    "parent",
    "config_path",
    "review_path",
}
BASE_ARTIFACT_FIELDS = {
    "kind",
    "model",
    "config",
    "scalers",
    "bindings",
    "selected_pretrain_step",
    "architecture",
    "evidence_kind",
    "correctness_smoke",
    "downstream_config",
    "supervised_ancestry",
    "review_sha256",
}
SELECTED_FIELDS = {
    "kind",
    "encoder",
    "config",
    "scalers",
    "bindings",
    "architecture",
    "evidence_kind",
    "correctness_smoke",
    "selected_pretrain_step",
    "supervised_ancestry",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def json_document(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON metadata key.")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Nonfinite JSON metadata.")

    document = json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(document, dict):
        raise TypeError("Metadata object required.")
    return document


def _regular(path):
    path = Path(os.path.abspath(path))
    for parent in (path, *path.parents):
        if parent.exists() or parent.is_symlink():
            info = parent.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError("Symlink/reparse metadata identity is unsafe.")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Regular immutable input required.")
    return path


def _digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def source_root():
    checkout = Path(__file__).resolve().parents[3]
    return (
        checkout.parent / "marine-echo-jepa"
        if checkout.name == "marine-jepa-vnext-builder"
        else checkout
    )


def required_sources():
    """Hash-bind exact static local source closure, without importing trainers."""
    root = source_root()
    pending = [
        Path(__file__),
        Path(__file__).resolve().parents[3] / "tools/prepare_native_research_inventory.py",
    ]
    pending += [
        root / name
        for name in (
            "tools/prepare_native_seed7_assessment_metadata.py",
            "src/marine_echo/evaluation/native_assessment_replication.py",
            "src/marine_echo/training/native_prefix_transfer.py",
            "src/marine_echo/data/native_transfer_corpus.py",
            "src/marine_echo/training/native_references.py",
        )
    ]
    found = set()
    while pending:
        path = pending.pop().resolve()
        if path in found:
            continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_bytes())):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] + [node.module + "." + item.name for item in node.names]
            elif isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            for name in names:
                if not name.startswith("marine_echo"):
                    continue
                parts = name.split(".")
                for n in range(1, len(parts) + 1):
                    candidate = root / "src" / Path(*parts[:n])
                    for file in (candidate.with_suffix(".py"), candidate / "__init__.py"):
                        if file.is_file():
                            pending.append(file)
    return sorted(found)


def validate_parent_graph(entries):
    active, done, order = set(), set(), []

    def visit(name):
        if name not in entries:
            raise ValueError("Missing declared completed parent.")
        if name in active:
            raise ValueError("Cyclic local ancestry is incomplete.")
        if name in done:
            return
        active.add(name)
        parent = entries[name].get("parent")
        if parent is not None:
            if not isinstance(parent, str):
                raise ValueError("Explicit parent name required.")
            visit(parent)
        active.remove(name)
        done.add(name)
        order.append(name)

    for name in entries:
        visit(name)
    return order


def _identities(sources):
    return [
        {
            "archive_id": str(s["file_id"]),
            "deployment_id": s["deployment"],
            "site_id": s["site"],
            "source_ids": [s["archive_sha256"]],
        }
        for s in sources
    ]


def _identity_set(entries):
    if not isinstance(entries, list) or not entries:
        raise ValueError("Whole-deployment identities are incomplete.")
    result = set()
    for item in entries:
        if (
            set(item) != {"archive_id", "deployment_id", "site_id", "source_ids"}
            or not isinstance(item["source_ids"], list)
            or not item["source_ids"]
        ):
            raise ValueError("Explicit archive/deployment/site/raw source identities required.")
        identity = (
            item["archive_id"],
            item["deployment_id"],
            item["site_id"],
            tuple(item["source_ids"]),
        )
        if identity in result:
            raise ValueError("Duplicate whole-deployment identities.")
        result.add(identity)
    return result


def _rows(values):
    if not isinstance(values, list) or not values:
        raise ValueError("Complete issued TRAIN row identities required.")
    if any(
        not isinstance(row, list)
        or len(row) != 2
        or any(not isinstance(v, str) or not v for v in row)
        for row in values
    ):
        raise ValueError("Invalid raw TRAIN row identities.")
    rows = [tuple(row) for row in values]
    if len(set(rows)) != len(rows):
        raise ValueError("Duplicate TRAIN issuance identity.")
    return rows


def _fields(module, class_name):
    path = source_root() / f"src/marine_echo/training/{module}.py"
    tree = ast.parse(path.read_bytes())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)}


def _integer_fields(module, class_name):
    tree = ast.parse((source_root() / f"src/marine_echo/training/{module}.py").read_bytes())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return {
        n.target.id
        for n in cls.body
        if isinstance(n, ast.AnnAssign) and "int" in ast.unparse(n.annotation)
    }


def _scalers(value):
    expected = {"channel_mean": 4, "channel_std": 4, "target_mean": 3, "target_std": 3}
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("Complete original TRAIN scalers required.")
    for name, length in expected.items():
        values = value[name]
        if (
            not isinstance(values, list)
            or len(values) != length
            or any(type(v) not in (float, int) or not math.isfinite(v) for v in values)
        ):
            raise ValueError("Invalid or nonfinite scaler metadata.")
        if name.endswith("std") and any(v <= 0 for v in values):
            raise ValueError("Scaler standard deviations must be positive.")


def _config(config, entry, *, downstream=False):
    modules = KINDS[entry["kind"]]
    cls, module = ("DownstreamConfig", modules[3]) if downstream else ("Config", modules[2])
    if not isinstance(config, dict) or set(config) != _fields(module, cls):
        raise ValueError("Exact saved config schema required; no unknown/default fields.")
    for value in config.values():
        if type(value) not in (str, int, float) or (
            type(value) is float and not math.isfinite(value)
        ):
            raise ValueError("Unsafe or nonfinite config metadata.")
    for name in _integer_fields(module, cls):
        if type(config[name]) is not int:
            raise ValueError("Exact integer config dimensions required.")
    if (
        config["method"] != entry["method"]
        or config["seed"] != entry["seed"]
        or config["history"] != 96
    ):
        raise ValueError("Typed method/seed/history mismatch.")
    if entry["seed"] not in (7, 13, 23) or type(entry["seed"]) is not int:
        raise ValueError("Unadmitted fixed seed.")
    band = entry["kind"] != "native_ssl_weights_only_inference_v1"
    if band and (config["architecture"] != ARCHITECTURE or entry["method"] == "cf_jepa"):
        raise ValueError("Band architecture/method mismatch.")
    if entry["kind"] == "native_band_ssl_weights_only_inference_v1" and entry["seed"] != 7:
        raise ValueError("Band V1 has seed7 only; preserve typed V2 distinction.")
    if (
        config["batch_size"],
        config["lr"],
        config["weight_decay"],
        config["patience"],
        config["min_daily_anchors"],
    ) != (64, 0.0003, 0.0001, 4, 18):
        raise ValueError("Unknown fixed optimizer/selection variant.")
    if downstream:
        frozen = entry["mode"] == "frozen_readout"
        if (
            config["mode"],
            config["updates"],
            config["cadence"],
            config["gradient_clip"],
            config["selection_policy"],
        ) != (
            entry["mode"],
            2000 if frozen else 3000,
            500 if frozen else 750,
            1.0,
            "scheduled_native_daily_pinball_earliest_strict_improvement_v1",
        ):
            raise ValueError("Downstream configuration variant or wrong mode.")
    elif (
        config["width"],
        config["latent"],
        config["blocks"],
        config["heads"],
        config["sigreg_weight"],
        config["pretrain_cadence"],
        config["readout_cadence"],
    ) != (
        192, 64, 4, 4, 0.03,
        750 if entry["method"] == "direct" and entry["mode"] != "direct_end_to_end" else 1500,
        250,
    ):
        raise ValueError("Unknown backbone/objective/schedule variant.")
    if not downstream and (
        config["selection_policy"] != "train_only_frozen_probe_per_candidate_daily_dev_patience4"
        or (
            config["cf_width"],
            config["cf_latent"],
            config["cf_blocks"],
            config["cf_lr"],
            config["cf_weight_decay"],
        )
        != (256, 128, 5, 0.00034, 0.05)
    ):
        raise ValueError("Unknown saved source-defined selection/CF variant.")
    if (
        not downstream
        and entry["mode"] == "core_frozen_readout"
        and (config["pretrain_updates"], config["readout_updates"])
        != (
            0
            if entry["method"] == "random_frozen"
            else 3000
            if entry["method"] == "direct"
            else 6000,
            500,
        )
    ):
        raise ValueError("Unknown completed pretraining/probe recipe.")


def _memory():
    import psutil

    # This module never spawns a child. Check this single CPU process without
    # enumerating unrelated system processes for every state tensor.
    rss = psutil.Process().memory_info().rss
    if rss >= RAM_LIMIT:
        raise MemoryError("Metadata inventory owned RSS reaches 22 GiB.")
    return rss


def _safe_load(path):
    import torch

    _memory()
    artifact = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    _memory()
    if not isinstance(artifact, dict):
        raise TypeError("Safe owned artifact mapping required.")
    return artifact


def _state(value):
    import torch

    if not isinstance(value, dict) or not value:
        raise ValueError("Complete tensor state required.")
    for key, tensor in value.items():
        if (
            not isinstance(key, str)
            or not key
            or type(tensor) is not torch.Tensor
            or tensor.device.type != "cpu"
            or tensor.layout != torch.strided
            or tensor.dtype not in (torch.float32, torch.int64)
        ):
            raise ValueError("Unsafe tensor state/dtype/layout.")
        if tensor.numel() == 0 or not torch.isfinite(tensor).all().item():
            raise ValueError("Missing or nonfinite state tensor.")
        _memory()


def _equal_state(left, right):
    import torch

    return set(left) == set(right) and all(
        left[k].dtype == right[k].dtype
        and left[k].shape == right[k].shape
        and torch.equal(left[k], right[k])
        for k in left
    )


def _subset(state, prefix):
    result = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    if not result:
        raise ValueError("Actual encoder/readout tensor branch is absent.")
    return result


def _check_membership(member, cohort_rows, train_sources, report):
    names = ("train_deployments", "train_row_ids", "train_archive_sha256")
    if (
        any(not isinstance(member.get(k), list) for k in names)
        or len({len(member[k]) for k in names}) != 1
    ):
        raise ValueError("Partial fitted TRAIN membership.")
    rows = list(zip(member[names[0]], member[names[1]], strict=True))
    if rows != cohort_rows or len(rows) != report["issued"]:
        raise ValueError("Exact complete TRAIN row membership differs.")
    allowed = {s["deployment"]: s["archive_sha256"] for s in train_sources}
    if any(
        allowed.get(dep) != sha for dep, sha in zip(member[names[0]], member[names[2]], strict=True)
    ):
        raise ValueError("Fitted archive/deployment identity outside original TRAIN.")
    for field in ("supervised_indices", "ssl_eligible_indices"):
        if field not in member and field == "ssl_eligible_indices":
            continue
        values = member.get(field)
        if (
            not isinstance(values, list)
            or len(set(values)) != len(values)
            or any(type(i) is not int or not 0 <= i < len(rows) for i in values)
        ):
            raise ValueError("Invalid fitted row/eligibility indices.")
        if field == "ssl_eligible_indices" and len(values) != report["ssl_eligible"]:
            raise ValueError("TRAIN SSL eligibility is incomplete.")
    sequence = member.get("sequence")
    if not isinstance(sequence, list):
        raise TypeError("Actual fitted sample sequence missing.")
    for item in sequence:
        if not isinstance(item, dict) or ("indices" in item) == ("context_indices" in item):
            raise ValueError("Exactly one actual fitted sample layout required.")
        ids = item.get("context_indices", item.get("indices"))
        for values in (ids, *([item["target_indices"]] if "target_indices" in item else [])):
            if (
                not isinstance(values, list)
                or not values
                or any(type(i) is not int or not 0 <= i < len(rows) for i in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError("Fitted sequence escapes TRAIN or duplicates samples.")
        def sample_hash(values):
            return hashlib.sha256(struct.pack("<" + "q" * len(values), *values)).hexdigest()

        digest_key = "context_sha256" if "context_indices" in item else "sha256"
        if digest_key in item and item[digest_key] != sample_hash(ids):
            raise ValueError("Actual sample context hash differs.")
        if "context_indices" in item and digest_key not in item:
            raise ValueError("Actual SSL context hash missing.")
        if "target_indices" in item:
            targets = item["target_indices"]
            if sorted(targets) != sorted(ids) or item.get("target_multiset_sha256") != sample_hash(sorted(targets)):
                raise ValueError("Actual SSL paired/permuted target multiset differs.")
        elif "target_multiset_sha256" in item:
            raise ValueError("Target hash without actual target identities.")
        if "row_ids" in item and item["row_ids"] != [rows[i][1] for i in ids]:
            raise ValueError("Fitted sequence row identities differ.")
    if "sequence_sha256" in member:
        raw = json.dumps(sequence, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(raw).hexdigest() != member["sequence_sha256"]:
            raise ValueError("Actual sample-sequence hash differs.")
    return rows


def resolve_original_source_binding(path, expected, method, archives):
    """Find exact historical bytes; this grants no runtime or compatibility approval."""
    path = _regular(path)
    if sha256(path) == expected:
        return path
    matches = []
    for record in archives:
        if (not isinstance(record, dict)
                or set(record) != {"original_path", "path", "sha256", "methods", "status"}
                or record.get("status") != "ORIGINAL_SOURCE_PRESERVED_NOT_COMPATIBILITY_APPROVAL"
                or not isinstance(record.get("methods"), list)
                or not record["methods"] or any(value not in METHODS for value in record["methods"])
                or not _digest(record.get("sha256"))):
            raise ValueError("Exact historical source receipt required; no compatibility promotion.")
        original, archived = _regular(record["original_path"]), _regular(record["path"])
        allowed_source = original.is_relative_to(source_root() / "src/marine_echo") or original.is_relative_to(source_root() / "tools")
        if original.suffix != ".py" or not allowed_source:
            raise ValueError("Historical resolution is restricted to local Python source.")
        if sha256(archived) != record["sha256"]:
            raise ValueError("Preserved source bytes differ.")
        if original == path and record["sha256"] == expected and method in record["methods"]:
            matches.append(archived)
    if len(matches) != 1:
        raise ValueError("One exact method-scoped historical source archive required.")
    return matches[0]


def reserved_groups(split, *, evidence):
    """Validate original identities; development may share a site, final test may not."""
    if evidence not in (REAL, SYNTHETIC):
        raise ValueError("Unknown split evidence kind.")
    if (
        split.get("schema_version") != "native_acoustic_ssl_v1"
        or split.get("support_history") != 96
        or not isinstance(split.get("sources"), list)
    ):
        raise ValueError("Original native split schema/roles required.")
    groups = {role: [] for role in ("train", "development", "final_test")}
    all_ids = {key: set() for key in ("deployment", "archive_sha256", "file_id")}
    for source in split["sources"]:
        if (
            source.get("role") not in groups
            or not source.get("site")
            or not _digest(source.get("archive_sha256"))
            or type(source.get("file_id")) is not int
        ):
            raise ValueError("Unknown/unsafe original source identity or role.")
        for key, seen in all_ids.items():
            if source.get(key) in seen or not source.get(key):
                raise ValueError("Conflicting fitted/reserved source identity.")
            seen.add(source[key])
        groups[source["role"]].append(source)
    expected = {61937263, 61937272, 61937275, 61937281} if evidence == REAL else {18593}
    if any(not group for group in groups.values()) or {s["file_id"] for s in groups["train"]} != expected:
        raise ValueError("Exact historical TRAIN archive reservation required; row counts are not archive IDs.")
    if {s["site"] for s in groups["final_test"]} & {
        s["site"] for s in groups["train"] + groups["development"]
    }:
        raise ValueError("Reserved final-test site overlaps fitted or development sites.")
    return groups


def _admit(manifest_path, output_path):
    manifest_path = _regular(manifest_path)
    manifest = json_document(manifest_path)
    fields = {
        "kind",
        "purpose",
        "execution",
        "evidence_kind",
        "device",
        "output_path",
        "bindings",
        "split_path",
        "train",
        "endpoints",
        "references",
        "owner_session_id",
        "source_archives",
    }
    if (
        set(manifest) - fields
        or manifest.get("kind") != "native_research_inventory_manifest_v1"
        or manifest.get("purpose") != "LOCAL_METADATA_DERIVATION_ONLY"
        or manifest.get("execution") != "ROOT_LOCAL_COMPLETED_METADATA_AUDIT"
        or manifest.get("device") != "cpu"
        or not manifest.get("owner_session_id")
    ):
        raise ValueError("Explicit versioned CPU metadata manifest required.")
    evidence = manifest.get("evidence_kind")
    if evidence not in (REAL, SYNTHETIC):
        raise ValueError("Unknown inventory evidence kind.")
    if evidence == REAL and Path(__file__).resolve().parents[3].name == "marine-jepa-vnext-builder":
        raise ValueError("Actual completed weights are ROOT-only; builder cannot decode them.")
    output = Path(os.path.abspath(output_path))
    if (
        str(output) != manifest.get("output_path")
        or output.exists()
        or output.is_symlink()
        or not output.parent.is_dir()
    ):
        raise FileExistsError("Explicit fresh protected output required.")
    for parent in output.parents:
        info = parent.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Unsafe output parent identity.")
    if evidence == SYNTHETIC and (
        "SYNTHETIC_CORRECTNESS_ONLY" not in str(manifest_path.parent)
        or output.parent != manifest_path.parent
    ):
        raise ValueError("Private visibly synthetic fixture identity required.")
    bindings = manifest.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("Exact source/input/endpoint bindings required before safe decode.")
    for name, digest in bindings.items():
        p = _regular(name)
        if str(p) != name or not _digest(digest) or sha256(p) != digest:
            raise ValueError("Stale/unsafe immutable binding.")
        if output == p or output in p.parents:
            raise ValueError("Output overlaps immutable inputs.")

    def bound(path):
        p = _regular(path)
        if bindings.get(str(p)) != sha256(p):
            raise ValueError("Missing required source/artifact binding.")
        if evidence == SYNTHETIC and p not in sources and manifest_path.parent not in p.parents:
            raise ValueError("Synthetic fixtures cannot decode actual trained artifacts.")
        return p

    sources = set(required_sources())
    for source in sources:
        bound(source)
    source_archives = []
    for catalog in manifest.get("source_archives", []):
        record = json_document(bound(catalog))
        bound(record["path"])
        bound(record["original_path"])
        source_archives.append(record)
    split_path = bound(manifest["split_path"])
    if evidence == REAL and split_path != source_root() / "configs/native_ssl_split_v1.json":
        raise ValueError("Actual immutable native split path required.")
    split = json_document(split_path)
    groups = reserved_groups(split, evidence=evidence)
    train = manifest["train"]
    if set(train) != {"input_path", "report_path", "identity_cohort_path"}:
        raise ValueError("Exact original TRAIN input/report/identity-cohort paths required.")
    train = {key: bound(value) for key, value in train.items()}
    report, cohort = (
        json_document(train["report_path"]),
        json_document(train["identity_cohort_path"]),
    )
    train_sha, split_sha = sha256(train["input_path"]), sha256(split_path)
    if (
        report.get("role") != "train"
        or report.get("npz_sha256") != train_sha
        or report.get("split_sha256") != split_sha
        or type(report.get("issued")) is not int
        or report["issued"] <= 0
        or report.get("status")
        != ("MATERIALIZED_REAL_NATIVE_PRODUCTS" if evidence == REAL else SYNTHETIC)
    ):
        raise ValueError("Original TRAIN byte/report identity or evidence differs.")
    rows = _rows(cohort.get("rows"))
    if (
        cohort.get("role") != "train"
        or cohort.get("input_npz_sha256") != train_sha
        or cohort.get("split_sha256") != split_sha
        or len(rows) != report["issued"]
        or cohort.get("complete", True) is not True
        or _identity_set(cohort.get("identities")) != _identity_set(_identities(groups["train"]))
    ):
        raise ValueError("Complete independently bound TRAIN identity cohort differs.")
    if evidence == SYNTHETIC and cohort.get("evidence_kind") != SYNTHETIC:
        raise ValueError("Synthetic cohort must retain synthetic identity.")
    counts = {s["deployment"]: s["issued"] for s in report.get("source_reports", [])}
    if counts != {
        s["deployment"]: sum(d == s["deployment"] for d, _ in rows) for s in groups["train"]
    }:
        raise ValueError("Whole original TRAIN source-count identity is incomplete.")
    entries = manifest.get("endpoints")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Explicit completed endpoint inventory required; no enumeration.")
    order = validate_parent_graph(entries)
    records, seen_runs = {}, set()
    for name in order:
        entry = entries[name]
        if (
            not re.fullmatch("[A-Za-z0-9][A-Za-z0-9_.-]*", name)
            or not ENDPOINT_FIELDS <= set(entry)
            or set(entry) - ENDPOINT_FIELDS - {"scalers_path"}
            or entry["kind"] not in KINDS
            or entry["method"] not in METHODS
            or entry["mode"]
            not in {"core_frozen_readout", "frozen_readout", "full_finetune", "direct_end_to_end"}
        ):
            raise ValueError("Unknown endpoint kind/method/mode/metadata.")
        directory = Path(entry["directory"])
        paths = {
            key: bound(directory / filename)
            for key, filename in (
                ("run", "run.json"),
                ("membership", "membership.json"),
                ("model", "inference.pt"),
                ("selected", "selected_encoder.pt"),
            )
        }
        paths["scalers"] = bound(entry.get("scalers_path", directory / "scalers.json"))
        paths.update(config=bound(entry["config_path"]), review=bound(entry["review_path"]))
        run, member, review = (json_document(paths[k]) for k in ("run", "membership", "review"))
        if sha256(paths["run"]) in seen_runs:
            raise ValueError("Duplicate completed endpoint aliases.")
        seen_runs.add(sha256(paths["run"]))
        if (
            run.get("status") != "COMPLETED"
            or run.get("evidence_kind") != evidence
            or run.get("test_access") != "NOT_RUN"
            or run.get("historical_initial_weights", False) is not False
        ):
            raise ValueError("Incomplete/synthetic-promoted/historical fitted report.")
        mode = run.get("mode", "core_frozen_readout")
        if mode != entry["mode"] or (mode == "direct_end_to_end" and entry["method"] != "direct"):
            raise ValueError("Accurate supervised/control mode required.")
        for key in ("inference", "membership", "review"):
            path = paths[{"inference": "model"}.get(key, key)]
            if run.get(key + "_sha256") != sha256(path):
                raise ValueError("Actual completed report artifact hash differs.")
        if "selected_encoder_sha256" in run and run["selected_encoder_sha256"] != sha256(
            paths["selected"]
        ):
            raise ValueError("Actual selected encoder hash differs.")
        config = run.get("core_config", run.get("config"))
        _config(config, entry)
        original_config = json_document(paths["config"])
        if original_config != run.get("config"):
            raise ValueError("Original reviewed config differs from actual report.")
        _check_membership(member, rows, groups["train"], report)
        original_bindings = run.get("bindings")
        review_bindings = review.get("bindings")
        if (
            not isinstance(original_bindings, dict)
            or not original_bindings
            or not isinstance(review_bindings, dict)
            or any(review_bindings.get(path) != value for path, value in original_bindings.items())
        ):
            raise ValueError("Original review/run source bindings differ or are absent.")
        for path, expected in original_bindings.items():
            resolved = resolve_original_source_binding(path, expected, entry["method"], source_archives)
            if sha256(bound(resolved)) != expected:
                raise ValueError("Original fit source/config/parent binding changed.")
        for path, expected in review_bindings.items():
            resolved = resolve_original_source_binding(path, expected, entry["method"], source_archives)
            if sha256(bound(resolved)) != expected:
                raise ValueError("Original independent review binding changed.")
        if (
            original_bindings.get(str(train["input_path"])) != train_sha
            or original_bindings.get(str(split_path)) != split_sha
            or original_bindings.get(str(paths["config"])) != sha256(paths["config"])
        ):
            raise ValueError("Original TRAIN/split/config ancestry is unbound.")
        reviewer = review.get("reviewer_session_id")
        if (
            not isinstance(reviewer, str)
            or not reviewer
            or reviewer.casefold()
            in {
                str(review.get("implementer_session_id")).casefold(),
                str(manifest["owner_session_id"]).casefold(),
            }
            or review.get("status")
            != (
                "APPROVED_PREFIT" if mode == "core_frozen_readout" else "APPROVED_DOWNSTREAM_PREFIT"
            )
        ):
            raise ValueError("Actual distinct original prefit review identity required.")
        if mode != "core_frozen_readout":
            _config(run["config"], entry, downstream=True)
        records[name] = {
            "entry": entry,
            "paths": paths,
            "run": run,
            "config": config,
            "member": member,
            "source_archives": [record for record in source_archives
                                if original_bindings.get(record["original_path"]) == record["sha256"]],
        }
    for name in order:
        item = records[name]
        mode, parent = item["entry"]["mode"], item["entry"]["parent"]
        lineage = item["run"].get("supervised_ancestry")
        if mode == "core_frozen_readout":
            if parent is not None or lineage is not None:
                raise ValueError("Root SSL/control report has unexpected fitted parent.")
            if (
                item["entry"]["method"] == "random_frozen"
                and item["run"].get("pretrain_steps") != 0
            ):
                raise ValueError("Random control report declares trained features.")
        else:
            if (
                not isinstance(lineage, dict)
                or lineage.get("mode") != mode
                or lineage.get("ssl_only") is not False
            ):
                raise ValueError("Missing actual supervised report ancestry before tensor decode.")
            if mode == "direct_end_to_end":
                if parent is not None or any(
                    lineage.get(key, "MISSING") is not None
                    for key in ("ancestor_run_sha256", "ancestor_encoder_sha256")
                ):
                    raise ValueError("Fresh direct report inherits a fitted parent.")
            elif parent is None or any(
                lineage.get(key) != sha256(records[parent]["paths"][path])
                for key, path in (
                    ("ancestor_run_sha256", "run"),
                    ("ancestor_encoder_sha256", "selected"),
                )
            ):
                raise ValueError("Missing/substituted local report ancestor before tensor decode.")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "output": output,
        "bound": bound,
        "records": records,
        "order": order,
        "groups": groups,
        "train": train,
        "report": report,
        "rows": rows,
        "split_path": split_path,
        "train_sha": train_sha,
        "split_sha": split_sha,
        "evidence": evidence,
    }


def _decode_records(admitted):
    records = admitted["records"]
    for name in admitted["order"]:
        item = records[name]
        entry, paths, run, config = (item[k] for k in ("entry", "paths", "run", "config"))
        artifact, selected = _safe_load(paths["model"]), _safe_load(paths["selected"])
        if (
            set(artifact) - BASE_ARTIFACT_FIELDS
            or set(selected) - SELECTED_FIELDS
            or artifact.get("kind") != entry["kind"]
            or artifact.get("config") != config
            or artifact.get("bindings") != run["bindings"]
        ):
            raise ValueError("Unsafe/extra/mismatched typed inference metadata.")
        scalers = json_document(paths["scalers"])
        _scalers(scalers)
        if (
            artifact.get("scalers") != scalers
            or selected.get("scalers") != scalers
            or selected.get("config") != config
        ):
            raise ValueError("Embedded config/TRAIN scalers differ from original files.")
        if entry["kind"] != "native_ssl_weights_only_inference_v1" and any(
            value.get("architecture") != ARCHITECTURE
            or value.get("evidence_kind") != admitted["evidence"]
            for value in (artifact, selected, run)
        ):
            raise ValueError("Band typed architecture/evidence differs.")
        if (
            artifact.get("evidence_kind", admitted["evidence"]) != admitted["evidence"]
            or selected.get("evidence_kind", admitted["evidence"]) != admitted["evidence"]
        ):
            raise ValueError("Artifact evidence was promoted.")
        _state(artifact.get("model"))
        _state(selected.get("encoder"))
        encoder = _subset(artifact["model"], "encoder.")
        if not _equal_state(encoder, selected["encoder"]):
            raise ValueError("Selected tensors differ from actual inference.encoder branch.")
        head = _subset(artifact["model"], "readout.")
        mode, parent = entry["mode"], entry["parent"]
        lineage = artifact.get("supervised_ancestry")
        if mode == "core_frozen_readout":
            if (
                parent is not None
                or lineage is not None
                or selected.get("kind") != KINDS[entry["kind"]][0]
                or selected.get("bindings") != run["bindings"]
            ):
                raise ValueError(
                    "SSL/control parent cannot be relabelled supervised or inherit an undeclared ancestor."
                )
            pretrain = run.get("pretrain_steps")
            if (
                type(pretrain) is not int
                or pretrain < 0
                or (entry["method"] == "random_frozen" and pretrain != 0)
            ):
                raise ValueError("Random encoder must remain an untrained zero-pretrain control.")
            item["feature_label"] = LABELS[entry["method"]]
            item["tensor_parent_relation"] = "ROOT_ENCODER_NO_FITTED_PARENT"
        else:
            if (
                not isinstance(lineage, dict)
                or set(lineage)
                != {
                    "mode",
                    "ssl_only",
                    "supervised_updates",
                    "selected_supervised_step",
                    "ancestor_encoder_sha256",
                    "ancestor_run_sha256",
                }
                or lineage != run.get("supervised_ancestry")
                or lineage.get("mode") != mode
                or lineage.get("ssl_only") is not False
                or artifact.get("downstream_config") != run["config"]
            ):
                raise ValueError("Explicit intact supervised label ancestry required.")
            steps, selected_step = (
                lineage["supervised_updates"],
                lineage["selected_supervised_step"],
            )
            if (
                type(steps) is not int
                or type(selected_step) is not int
                or not 1 <= selected_step <= steps <= run["config"]["updates"]
                or selected_step % run["config"]["cadence"]
            ):
                raise ValueError("Supervised endpoint update/selection ancestry differs.")
            if steps != run.get("supervised_updates") or selected_step != run.get(
                "selected_supervised_step"
            ):
                raise ValueError("Actual report supervised label counters differ.")
            if mode == "direct_end_to_end":
                if (
                    parent is not None
                    or any(
                        lineage[k] is not None
                        for k in ("ancestor_run_sha256", "ancestor_encoder_sha256")
                    )
                    or selected.get("kind") != KINDS[entry["kind"]][1]
                ):
                    raise ValueError(
                        "Fresh direct supervised endpoint must have no selected ancestor."
                    )
                item["tensor_parent_relation"] = "FRESH_DIRECT_NO_SELECTED_PARENT"
            else:
                if parent is None:
                    raise ValueError("Missing completed local fitted parent.")
                original = records[parent]
                if lineage["ancestor_run_sha256"] != sha256(original["paths"]["run"]) or lineage[
                    "ancestor_encoder_sha256"
                ] != sha256(original["paths"]["selected"]):
                    raise ValueError("Substituted/foreign/incomplete fitted parent identity.")
                same_family = (
                    original["entry"]["kind"] == "native_ssl_weights_only_inference_v1"
                ) == (entry["kind"] == "native_ssl_weights_only_inference_v1")
                if (
                    not same_family
                    or original["entry"]["method"] != entry["method"]
                    or original["entry"]["seed"] != entry["seed"]
                    or original["scalers"] != scalers
                ):
                    raise ValueError("Typed parent config/method/seed/scaler identity differs.")
                if original["selected_kind"] not in {k[0] for k in KINDS.values()}:
                    raise ValueError(
                        "This frozen downstream factory does not admit supervised selected parents."
                    )
                expected_core = dict(original["config"])
                for field in ("method", "seed", "history", "batch_size", "lr", "weight_decay"):
                    expected_core[field] = artifact["downstream_config"][field]
                if config != expected_core:
                    raise ValueError(
                        "Actual downstream core_config differs from selected parent factory."
                    )
                equality = _equal_state(encoder, original["encoder"])
                if mode == "frozen_readout":
                    if not equality or sha256(paths["selected"]) != sha256(
                        original["paths"]["selected"]
                    ):
                        raise ValueError("Frozen encoder weights/buffers/bytes changed.")
                    if _equal_state(head, original["head"]):
                        raise ValueError(
                            "Unchanged parent probe head cannot establish a fresh readout."
                        )
                    item["tensor_parent_relation"] = "FROZEN_TENSORS_AND_SELECTED_BYTES_IDENTICAL"
                elif equality:
                    raise ValueError(
                        "Full-finetuned encoder has no actual selected tensor differences."
                    )
                else:
                    if selected.get("kind") != KINDS[entry["kind"]][1]:
                        raise ValueError(
                            "Full-finetuned features require explicit supervised kind."
                        )
                    item["tensor_parent_relation"] = "SUPERVISED_ENCODER_TENSORS_DIFFER"
            if mode != "frozen_readout" and (
                selected.get("supervised_ancestry") != lineage
                or selected.get("bindings") != run["bindings"]
            ):
                raise ValueError("Supervised selected encoder lost label/source ancestry.")
            item["feature_label"] = (
                "Fresh direct supervised features"
                if mode == "direct_end_to_end"
                else "Frozen " + LABELS[entry["method"]]
                if mode == "frozen_readout"
                else "Supervised full-finetuned " + LABELS[entry["method"]]
            )
        item.update(
            encoder=encoder,
            head=head,
            scalers=scalers,
            artifact=artifact,
            selected_kind=selected["kind"],
        )


def _write(path, document):
    with path.open("xb") as stream:
        stream.write(encoded(document))
    return sha256(path)


def derive_inventory(manifest_path, output_path):
    """Root-executed safe metadata derivation; never np.load or model construction."""
    started = time.monotonic()
    admitted = _admit(manifest_path, output_path)
    # Reference kinds use a separate accurate schema; external ancestry is never clean-local.
    references = _references(admitted)
    _decode_records(admitted)
    for name, expected in admitted["manifest"]["bindings"].items():
        if sha256(name) != expected:
            raise ValueError("Inventory input changed during safe metadata audit.")
    output = admitted["output"]
    output.mkdir()
    synthetic = admitted["evidence"] == SYNTHETIC
    evidence = SYNTHETIC if synthetic else ASSESSMENT_EVIDENCE
    train_cohort = output / "train_cohort.json"
    split_receipt = output / "whole_deployment_split.json"
    _write(
        split_receipt,
        {
            "train": _identities(admitted["groups"]["train"]),
            "reserved_test": _identities(admitted["groups"]["final_test"]),
            "development": _identities(admitted["groups"]["development"]),
            "original_split_path": str(admitted["split_path"]),
            "original_split_sha256": admitted["split_sha"],
            "status": "DERIVED_IDENTITY_RECEIPT_NOT_A_NEW_PARTITION",
        },
    )
    assessment_split_sha = sha256(split_receipt)
    _write(
        train_cohort,
        {
            "role": "train",
            "evidence_kind": evidence,
            "complete": True,
            "input_npz_sha256": admitted["train_sha"],
            "split_sha256": assessment_split_sha,
            "original_split_path": str(admitted["split_path"]),
            "original_split_sha256": admitted["split_sha"],
            "rows": admitted["rows"],
            "identities": _identities(admitted["groups"]["train"]),
            "original_identity_cohort_path": str(admitted["train"]["identity_cohort_path"]),
            "original_identity_cohort_sha256": sha256(admitted["train"]["identity_cohort_path"]),
            "derivation": "Exact original TRAIN byte/report/cohort and every fitted membership agree; no corpus values decoded",
        },
    )
    inventory, proposed = {}, {}
    for name in admitted["order"]:
        item = admitted["records"][name]
        paths, entry = item["paths"], item["entry"]
        config_path, stats_path = output / f"{name}.config.json", output / f"{name}.statistics.json"
        _write(config_path, item["config"])
        _write(stats_path, item["scalers"])
        fit_path, ancestry_path, selection_path = (
            output / f"{name}.{suffix}.json" for suffix in ("fit_input", "ancestry", "selection")
        )
        _write(
            fit_path,
            {
                "role": "train",
                "input_sha256": admitted["train_sha"],
                "cohort_sha256": sha256(train_cohort),
                "statistics_sha256": sha256(stats_path),
                "config_sha256": sha256(config_path),
                "split_sha256": assessment_split_sha,
                "original_split_sha256": admitted["split_sha"],
                "actual_run_path": str(paths["run"]),
                "actual_run_sha256": sha256(paths["run"]),
                "actual_membership_path": str(paths["membership"]),
                "actual_membership_sha256": sha256(paths["membership"]),
                "original_embedded_bindings": item["run"]["bindings"],
                "fitted_operations": [
                    "TRAIN_scalers",
                    "pretraining_or_supervised_features",
                    "TRAIN_probe_or_readout_labels",
                ]
                if entry["mode"] == "core_frozen_readout"
                else ["original_TRAIN_scalers", "TRAIN_fresh_readout_labels"]
                + (
                    ["TRAIN_supervised_encoder_labels"] if entry["mode"] != "frozen_readout" else []
                ),
            },
        )
        parent = entry["parent"]
        _write(
            ancestry_path,
            {
                "kind": "native_assessment_ancestry_v1",
                "locally_fitted": True,
                "historical_initial_weights": False,
                "completeness": "COMPLETE_LOCAL_ANCESTRY",
                "parents": [str(output / f"{parent}.ancestry.json")] if parent else [],
                "artifacts": [
                    {"path": str(paths[key]), "sha256": sha256(paths[key])}
                    for key in (
                        "model",
                        "selected",
                        "run",
                        "membership",
                        "config",
                        "review",
                        "scalers",
                    )
                ],
                "config_sha256": sha256(config_path),
                "statistics_sha256": sha256(stats_path),
                "fit_inputs": [
                    {
                        "input_path": str(admitted["train"]["input_path"]),
                        "cohort_path": str(train_cohort),
                        "manifest_path": str(fit_path),
                        "statistics_path": str(stats_path),
                        "config_path": str(config_path),
                        "split_path": str(split_receipt),
                    }
                ],
                "original_run_kind": entry["mode"],
                "feature_training_kind": item["feature_label"],
                "encoder_parent_relation": item["tensor_parent_relation"],
                "scientific_quality": "NOT_ESTABLISHED",
                "preserved_original_source_versions": item["source_archives"],
                "historical_source_resolution_is_compatibility_approval": False,
            },
        )
        _write(
            selection_path,
            {
                "status": "NOT_FINAL_SELECTION",
                "selection_role": "development",
                "frozen_before_numeric_access": True,
                "model_sha256": {str(paths["model"]): sha256(paths["model"])},
                "config_sha256": sha256(config_path),
                "statistics_sha256": sha256(stats_path),
                "actual_run_path": str(paths["run"]),
                "actual_run_sha256": sha256(paths["run"]),
                "is_finalist_selection": False,
                "derivation_note": "Completed endpoint identity only; not an owner freeze or access approval",
            },
        )
        inventory[name] = {
            "model_paths": [str(paths["model"])],
            "config_path": str(config_path),
            "statistics_path": str(stats_path),
            "ancestry_path": str(ancestry_path),
            "selection_path": str(selection_path),
            "parent_names": [parent] if parent else [],
            "method": entry["method"],
            "seed": entry["seed"],
            "mode": entry["mode"],
            "kind": entry["kind"],
            "loader": "builtin",
            "feature_training_kind": item["feature_label"],
        }
        proposed[name] = {
            "status": "COMPLETED",
            "artifact_family": "native_neural",
            "model": str(paths["model"]),
            "readout": str(paths["model"]),
            "scalers": str(stats_path),
            "config": str(config_path),
            "completion": str(paths["run"]),
            "ancestry": str(ancestry_path),
        }
    for name, reference in references.items():
        _write(output / f"{name}.reference.json", reference)
        if reference["kind"] == "external_chronos_unknown":
            continue
        config_path, stats_path = (
            output / f"{name}.reference-config.json",
            output / f"{name}.reference-statistics.json",
        )
        _write(config_path, reference["config"])
        _write(stats_path, reference["statistics"])
        fit_inputs = []
        if reference["locally_fitted"]:
            fit_path = output / f"{name}.reference-fit_input.json"
            _write(
                fit_path,
                {
                    "role": "train",
                    "input_sha256": admitted["train_sha"],
                    "cohort_sha256": sha256(train_cohort),
                    "statistics_sha256": sha256(stats_path),
                    "config_sha256": sha256(config_path),
                    "split_sha256": assessment_split_sha,
                    "original_split_sha256": admitted["split_sha"],
                    "fitted_operations": [
                        "TRAIN_fifteen_quantile_boosters_no_fitted_normalization"
                    ],
                },
            )
            fit_inputs = [
                {
                    "input_path": str(admitted["train"]["input_path"]),
                    "cohort_path": str(train_cohort),
                    "manifest_path": str(fit_path),
                    "statistics_path": str(stats_path),
                    "config_path": str(config_path),
                    "split_path": str(split_receipt),
                }
            ]
        ancestry_path = output / f"{name}.reference-ancestry.json"
        _write(
            ancestry_path,
            {
                "kind": "native_assessment_ancestry_v1",
                "locally_fitted": reference["locally_fitted"],
                "historical_initial_weights": False,
                "completeness": "COMPLETE_LOCAL_ANCESTRY",
                "parents": [],
                "artifacts": [
                    {"path": value, "sha256": sha256(value)}
                    for value in reference["paths"].values()
                ]
                + reference["boosters"],
                "fit_inputs": fit_inputs,
                "reference_kind": reference["kind"],
                "scientific_quality": "NOT_ESTABLISHED",
            },
        )
        reference["ancestry_path"] = str(ancestry_path)
    _write(
        output / "proposed-freeze-inputs.json",
        {
            "kind": "native_proposed_freeze_inputs_v1",
            "status": "PROPOSED_NOT_FINAL_SELECTION",
            "methods": proposed,
        },
    )
    result = {
        "kind": "native_research_inventory_v1",
        "status": "DERIVED_METADATA_NOT_FINAL_SELECTION",
        "evidence_kind": admitted["evidence"],
        "models": inventory,
        "references": references,
        "train_rows": len(admitted["rows"]),
        "manifest_sha256": sha256(admitted["manifest_path"]),
        "bindings": admitted["manifest"]["bindings"],
        "elapsed_seconds": time.monotonic() - started,
        "owned_rss_bytes_at_completion": _memory(),
        "rss_limit_bytes": RAM_LIMIT,
        "numeric_corpus_decoded": False,
        "model_constructed": False,
        "independent_ancestry_review": "NOT_RUN",
        "final_numeric_access": "NOT_RUN",
        "programme_completion": "NOT_ESTABLISHED",
        "sota": "NOT_ESTABLISHED",
    }
    _write(output / "inventory.json", result)
    return result


def _references(admitted):
    result = {}
    refs = admitted["manifest"].get("references", {})
    if not isinstance(refs, dict):
        raise TypeError("Explicit separate reference metadata required.")
    for name, spec in refs.items():
        if (
            not re.fullmatch("[A-Za-z0-9][A-Za-z0-9_.-]*", name)
            or name in admitted["records"]
            or not isinstance(spec, dict)
            or set(spec) != {"kind", "report_path", "config_path", "statistics_path", "review_path"}
            or spec["kind"]
            not in {"persistence", "seasonal24", "lightgbm15_utf8", "external_chronos_unknown"}
        ):
            raise ValueError("Unsupported or unsafe reference inventory schema.")
        paths = {
            k: admitted["bound"](spec[k + "_path"])
            for k in ("report", "config", "statistics", "review")
        }
        report = json_document(paths["report"])
        config, stats = json_document(paths["config"]), json_document(paths["statistics"])
        if spec["kind"] == "external_chronos_unknown":
            if (
                set(config) != {"kind", "external_model_identity"}
                or config.get("kind") != "external_chronos_unknown"
                or stats != {"ancestry": "UNKNOWN_EXTERNAL"}
            ):
                raise ValueError(
                    "External unknown lineage must remain explicit reference-only metadata."
                )
            result[name] = {
                "kind": spec["kind"],
                "ancestry": "UNKNOWN_EXTERNAL",
                "clean_local_ancestry_guarantee": False,
                "assessment_neural_kind": False,
                "status": "REFERENCE_METADATA_ONLY_UNSUPPORTED_NEURAL_LOADER",
                "paths": {k: str(v) for k, v in paths.items()},
            }
            continue
        review = json_document(paths["review"])
        if (
            report.get("status") != "COMPLETED_DEVELOPMENT_REFERENCE"
            or report.get("method")
            != ("lightgbm" if spec["kind"] == "lightgbm15_utf8" else spec["kind"])
            or report.get("history") != 96
            or report.get("seed") not in (7, 13, 23)
            or report.get("tuning") != "one_fixed_development_recipe_no_test_access"
            or report.get("train_sha256") != admitted["train_sha"]
            or report.get("review_sha256") != sha256(paths["review"])
        ):
            raise ValueError("Actual completed fixed-reference lineage differs.")
        if (
            review.get("status") != "APPROVED_PREFIT"
            or not review.get("reviewer_session_id")
            or review.get("reviewer_session_id")
            in (review.get("implementer_session_id"), admitted["manifest"]["owner_session_id"])
        ):
            raise ValueError("Actual distinct reference review required.")
        for path, value in review.get("bindings", {}).items():
            if sha256(admitted["bound"](path)) != value:
                raise ValueError("Actual reference source/TRAIN/DEV lineage changed.")
        models = report.get("models")
        if not isinstance(models, list) or len(models) != (
            15 if spec["kind"] == "lightgbm15_utf8" else 0
        ):
            raise ValueError("Exact fifteen fixed boosters or no-fit reference required.")
        boosters = []
        for model in models:
            path = admitted["bound"](paths["report"].parent / model["path"])
            if sha256(path) != model["sha256"]:
                raise ValueError("Fixed booster byte identity differs.")
            boosters.append({"path": str(path), "sha256": sha256(path)})
        expected_config = {"method", "history", "seed", "feature_schema"}
        expected_stats = {"fit_role", "input_npz_sha256", "normalization"}
        if (
            set(config) != expected_config
            or set(stats) != expected_stats
            or config.get("history") != 96
            or config.get("method") != report["method"]
            or config.get("seed") != report["seed"]
            or stats.get("fit_role") != ("train" if models else "none")
            or stats.get("input_npz_sha256") != (admitted["train_sha"] if models else None)
            or stats.get("normalization") != "NO_FITTED_NORMALIZATION"
            or config.get("feature_schema")
            != (
                "native_context_mask_metadata_summary_v1"
                if models
                else "native_primary_context_source_offsets_v1"
            )
        ):
            raise ValueError("Explicit reference feature/statistics/no-fit schema required.")
        result[name] = {
            "kind": spec["kind"],
            "locally_fitted": bool(models),
            "fit_input_sha256": admitted["train_sha"] if models else None,
            "ancestry": "LOCAL_TRAIN_BOOSTERS" if models else "NO_FIT_CONTEXT_REFERENCE",
            "config": config,
            "statistics": stats,
            "boosters": boosters,
            "paths": {k: str(v) for k, v in paths.items()},
            "neural_kind": False,
        }
    return result
