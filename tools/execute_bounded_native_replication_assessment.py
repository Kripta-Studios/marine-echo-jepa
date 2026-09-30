"""Owned version2 frozen assessment; stdlib admission precedes scientific imports.

Production root is the actual integrated checkout. Private injected roots and
process functions exist only in tests, never as CLI arguments. An owner budget
resolution is not scientific/numeric approval.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

IMPLEMENTER_SESSION_ID = "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1"
STATUSES = {
    "final_test": "APPROVED_FINAL_ASSESSMENT_EXECUTION",
    "development": "APPROVED_DEVELOPMENT_ASSESSMENT_EXECUTION",
}
NEURAL_KINDS = {
    "native_ssl_weights_only_inference_v1",
    "native_band_ssl_weights_only_inference_v1",
    "native_band_replication_ssl_weights_only_inference_v2",
}
REFERENCES = {
    "persistence": "persistence",
    "seasonal24": "seasonal24",
    "lightgbm15_utf8": "lightgbm",
}
METHODS = {"shared_ssl", "masked_ssl", "permuted_ssl", "random_frozen", "direct", "cf_jepa"}
MODES = {"core_frozen_readout", "frozen_readout", "full_finetune", "direct_end_to_end"}
ROBUSTNESS = {
    "kind": "secondary_observation_dropout_v1",
    "probability": 0.25,
    "seed": 20260929,
    "channels": [1, 2, 3],
}
LIMITS = {"rss_gib": 22, "cuda_gib": 10, "one_gpu_process": True}
ROOT = Path(__file__).resolve().parents[1]
ROOT_PACKAGE = ROOT / "src/marine_echo"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


IMPORTED_HASHES = {str(Path(__file__).resolve()): sha(Path(__file__).read_bytes())}


def _pairs(values):
    result = {}
    for key, value in values:
        if key in result:
            raise ValueError(f"Duplicate JSON identity/key: {key}")
        result[key] = value
    return result


def _bad_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _json(raw):
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_bad_constant)
    if not isinstance(value, dict):
        raise TypeError("Policy/receipt/config must be a JSON object.")
    return value


def _path(value, base):
    if not isinstance(value, str) or not value:
        raise ValueError("Explicit immutable path required.")
    p = Path(value)
    return (p if p.is_absolute() else base / p).resolve()


def _identity(value):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("Explicit stable identity required.")
    return value


def _bindings(value):
    if not isinstance(value, dict) or not value:
        raise ValueError("Exact nonempty immutable bindings required.")
    for p, h in value.items():
        if (
            not isinstance(p, str)
            or not Path(p).is_absolute()
            or str(Path(p).resolve()) != p
            or not isinstance(h, str)
            or not re.fullmatch("[0-9a-f]{64}", h)
        ):
            raise ValueError("Bindings require canonical absolute paths and lowercase SHA256.")
    return value


def _review(review, manifest, status, scope):
    excluded = {
        IMPLEMENTER_SESSION_ID.casefold(),
        _identity(manifest.get("implementer_session_id")).casefold(),
        _identity(manifest.get("root_coordinator_session_id")).casefold(),
    }
    if _identity(review.get("reviewer_session_id")).casefold() in excluded:
        raise ValueError("Review must be distinct from builder/coordinator/implementer.")
    if (
        review.get("status") != status
        or review.get("scope") != scope
        or review.get("allowed_roles") != [manifest["role"]]
        or review.get("evidence_kind") != manifest["evidence_kind"]
        or any(
            review.get(k) != manifest[k]
            for k in ("implementer_session_id", "root_coordinator_session_id")
        )
    ):
        raise ValueError("Review status, exact role/scope or identity differs.")
    return _bindings(review.get("bindings"))


def required_sources(loaders=None, *, package=None):
    """Static local import closure including lazy imports; never import manifest paths.

    External dependency bytes are represented by the separately admitted lockfile.
    Registered caller factories must be supplied as functions in reviewed modules.
    """
    package = ROOT_PACKAGE if package is None else package
    pending = [
        Path(__file__).resolve(),
        package / "evaluation/native_product.py",
        package / "evaluation/native_assessment_replication.py",
        package.parents[1] / "tools/execute_native_replication_assessment_worker.py",
        package.parents[1] / "tools/native_reference_supervisor.py",
        package.parents[1] / "tools/execute_native_band_replication_job.py",
        package / "inference/native_acoustic.py",
        package / "inference/native_encoder.py",
        package / "inference/native_band_acoustic.py",
        package / "evaluation/native_assessment.py",
        package / "inference/native_band_replication_acoustic.py",
        package / "training/native_band_replication_downstream.py",
        package / "training/native_downstream.py",
        package / "training/native_references.py",
    ]
    pending.extend(Path(inspect.getsourcefile(f)).resolve() for f in (loaders or {}).values())
    found = set()
    while pending:
        p = pending.pop()
        if p in found:
            continue
        if not p.is_file():
            raise ValueError(f"Missing required source: {p}")
        found.add(p)
        for node in ast.walk(ast.parse(p.read_bytes(), filename=str(p))):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if not p.is_relative_to(package):
                        continue
                    prefix = ["marine_echo", *p.relative_to(package).parts[:-1]]
                    prefix = prefix[: len(prefix) - node.level + 1]
                    name = ".".join([*prefix, *(node.module or "").split(".")]).rstrip(".")
                else:
                    name = node.module or ""
                modules = [name, *(name + "." + a.name for a in node.names)]
            for name in modules:
                if name != "marine_echo" and not name.startswith("marine_echo."):
                    continue
                parts = name.split(".")[1:]
                f, init = (
                    package.joinpath(*parts).with_suffix(".py"),
                    package.joinpath(*parts) / "__init__.py",
                )
                candidate = f if parts and f.is_file() else init
                if candidate.is_file():
                    pending.append(candidate)
                for n in range(len(parts) + 1):
                    init = package.joinpath(*parts[:n]) / "__init__.py"
                    if init.is_file():
                        pending.append(init)
    return tuple(sorted(found))


def _members(document):
    identities = document.get("identities")
    if not isinstance(identities, list) or not identities:
        raise ValueError("Complete archive/deployment/site/source membership required.")
    seen = set()
    result = {k: set() for k in ("archive_id", "deployment_id", "site_id", "source_id")}
    for item in identities:
        if not isinstance(item, dict):
            raise TypeError("Invalid membership identity.")
        triple = tuple(_identity(item.get(k)) for k in ("archive_id", "deployment_id", "site_id"))
        if triple in seen:
            raise ValueError("Duplicate deployment membership identity.")
        seen.add(triple)
        sources = item.get("source_ids")
        if not isinstance(sources, list) or not sources or len(set(sources)) != len(sources):
            raise ValueError("Exact source identities required.")
        for key, value in zip(("archive_id", "deployment_id", "site_id"), triple, strict=True):
            result[key].add(value)
        result["source_id"].update(_identity(v) for v in sources)
    return result


def _split_members(split):
    """Use original native-reader split bytes; never rewrite its NPZ digest."""
    if split.get("schema_version") != "native_acoustic_ssl_v1":
        if "sources" in split:
            raise ValueError("Unknown native source split schema.")
        return (
            _members({"identities": split.get("train")}),
            _members({"identities": split.get("reserved_test")}),
        )
    if "train" in split or "reserved_test" in split:
        raise ValueError("Ambiguous native source and derived split memberships.")
    sources = split.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Original native source split requires complete source identities.")
    by_role = {"train": [], "development": [], "final_test": []}
    archive_ids, deployments, digests = set(), set(), set()
    for source in sources:
        if not isinstance(source, dict) or source.get("role") not in by_role:
            raise ValueError("Unknown native source role.")
        file_id, digest = source.get("file_id"), source.get("archive_sha256")
        deployment, site = _identity(source.get("deployment")), _identity(source.get("site"))
        if (
            type(file_id) is not int
            or file_id <= 0
            or not isinstance(digest, str)
            or not re.fullmatch("[0-9a-f]{64}", digest)
            or file_id in archive_ids
            or deployment in deployments
            or digest in digests
        ):
            raise ValueError("Missing, duplicate or invalid native archive/deployment identity.")
        archive_ids.add(file_id)
        deployments.add(deployment)
        digests.add(digest)
        by_role[source["role"]].append(
            {
                "archive_id": str(file_id),
                "deployment_id": deployment,
                "site_id": site,
                "source_ids": [digest],
            }
        )
    return (
        _members({"identities": by_role["train"]}),
        _members({"identities": by_role["final_test"]}),
    )


@dataclass(frozen=True)
class Admission:
    manifest: dict
    base: Path
    documents: dict
    snapshots: dict
    paths: dict
    provenance: dict


def admit(manifest_path, review_path, output_path, *, loaders=None, source_package=None):
    """Verify complete immutable graph before ANY NPZ/tensor decoding."""
    manifest_path, review_path, output_path = (
        Path(p).resolve() for p in (manifest_path, review_path, output_path)
    )
    manifest_raw, review_raw = manifest_path.read_bytes(), review_path.read_bytes()
    m, review = _json(manifest_raw), _json(review_raw)
    if m.get("role") not in STATUSES or m.get("evidence_kind") not in {
        "SYNTHETIC_CORRECTNESS_ONLY",
        "REVIEWED_FROZEN_ASSESSMENT",
    }:
        raise ValueError("Explicit supported role/evidence kind required.")
    if m["evidence_kind"] == "SYNTHETIC_CORRECTNESS_ONLY" and (
        m.get("fixture_identity") != "SYNTHETIC_CORRECTNESS_ONLY" or m.get("device") != "cpu"
    ):
        raise ValueError("Synthetic identity is CPU only and changes no gates.")
    bindings = _review(review, m, STATUSES[m["role"]], "model_only_frozen_assessment")
    if (
        type(m.get("floor")) is not int
        or m["floor"] != 18
        or type(m.get("batch_size")) is not int
        or m["batch_size"] < 1
        or m.get("device") not in {"cpu", "cuda:0"}
        or m.get("resource_limits") != LIMITS
        or review.get("device") != m["device"]
        or review.get("output_path") != str(output_path)
        or review.get("batch_size") != m["batch_size"]
        or m.get("robustness") not in (None, ROBUSTNESS)
        or review.get("robustness") != m.get("robustness")
    ):
        raise ValueError("Fixed floor/device/batch/resource/robustness/output recipe differs.")
    if m.get("input_schema") not in {"native_corpus_v1", "assessment_context_v1"}:
        raise ValueError("Explicit supported input schema required.")
    if m.get("input_corpus_role") not in (
        {"test", "final_test"} if m["role"] == "final_test" else {"development"}
    ):
        raise ValueError("Explicit input role cannot promote test access through development.")
    if "input_evidence_kind" not in m or (
        m["input_evidence_kind"] is not None
        and (not isinstance(m["input_evidence_kind"], str) or not m["input_evidence_kind"].strip())
    ):
        raise ValueError("Explicit original evidence header or admitted absence required.")
    if m["evidence_kind"] == "SYNTHETIC_CORRECTNESS_ONLY" and m["input_evidence_kind"] not in (
        None,
        "SYNTHETIC_CORRECTNESS_ONLY",
    ):
        raise ValueError("Synthetic correctness cannot admit a real evidence header.")
    if (
        m["evidence_kind"] != "SYNTHETIC_CORRECTNESS_ONLY"
        and m["input_evidence_kind"] == "SYNTHETIC_CORRECTNESS_ONLY"
    ):
        raise ValueError("Synthetic inputs cannot become scientific evidence.")
    if (
        not isinstance(m.get("historical_metadata_exposure_disclosure"), str)
        or not m["historical_metadata_exposure_disclosure"].strip()
    ):
        raise ValueError("Explicit historical compatibility/metadata exposure disclosure required.")
    if output_path.exists():
        raise FileExistsError("New protected output directory required.")
    snapshots, documents, exact = {}, {}, {}

    def bind(p, *, raw=None, document=False):
        p = Path(p).resolve()
        if p not in snapshots:
            if str(p) not in bindings:
                raise ValueError(f"Missing immutable binding: {p}")
            payload = p.read_bytes() if raw is None else raw
            digest = sha(payload)
            if digest != bindings[str(p)] or (
                str(p) in IMPORTED_HASHES and digest != IMPORTED_HASHES[str(p)]
            ):
                raise ValueError(f"Stale immutable source/artifact binding: {p}")
            snapshots[p], exact[str(p)] = payload, digest
        if document and p not in documents:
            documents[p] = _json(snapshots[p])
        return p

    bind(manifest_path, raw=manifest_raw, document=True)
    for p in required_sources(loaders, package=source_package):
        bind(p)
    base = manifest_path.parent
    paths = {
        k: bind(
            _path(m.get(k + "_path"), base),
            document=k in {"cohort", "split", "selection", "source", "numeric_access_review"},
        )
        for k in (
            "input",
            "cohort",
            "split",
            "protocol",
            "selection",
            "source",
            "dependency_lock",
            "numeric_access_review",
        )
    }
    source_receipt = documents[paths["source"]]
    declared_source_bindings = _bindings(source_receipt.get("bindings"))
    for p, h in declared_source_bindings.items():
        bind(Path(p))
        if exact[p] != h:
            raise ValueError("Source acquisition/pin receipt differs.")
    numeric = documents[paths["numeric_access_review"]]
    nb = _review(
        numeric, m, "APPROVED_NUMERIC_CORPUS_PROVENANCE", "native_assessment_numeric_decode"
    )
    if any(
        k not in numeric or numeric[k] != m[k] for k in ("input_corpus_role", "input_evidence_kind")
    ):
        raise ValueError(
            "Numeric provenance must explicitly admit original source role/evidence headers."
        )
    finalist_selection = documents[paths["selection"]]
    if (
        finalist_selection.get("selection_role")
        not in {"development", "prespecified_without_selection"}
        or finalist_selection.get("frozen_before_numeric_access") is not True
    ):
        raise ValueError("Frozen selection must precede numeric access and cannot use final_test.")
    for key in ("input", "cohort", "split", "protocol", "source", "selection"):
        if nb.get(str(paths[key])) != exact[str(paths[key])]:
            raise ValueError("Separately approved numeric corpus/selection provenance differs.")
    cohort, split = documents[paths["cohort"]], documents[paths["split"]]
    if (
        cohort.get("role") != m["role"]
        or cohort.get("evidence_kind") != m["evidence_kind"]
        or cohort.get("input_npz_sha256") != exact[str(paths["input"])]
        or cohort.get("split_sha256") != exact[str(paths["split"])]
        or not isinstance(cohort.get("rows"), list)
        or not cohort["rows"]
    ):
        raise ValueError("Corpus role/identity/cohort/split provenance differs.")
    members = _members(cohort)
    allowed_train, reserved = _split_members(split)
    if any(allowed_train[k] & reserved[k] for k in reserved):
        raise ValueError("Split TRAIN and whole test reservations overlap.")
    if m["role"] == "final_test" and any(not members[k] <= reserved[k] for k in members):
        raise ValueError("Final corpus is outside whole deployment/site reservations.")
    methods = m.get("methods")
    if not isinstance(methods, dict) or not methods or review.get("methods") != methods:
        raise ValueError("Exact frozen method/seed/mode/artifact identities required.")
    done, active, ancestry_proof = set(), set(), []

    def ancestry(p):
        p = bind(p, document=True)
        if p in active:
            raise ValueError("Cyclic ancestry; completeness cannot be established.")
        if p in done:
            return
        active.add(p)
        node = documents[p]
        fitted = node.get("locally_fitted")
        if (
            node.get("kind") != "native_assessment_ancestry_v1"
            or type(fitted) is not bool
            or node.get("historical_initial_weights") is not False
            or node.get("completeness") != "COMPLETE_LOCAL_ANCESTRY"
            or not isinstance(node.get("parents"), list)
            or not isinstance(node.get("artifacts"), list)
            or not node["artifacts"]
            or not isinstance(node.get("fit_inputs"), list)
        ):
            raise ValueError("Unknown/missing/historical local ancestry; root receipt required.")
        for a in node["artifacts"]:
            artifact = bind(_path(a.get("path"), p.parent))
            if a.get("sha256") != exact[str(artifact)]:
                raise ValueError("Ancestor artifact identity differs.")
        if fitted != bool(node["fit_inputs"]):
            raise ValueError("Every fitted numerical/scaler input must be declared.")
        fits = []
        for item in node["fit_inputs"]:
            fp = {
                k: bind(_path(item.get(k + "_path"), p.parent), document=k != "input")
                for k in ("input", "cohort", "manifest", "statistics", "config", "split")
            }
            c = documents[fp["cohort"]]
            if (
                c.get("role") != "train"
                or c.get("input_npz_sha256") != exact[str(fp["input"])]
                or c.get("split_sha256") != exact[str(paths["split"])]
                or fp["split"] != paths["split"]
                or c.get("evidence_kind") != m["evidence_kind"]
            ):
                raise ValueError("Every fitted input must be TRAIN on this exact split.")
            train_members = _members(c)
            if any(
                train_members[k] & reserved[k]
                or train_members[k] & members[k]
                or not train_members[k] <= allowed_train[k]
                for k in members
            ):
                raise ValueError(
                    "Fitted ancestry overlaps assessment/test reservations or is outside admitted TRAIN."
                )
            fit_manifest = documents[fp["manifest"]]
            if fit_manifest.get("role") != "train" or any(
                fit_manifest.get(k + "_sha256") != exact[str(fp[k])]
                for k in ("input", "cohort", "statistics", "config", "split")
            ):
                raise ValueError("Fitted-input manifest identities differ.")
            fits.append(
                {
                    "paths": {k: str(v) for k, v in fp.items()},
                    "members": {k: sorted(v) for k, v in train_members.items()},
                }
            )
        for parent in node["parents"]:
            ancestry(_path(parent, p.parent))
        active.remove(p)
        done.add(p)
        ancestry_proof.append(
            {
                "receipt_path": str(p),
                "sha256": exact[str(p)],
                "locally_fitted": fitted,
                "fit_inputs": fits,
            }
        )

    frozen_models, histories = {}, set()
    for name, spec in methods.items():
        if (
            not isinstance(name, str)
            or not re.fullmatch("[A-Za-z0-9][A-Za-z0-9_.-]*", name)
            or not isinstance(spec, dict)
        ):
            raise ValueError("Safe fixed method names/specifications required.")
        kind, method = spec.get("kind"), spec.get("method")
        if kind not in NEURAL_KINDS | REFERENCES.keys() or (
            method != REFERENCES[kind] if kind in REFERENCES else method not in METHODS
        ):
            raise ValueError(
                "Unsupported safe kind/method; no Chronos, encoder-only or resume kinds."
            )
        if kind.startswith("native_band") and method == "cf_jepa":
            raise ValueError("CF is not part of the band family.")
        if (
            type(spec.get("seed")) is not int
            or spec["seed"] < 0
            or spec.get("mode") not in ({"reference"} if kind in REFERENCES else MODES)
        ):
            raise ValueError("Frozen mode/seed required.")
        loader = spec.get("loader")
        if loader != "builtin" and (
            loader not in (loaders or {}) or loader not in review.get("allowed_loader_ids", [])
        ):
            raise ValueError(
                "Loader must be caller-supplied, enumerated and source-bound; no manifest imports."
            )
        if not isinstance(spec.get("model_paths"), list) or not spec["model_paths"]:
            raise ValueError("Exact frozen artifact files required, including stateless recipe.")
        models = [bind(_path(p, base)) for p in spec["model_paths"]]
        if len(set(models)) != len(models):
            raise ValueError("Duplicate model artifact paths.")
        frozen_models.update({str(p): exact[str(p)] for p in models})
        side = {
            k: bind(_path(spec.get(k + "_path"), base), document=True)
            for k in ("config", "statistics", "selection", "ancestry")
        }
        config, statistics, selected = (
            documents[side[k]] for k in ("config", "statistics", "selection")
        )
        if (
            selected.get("selection_role") not in {"development", "prespecified_without_selection"}
            or selected.get("frozen_before_numeric_access") is not True
        ):
            raise ValueError(
                "Every method selection must be frozen before assessment numeric access."
            )
        if (
            any(config.get(k) != spec[k] for k in ("method", "seed"))
            or config.get("batch_size") != m["batch_size"]
            or config.get("history") not in (24, 96)
        ):
            raise ValueError("Frozen config/batch/method/seed differs.")
        _typed_band_config(kind, config, spec, m["evidence_kind"])
        histories.add(config["history"])
        if selected.get("model_sha256") != {str(p): exact[str(p)] for p in models} or any(
            selected.get(k + "_sha256") != exact[str(side[k])] for k in ("config", "statistics")
        ):
            raise ValueError("Frozen selection/statistics/model identities differ.")
        ancestry(side["ancestry"])
        node = documents[side["ancestry"]]
        if not set(models) <= {
            _path(v["path"], side["ancestry"].parent) for v in node["artifacts"]
        } or any(node.get(k + "_sha256") != exact[str(side[k])] for k in ("config", "statistics")):
            raise ValueError("Ancestry does not identify this artifact/config/statistics.")
        if kind not in {"persistence", "seasonal24"} and (
            node["locally_fitted"] is not True
            or not any(
                _path(v["statistics_path"], side["ancestry"].parent) == side["statistics"]
                for v in node["fit_inputs"]
            )
        ):
            raise ValueError("Learned/readout model requires exact fitted scaler ancestry.")
        if kind in REFERENCES and (
            statistics.get("fit_role") != "train"
            or config.get("feature_schema") != "native_references.feature_matrix_v1"
        ):
            raise ValueError("Reference requires frozen TRAIN statistics and exact feature schema.")
    if len(histories) != 1:
        raise ValueError("Methods require identical context history/support.")
    if (
        numeric.get("frozen_model_sha256") != frozen_models
        or numeric.get("frozen_selection_sha256") != exact[str(paths["selection"])]
    ):
        raise ValueError("Models/selection must be frozen before numeric access.")
    if output_path == review_path or any(
        output_path == p or p.is_relative_to(output_path) for p in snapshots
    ):
        raise ValueError("Output must be distinct from every immutable input/source.")
    provenance = {
        "bindings": exact,
        "manifest_sha256": sha(manifest_raw),
        "review_sha256": sha(review_raw),
        "reviewer_session_id": review["reviewer_session_id"],
        "numeric_reviewer_session_id": numeric["reviewer_session_id"],
        "builder_session_id": IMPLEMENTER_SESSION_ID,
        "recursive_local_ancestry": ancestry_proof,
        "historical_metadata_exposure_disclosure": m["historical_metadata_exposure_disclosure"],
        "no_universal_sealed_claim": True,
        "external_ancestry_guarantee": False,
    }
    return Admission(m, base, documents, snapshots, paths, provenance)


def _typed_band_config(kind, config, spec, evidence):
    """Metadata-only finite seed/type admission, before any artifact decoding."""
    if kind not in {
        "native_band_ssl_weights_only_inference_v1",
        "native_band_replication_ssl_weights_only_inference_v2",
    }:
        return
    seeds = (7,) if kind.endswith("_v1") else (7, 13, 23)
    if (
        type(spec.get("seed")) is not int
        or spec["seed"] not in seeds
        or config.get("architecture") != "nonlinear_frequency_conditioned_v1"
        or config.get("method") == "cf_jepa"
        or config.get("history") != 96
        or (config.get("lr"), config.get("weight_decay"), config.get("sigreg_weight"))
        != (0.0003, 0.0001, 0.03)
    ):
        raise ValueError("Exact typed band architecture/seed/objective/config required.")
    if evidence != "SYNTHETIC_CORRECTNESS_ONLY" and any(
        config.get(k) != v
        for k, v in {"width": 192, "latent": 64, "blocks": 4, "heads": 4, "batch_size": 64}.items()
    ):
        raise ValueError("Frozen real band dimensions/batch required.")


FAMILY = "native_band_v1"
RSS_LIMIT = 22 * 2**30
BAND_KINDS = {
    "native_band_ssl_weights_only_inference_v1",
    "native_band_replication_ssl_weights_only_inference_v2",
}
OWNER = "orchestration/native_band_budget_owner_resolution_v1.json"
LEDGER = "orchestration/native_ssl_run_ledger_v1.json"
LOCK = "evidence/ssl-builder-v1/gpu-owner.lock"


def digest(path):
    return sha(Path(path).read_bytes())


def read_json(path):
    return _json(Path(path).read_bytes())


def _number(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("Invalid/nonfinite " + label)
    return float(value)


def _policy(root, bindings):
    """Import only this fixed reviewed stdlib budget helper, never manifest code."""
    path = root / "tools/execute_native_band_replication_job.py"
    if bindings.get(str(path)) != digest(path):
        raise ValueError("Budget helper source changed.")
    spec = importlib.util.spec_from_file_location("reviewed_replication_budget", path)
    module = importlib.util.module_from_spec(spec)
    # Dataclass resolution needs the genuine module entry, not a source alias.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if bindings[str(path)] != digest(path):
        raise ValueError("Budget helper changed during import.")
    return module


@dataclass(frozen=True)
class Job:
    manifest: Path
    review: Path
    output: Path
    receipt: Path


@dataclass
class Plan:
    job: Job
    root: Path
    admission: Admission
    review: dict
    bindings: dict
    ledger: dict
    ledger_hash: str
    command: list
    deadline_seconds: float
    charge_band: bool


def prelaunch(job, *, _root=None):
    """No subprocess, output mutation, NumPy, Torch or scientific import."""
    root = ROOT if _root is None else Path(_root).resolve()
    job = Job(*(Path(p).resolve() for p in (job.manifest, job.review, job.output, job.receipt)))
    review, manifest = read_json(job.review), read_json(job.manifest)
    # CLI permits genuine reviewed evidence only. CPU private fixtures must use
    # the same complete graph with an explicit synthetic identity.
    if _root is None and manifest.get("evidence_kind") != "REVIEWED_FROZEN_ASSESSMENT":
        raise ValueError("Production execution requires genuine reviewed frozen assessment.")
    admission = admit(job.manifest, job.review, job.output, source_package=root / "src/marine_echo")
    if any(spec["loader"] != "builtin" for spec in manifest["methods"].values()):
        raise ValueError("Production worker supports only enumerated immutable loaders.")
    if review.get("receipt_path") != str(job.receipt) or review.get("runtime_arguments") != {
        "manifest": str(job.manifest),
        "review": str(job.review),
        "output": str(job.output),
        "receipt": str(job.receipt),
        "device": manifest["device"],
    }:
        raise ValueError("Exact reviewed receipt/runtime required.")
    if (
        job.receipt.exists()
        or job.output.exists()
        or (
            job.receipt == job.output
            or job.receipt.is_relative_to(job.output)
            or job.output.is_relative_to(job.receipt)
        )
    ):
        raise FileExistsError("Fresh disjoint protected output and receipt required.")
    if not job.output.is_relative_to(root / "evidence") or not job.receipt.is_relative_to(
        root / "evidence"
    ):
        raise ValueError("Root-owned evidence output/receipt required before mutation.")
    bindings = _bindings(review["bindings"])
    for p, h in bindings.items():
        if digest(p) != h:
            raise ValueError("Stale binding before launch/mutation: " + p)
    required = [
        root / ".venv/Scripts/python.exe",
        root / OWNER,
        root / "docs/adr/0023-owner-resolved-band-budget.md",
        Path(__file__).resolve(),
        root / "tools/execute_native_replication_assessment_worker.py",
        root / "tools/native_reference_supervisor.py",
        root / "tools/execute_native_band_replication_job.py",
    ]
    if any(str(p.resolve()) not in bindings for p in required):
        raise ValueError("Missing wrapper/worker/python/supervisor/budget binding.")
    for target in (job.output, job.receipt):
        if any(Path(p) == target or Path(p).is_relative_to(target) for p in bindings):
            raise ValueError("Protected destination contains immutable source/input.")
        if not target.parent.is_dir():
            raise ValueError("Root must prepare output/receipt parent directories.")
    lock, ledger_path = root / LOCK, root / LEDGER
    if lock.exists():
        raise ValueError("An owner lock exists; never remove an unknown lock.")
    if any(
        ledger_path.with_suffix(s).exists()
        for s in (
            ".pending",
            ".band-pending",
            ".prefix-pending",
            ".assessment-pending",
        )
    ):
        raise ValueError("Pending journal requires root reconciliation.")
    ledger = read_json(ledger_path)
    if ledger.get("gpu_limit_hours") != 96 or isinstance(ledger.get("gpu_limit_hours"), bool):
        raise ValueError("Aggregate limit must remain96.")
    aggregate = _number(ledger.get("gpu_hours_spent_owned_scientific_jobs"), "aggregate hours")
    if not isinstance(ledger.get("runs"), list) or any(
        not isinstance(r, dict)
        or str(r.get("status", "")).startswith("RUNNING_")
        or r.get("requires_reconciliation")
        or r.get("reconciliation_required")
        for r in ledger["runs"]
    ):
        raise ValueError("Active/unknown/unreconciled run ledger.")
    policy = _policy(root, bindings)
    resolution = read_json(root / OWNER)
    policy.validate_budget(resolution)
    if review.get("band_budget_status") != "ROOT_RESOLVED" or review.get(
        "budget_resolution_path"
    ) != str(root / OWNER):
        raise ValueError("Exact ROOT_RESOLVED owner binding required.")
    # Validate accounting even for CPU/non-band jobs, without reopening an
    # exhausted band allowance when only a genuine CPU assessment is requested.
    band_seconds = 0.0
    for record in ledger["runs"]:
        signature = (
            record.get("architecture") == "nonlinear_frequency_conditioned_v1"
            or "native_band" in str(record.get("config", "")).lower()
            or any("native_band" in str(v) for v in record.get("command", []))
            or record.get("family") == "band"
            or isinstance(record.get("config"), dict)
            and record["config"].get("family") == "band"
            or record.get("uses_band_model") is True
            and record.get("device") == "cuda:0"
        )
        if signature and record.get("budget_family") != FAMILY:
            raise ValueError("Unaccounted historical band attempt.")
        if record.get("budget_family") == FAMILY:
            if record.get("status") not in {
                "COMPLETED_REAL_BAND_CUDA",
                "FAILED_REAL_BAND_CUDA_ATTEMPT",
                "COMPLETED_REAL_CUDA",
                "FAILED_REAL_CUDA_ATTEMPT",
                "COMPLETED_REAL_PREFIX_TRANSFER",
                "PREFIX_TRANSFER_NOT_ASSESSABLE",
                "PREFIX_TRANSFER_FAILED_OR_BLOCKED",
            }:
                raise ValueError("Unknown band job requires reconciliation.")
            full = record.get("resources_full_attempt", {}).get("elapsed_full_attempt_seconds")
            seconds = _number(record.get("elapsed_owned_seconds", full), "owned attempt")
            if full is not None and not math.isclose(
                seconds, _number(full, "full time"), abs_tol=1e-6, rel_tol=1e-9
            ):
                raise ValueError("Conflicting full-owned counters.")
            band_seconds += seconds
    if aggregate * 3600 + 1e-6 < band_seconds:
        raise ValueError("Aggregate undercounts band attempts.")
    if "native_band_gpu_hours_spent_full_owned" in ledger and not math.isclose(
        _number(ledger["native_band_gpu_hours_spent_full_owned"], "Band cumulative hours"),
        band_seconds / 3600,
        abs_tol=1e-9,
        rel_tol=1e-9,
    ):
        raise ValueError("Band cumulative counter conflicts with full-owned records.")
    is_cuda = manifest["device"] == "cuda:0"
    charge_band = is_cuda and any(s["kind"] in BAND_KINDS for s in manifest["methods"].values())
    remaining = (96 - aggregate) * 3600
    if charge_band:
        remaining = min(remaining, 12 * 3600 - band_seconds)
    if is_cuda and remaining <= 0:
        raise ValueError("Band or aggregate owned allowance exhausted.")
    deadline = min(3 * 3600.0, remaining) if is_cuda else 3 * 3600.0
    command = [
        str(root / ".venv/Scripts/python.exe"),
        "-B",
        "-u",
        str(root / "tools/execute_native_replication_assessment_worker.py"),
    ]
    for key, value in {
        "manifest": job.manifest,
        "review": job.review,
        "output": job.output,
        "receipt": job.receipt,
        "ownership-output": job.output,
    }.items():
        command.extend(["--" + key, str(value)])
    return Plan(
        job,
        root,
        admission,
        review,
        dict(bindings),
        ledger,
        digest(ledger_path),
        command,
        deadline,
        charge_band,
    )


def _write_ledger(root, ledger):
    target = root / LEDGER
    pending = target.with_suffix(".assessment-pending")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(ledger, stream, indent=2, allow_nan=False)
        stream.write("\n")
    pending.replace(target)


def _exclusive_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _supervisor(root, bindings):
    path = root / "tools/native_reference_supervisor.py"
    if digest(path) != bindings[str(path)]:
        raise ValueError("Supervisor changed before import.")
    spec = importlib.util.spec_from_file_location("native_reference_supervisor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if digest(path) != bindings[str(path)]:
        raise ValueError("Supervisor changed during import.")
    return module.supervise_owned


def _completion(plan, child, resources):
    if (
        type(child.pid) is not int
        or child.pid < 1
        or type(resources.get("exit_code")) is not int
        or child.poll() != resources.get("exit_code")
    ):
        raise ValueError("Actual child/exit identity missing.")
    pids = resources.get("owned_process_pids")
    if (
        not isinstance(pids, list)
        or child.pid not in pids
        or len(pids) != len(set(pids))
        or any(type(p) is not int or p < 1 for p in pids)
        or resources.get("owned_tree_cleanup_verified") is not True
    ):
        raise ValueError("Owned tree cleanup not proven.")
    for key in (
        "peak_process_rss_bytes",
        "elapsed_full_attempt_seconds",
        "deadline_seconds",
        "rss_limit_bytes",
    ):
        _number(resources.get(key), key)
    if (
        resources["rss_limit_bytes"] != RSS_LIMIT
        or resources["deadline_seconds"] != plan.deadline_seconds
    ):
        raise ValueError("Supervisor policy changed.")
    if resources["exit_code"] != 0 or resources.get("stopped_for") is not None:
        return None
    path = plan.job.output / "completion.json"
    if not path.is_file():
        return None
    report = read_json(path)
    manifest = plan.admission.manifest
    if (
        report.get("kind") != "native_replication_frozen_assessment_completion_v2"
        or report.get("status") not in {"COMPLETED_FORECASTS", "COMPLETED_FORECASTS_NOT_ASSESSABLE"}
        or report.get("role") != manifest["role"]
        or report.get("evidence_kind") != manifest["evidence_kind"]
        or report.get("provenance", {}).get("manifest_sha256") != digest(plan.job.manifest)
        or report.get("provenance", {}).get("review_sha256") != digest(plan.job.review)
        or report.get("provenance", {}).get("bindings") != plan.admission.provenance["bindings"]
        or set(report.get("methods", {})) != set(manifest["methods"])
    ):
        return None
    for name, spec in manifest["methods"].items():
        result = report["methods"][name]
        artifact = plan.job.output / (name + ".npz")
        if any(result.get(k) != spec[k] for k in ("kind", "method", "seed", "mode")) or (
            result.get("prediction_path") != str(artifact)
            or not artifact.is_file()
            or result.get("prediction_sha256") != digest(artifact)
        ):
            return None
    runtime = report.get("resources", {})
    if runtime.get("device") != manifest["device"] or runtime.get("process_id") not in pids:
        return None
    for key, cap in (
        ("sampled_peak_rss_bytes", RSS_LIMIT),
        ("process_cuda_peak_allocated_bytes", 10 * 2**30),
        ("process_cuda_peak_reserved_bytes", 10 * 2**30),
    ):
        if _number(runtime.get(key), key) >= cap:
            return None
    if manifest["device"] == "cpu" and any(
        runtime[k] != 0
        for k in ("process_cuda_peak_allocated_bytes", "process_cuda_peak_reserved_bytes")
    ):
        return None
    return report


def _execute(plan, *, popen, supervisor, clock=time.monotonic, write_ledger=_write_ledger):
    """Private injection permits virtual policy checks, never a production bypass."""
    root, job = plan.root, plan.job
    if (root / LOCK).exists() or digest(root / LEDGER) != plan.ledger_hash:
        raise ValueError("Owner/ledger changed before mutation.")
    for p, h in plan.bindings.items():
        if digest(p) != h:
            raise ValueError("Binding changed before launch/mutation.")
    job.receipt.mkdir(exist_ok=False)
    started = clock()
    m = plan.admission.manifest
    cuda = m["device"] == "cuda:0"
    record = {
        "id": job.receipt.name,
        "attempt_id": str(uuid.uuid4()),
        "status": "RUNNING_CUDA" if cuda else "RUNNING_CPU_FIT",
        "operation": "FROZEN_ASSESSMENT_NO_FITTING",
        "fitting": False,
        "scheduler_status_compatibility": "CPU_FIT blocks legacy owners; device/operation remain CPU assessment",
        "device": m["device"],
        "uses_band_model": any(s["kind"] in BAND_KINDS for s in m["methods"].values()),
        "budget_family": FAMILY if plan.charge_band else None,
        "evidence_kind": m["evidence_kind"],
        "role": m["role"],
        "command": plan.command,
        "pid": None,
        "output": str(job.output),
        "receipt": str(job.receipt),
        "bindings": plan.bindings,
        "manifest_sha256": digest(job.manifest),
        "review_sha256": digest(job.review),
        "requires_reconciliation": True,
    }
    ledger = copy.deepcopy(plan.ledger)
    ledger["runs"].append(record)
    journal_failed = False
    charged = 0.0
    band_base = sum(
        _number(
            r.get(
                "elapsed_owned_seconds",
                r.get("resources_full_attempt", {}).get("elapsed_full_attempt_seconds"),
            ),
            "previous Band full time",
        )
        / 3600
        for r in plan.ledger["runs"]
        if r.get("budget_family") == FAMILY
    )

    def journal():
        nonlocal journal_failed
        try:
            write_ledger(root, ledger)
        except BaseException:
            journal_failed = True
            raise

    def charge(elapsed):
        nonlocal charged
        elapsed = _number(elapsed, "full owned elapsed")
        delta = max(0.0, elapsed - charged)
        if cuda:
            ledger["gpu_hours_spent_owned_scientific_jobs"] += delta / 3600
        else:
            ledger["cpu_assessment_hours_owned"] = (
                _number(ledger.get("cpu_assessment_hours_owned", 0), "CPU hours") + delta / 3600
            )
        charged = max(charged, elapsed)
        if plan.charge_band:
            ledger["native_band_gpu_hours_spent_full_owned"] = band_base + charged / 3600
        record["elapsed_owned_seconds"] = charged
        if "resources_full_attempt" in record:
            record["resources_full_attempt"]["elapsed_full_attempt_seconds"] = charged

    journal()
    try:
        with (job.receipt / "console.log").open("x", encoding="utf-8") as log:
            child = popen(plan.command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            record["pid"] = child.pid
            journal()
            resources = supervisor(
                child,
                started=started,
                deadline_seconds=plan.deadline_seconds,
                rss_limit_bytes=RSS_LIMIT,
            )
        record["resources_full_attempt"] = resources
        report = _completion(plan, child, resources)
        lock = root / LOCK
        if resources.get("stopped_for") is not None and lock.exists():
            raw = lock.read_bytes()
            owner = _json(raw)
            if (
                owner.get("pid") in resources["owned_process_pids"]
                and type(owner.get("pid")) is int
                and owner.get("output") == str(job.output)
                and lock.read_bytes() == raw
                and resources["owned_tree_cleanup_verified"] is True
            ):
                lock.unlink()
                record["owned_terminated_tree_lock_cleanup"] = True
            else:
                record["unknown_lock_preserved"] = True
        elapsed = max(
            _number(resources["elapsed_full_attempt_seconds"], "full time"), clock() - started
        )
        charge(elapsed)
        completed = (
            report is not None
            and not lock.exists()
            and elapsed < plan.deadline_seconds
            and resources["peak_process_rss_bytes"] < RSS_LIMIT
        )
        record.update(
            exit_code=resources["exit_code"],
            completion_verified=completed,
            report_sha256=digest(job.output / "completion.json") if report else None,
            owned_tree_cleanup_verified=resources["owned_tree_cleanup_verified"],
        )
        record["requires_reconciliation"] = lock.exists()
        record["status"] = (
            ("COMPLETED_REAL_BAND_CUDA" if completed else "FAILED_REAL_BAND_CUDA_ATTEMPT")
            if plan.charge_band
            else ("COMPLETED_REAL_CUDA" if completed else "FAILED_REAL_CUDA_ATTEMPT")
            if cuda
            else ("COMPLETED_CPU_ASSESSMENT" if completed else "FAILED_CPU_ASSESSMENT_ATTEMPT")
        )
        _exclusive_json(job.receipt / "assessment-attempt.json", record)
        charge(max(charged, clock() - started))
        journal()
        return 0 if completed else resources["exit_code"] if resources["exit_code"] != 0 else 1
    except BaseException as error:
        record.update(
            status="RUNNING_CUDA" if cuda else "RUNNING_CPU_FIT",
            requires_reconciliation=True,
            parent_exception_type=type(error).__name__,
            completion_verified=False,
        )
        charge(max(charged, clock() - started))
        if not journal_failed:
            journal()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "review", "output", "receipt"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args(argv)
    plan = prelaunch(Job(args.manifest, args.review, args.output, args.receipt))
    return _execute(plan, popen=subprocess.Popen, supervisor=_supervisor(plan.root, plan.bindings))


if __name__ == "__main__":
    raise SystemExit(main())
