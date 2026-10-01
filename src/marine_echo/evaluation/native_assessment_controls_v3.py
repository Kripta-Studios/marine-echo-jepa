"""Version3 matched-control frozen assessment; no fitting, selection, or provenance execution.

Root supplies genuine distinct review documents and complete immutable recursive
local ancestry receipts. Historical metadata exposure is disclosed separately.
Source-calendar dates are not verified UTC. Chronos/external ancestry is unsupported.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import io
import json
import re
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from marine_echo.evaluation import native_product

IMPLEMENTER_SESSION_ID = "01a0f72d-927a-77a3-8b62-21d456f7ed85"
ORIGINAL_BUILDER_SESSION_ID = "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1"
STATUSES = {
    "final_test": "APPROVED_FINAL_ASSESSMENT_EXECUTION",
    "development": "APPROVED_DEVELOPMENT_ASSESSMENT_EXECUTION",
}
NEURAL_KINDS = {
    "native_cf_control_weights_only_inference_v1",
    "native_ssl_weights_only_inference_v1",
    "native_band_ssl_weights_only_inference_v1",
    "native_band_replication_ssl_weights_only_inference_v2",
}
REFERENCES = {
    "persistence": "persistence",
    "seasonal24": "seasonal24",
    "lightgbm15_utf8": "lightgbm",
}
METHODS = {
    "cf_random_frozen",
    "cf_direct_supervised",
    "shared_ssl",
    "masked_ssl",
    "permuted_ssl",
    "random_frozen",
    "direct",
    "cf_jepa",
}
MODES = {"core_frozen_readout", "frozen_readout", "full_finetune", "direct_end_to_end"}
ROBUSTNESS = {
    "kind": "secondary_observation_dropout_v1",
    "probability": 0.25,
    "seed": 20260929,
    "channels": [1, 2, 3],
}
LIMITS = {"rss_gib": 22, "cuda_gib": 10, "one_gpu_process": True}
ROOT_PACKAGE = Path(native_product.__file__).resolve().parents[1]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


IMPORTED_HASHES = {
    str(Path(p).resolve()): sha(Path(p).read_bytes()) for p in (__file__, native_product.__file__)
}


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
        ORIGINAL_BUILDER_SESSION_ID.casefold(),
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


def required_sources(loaders=None):
    """Static local import closure including lazy imports; never import manifest paths.

    External dependency bytes are represented by the separately admitted lockfile.
    Registered caller factories must be supplied as functions in reviewed modules.
    """
    pending = [
        Path(__file__).resolve(),
        Path(native_product.__file__).resolve(),
        ROOT_PACKAGE / "inference/native_cf_controls.py",
        ROOT_PACKAGE / "inference/native_acoustic.py",
        ROOT_PACKAGE / "inference/native_encoder.py",
        ROOT_PACKAGE / "inference/native_band_acoustic.py",
        ROOT_PACKAGE / "evaluation/native_assessment.py",
        ROOT_PACKAGE / "inference/native_band_replication_acoustic.py",
        ROOT_PACKAGE / "training/native_band_replication_downstream.py",
        ROOT_PACKAGE / "training/native_downstream.py",
        ROOT_PACKAGE / "training/native_references.py",
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
                    if not p.is_relative_to(ROOT_PACKAGE):
                        continue
                    prefix = ["marine_echo", *p.relative_to(ROOT_PACKAGE).parts[:-1]]
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
                    ROOT_PACKAGE.joinpath(*parts).with_suffix(".py"),
                    ROOT_PACKAGE.joinpath(*parts) / "__init__.py",
                )
                candidate = f if parts and f.is_file() else init
                if candidate.is_file():
                    pending.append(candidate)
                for n in range(len(parts) + 1):
                    init = ROOT_PACKAGE.joinpath(*parts[:n]) / "__init__.py"
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


def admit(manifest_path, review_path, output_path, *, loaders=None):
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
    for p in required_sources(loaders):
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
        _typed_control_config(kind, config, spec, m["evidence_kind"])
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
        "implementer_session_id": IMPLEMENTER_SESSION_ID,
        "original_builder_session_id": ORIGINAL_BUILDER_SESSION_ID,
        "recursive_local_ancestry": ancestry_proof,
        "historical_metadata_exposure_disclosure": m["historical_metadata_exposure_disclosure"],
        "no_universal_sealed_claim": True,
        "external_ancestry_guarantee": False,
    }
    return Admission(m, base, documents, snapshots, paths, provenance)


def _typed_control_config(kind, config, spec, evidence):
    """Stdlib metadata checks before tensor/array decode; no method relabelling."""
    control_kind = "native_cf_control_weights_only_inference_v1"
    methods = {"cf_random_frozen", "cf_direct_supervised"}
    if kind != control_kind and spec.get("method") not in methods:
        return
    if kind != control_kind or spec.get("method") not in methods:
        raise ValueError("Distinct CF control kind and method required")
    frozen = spec["method"] == "cf_random_frozen"
    expected_mode = "frozen_readout" if frozen else "direct_end_to_end"
    fixed = {
        "method": spec["method"],
        "seed": spec.get("seed"),
        "history": 96,
        "heads": 4,
        "lr": 0.0003,
        "weight_decay": 0.0001,
        "gradient_clip": 1.0,
        "patience": 4,
        "min_daily_anchors": 18,
        "selection_policy": "scheduled_native_daily_pinball_earliest_strict_improvement_v1",
    }
    dimensions = {
        "width": 256,
        "latent": 128,
        "blocks": 5,
        "batch_size": 64,
        "updates": 2000 if frozen else 3000,
        "cadence": 500 if frozen else 750,
    }
    if (
        spec.get("mode") != expected_mode
        or type(spec.get("seed")) is not int
        or spec["seed"] not in (7, 13, 23)
        or set(config) != set(fixed) | set(dimensions)
        or any(config.get(k) != v or type(config.get(k)) is not type(v) for k, v in fixed.items())
    ):
        raise ValueError("Exact CF control recipe, mode, seed and fields required")
    if evidence == "SYNTHETIC_CORRECTNESS_ONLY":
        if (
            any(
                type(config[k]) is not int or not 1 <= config[k] <= ceiling
                for k, ceiling in dimensions.items()
            )
            or config["cadence"] > config["updates"]
        ):
            raise ValueError("Bounded synthetic control dimensions required")
    elif any(config[k] != v or type(config[k]) is not int for k, v in dimensions.items()):
        raise ValueError("Exact real CF control dimensions and supervision required")


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


def array_descriptor(value):
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "content_sha256": sha(np.ascontiguousarray(value).tobytes()),
        "encoding": "numpy_C_order_bytes",
    }


def validate_context(x, observed, metadata, query, history):
    if x.ndim != 3 or x.shape[1:] != (history, 4) or not len(x) or x.dtype.kind != "f":
        raise ValueError("Exact raw context [N,H,4] required; no future suffix.")
    if (
        observed.dtype != np.bool_
        or observed.shape != x.shape
        or not observed[:, :, 0].all()
        or not np.isfinite(x[observed]).all()
    ):
        raise ValueError("Complete finite primary issuance and Boolean context mask required.")
    if (
        metadata.shape != (len(x), 4, 10)
        or metadata.dtype.kind != "f"
        or not np.isfinite(metadata).all()
        or query.shape != (len(x), 3, 10)
        or query.dtype.kind != "f"
        or not np.isfinite(query).all()
    ):
        raise ValueError("Finite native metadata/query shapes required.")
    fixed = {1: 1, 2: 1, 5: 0, 9: 0}
    if any(
        not np.allclose(metadata[..., k], v, rtol=0, atol=1e-6) for k, v in fixed.items()
    ) or not np.allclose(
        metadata[..., 0], np.array([38000, 125000, 200000, 455000]) / 455000, rtol=0, atol=1e-6
    ):
        raise ValueError("Native frequency/order/60min/integrated/offset differs.")
    if (
        not np.isin(metadata[..., 6:9], [0, 1]).all()
        or (metadata[..., 3] < 0).any()
        or (metadata[..., 4] <= metadata[..., 3]).any()
        or not np.allclose(metadata[:, 0, 3], 0, rtol=0, atol=1e-6)
        or not np.allclose(metadata[:, 0, 4], 230 / 250, rtol=0, atol=1e-6)
    ):
        raise ValueError("This assessment retains whole-site native 0–230 m and known flags.")
    if not np.allclose(query[..., :9], metadata[:, :1, :9], rtol=0, atol=1e-6) or not np.allclose(
        query[..., 9], [1, 3, 6], rtol=0, atol=1e-6
    ):
        raise ValueError("Issued primary geometry/horizons differ.")


def load_input(admitted):
    m = admitted.manifest
    core = m["input_schema"] == "native_corpus_v1"
    ck, tk = ("observed", "y") if core else ("context_observed", "targets")
    with np.load(
        io.BytesIO(admitted.snapshots[admitted.paths["input"]]), allow_pickle=False
    ) as saved:
        keys = [
            "x",
            ck,
            "metadata",
            "query",
            tk,
            "row_id",
            "deployment",
            "target_dates",
            "corpus_role",
        ]
        if core:
            keys.append("split_sha256")
        if m["input_evidence_kind"] is not None:
            keys.append("evidence_kind")
        elif "evidence_kind" in saved.files:
            raise ValueError("Source evidence header exists but manifest claimed absence.")
        if m["evidence_kind"] == "SYNTHETIC_CORRECTNESS_ONLY":
            keys.append("fixture_identity")
        if not set(keys) <= set(saved.files):
            raise ValueError("Missing required context/support/role provenance fields.")
        data = {k: saved[k].copy() for k in keys}
        aliases = [
            k for k in ("y_observed", "target_observed", "observed") if k in saved.files and k != ck
        ]
        masks = {k: saved[k].copy() for k in aliases}
        optional = {
            k: saved[k].copy() for k in ("cutoff", "archive_id", "source_id") if k in saved.files
        }
        # Future/future_observed and all unlisted arrays are NEVER decoded.
    header_expectations = {"corpus_role": m["input_corpus_role"]}
    if m["input_evidence_kind"] is not None:
        header_expectations["evidence_kind"] = m["input_evidence_kind"]
    if core:
        header_expectations["split_sha256"] = sha(admitted.snapshots[admitted.paths["split"]])
    if m["evidence_kind"] == "SYNTHETIC_CORRECTNESS_ONLY":
        header_expectations["fixture_identity"] = "SYNTHETIC_CORRECTNESS_ONLY"
    for k, expected in header_expectations.items():
        if data[k].shape != () or data[k].item() != expected:
            raise ValueError("Numerical corpus role/evidence differs.")
    history = admitted.documents[
        _path(next(iter(m["methods"].values()))["config_path"], admitted.base)
    ]["history"]
    x, context_mask, targets = data["x"], data[ck], data[tk]
    validate_context(x, context_mask, data["metadata"], data["query"], history)
    if targets.shape != (len(x), 3) or targets.dtype.kind != "f" or not masks:
        raise ValueError("Floating targets and explicit target label masks required.")
    for name, mask in masks.items():
        if mask.shape != targets.shape or mask.dtype != np.bool_:
            raise ValueError(f"Forecast label mask {name} must be Boolean [N,3].")
    observed = next(iter(masks.values()))
    if any(not np.array_equal(observed, v) for v in masks.values()):
        raise ValueError("Conflicting dual forecast masks.")
    if not np.isfinite(targets[observed]).all():
        raise ValueError("Observed targets must be finite; support cannot shrink.")
    for k in ("deployment", "row_id"):
        if data[k].shape != (len(x),) or data[k].dtype.kind != "U" or any(not v for v in data[k]):
            raise ValueError("Nonempty Unicode deployment/row identities required.")
    pairs = list(zip(data["deployment"].tolist(), data["row_id"].tolist(), strict=True))
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate issuance identity.")
    cohort = admitted.documents[admitted.paths["cohort"]]
    declared = [tuple(v) for v in cohort["rows"]]
    if (
        len(set(declared)) != len(declared)
        or set(pairs) != set(declared)
        or set(data["deployment"]) != _members(cohort)["deployment_id"]
    ):
        raise ValueError("Exact cohort issuance set/deployments differs; no intersections.")
    dates = data["target_dates"]
    if dates.shape != (len(x), 3) or dates.dtype.kind != "U":
        raise ValueError("Source-calendar target dates must be Unicode [N,3].")
    for v in set(dates.ravel().tolist()):
        if v and date.fromisoformat(v).isoformat() != v:
            raise ValueError("Exact source-calendar dates required.")
    provenance = {
        "original_fields": list(masks),
        "canonical_field": "observed",
        "equal_dual_masks": len(masks) > 1,
        "context_mask_field": ck,
        "original_corpus_role": m["input_corpus_role"],
        "original_evidence_header": m["input_evidence_kind"],
    }
    return {
        "x": x,
        "context_observed": context_mask,
        "metadata": data["metadata"],
        "query": data["query"],
        "targets": targets,
        "observed": observed,
        "row_id": data["row_id"],
        "deployment": data["deployment"],
        "target_dates": dates,
        **optional,
    }, provenance


def secondary_dropout(x, observed, recipe):
    if recipe not in (None, ROBUSTNESS):
        raise ValueError("Only the predeclared secondary25% drop recipe supported.")
    values, remaining, removed = x.copy(), observed.copy(), np.zeros_like(observed)
    if recipe is not None:
        draws = np.random.default_rng(20260929).random(observed[:, :, 1:].shape)
        removed[:, :, 1:] = (draws < 0.25) & observed[:, :, 1:]
        remaining &= ~removed
    values[~remaining] = 0
    return values, remaining, removed


class _NativeReplay:
    """Immutable legacy forecast math using meta construction, never fit setup."""

    def __init__(self, artifact, device):
        import torch

        from marine_echo.inference.native_encoder import _config, _scalers
        from marine_echo.models.native_temporal import CFNativeModel, NativeTemporalModel

        self.config, self.scalers = _config(artifact["config"]), _scalers(artifact["scalers"])
        self.device, c = device, self.config
        with torch.device("meta"):
            model = (
                CFNativeModel(c.cf_width, c.cf_latent, c.cf_blocks)
                if c.method == "cf_jepa"
                else NativeTemporalModel(c.width, c.latent, c.blocks, c.heads)
            )
            model = model.float()
        state, expected = artifact.get("model"), model.state_dict()
        if not isinstance(state, dict) or set(state) != set(expected):
            raise ValueError("Exact complete safe inference tensor state required.")
        for key, template in expected.items():
            tensor = state[key]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or tensor.layout != torch.strided
                or tensor.shape != template.shape
                or tensor.dtype != template.dtype
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(f"Invalid inference tensor: {key}")
        model.load_state_dict(state, strict=True, assign=True)
        self.model = model.to(device).eval().requires_grad_(False)

    def forecast(self, x, observed, metadata, query):
        import torch

        self.model.eval()
        with torch.inference_mode():
            p = (
                self.model.forecast(
                    torch.as_tensor(self.scalers.channels(x, observed), device=self.device),
                    torch.as_tensor(observed.copy(), device=self.device),
                    torch.as_tensor(metadata.copy(), device=self.device),
                    torch.as_tensor(query.copy(), device=self.device),
                )
                .cpu()
                .numpy()
            )
        return p * self.scalers.target_std[None, :, None] + self.scalers.target_mean[None, :, None]


def _builtin(
    spec,
    snapshots,
    config,
    statistics,
    device,
    base,
    *,
    ancestor_artifact_hashes=None,
    expected_evidence=None,
):
    kind = spec["kind"]
    payloads = [snapshots[_path(p, base)] for p in spec["model_paths"]]
    if kind in NEURAL_KINDS:
        import torch

        if len(payloads) != 1:
            raise ValueError("Exactly one owned inference artifact required.")
        artifact = torch.load(io.BytesIO(payloads[0]), weights_only=True, map_location="cpu")
        if (
            not isinstance(artifact, dict)
            or artifact.get("kind") != kind
            or artifact.get("config") != config
            or artifact.get("scalers") != statistics
        ):
            raise ValueError("Safe artifact kind/config/scaler identities differ.")
        if kind == "native_cf_control_weights_only_inference_v1":
            from marine_echo.inference.native_cf_controls import NativeCFControlPredictor

            evidence = artifact.get("evidence_kind")
            if expected_evidence is not None and evidence != (
                "SYNTHETIC_CORRECTNESS_ONLY"
                if expected_evidence == "SYNTHETIC_CORRECTNESS_ONLY"
                else "REAL_TRAIN_DEVELOPMENT_FIT"
            ):
                raise ValueError("Control artifact evidence differs from admitted assessment")
            _typed_control_config(kind, config, spec, evidence)
            predictor = NativeCFControlPredictor(io.BytesIO(payloads[0]), device=device)
            if predictor.config.mode != spec["mode"]:
                raise ValueError("Control inference mode differs from frozen manifest")
            return predictor.forecast
        if spec["mode"] == "core_frozen_readout":
            if artifact.get("supervised_ancestry") is not None:
                raise ValueError("Downstream supervised cannot be labeled core readout.")
        else:
            supervised = artifact.get("supervised_ancestry")
            if (
                not isinstance(supervised, dict)
                or supervised.get("mode") != spec["mode"]
                or supervised.get("ssl_only") is not False
            ):
                raise ValueError("Explicit accurate downstream supervised ancestry required.")
            downstream = artifact.get("downstream_config")
            if (
                not isinstance(downstream, dict)
                or downstream.get("mode") != spec["mode"]
                or any(
                    downstream.get(k) != config.get(k)
                    for k in ("method", "seed", "history", "batch_size")
                )
            ):
                raise ValueError("Exact saved downstream config/identity required.")
            total, steps, selected = (
                downstream.get("updates"),
                supervised.get("supervised_updates"),
                supervised.get("selected_supervised_step"),
            )
            if (
                any(type(v) is not int for v in (total, steps, selected))
                or not 1 <= selected <= steps <= total <= 5000
            ):
                raise ValueError("Invalid frozen supervised update/selection ancestry.")
            for key in ("ancestor_encoder_sha256", "ancestor_run_sha256"):
                value = supervised.get(key)
                if spec["mode"] == "direct_end_to_end":
                    if value is not None:
                        raise ValueError("Direct endpoint must not inherit a selected ancestor.")
                elif (
                    not isinstance(value, str)
                    or not re.fullmatch("[0-9a-f]{64}", value)
                    or (
                        ancestor_artifact_hashes is not None
                        and value not in ancestor_artifact_hashes
                    )
                ):
                    raise ValueError(
                        "Encoded selected/run ancestor is absent from the admitted recursive graph."
                    )
        if kind == "native_band_replication_ssl_weights_only_inference_v2":
            from marine_echo.inference.native_band_replication_acoustic import (
                NativeBandAcousticPredictor,
            )

            predictor = object.__new__(NativeBandAcousticPredictor)
            predictor._initialize(artifact, device)
        elif kind == "native_band_ssl_weights_only_inference_v1":
            from marine_echo.inference.native_band_acoustic import NativeBandAcousticPredictor

            predictor = object.__new__(NativeBandAcousticPredictor)
            predictor._initialize(artifact, device)
        else:
            predictor = _NativeReplay(artifact, device)
        return predictor.forecast
    from marine_echo.training.native_references import feature_matrix

    if device != "cpu":
        raise ValueError("Reference replay is CPU only.")
    if kind == "lightgbm15_utf8":
        import lightgbm as lgb

        expected = [[h, float(q)] for h in (1, 3, 6) for q in native_product.QUANTILES]
        if spec.get("booster_slots") != expected or len(payloads) != 15:
            raise ValueError("Exact stored fifteen horizon/quantile boosters required.")
        boosters = [lgb.Booster(model_str=p.decode("utf-8")) for p in payloads]

        def forecast(x, observed, metadata, query):
            features = feature_matrix(
                {"x": x, "observed": observed, "metadata": metadata}, config["history"]
            )
            if any(b.num_feature() != features.shape[1] for b in boosters):
                raise ValueError("Stored booster feature schema/dimensions differ.")
            prediction = np.stack([b.predict(features) for b in boosters], axis=1).reshape(
                len(x), 3, 5
            )
            return np.sort(prediction, axis=-1)

        return forecast

    def forecast(x, observed, metadata, query):
        # Exact native_references.run indexing; no standalone immutable helper exists.
        values = (
            np.repeat(x[:, -1, 0, None], 3, axis=1)
            if kind == "persistence"
            else x[:, -25 + np.asarray([1, 3, 6]), 0]
        )
        return np.repeat(values[..., None], 5, axis=-1)

    return forecast


def _exclusive_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def execute_assessment(manifest_path, review_path, output_path, *, loaders=None):
    """Guard, replay context-only forecasts, then score on identical support.

    Registered factories are trusted, explicitly reviewed caller code. They receive
    artifact/config bytes, NEVER a corpus. Forecast calls receive exactly four
    arrays. IO errors propagate without retries; partial runs have no completion.
    Paired intervals need a later distinct exact native_comparison review.
    """
    started = time.perf_counter()
    admitted = admit(manifest_path, review_path, output_path, loaders=loaders)
    m, base = admitted.manifest, admitted.base
    import psutil

    peak_rss, allocated, reserved = 0, 0, 0
    process = psutil.Process()

    def resources():
        nonlocal peak_rss, allocated, reserved
        peak_rss = max(peak_rss, process.memory_info().rss)
        if m["device"] == "cuda:0":
            import torch

            torch.cuda.synchronize("cuda:0")
            allocated = max(allocated, torch.cuda.max_memory_allocated("cuda:0"))
            reserved = max(reserved, torch.cuda.max_memory_reserved("cuda:0"))
        if peak_rss >= 22 * 1024**3 or max(allocated, reserved) >= 10 * 1024**3:
            raise RuntimeError("Resource ceiling exceeded; no unsafe foreign termination.")

    forecasters, load_seconds = {}, {}

    def parent_hashes(receipt):
        node = admitted.documents[receipt]
        hashes = set()
        for parent in node["parents"]:
            p = _path(parent, receipt.parent)
            hashes.update(v["sha256"] for v in admitted.documents[p]["artifacts"])
            hashes.update(parent_hashes(p))
        return hashes

    for name, spec in m["methods"].items():
        resources()
        load_started = time.perf_counter()
        config = admitted.documents[_path(spec["config_path"], base)]
        statistics = admitted.documents[_path(spec["statistics_path"], base)]
        factory = _builtin if spec["loader"] == "builtin" else loaders[spec["loader"]]
        model_snapshots = {
            _path(p, base): admitted.snapshots[_path(p, base)] for p in spec["model_paths"]
        }
        # Factories see only model bytes, never the corpus/target snapshots.
        arguments = (spec, model_snapshots, config, statistics, m["device"], base)
        forecasters[name] = (
            factory(
                *arguments,
                ancestor_artifact_hashes=parent_hashes(_path(spec["ancestry_path"], base)),
                expected_evidence=m["evidence_kind"],
            )
            if spec["loader"] == "builtin"
            else factory(*arguments)
        )
        resources()
        load_seconds[name] = time.perf_counter() - load_started
    # Exact owned artifact kind/config/tensors are checked BEFORE corpus decoding.
    data, mask_provenance = load_input(admitted)
    x, mask, dropped = secondary_dropout(data["x"], data["context_observed"], m.get("robustness"))
    context = (x, mask, data["metadata"].copy(), data["query"].copy())
    for v in context:
        v.flags.writeable = False
    output = Path(output_path).resolve()
    prediction_evidence_kind = (
        "SYNTHETIC_CORRECTNESS_ONLY"
        if m["evidence_kind"] == "SYNTHETIC_CORRECTNESS_ONLY"
        else "REVIEWED_SAVED_PREDICTIONS"
    )
    output.mkdir(parents=False, exist_ok=False)
    with (output / "context-diagnostics.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            dropped=dropped,
            drop_positions=np.argwhere(dropped),
            row_id=data["row_id"],
            deployment=data["deployment"],
        )
    results = {}
    for name, spec in m["methods"].items():
        resources()
        forecast = forecasters.pop(name)
        forecast_started = time.perf_counter()
        chunks = []
        for start in range(0, len(x), m["batch_size"]):
            end = min(start + m["batch_size"], len(x))
            prediction = np.asarray(forecast(*(v[start:end] for v in context)))
            if (
                prediction.shape != (end - start, 3, 5)
                or prediction.dtype.kind != "f"
                or not np.isfinite(prediction).all()
                or (np.diff(prediction, axis=-1) < 0).any()
            ):
                raise ValueError("Every forecast must be finite ordered floating [N,3,5].")
            chunks.append(prediction.copy())
            resources()
        predictions = np.concatenate(chunks)
        forecast_seconds = time.perf_counter() - forecast_started
        metrics = native_product.native_scores(
            predictions,
            data["targets"],
            data["observed"],
            data["target_dates"],
            data["deployment"],
            minimum_daily_rows=18,
        )
        artifact = output / f"{name}.npz"
        with artifact.open("xb") as stream:
            np.savez_compressed(
                stream,
                predictions=predictions,
                **{k: v for k, v in data.items() if k not in {"x", "context_observed", "metadata"}},
                corpus_role=np.asarray(m["role"]),
                evidence_kind=np.asarray(prediction_evidence_kind),
                source_evidence_kind=np.asarray(m["evidence_kind"]),
                query_native_bounds_m=data["query"][..., 3:5] * 250,
                query_frequency_hz=data["query"][..., 0] * 455000,
                query_interval_seconds=data["query"][..., 1] * 3600,
            )
        feature_kind = {
            "shared_ssl": "ssl_pretrained",
            "masked_ssl": "ssl_pretrained",
            "cf_jepa": "ssl_pretrained",
            "permuted_ssl": "permuted_pairing_ssl_control",
            "direct": "supervised_feature_encoder",
            "random_frozen": "untrained_control",
            "cf_random_frozen": "untrained_control",
            "cf_direct_supervised": "supervised_feature_encoder",
        }.get(spec["method"], "conventional_reference")
        if spec["mode"] in {"full_finetune", "direct_end_to_end"}:
            feature_kind = "supervised_feature_encoder"
        results[name] = {
            "method": spec["method"],
            "seed": spec["seed"],
            "mode": spec["mode"],
            "kind": spec["kind"],
            "feature_training_kind": feature_kind,
            "forecast_training_kind": "conventional_reference"
            if spec["mode"] == "reference"
            else "supervised_readout_on_" + feature_kind,
            "model_load_elapsed_seconds": load_seconds[name],
            "device_forecast_wall_seconds": forecast_seconds,
            "prediction_path": str(artifact),
            "prediction_sha256": sha(artifact.read_bytes()),
            "metrics": metrics,
            "forecast_variation_diagnostic": {
                "constant_across_issuance": bool((np.ptp(predictions, axis=0) == 0).all()),
                "quantile_range_db_per_horizon": np.ptp(predictions, axis=0).tolist(),
                "diagnostic_only": True,
            },
            "status": "ASSESSABLE"
            if metrics["primary_pinball_db"] is not None
            else "NOT_ASSESSABLE",
        }
        del forecast
    resources()
    result = {
        "kind": "native_control_frozen_assessment_completion_v3",
        "status": "COMPLETED_FORECASTS"
        if all(v["status"] == "ASSESSABLE" for v in results.values())
        else "COMPLETED_FORECASTS_NOT_ASSESSABLE",
        "role": m["role"],
        "evidence_kind": m["evidence_kind"],
        "methods": results,
        "provenance": admitted.provenance,
        "prediction_artifact_evidence_kind": prediction_evidence_kind,
        "mask_provenance": mask_provenance,
        "support": {
            k: array_descriptor(data[k])
            for k in ("row_id", "deployment", "targets", "observed", "target_dates", "query")
        },
        "context_diagnostics": {
            "recipe": m.get("robustness"),
            "drop_mask": array_descriptor(dropped),
            "dropped_observations": int(dropped.sum()),
            "observed_before_per_channel": data["context_observed"].sum(axis=(0, 1)).tolist(),
            "observed_after_per_channel": mask.sum(axis=(0, 1)).tolist(),
            "positions_artifact": str(output / "context-diagnostics.npz"),
            "positions_artifact_sha256": sha((output / "context-diagnostics.npz").read_bytes()),
            "summarized_arrays": ["context_observed", "metadata", "drop_mask"],
            "original_context_mask": array_descriptor(data["context_observed"]),
            "metadata": array_descriptor(data["metadata"]),
        },
        "native_geometry": {
            "upper_m": 0,
            "lower_m": 230,
            "frequency_hz": 38000,
            "interval_seconds": 3600,
            "horizons": [1, 3, 6],
            "quantiles": native_product.QUANTILES.tolist(),
            "dB_rescaling": False,
        },
        "resources": {
            "process_id": process.pid,
            "device": m["device"],
            "elapsed_seconds": time.perf_counter() - started,
            "sampled_peak_rss_bytes": peak_rss,
            "process_cuda_peak_allocated_bytes": allocated,
            "process_cuda_peak_reserved_bytes": reserved,
            "supervisor_owns_peak_rss_deadline_one_gpu_full_accounting": True,
        },
        "paired_intervals": "NOT_RUN_requires_separate_native_comparison_exact_prediction_review",
        "date_basis": "source_calendar_proxy_not_verified_UTC",
        "scientific_claim": None,
    }
    _exclusive_json(output / "completion.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "review", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    result = execute_assessment(args.manifest, args.review, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "role": result["role"],
                "evidence_kind": result["evidence_kind"],
            }
        )
    )


if __name__ == "__main__":
    main()
