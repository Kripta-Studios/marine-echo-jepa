"""Separately admitted adapted suffix replay. No fitting or zero-shot gate waiver.

One true deployment per invocation; all declared cells retain its identical
source-calendar suffix. A coordinator-owned process wrapper supplies lifetime,
whole-tree ownership and aggregate/Band/CF accounting. This module never kills
processes or deletes locks. Provenance paths are metadata, never executable code.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import stat
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import torch

from marine_echo.evaluation import native_comparison as comparison
from marine_echo.evaluation.native_product import native_scores
from marine_echo.inference.native_band_acoustic import _query
from marine_echo.inference.native_encoder import _inputs
from marine_echo.training import native_prefix_matched_transfer_v4 as prefix

ROLE = "adapted_suffix"
EVIDENCE = "SYNTHETIC_CORRECTNESS_ONLY"
REAL = "REVIEWED_PREFIX_SUFFIX_ASSESSMENT"
NEURAL_KIND = "native_prefix_matched_transfer_inference_v4"
REFERENCE_KIND = "native_prefix_lightgbm_inference_v1"
UNFITTED = {"persistence", "seasonal24"}
RECIPE = {
    "bootstrap_seed": 20260929,
    "bootstrap_replicates": 2000,
    "block_hours": 48,
    "block_days": 2,
    "floor": 18,
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def regular(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("No symlink/reparse artifact or source paths.")
    if not path.is_file():
        raise ValueError("Regular immutable input required.")
    return path


def verify_review(review, manifest, status, scope):
    reviewer = review.get("reviewer_session_id")
    excluded = [
        prefix.IMPLEMENTER_SESSION_ID,
        manifest.get("implementer_session_id"),
        manifest.get("coordinator_session_id"),
    ]
    if (
        not isinstance(reviewer, str)
        or not reviewer.strip()
        or reviewer.casefold() in {s.casefold() for s in excluded if isinstance(s, str)}
    ):
        raise ValueError("Genuinely distinct reviewer required; no self/coordinator review.")
    expected = {
        "status": status,
        "scope": scope,
        "allowed_roles": [ROLE],
        "evidence_kind": manifest.get("evidence_kind"),
        "implementer_session_id": manifest.get("implementer_session_id"),
        "coordinator_session_id": manifest.get("coordinator_session_id"),
    }
    if (
        manifest.get("role") != ROLE
        or any(not manifest.get(k) for k in ("implementer_session_id", "coordinator_session_id"))
        or any(review.get(k) != v for k, v in expected.items())
    ):
        raise ValueError("Exact adapted role/status/scope/evidence/identity required.")
    bindings = review.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("Every immutable source/model/data/partition binding required.")
    for name, digest in bindings.items():
        if (
            not isinstance(name, str)
            or not Path(name).is_absolute()
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or sha(regular(name).read_bytes()) != digest
        ):
            raise ValueError("Missing/stale/unsafe exact binding.")
    return bindings


def required_sources():
    root = Path(prefix.core.__file__).resolve().parents[1]
    pending = [
        Path(__file__).resolve(),
        *prefix.required_sources(),
        Path(comparison.__file__),
        root / "evaluation/native_product.py",
    ]
    found = set()
    while pending:
        path = regular(pending.pop())
        if path in found:
            continue
        found.add(path)
        if path.suffix != ".py":
            continue
        for node in ast.walk(ast.parse(path.read_bytes())):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or "", *((node.module or "") + "." + a.name for a in node.names)]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            for name in names:
                if name.startswith("marine_echo."):
                    candidate = root.joinpath(*name.split(".")[1:]).with_suffix(".py")
                    if candidate.is_file():
                        pending.append(candidate)
    return sorted(found)


LOADED_SOURCES = {str(p): sha(p.read_bytes()) for p in required_sources()}


def check_resources(device):
    rss = psutil.Process().memory_info().rss
    if rss >= 22 * 2**30:
        raise RuntimeError("RSS22GiB limit reached.")
    values = {"rss_bytes": rss, "cuda_allocated_bytes": 0, "cuda_reserved_bytes": 0}
    if device == "cuda:0" and torch.cuda.is_initialized():
        values.update(
            cuda_allocated_bytes=torch.cuda.memory_allocated(0),
            cuda_reserved_bytes=torch.cuda.memory_reserved(0),
        )
        if max(values["cuda_allocated_bytes"], values["cuda_reserved_bytes"]) >= 10 * 2**30:
            raise RuntimeError("CUDA allocated/reserved10GiB limit reached.")
    return values


@dataclass
class Admission:
    manifest: dict
    snapshots: dict
    documents: dict
    partition: dict
    fitted: dict
    identities: dict


def admit(manifest_path, review_path, output_path):
    """Complete software/access/execution/source/ancestry admission before decode."""
    manifest_path, review_path = regular(manifest_path), regular(review_path)
    output = Path(output_path).absolute()
    if output.exists():
        raise FileExistsError("Fresh immutable suffix output only.")
    for part in output.parents:
        if part.exists() and (
            part.is_symlink() or getattr(part.lstat(), "st_file_attributes", 0) & 0x400
        ):
            raise ValueError("No reparse output paths.")
    base = manifest_path.parent
    manifest = prefix._json(manifest_path.read_bytes())
    if (
        manifest.get("kind") != "native_prefix_suffix_assessment_manifest_v1"
        or manifest.get("role") != ROLE
        or manifest.get("evidence_kind") not in (EVIDENCE, REAL)
        or manifest.get("device") not in ("cpu", "cuda:0")
    ):
        raise ValueError("Separately scoped adapted suffix manifest/device required.")
    if manifest.get("recipe") != RECIPE:
        raise ValueError("Frozen48 nominal source-hour recipe/seed20260929/2000/floor18 required.")
    if manifest["evidence_kind"] == EVIDENCE:
        folder = (
            Path(__file__).resolve().parents[3] / "evidence/ssl-transfer-integration-builder-v1"
        )
        if (
            manifest["device"] != "cpu"
            or manifest.get("fixture_identity") != EVIDENCE
            or not base.is_relative_to(folder)
            or not base.name.startswith(EVIDENCE + "-")
            or not output.is_relative_to(base)
        ):
            raise ValueError("Synthetic admission confined to private CPU fixture identity.")
    review = prefix._json(review_path.read_bytes())
    bindings = verify_review(
        review,
        manifest,
        "APPROVED_PREFIX_SUFFIX_ASSESSMENT_EXECUTION",
        "prefix_suffix_assessment_execution",
    )
    if review.get("runtime") != {"device": manifest["device"], "output": str(output)} or review.get(
        "allowed_cells"
    ) != manifest.get("cells"):
        raise ValueError("Exact frozen cells/runtime/output admission required.")
    snapshots, documents = {}, {}

    def bound(value):
        path = regular(value if isinstance(value, Path) else prefix._path(value, base))
        if path not in snapshots:
            raw = path.read_bytes()
            if bindings.get(str(path)) != sha(raw):
                raise ValueError("Required exact binding missing/stale: " + str(path))
            if (
                manifest["evidence_kind"] == EVIDENCE
                and path.suffix in (".pt", ".npz")
                and not path.is_relative_to(base)
            ):
                raise ValueError("Synthetic flag cannot admit public data/weights.")
            snapshots[path] = raw
            check_resources(manifest["device"])
        return snapshots[path]

    def doc(value):
        path = prefix._path(value, base)
        if path not in documents:
            documents[path] = prefix._json(bound(path))
        return documents[path]

    bound(manifest_path)
    for path in required_sources():
        if sha(bound(path)) != LOADED_SOURCES[str(path)]:
            raise ValueError("Loaded scientific source bytes changed.")
    for key in (
        "protocol",
        "split",
        "selection_freeze",
        "dependency_lock",
        "executor",
        "cohort",
        "raw_intervals",
    ):
        bound(manifest[key])
    freeze = doc(manifest["selection_freeze"])
    if (
        freeze.get("kind") != "native_prefix_suffix_selection_freeze_v1"
        or freeze.get("cells") != manifest["cells"]
        or freeze.get("selection") != "original_development_only"
        or freeze.get("suffix_selection") is not False
    ):
        raise ValueError("All model/readout/scaler/cell identities frozen before suffix access.")
    frozen_at = datetime.fromisoformat(freeze["frozen_at"])
    if frozen_at.tzinfo is None:
        raise ValueError("Approval audit timestamp must have an explicit timezone.")
    model_bindings = freeze.get("model_bindings")
    if not isinstance(model_bindings, dict) or not model_bindings:
        raise ValueError("Frozen actual model/config/scaler/readout byte identities required.")
    for path, expected in model_bindings.items():
        if sha(bound(path)) != expected:
            raise ValueError("Frozen model/input identity changed after selection.")
    for key, status, scope in (
        ("software_review", "APPROVED_PREFIX_SUFFIX_SOFTWARE", "prefix_suffix_assessment_software"),
        (
            "numeric_access_review",
            "APPROVED_PREFIX_SUFFIX_NUMERIC_ACCESS",
            "prefix_suffix_numeric_assessment",
        ),
    ):
        other = doc(manifest[key])
        other_bindings = verify_review(other, manifest, status, scope)
        if key == "numeric_access_review" and (
            other.get("allowed_uses") != ["frozen_adapted_suffix_assessment"]
            or other.get("selection_freeze_sha256") != sha(bound(manifest["selection_freeze"]))
        ):
            raise ValueError("Separate suffix numeric scope/freeze identity required.")
        if key == "numeric_access_review":
            issued = datetime.fromisoformat(other["issued_at"])
            if issued.tzinfo is None or issued <= frozen_at:
                raise ValueError("Selection freeze must precede suffix numerical admission.")
        if any(
            other_bindings.get(str(p)) != sha(raw)
            for p, raw in snapshots.items()
            if p != prefix._path(manifest[key], base)
        ):
            raise ValueError("Software/access review missing exact closure.")
    metadata = doc(manifest["raw_intervals"])
    if len(metadata.get("sources", {})) != 1:
        raise ValueError("One true deployment per suffix invocation; no pooled adaptation.")
    deployment = next(iter(metadata["sources"]))
    partition = prefix.partitions(metadata, 30)
    cohort = doc(manifest["cohort"])
    if (
        cohort.get("role") != "suffix"
        or cohort.get("complete") is not True
        or cohort.get("rows") != [list(v) for v in sorted(partition["suffix_rows"])]
    ):
        raise ValueError("Exact fixed suffix cohort; no intersection or filtering.")
    cells, fitted, names = manifest.get("cells"), {}, set()
    if not isinstance(cells, list) or not cells:
        raise ValueError("Explicit nonempty frozen cells required.")
    for cell in cells:
        name = cell.get("name")
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for c in name
            )
        ):
            raise ValueError("Unique safe output cell names required.")
        names.add(name)
        if cell.get("deployment") != deployment:
            raise ValueError("Every cell must assess its own declared deployment.")
        kind = cell.get("kind")
        if kind in UNFITTED:
            if cell.get("zero_shot") is not True or cell.get("model") is not None:
                raise ValueError("Unchanged no-fit references are explicit zero-shot controls.")
            continue
        if kind not in (NEURAL_KIND, REFERENCE_KIND) or cell.get("zero_shot") is not False:
            raise ValueError("Enumerated adapted kind; never call adaptation zero-shot.")
        bound(cell["model"])
        for key in ("model", "completion", "fit_manifest"):
            if model_bindings.get(str(prefix._path(cell[key], base))) != sha(bound(cell[key])):
                raise ValueError("Every selected model and actual fitted receipt must be frozen.")
        receipt = doc(cell["completion"])
        if (
            receipt.get("kind") != "native_prefix_matched_transfer_completion_v4"
            or receipt.get("status") != "COMPLETED"
            or receipt.get("zero_shot") is not False
            or receipt.get("suffix_numerical_access") is not False
        ):
            raise ValueError("Actual completed prefix-only ancestor required.")
        fit_manifest = doc(cell["fit_manifest"])
        bound(cell["fit_review"])
        if fit_manifest.get("evidence_kind") != (
            EVIDENCE if manifest["evidence_kind"] == EVIDENCE else "REVIEWED_PREFIX_TRANSFER"
        ):
            raise ValueError("Synthetic completion cannot be promoted to scientific ancestry.")
        cfg = prefix.PrefixConfig(**fit_manifest["config"])
        if receipt.get("config") != cfg.to_dict() or any(
            cell.get(k) != getattr(cfg, k) for k in ("method", "seed", "mode", "prefix_days")
        ):
            raise ValueError("Own prefix length/seed/method/config receipt mismatch.")
        fit_admission = prefix.admit(
            prefix._path(cell["fit_manifest"], base),
            prefix._path(cell["fit_review"], base),
            base / ("admission-only-" + name),
        )
        if fit_admission.identities != receipt.get("bindings"):
            raise ValueError("Fitted exact recursive TRAIN/scaler/source/parent identities differ.")
        fit_base = prefix._path(cell["fit_manifest"], base).parent
        for key in ("scalers", "config_path", "backbone_config"):
            p = prefix._path(fit_manifest[key], fit_base)
            if model_bindings.get(str(p)) != sha(fit_admission.snapshots[p]):
                raise ValueError(
                    "Actual fitted config/backbone/original TRAIN scalers must be frozen."
                )
        for p, raw in fit_admission.snapshots.items():
            if bindings.get(str(p)) != sha(raw):
                raise ValueError("Suffix review must bind every prefix ancestor/source/data byte.")
            snapshots[p] = raw
        own = fit_admission.documents[
            prefix._path(
                fit_manifest["raw_intervals"], prefix._path(cell["fit_manifest"], base).parent
            )
        ]
        own_partition = prefix.partitions(own, cfg.prefix_days)
        if (
            own.get("sources") != metadata["sources"]
            or json.dumps(own_partition, sort_keys=True)
            != json.dumps(receipt.get("partition"), sort_keys=True)
            or own_partition["suffix_support_sha256"] != partition["suffix_support_sha256"]
            or own_partition["suffix_interval_ids"] != partition["suffix_interval_ids"]
        ):
            raise ValueError(
                "Raw fitted/suffix IDs, source configuration and fixed boundaries differ."
            )
        declared = {(r["deployment"], r["row_id"]): r for r in own["rows"]}
        counts = (
            np.asarray(
                [declared[v]["target_observed"] for v in own_partition["fit_rows"]], dtype=bool
            )
            .sum(0)
            .tolist()
        )
        if receipt.get("label_counts") != counts or receipt.get("unique_prefix_rows") != len(
            own_partition["fit_rows"]
        ):
            raise ValueError("Actual prefix label/row counts differ from bound interval support.")
        fitted[name] = fit_admission
    if manifest.get("reference_name") not in names:
        raise ValueError("One prespecified paired reference name required.")
    bound(manifest["input_npz"])
    # Access review must include all recursively discovered ancestors and input.
    for key in ("software_review", "numeric_access_review"):
        other = doc(manifest[key])["bindings"]
        if any(
            other.get(str(p)) != sha(raw)
            for p, raw in snapshots.items()
            if p
            not in (
                prefix._path(manifest["software_review"], base),
                prefix._path(manifest["numeric_access_review"], base),
            )
        ):
            raise ValueError("Final software/access closure incomplete.")
    identities = {str(p): sha(raw) for p, raw in snapshots.items()}
    identities[str(review_path)] = sha(review_path.read_bytes())
    return Admission(manifest, snapshots, documents, partition, fitted, identities)


def _label_masks(arrays, n):
    masks = [(name, arrays[name]) for name in ("target_observed", "y_observed") if name in arrays]
    if "observed" in arrays and arrays["observed"].shape == (n, 3):
        masks.append(("observed", arrays["observed"]))
    if (
        not masks
        or any(v.shape != (n, 3) or v.dtype != bool for _, v in masks)
        or any(not np.array_equal(v, masks[0][1]) for _, v in masks)
    ):
        raise ValueError("Strict Boolean forecast mask aliases; context is never a target mask.")
    return masks[0][1], [name for name, _ in masks]


def load_input(admission):
    m = admission.manifest
    # The manifest path is not executed; input bytes were admitted in full above.
    candidates = [p for p in admission.snapshots if str(p) == str(Path(m["input_npz"]).absolute())]
    if not candidates:
        raise ValueError("Input NPZ path must be explicit absolute identity.")
    with np.load(io.BytesIO(admission.snapshots[candidates[0]]), allow_pickle=False) as archive:
        if str(archive["corpus_role"].item()) != "suffix":
            raise ValueError("Suffix-role corpus only.")
        if m["evidence_kind"] == EVIDENCE and (
            str(archive["evidence_kind"].item()) != EVIDENCE
            or str(archive["fixture_identity"].item()) != EVIDENCE
        ):
            raise ValueError("Synthetic NPZ evidence/identity marker required.")
        arrays = {
            k: archive[k].copy()
            for k in archive.files
            if k
            in (
                "x",
                "observed",
                "context_observed",
                "metadata",
                "query",
                "targets",
                "y",
                "target_observed",
                "y_observed",
                "row_id",
                "deployment",
                "target_dates",
                "cutoff",
            )
        }
    n = len(arrays["x"])
    mask, mask_sources = _label_masks(arrays, n)
    context = arrays.get("context_observed")
    if context is None and arrays.get("observed", np.zeros(0)).shape == (n, 96, 4):
        context = arrays["observed"]
    x, context, metadata = _inputs(arrays["x"], context, arrays["metadata"], 96)
    query = _query(arrays["query"], metadata)
    if not np.allclose(metadata[:, 0, 3:5] * 250, [0, 230], atol=1e-5, rtol=0):
        raise ValueError("Suffix primary native230 must never be relabelled200.")
    targets = arrays.get("targets", arrays.get("y"))
    if (
        targets is None
        or targets.shape != (n, 3)
        or targets.dtype.kind != "f"
        or not np.isfinite(targets[mask]).all()
    ):
        raise ValueError("Finite observed native targets required.")
    if (
        "targets" in arrays
        and "y" in arrays
        and not np.array_equal(arrays["targets"][mask], arrays["y"][mask])
    ):
        raise ValueError("Conflicting observed target aliases.")
    rows = list(zip(arrays["deployment"].tolist(), arrays["row_id"].tolist(), strict=True))
    if len(set(rows)) != n or set(rows) != set(admission.partition["suffix_rows"]):
        raise ValueError("Exact unique common issuance set required.")
    metadata_registry = next(
        v
        for v in admission.documents.values()
        if v.get("kind") in ("native_prefix_raw_intervals_v2", "native_prefix_raw_intervals_v1")
    )
    declared = {(r["deployment"], r["row_id"]): r for r in metadata_registry["rows"]}
    for i, identity in enumerate(rows):
        row = declared[identity]
        dates = [
            metadata_registry["intervals"][v]["timestamp"][:10] if isinstance(v, str) else ""
            for v in row["target_ids"]
        ]
        cutoff_index = metadata_registry["intervals"][row["context_ids"][-1][0]][
            "source_interval_index"
        ]
        if (
            mask[i].tolist() != row["target_observed"]
            or arrays["target_dates"][i].tolist() != dates
            or arrays["cutoff"][i] != cutoff_index
        ):
            raise ValueError("Numeric suffix masks/dates/cutoff differ from exact raw intervals.")
        if "context_observed" in row and not np.array_equal(context[i], row["context_observed"]):
            raise ValueError("Actual observed source context mask differs.")
        catalog = metadata_registry.get("configuration_map")
        if catalog:
            cfg = catalog["sources"][identity[0]]["configurations"][row["configuration"]]
            if not np.allclose(
                metadata[i, :, 3:5] * 250, cfg["channel_bounds_m"], atol=1e-5, rtol=0
            ) or not np.array_equal(
                metadata[i, :, 6], [v != "UNKNOWN" for v in cfg["processing_id_or_unknown"]]
            ):
                raise ValueError("Actual native per-channel configuration differs.")
    order = np.array(sorted(range(n), key=lambda i: rows[i]))
    data = {
        "x": x,
        "context_observed": context,
        "metadata": metadata,
        "query": query,
        "targets": targets,
        "observed": mask,
        "row_id": arrays["row_id"],
        "deployment": arrays["deployment"],
        "target_dates": arrays["target_dates"],
        "cutoff": arrays["cutoff"],
    }
    return {k: v[order] for k, v in data.items()}, mask_sources


def _forecast(cell, artifact, data, device):
    kind = cell["kind"]
    if kind in UNFITTED:
        values = (
            np.repeat(data["x"][:, -1, 0, None], 3, axis=1)
            if kind == "persistence"
            else data["x"][:, -25 + np.asarray([1, 3, 6]), 0]
        )
        return np.repeat(values[..., None], 5, axis=-1)
    if kind == NEURAL_KIND:
        predictor = prefix.PrefixPredictor(io.BytesIO(artifact), device=device)
    else:
        predictor = prefix.ReferencePredictor(prefix._json(artifact))
    # Strict context-only call. Targets/masks/dates/future arrays never enter it.
    return predictor.forecast(data["x"], data["context_observed"], data["metadata"], data["query"])


def execute_assessment(manifest_path, review_path, output_path):
    started = time.perf_counter()
    admitted = admit(manifest_path, review_path, output_path)
    m = admitted.manifest
    data, aliases = load_input(admitted)
    base, output = Path(manifest_path).absolute().parent, Path(output_path).absolute()
    forecasts, results, peaks = {}, {}, check_resources(m["device"])
    for cell in m["cells"]:
        raw = (
            admitted.snapshots[prefix._path(cell["model"], base)]
            if cell["kind"] not in UNFITTED
            else None
        )
        if raw is not None:
            artifact = (
                prefix.decode_checkpoint(raw) if cell["kind"] == NEURAL_KIND else prefix._json(raw)
            )
            fitted = admitted.fitted[cell["name"]]
            if (
                artifact.get("config") != fitted.config.to_dict()
                or artifact.get("identities") != fitted.identities
                or artifact.get("scalers")
                != fitted.documents[
                    prefix._path(
                        fitted.manifest["scalers"], prefix._path(cell["fit_manifest"], base).parent
                    )
                ]
            ):
                raise ValueError("Actual saved inference config/scalers/fit identities differ.")
            completion = admitted.documents[prefix._path(cell["completion"], base)]
            if cell["kind"] == NEURAL_KIND and (
                artifact.get("selected_step") != completion.get("selected_step")
                or artifact.get("feature_ancestor") != completion.get("feature_ancestor")
            ):
                raise ValueError("Saved selected endpoint and exact feature ancestry differ.")
            if cell["kind"] == REFERENCE_KIND and artifact.get(
                "selected_iteration"
            ) != completion.get("selected_iteration"):
                raise ValueError("Saved fixed conventional DEV selection differs.")
        predictions = _forecast(cell, raw, data, m["device"])
        if (
            predictions.shape != (len(data["x"]), 3, 5)
            or not np.isfinite(predictions).all()
            or (np.diff(predictions, axis=-1) < 0).any()
        ):
            raise ValueError("Finite monotonic forecasts on every common row required.")
        forecasts[cell["name"]] = predictions
        metrics = native_scores(
            predictions,
            data["targets"],
            data["observed"],
            data["target_dates"],
            data["deployment"],
            minimum_daily_rows=18,
        )
        fit = admitted.documents.get(prefix._path(cell.get("completion", "unused"), base), {})
        results[cell["name"]] = {
            "metrics": metrics,
            "zero_shot": cell["zero_shot"],
            "label_counts": fit.get("label_counts", [0, 0, 0]),
            "config": fit.get("config"),
            "adaptation_ancestry": fit.get("bindings", {}),
            "constant_diagnostic": comparison._variation(predictions, data["deployment"]),
        }
        current = check_resources(m["device"])
        peaks = {k: max(peaks[k], current[k]) for k in peaks}
    uncertainty, draws = comparison._bootstrap(data, list(results), results)
    reference = m["reference_name"]
    paired = {}
    for i, name in enumerate(results):
        score, base_score = (
            results[name]["metrics"]["primary_pinball_db"],
            results[reference]["metrics"]["primary_pinball_db"],
        )
        paired[name] = {
            "method_minus_reference": None
            if score is None or base_score is None
            else score - base_score,
            "reference": reference,
            "interval": None
            if draws is None
            else np.percentile(
                draws[:, i] - draws[:, list(results).index(reference)], [2.5, 97.5]
            ).tolist(),
        }
    output.mkdir(exist_ok=False)
    hashes = {}
    for name, prediction in forecasts.items():
        path = output / (name + ".npz")
        with path.open("xb") as stream:
            np.savez_compressed(
                stream,
                predictions=prediction,
                targets=data["targets"],
                observed=data["observed"],
                target_dates=data["target_dates"],
                row_id=data["row_id"],
                deployment=data["deployment"],
                cutoff=data["cutoff"],
                query=data["query"],
                query_native_bounds_m=data["query"][..., 3:5] * 250,
            )
        hashes[str(path)] = sha(path.read_bytes())
    report = {
        "kind": "native_prefix_suffix_assessment_completion_v1",
        "status": "COMPLETED"
        if all(v["metrics"]["primary_pinball_db"] is not None for v in results.values())
        else "NOT_ASSESSABLE",
        "evidence_kind": m["evidence_kind"],
        "role": ROLE,
        "zero_shot": False,
        "results": results,
        "paired": paired,
        "uncertainty": uncertainty,
        "partition": admitted.partition,
        "prediction_hashes": hashes,
        "bindings": admitted.identities,
        "mask_field_provenance": aliases,
        "native_geometry": comparison._geometry(data["query"]),
        "resource_peaks": peaks,
        "device": m["device"],
        "elapsed_process_seconds": time.perf_counter() - started,
        "scope": "sampled process memory; root owns full-tree caps/lifetime/12-in96 attempt accounting",
        "source_clock": "source_calendar_not_verified_UTC",
        "scientific_claim": None,
    }
    with (output / "completion.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, allow_nan=False, indent=2)
    return report


run = execute_assessment
