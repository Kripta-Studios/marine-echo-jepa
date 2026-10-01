"""Synthetic-only independent saved-score reconstruction checks."""

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from marine_echo.evaluation import native_comparison as comparison
from marine_echo.evaluation import native_suffix_reconstruction_v1 as reconstruction
from marine_echo.evaluation.native_product import native_scores


def arrays():
    dates = np.repeat(np.array(["2026-01-01", "2026-01-02", "2026-01-03"]), 24)
    n = len(dates)
    query = np.zeros((n, 3, 10))
    query[..., 0] = 38000 / 455000
    query[..., 1:3] = 1
    query[..., 4] = 230 / 250
    query[..., 9] = [1, 3, 6]
    targets = np.arange(n * 3, dtype=float).reshape(n, 3) / 50
    predictions = targets[..., None] + np.array([-2, -1, 0, 1, 2])
    observed = np.ones((n, 3), bool)
    observed[:4, 1] = False
    targets[~observed] = np.nan
    return {
        "predictions": predictions,
        "targets": targets,
        "observed": observed,
        "target_dates": np.tile(dates[:, None], (1, 3)),
        "deployment": np.full(n, "SYNTHETIC_DEPLOYMENT"),
        "row_id": np.array([f"row{i:03}" for i in range(n)]),
        "cutoff": np.arange(n),
        "query": query,
    }


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def case(tmp_path):
    reconstruction.FIXTURE_ROOT.mkdir(exist_ok=True)
    folder = (
        reconstruction.FIXTURE_ROOT
        / Path(
            tempfile.mkdtemp(prefix="SYNTHETIC_CORRECTNESS_ONLY-", dir=reconstruction.FIXTURE_ROOT)
        ).name
    )
    a = arrays()
    cells = []
    results, hashes = {}, {}
    for name, zero_shot in (("adapted", False), ("persistence", True)):
        path = folder / (name + ".npz")
        np.savez_compressed(path, **a)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        results[name] = {
            "metrics": native_scores(
                *[
                    a[k]
                    for k in ("predictions", "targets", "observed", "target_dates", "deployment")
                ]
            ),
            "zero_shot": zero_shot,
        }
        cells.append(
            {
                "name": name,
                "zero_shot": zero_shot,
                "deployment": "SYNTHETIC_DEPLOYMENT",
                "kind": "persistence"
                if zero_shot
                else "native_prefix_matched_transfer_inference_v4",
                "prefix_days": 1,
            }
        )
    assessment = {
        "kind": "native_prefix_suffix_assessment_manifest_v1",
        "role": "adapted_suffix",
        "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
        "implementer_session_id": "fixture-builder",
        "coordinator_session_id": "fixture-coordinator",
        "cells": cells,
        "reference_name": "persistence",
        "recipe": comparison.RECIPE,
        "software_review": str(folder / "software.json"),
        "numeric_access_review": str(folder / "access.json"),
        "selection_freeze": str(folder / "freeze.json"),
    }
    dump(folder / "assessment.json", assessment)
    dump(
        folder / "freeze.json",
        {
            "kind": "native_prefix_suffix_selection_freeze_v1",
            "cells": cells,
            "selection": "original_development_only",
            "suffix_selection": False,
            "frozen_at": "2026-01-01T00:00:00+00:00",
        },
    )
    execution = {
        "status": "APPROVED_PREFIX_SUFFIX_ASSESSMENT_EXECUTION",
        "scope": "prefix_suffix_assessment_execution",
        "allowed_roles": ["adapted_suffix"],
        "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
        "reviewer_session_id": "fixture-reviewer",
        "implementer_session_id": "fixture-builder",
        "coordinator_session_id": "fixture-coordinator",
        "allowed_cells": cells,
        "bindings": {
            str(folder / "assessment.json"): hashlib.sha256(
                (folder / "assessment.json").read_bytes()
            ).hexdigest()
        },
    }
    for filename, status, scope in (
        ("software.json", "APPROVED_PREFIX_SUFFIX_SOFTWARE", "prefix_suffix_assessment_software"),
        (
            "access.json",
            "APPROVED_PREFIX_SUFFIX_NUMERIC_ACCESS",
            "prefix_suffix_numeric_assessment",
        ),
    ):
        original = dict(execution, status=status, scope=scope)
        if filename == "access.json":
            original.update(
                allowed_uses=["frozen_adapted_suffix_assessment"],
                selection_freeze_sha256=hashlib.sha256(
                    (folder / "freeze.json").read_bytes()
                ).hexdigest(),
                issued_at="2026-01-02T00:00:00+00:00",
            )
        dump(folder / filename, original)
        execution["bindings"][str(folder / filename)] = hashlib.sha256(
            (folder / filename).read_bytes()
        ).hexdigest()
    dump(folder / "execution.json", execution)
    completion = {
        "kind": "native_prefix_suffix_assessment_completion_v1",
        "status": "COMPLETED",
        "role": "adapted_suffix",
        "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
        "zero_shot": False,
        "results": results,
        "prediction_hashes": hashes,
        "bindings": {
            str(folder / "assessment.json"): hashlib.sha256(
                (folder / "assessment.json").read_bytes()
            ).hexdigest(),
            str(folder / "execution.json"): hashlib.sha256(
                (folder / "execution.json").read_bytes()
            ).hexdigest(),
        },
        "partition": {
            "fit_rows": [["SYNTHETIC_DEPLOYMENT", "prefix"]],
            "suffix_rows": [
                [str(d), str(r)] for d, r in zip(a["deployment"], a["row_id"], strict=True)
            ],
            "fit_interval_ids": ["prefix-only"],
            "suffix_interval_ids": ["suffix-only"],
        },
    }
    for filename in ("software.json", "access.json", "freeze.json"):
        completion["bindings"][str(folder / filename)] = hashlib.sha256(
            (folder / filename).read_bytes()
        ).hexdigest()
    scores = {k: {"metrics": v["metrics"]} for k, v in results.items()}
    completion["uncertainty"], draws = comparison._bootstrap(a, list(results), scores)
    completion["paired"] = {
        k: {
            "reference": "persistence",
            "method_minus_reference": 0.0,
            "interval": np.percentile(draws[:, i] - draws[:, 1], [2.5, 97.5]).tolist(),
        }
        for i, k in enumerate(results)
    }
    dump(folder / "completion.json", completion)
    manifest = {
        "kind": "native_prefix_suffix_reconstruction_manifest_v1",
        "role": "adapted_suffix",
        "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
        "fixture_identity": "SYNTHETIC_CORRECTNESS_ONLY",
        "implementer_session_id": "fixture-author",
        "coordinator_session_id": "fixture-coordinator",
        "recipe": comparison.RECIPE,
        "reference_name": "persistence",
        "assessment_manifest": str(folder / "assessment.json"),
        "assessment_execution_review": str(folder / "execution.json"),
        "assessment_completion": str(folder / "completion.json"),
    }
    dump(folder / "manifest.json", manifest)
    rebind(folder)
    return folder, a


def rebind(folder):
    paths = list(reconstruction.SOURCE_PATHS) + list(folder.glob("*.json"))
    paths = [p for p in paths if p.name != "review.json"] + list(folder.glob("*.npz"))
    review = {
        "status": "APPROVED_PREFIX_SUFFIX_RECONSTRUCTION",
        "scope": "saved_adapted_suffix_reconstruction",
        "allowed_roles": ["adapted_suffix"],
        "allowed_uses": ["saved_adapted_suffix_reconstruction"],
        "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
        "reviewer_session_id": "fixture-reviewer",
        "implementer_session_id": "fixture-author",
        "coordinator_session_id": "fixture-coordinator",
        "bindings": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    dump(folder / "review.json", review)


def run(folder):
    return reconstruction.reconstruct(
        folder / "manifest.json", folder / "review.json", folder / "result.json"
    )


def test_independent_daily_equal_weighting_and_masked_nan():
    a = arrays()
    # Unequal per-day counts must not silently become pooled-row weighting.
    a["predictions"][-24:] += 2
    a["observed"][-6:, 0] = False
    expected = native_scores(
        *[a[k] for k in ("predictions", "targets", "observed", "target_dates", "deployment")]
    )
    actual = reconstruction.independent_scores(a)
    reconstruction.assert_numerical_equal(actual, expected)
    assert len(actual["daily_rows"]) == 9


def test_reconstruct_real_saved_arrays_in_synthetic_container(tmp_path, monkeypatch):
    folder, a = case(tmp_path)
    monkeypatch.setattr(
        comparison.native_product,
        "native_scores",
        lambda *a, **k: pytest.fail("Reconstruction must not call the original scorer"),
    )
    report = run(folder)
    assert report["status"] == "RECONSTRUCTED"
    assert report["role"] == "adapted_suffix" and report["zero_shot"] is False
    assert report["results"]["persistence"]["zero_shot"] is True
    assert report["paired"]["adapted"]["interval"] == [0.0, 0.0]
    assert report["native_geometry"]["unique_products"][0]["lower_m"] == 230
    assert report["independent_scientific_approval"] is False
    with pytest.raises(FileExistsError):
        run(folder)
    np.testing.assert_array_equal(a["targets"], arrays()["targets"])


@pytest.mark.parametrize("damage", ["self", "status", "role", "binding", "recipe", "evidence"])
def test_reject_before_numeric_decode(tmp_path, monkeypatch, damage):
    folder, _ = case(tmp_path)
    path = folder / ("manifest.json" if damage == "recipe" else "review.json")
    doc = json.loads(path.read_text())
    if damage == "self":
        doc["reviewer_session_id"] = "fixture-author"
    elif damage == "status":
        doc["status"] = "APPROVED_FINAL_ASSESSMENT"
    elif damage == "role":
        doc["allowed_roles"] = ["final_test"]
    elif damage == "binding":
        doc["bindings"][str(folder / "adapted.npz")] = "0" * 64
    elif damage == "evidence":
        doc["evidence_kind"] = "REVIEWED_PREFIX_SUFFIX_ASSESSMENT"
    else:
        doc["recipe"]["floor"] = 1
    dump(path, doc)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Denied before NPZ decode"))
    with pytest.raises(ValueError):
        run(folder)


@pytest.mark.parametrize("damage", ["daily", "paired", "label", "receipt", "cutoff", "rows"])
def test_no_silent_numerical_or_ancestry_correction(tmp_path, damage):
    folder, a = case(tmp_path)
    path = folder / "completion.json"
    doc = json.loads(path.read_text())
    if damage == "daily":
        doc["results"]["adapted"]["metrics"]["daily_rows"][0]["pinball_db"] += 0.01
    elif damage == "paired":
        doc["paired"]["adapted"]["method_minus_reference"] = 0.01
    elif damage == "label":
        doc["results"]["adapted"]["zero_shot"] = True
    elif damage == "receipt":
        doc["bindings"][str(folder / "execution.json")] = "0" * 64
    else:
        a["cutoff"][0] += 10 if damage == "cutoff" else 0
        if damage == "rows":
            a["row_id"][0] = a["row_id"][1]
        np.savez_compressed(folder / "adapted.npz", **a)
        doc["prediction_hashes"][str(folder / "adapted.npz")] = hashlib.sha256(
            (folder / "adapted.npz").read_bytes()
        ).hexdigest()
    dump(path, doc)
    rebind(folder)
    with pytest.raises(ValueError):
        run(folder)


def test_independent_bootstrap_matches_original_sequence_and_draws():
    a = arrays()
    a["predictions"][-24:] += 3
    methods = {
        "a": {"metrics": reconstruction.independent_scores(a)},
        "b": {"metrics": reconstruction.independent_scores(arrays())},
    }
    expected, draws = comparison._bootstrap(a, list(methods), methods)
    actual, independent_draws = reconstruction.independent_bootstrap(a, methods)
    assert actual["sequence_sha256"] == expected["sequence_sha256"]
    assert actual["valid_replicates"] == 2000
    np.testing.assert_allclose(independent_draws, draws, atol=1e-12, rtol=0)


def test_missing_horizon_or_zero_span_does_not_get_interval():
    a = arrays()
    a["observed"][:, 2] = False
    methods = {"a": {"metrics": reconstruction.independent_scores(a)}}
    assert methods["a"]["metrics"]["primary_pinball_db"] is None
    uncertainty, draws = reconstruction.independent_bootstrap(a, methods)
    assert draws is None and uncertainty["interval_status"] == "NOT_ASSESSABLE"
    a = arrays()
    a["target_dates"][:] = "2026-01-01"
    uncertainty, draws = reconstruction.independent_bootstrap(
        a, {"a": {"metrics": reconstruction.independent_scores(a)}}
    )
    assert draws is None and uncertainty["valid_replicates"] == 2000


def test_reordered_saved_rows_are_canonically_aligned(tmp_path):
    folder, a = case(tmp_path)
    np.savez_compressed(folder / "adapted.npz", **{k: v[::-1] for k, v in a.items()})
    doc = json.loads((folder / "completion.json").read_text())
    doc["prediction_hashes"][str(folder / "adapted.npz")] = hashlib.sha256(
        (folder / "adapted.npz").read_bytes()
    ).hexdigest()
    dump(folder / "completion.json", doc)
    rebind(folder)
    assert run(folder)["status"] == "RECONSTRUCTED"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_reconstructed_scalar_is_rejected(value):
    with pytest.raises(ValueError):
        reconstruction.assert_numerical_equal(value, 0.1)


@pytest.mark.parametrize("damage", ["software", "access", "freeze", "partition", "geometry"])
def test_original_access_freeze_partition_and_native_geometry_retained(
    tmp_path, damage, monkeypatch
):
    folder, a = case(tmp_path)
    path = folder / (
        damage + ".json" if damage in ("software", "access", "freeze") else "completion.json"
    )
    doc = json.loads(path.read_text())
    if damage in ("software", "access"):
        doc["status"] = "PENDING"
    elif damage == "freeze":
        doc["suffix_selection"] = True
    elif damage == "partition":
        doc["partition"]["fit_interval_ids"].append("suffix-only")
    else:
        a["query"][..., 4] = 200 / 250
        np.savez_compressed(folder / "adapted.npz", **a)
        doc["prediction_hashes"][str(folder / "adapted.npz")] = hashlib.sha256(
            (folder / "adapted.npz").read_bytes()
        ).hexdigest()
    dump(path, doc)
    rebind(folder)
    if damage != "geometry":
        monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Denied before decode"))
    with pytest.raises(ValueError):
        run(folder)


def test_multiple_deployments_unequal_days_and_nonzero_paired_draws():
    a = arrays()
    a = {k: np.concatenate([v, v], axis=0) for k, v in a.items()}
    a["deployment"][72:] = "SYNTHETIC_OTHER"
    a["observed"][96:120, 0] = False
    a["predictions"][120:] += 4
    b = {k: v.copy() for k, v in a.items()}
    b["predictions"][48:72] += 2
    methods = {
        "a": {"metrics": reconstruction.independent_scores(a)},
        "b": {"metrics": reconstruction.independent_scores(b)},
    }
    reconstruction.assert_numerical_equal(
        methods["a"]["metrics"],
        native_scores(
            *[a[k] for k in ("predictions", "targets", "observed", "target_dates", "deployment")]
        ),
    )
    expected, draws = comparison._bootstrap(a, list(methods), methods)
    actual, independent = reconstruction.independent_bootstrap(a, methods)
    assert actual["sequence_sha256"] == expected["sequence_sha256"]
    np.testing.assert_allclose(independent, draws, atol=1e-12, rtol=0)
    assert np.ptp(independent[:, 0] - independent[:, 1]) > 0
