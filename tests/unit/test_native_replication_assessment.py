"""SYNTHETIC_CORRECTNESS_ONLY: CPU target-free assessment checks."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

WRAPPER_SPEC = importlib.util.spec_from_file_location(
    "native_replication_assessment_wrapper",
    Path(__file__).resolve().parents[2] / "tools/execute_bounded_native_replication_assessment.py",
)
wrapper = importlib.util.module_from_spec(WRAPPER_SPEC)
sys.modules[WRAPPER_SPEC.name] = wrapper
WRAPPER_SPEC.loader.exec_module(wrapper)


def policy_support():
    spec = importlib.util.spec_from_file_location(
        "native_replication_resource_support",
        Path(__file__).resolve().parents[2]
        / "evidence/ssl-replication-assessment-prefix-builder-v4/resource_test_support.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("aggregate,band_hours,remaining", [(95, 2, 3600), (12, 11.5, 1800)])
def test_mixed_band_versions_use_lesser_full_owned_allowance(
    monkeypatch, aggregate, band_hours, remaining
):
    helper = policy_support()
    f = helper.resource_case(sys.modules[__name__], monkeypatch, cuda=True, band=True)
    helper.add_attempt(sys.modules[__name__], f, band_hours * 3600)
    ledger = wrapper.read_json(f.root / wrapper.LEDGER)
    ledger["gpu_hours_spent_owned_scientific_jobs"] = aggregate
    f.fs.files[f.root / wrapper.LEDGER] = _encoded(ledger)
    # V1 and V2 are distinct typed artifacts; the entire mixed attempt is Band.
    helper.add_v1_method(sys.modules[__name__], f)
    plan = f.plan()
    assert plan.charge_band and plan.deadline_seconds == remaining
    assert {s["kind"] for s in plan.admission.manifest["methods"].values()} == {
        "native_band_ssl_weights_only_inference_v1",
        "native_band_replication_ssl_weights_only_inference_v2",
    }


@pytest.mark.parametrize(
    "damage",
    [
        "active",
        "pending",
        "lock",
        "bool_time",
        "negative_time",
        "unknown_band",
        "unaccounted",
        "band_exhausted",
        "aggregate_exhausted",
        "missing_source",
        "self_review",
        "owner_limit",
    ],
)
def test_resource_or_admission_damage_refuses_before_mutation(monkeypatch, damage):
    helper = policy_support()
    f = helper.resource_case(sys.modules[__name__], monkeypatch, cuda=True, band=True)
    if damage == "pending":
        f.fs.files[(f.root / wrapper.LEDGER).with_suffix(".prefix-pending")] = b"pending"
    elif damage == "lock":
        f.fs.files[f.root / wrapper.LOCK] = b"unknown owner"
    elif damage in (
        "bool_time",
        "negative_time",
        "unknown_band",
        "unaccounted",
        "band_exhausted",
        "active",
    ):
        helper.add_attempt(
            sys.modules[__name__],
            f,
            True
            if damage == "bool_time"
            else -1
            if damage == "negative_time"
            else 12 * 3600
            if damage == "band_exhausted"
            else 1,
            status="UNKNOWN"
            if damage == "unknown_band"
            else "RUNNING_CUDA"
            if damage == "active"
            else "FAILED_REAL_BAND_CUDA_ATTEMPT",
            family=damage != "unaccounted",
        )
    elif damage == "aggregate_exhausted":
        ledger = wrapper.read_json(f.root / wrapper.LEDGER)
        ledger["gpu_hours_spent_owned_scientific_jobs"] = 96
        f.fs.files[f.root / wrapper.LEDGER] = _encoded(ledger)
    elif damage == "missing_source":
        f.case.review["bindings"].pop(str(f.root / "tools/native_reference_supervisor.py"))
        f.fs.files[f.case.review_path] = _encoded(f.case.review)
    elif damage == "self_review":
        f.case.review["reviewer_session_id"] = wrapper.IMPLEMENTER_SESSION_ID
        f.fs.files[f.case.review_path] = _encoded(f.case.review)
    else:
        receipt = wrapper.read_json(f.root / wrapper.OWNER)
        receipt["revision_gpu_hours_limit_full_owned"] = 13
        f.fs.files[f.root / wrapper.OWNER] = _encoded(receipt)
        f.case.review["bindings"][str(f.root / wrapper.OWNER)] = wrapper.digest(
            f.root / wrapper.OWNER
        )
        f.fs.files[f.case.review_path] = _encoded(f.case.review)
    before = set(f.fs.files), set(f.fs.dirs)
    with pytest.raises((ValueError, FileExistsError)):
        f.plan()
    assert (set(f.fs.files), set(f.fs.dirs)) == before


@pytest.mark.parametrize(
    "exit_code,report,expected", [(0, True, 0), (17, False, 17), (0, False, 1)]
)
def test_virtual_failure_full_time_is_charged_and_no_false_completion(
    monkeypatch, exit_code, report, expected
):
    helper = policy_support()
    f = helper.resource_case(sys.modules[__name__], monkeypatch, cuda=True, band=True)
    result = helper.virtual_execute(sys.modules[__name__], f, exit_code=exit_code, report=report)
    ledger = wrapper.read_json(f.root / wrapper.LEDGER)
    record = ledger["runs"][-1]
    assert result == expected
    assert record["budget_family"] == "native_band_v1"
    assert record["elapsed_owned_seconds"] >= 120
    assert ledger["gpu_hours_spent_owned_scientific_jobs"] == record["elapsed_owned_seconds"] / 3600
    assert (
        ledger["native_band_gpu_hours_spent_full_owned"] == record["elapsed_owned_seconds"] / 3600
    )
    assert record["completion_verified"] is (expected == 0)
    assert record["fitting"] is False


def test_virtual_parent_exception_keeps_active_charge_and_reconciliation(monkeypatch):
    helper = policy_support()
    f = helper.resource_case(sys.modules[__name__], monkeypatch, cuda=True, band=True)
    with pytest.raises(RuntimeError, match="SYNTHETIC supervisor"):
        helper.virtual_execute(sys.modules[__name__], f, exception=True)
    ledger = wrapper.read_json(f.root / wrapper.LEDGER)
    assert ledger["runs"][-1]["status"] == "RUNNING_CUDA"
    assert ledger["runs"][-1]["requires_reconciliation"]
    assert ledger["gpu_hours_spent_owned_scientific_jobs"] > 0
    with pytest.raises((ValueError, FileExistsError)):
        f.plan()


@pytest.mark.parametrize("lock,removed", [("owned", True), ("unknown", False)])
def test_virtual_cleanup_is_only_exact_dead_owned_tree(monkeypatch, lock, removed):
    helper = policy_support()
    f = helper.resource_case(sys.modules[__name__], monkeypatch, cuda=True, band=True)
    assert helper.virtual_execute(sys.modules[__name__], f, report=False, lock=lock) != 0
    assert (f.root / wrapper.LOCK).exists() is (not removed)


def test_virtual_cpu_identity_is_not_cuda_or_fitting(monkeypatch):
    helper = policy_support()
    f = helper.resource_case(sys.modules[__name__], monkeypatch)
    assert helper.virtual_execute(sys.modules[__name__], f) == 0
    ledger = wrapper.read_json(f.root / wrapper.LEDGER)
    record = ledger["runs"][-1]
    assert record["device"] == "cpu" and record["fitting"] is False
    assert record["status"] == "COMPLETED_CPU_ASSESSMENT"
    assert record["budget_family"] is None
    assert ledger["gpu_hours_spent_owned_scientific_jobs"] == 0
    assert ledger["cpu_assessment_hours_owned"] > 0


@pytest.mark.parametrize("changed", [False, True])
def test_worker_resource_exit_preserves_unknown_lock(monkeypatch, changed):
    import os

    helper = policy_support()
    f = helper.resource_case(sys.modules[__name__], monkeypatch)
    sys.path.insert(0, str(BUILDER / "tools"))
    spec = importlib.util.spec_from_file_location(
        "replication_worker_test_source",
        BUILDER / "tools/execute_native_replication_assessment_worker.py",
    )
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    lock = f.root / wrapper.LOCK
    exits = []

    class Resource:
        def __init__(self):
            self.lock, self.output = lock, f.case.output

        def __enter__(self):
            f.fs.files[lock] = _encoded({"pid": os.getpid(), "output": str(self.output)})
            return self

        def __exit__(self, *args):
            exits.append(args)
            lock.unlink()

    if changed:
        with pytest.raises(RuntimeError, match="changed"), worker.owned_resources(Resource()):
            f.fs.files[lock] = _encoded({"pid": 999, "output": "unknown"})
        assert not exits and lock.exists()
    else:
        with worker.owned_resources(Resource()):
            assert lock.exists()
        assert len(exits) == 1 and not lock.exists()


BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
sys.path.insert(0, str(MAIN / "src"))
SPEC = importlib.util.spec_from_file_location(
    "marine_echo.evaluation.native_assessment_replication",
    BUILDER / "src/marine_echo/evaluation/native_assessment_replication.py",
)
assessment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = assessment
SPEC.loader.exec_module(assessment)

RECIPE = {
    "kind": "secondary_observation_dropout_v1",
    "probability": 0.25,
    "seed": 20260929,
    "channels": [1, 2, 3],
}


def test_secondary_dropout_removes_values_and_masks_together():
    x = np.arange(24 * 96 * 4, dtype=np.float32).reshape(24, 96, 4)
    mask = np.ones_like(x, dtype=bool)
    mask[::2, ::3, 2] = False
    original_x, original_mask = x.copy(), mask.copy()
    values, remaining, removed = assessment.secondary_dropout(x, mask, RECIPE)
    assert removed.any(), "The declared robustness case must actually drop observations."
    expected = np.zeros_like(mask)
    draws = np.random.default_rng(20260929).random(mask[:, :, 1:].shape)
    expected[:, :, 1:] = (draws < 0.25) & mask[:, :, 1:]
    np.testing.assert_array_equal(removed, expected)
    np.testing.assert_array_equal(remaining, mask & ~expected)
    assert (values[~remaining] == 0).all()
    np.testing.assert_array_equal(values[:, :, 0], x[:, :, 0])
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(mask, original_mask)


def _encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def npz_bytes(data):
    stream = io.BytesIO()
    np.savez_compressed(stream, **data)
    return stream.getvalue()


def synthetic_arrays(n=24, schema="assessment_context_v1"):
    x = np.linspace(-70, -30, n * 96 * 4, dtype=np.float32).reshape(n, 96, 4)
    mask = np.ones_like(x, dtype=bool)
    mask[::2, ::3, 2] = False
    x[~mask] = np.nan
    meta = np.zeros((n, 4, 10), np.float32)
    meta[..., 0] = np.asarray([38000, 125000, 200000, 455000]) / 455000
    meta[..., 1:3] = 1
    meta[..., 4] = 230 / 250
    query = np.repeat(meta[:, :1], 3, axis=1)
    query[..., 9] = [1, 3, 6]
    target = np.full((n, 3), -50, np.float32)
    target_mask = np.ones_like(target, dtype=bool)
    target_mask[0, 1] = False
    target[0, 1] = np.nan
    data = {
        "x": x,
        "metadata": meta,
        "query": query,
        "row_id": np.asarray([f"synthetic-row-{i}" for i in range(n)]),
        "deployment": np.asarray(["synthetic-test"] * n),
        "target_dates": np.asarray([["2026-01-01"] * 3] * n),
        "corpus_role": np.asarray("final_test"),
        "evidence_kind": np.asarray("SYNTHETIC_CORRECTNESS_ONLY"),
        "fixture_identity": np.asarray("SYNTHETIC_CORRECTNESS_ONLY"),
        "future": np.asarray([{"forbidden": "future"}], dtype=object),
    }
    if schema == "native_corpus_v1":
        data.update(observed=mask, y=target, y_observed=target_mask)
    else:
        data.update(context_observed=mask, targets=target, target_observed=target_mask)
    return data


def spy_factory(spec, snapshots, config, statistics, device, base):
    assert device == "cpu"
    assert set(snapshots) == {Path(p) for p in spec["model_paths"]}
    assert all("corpus" not in str(p) for p in snapshots)

    def forecast(x, observed, metadata, query):
        assert x.ndim == 3 and observed.dtype == np.bool_
        assert (x[~observed] == 0).all()
        assert observed[:, :, 0].all()
        assert not x.flags.writeable and not observed.flags.writeable
        np.testing.assert_allclose(
            query[..., 3:5] * 250, np.broadcast_to([0, 230], (len(x), 3, 2)), atol=1e-5
        )
        return np.repeat(np.repeat(x[:, -1, 0, None], 3, axis=1)[..., None], 5, axis=-1)

    return forecast


TEST_SOURCE_PATHS = assessment.required_sources({"spy": spy_factory})


class MemoryFS:
    """Explicit SYNTHETIC_CORRECTNESS_ONLY transport; no denied disk retry."""

    def __init__(self, monkeypatch):
        self.base = (
            BUILDER
            / "evidence/ssl-replication-assessment-prefix-builder-v4/SYNTHETIC_CORRECTNESS_ONLY_virtual"
        )
        self.files, self.dirs = {}, {self.base}
        originals = {
            name: getattr(Path, name)
            for name in ("read_bytes", "exists", "is_file", "open", "mkdir")
        }

        def virtual(p):
            return p.is_relative_to(self.base)

        def read(p):
            if virtual(p):
                if p not in self.files:
                    raise FileNotFoundError(p)
                return self.files[p]
            return originals["read_bytes"](p)

        def exists(p):
            return p in self.files or p in self.dirs if virtual(p) else originals["exists"](p)

        def is_file(p):
            return p in self.files if virtual(p) else originals["is_file"](p)

        def mkdir(p, parents=False, exist_ok=False):
            if not virtual(p):
                return originals["mkdir"](p, parents=parents, exist_ok=exist_ok)
            if exists(p) and not exist_ok:
                raise FileExistsError(p)
            if not parents and p.parent not in self.dirs:
                raise FileNotFoundError(p.parent)
            self.dirs.add(p)

        def opened(p, mode="r", **kwargs):
            if not virtual(p):
                return originals["open"](p, mode, **kwargs)
            if mode not in ("x", "xb"):
                raise AssertionError("Only exclusive output transport is permitted.")
            if exists(p):
                raise FileExistsError(p)
            if p.parent not in self.dirs:
                raise FileNotFoundError(p.parent)
            owner = self

            class Writer(io.BytesIO):
                def close(self):
                    if not self.closed:
                        owner.files[p] = self.getvalue()
                    super().close()

            raw = Writer()
            return (
                raw
                if "b" in mode
                else io.TextIOWrapper(raw, encoding=kwargs.get("encoding", "utf-8"))
            )

        for name, value in (
            ("read_bytes", read),
            ("exists", exists),
            ("is_file", is_file),
            ("open", opened),
            ("mkdir", mkdir),
        ):
            monkeypatch.setattr(Path, name, value)


def identity(label):
    return {
        "archive_id": f"{label}-archive",
        "deployment_id": f"synthetic-{label}",
        "site_id": f"{label}-site",
        "source_ids": [f"{label}-source"],
    }


class Case:
    def __init__(self, fs, *, learned=False):
        self.fs, self.base = fs, fs.base
        self.manifest_path, self.review_path, self.output = (
            self.base / p for p in ("manifest.json", "review.json", "out")
        )
        self.arrays = synthetic_arrays()
        self.documents = {}

        def path(name):
            return str(self.base / name)

        self.config = {
            "method": "shared_ssl" if learned else "persistence",
            "seed": 7,
            "history": 96,
            "batch_size": 8,
            "feature_schema": "native_references.feature_matrix_v1",
        }
        self.statistics = {"fit_role": "train"}
        self.spec = {
            "method": self.config["method"],
            "seed": 7,
            "mode": "core_frozen_readout" if learned else "reference",
            "kind": "native_ssl_weights_only_inference_v1" if learned else "persistence",
            "loader": "spy",
            "model_paths": [path("model.bin")],
            **{
                k + "_path": path(k + ".json")
                for k in ("config", "statistics", "selection", "ancestry")
            },
        }
        self.manifest = {
            "role": "final_test",
            "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
            "fixture_identity": "SYNTHETIC_CORRECTNESS_ONLY",
            "implementer_session_id": assessment.IMPLEMENTER_SESSION_ID,
            "root_coordinator_session_id": "synthetic-root",
            "floor": 18,
            "device": "cpu",
            "batch_size": 8,
            "resource_limits": dict(assessment.LIMITS),
            "robustness": None,
            "input_schema": "assessment_context_v1",
            "input_corpus_role": "final_test",
            "input_evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
            "historical_metadata_exposure_disclosure": "Historical compatibility metadata exposure; not a universal sealed holdout.",
            "methods": {"fixed": self.spec},
            **{
                k + "_path": path(name)
                for k, name in {
                    "input": "corpus.npz",
                    "cohort": "cohort.json",
                    "split": "split.json",
                    "protocol": "protocol.md",
                    "source": "source.json",
                    "selection": "selection.json",
                    "dependency_lock": "dependencies.lock",
                    "numeric_access_review": "numeric-review.json",
                }.items()
            },
        }
        self.documents["split.json"] = {
            "reserved_test": [identity("test")],
            "train": [identity("train")],
        }
        self.documents["cohort.json"] = {
            "role": "final_test",
            "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
            "identities": [identity("test")],
            "rows": [["synthetic-test", r] for r in self.arrays["row_id"]],
        }
        self.documents["config.json"] = self.config
        self.documents["statistics.json"] = self.statistics
        self.documents["selection.json"] = {
            "selection_role": "development",
            "frozen_before_numeric_access": True,
        }
        self.documents["source.json"] = {
            "bindings": {
                str(Path(assessment.native_product.__file__).resolve()): assessment.sha(
                    Path(assessment.native_product.__file__).read_bytes()
                )
            }
        }
        self.documents["ancestry.json"] = {
            "kind": "native_assessment_ancestry_v1",
            "locally_fitted": learned,
            "historical_initial_weights": False,
            "completeness": "COMPLETE_LOCAL_ANCESTRY",
            "artifacts": [{"path": path("model.bin"), "sha256": ""}],
            "parents": [],
            "fit_inputs": [],
        }
        if learned:
            self.documents["train-cohort.json"] = {
                "role": "train",
                "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
                "identities": [identity("train")],
            }
            self.documents["fit-manifest.json"] = {"role": "train"}
            self.documents["ancestry.json"]["fit_inputs"] = [
                {
                    k + "_path": path(name)
                    for k, name in {
                        "input": "train-input.npz",
                        "cohort": "train-cohort.json",
                        "manifest": "fit-manifest.json",
                        "statistics": "statistics.json",
                        "config": "config.json",
                        "split": "split.json",
                    }.items()
                }
            ]
            fs.files[self.base / "train-input.npz"] = b"TRAIN bytes hashed, NEVER decoded"
        fs.files[self.base / "model.bin"] = b"explicitly reviewed SYNTHETIC fake loader artifact"
        fs.files[self.base / "protocol.md"] = b"Fixed source-calendar point assessment protocol"
        fs.files[self.base / "dependencies.lock"] = b"Frozen installed dependencies; not executed"
        self.review_overrides, self.numeric_overrides = {}, {}
        self.seal()

    def seal(self):
        fs, b = self.fs, self.base
        if self.manifest["input_schema"] == "native_corpus_v1":
            self.arrays.setdefault(
                "split_sha256", np.asarray(assessment.sha(_encoded(self.documents["split.json"])))
            )
        fs.files[b / "corpus.npz"] = npz_bytes(self.arrays)
        for name, value in self.documents.items():
            fs.files[b / name] = _encoded(value)
        digest = lambda name: assessment.sha(fs.files[b / name])
        c = self.documents["cohort.json"]
        c.update(input_npz_sha256=digest("corpus.npz"), split_sha256=digest("split.json"))
        fs.files[b / "cohort.json"] = _encoded(c)
        selected = self.documents["selection.json"]
        selected.update(
            model_sha256={str(b / "model.bin"): digest("model.bin")},
            config_sha256=digest("config.json"),
            statistics_sha256=digest("statistics.json"),
        )
        fs.files[b / "selection.json"] = _encoded(selected)
        node = self.documents["ancestry.json"]
        node["artifacts"][0]["sha256"] = digest("model.bin")
        node.update(
            config_sha256=digest("config.json"), statistics_sha256=digest("statistics.json")
        )
        if "train-cohort.json" in self.documents:
            train = self.documents["train-cohort.json"]
            train.update(
                input_npz_sha256=digest("train-input.npz"), split_sha256=digest("split.json")
            )
            fs.files[b / "train-cohort.json"] = _encoded(train)
            fit = self.documents["fit-manifest.json"]
            fit.update(
                {
                    k + "_sha256": digest(name)
                    for k, name in {
                        "input": "train-input.npz",
                        "cohort": "train-cohort.json",
                        "statistics": "statistics.json",
                        "config": "config.json",
                        "split": "split.json",
                    }.items()
                }
            )
            fs.files[b / "fit-manifest.json"] = _encoded(fit)
        fs.files[b / "ancestry.json"] = _encoded(node)
        identity_fields = {
            k: self.manifest[k]
            for k in ("implementer_session_id", "root_coordinator_session_id", "evidence_kind")
        }
        numeric = {
            **identity_fields,
            "reviewer_session_id": "synthetic-numeric-reviewer",
            "status": "APPROVED_NUMERIC_CORPUS_PROVENANCE",
            "scope": "native_assessment_numeric_decode",
            "input_corpus_role": self.manifest["input_corpus_role"],
            "input_evidence_kind": self.manifest["input_evidence_kind"],
            "allowed_roles": [self.manifest["role"]],
            "bindings": {
                str(b / name): digest(name)
                for name in (
                    "corpus.npz",
                    "cohort.json",
                    "split.json",
                    "protocol.md",
                    "source.json",
                    "selection.json",
                )
            },
            "frozen_model_sha256": {str(b / "model.bin"): digest("model.bin")},
            "frozen_selection_sha256": digest("selection.json"),
            **self.numeric_overrides,
        }
        fs.files[b / "numeric-review.json"] = _encoded(numeric)
        fs.files[self.manifest_path] = _encoded(self.manifest)
        bindings = {
            str(p): assessment.sha(raw)
            for p, raw in fs.files.items()
            if p != self.review_path and p.parent == b
        }
        bindings.update({str(p): assessment.sha(p.read_bytes()) for p in TEST_SOURCE_PATHS})
        self.review = {
            **identity_fields,
            "reviewer_session_id": "synthetic-distinct-reviewer",
            "status": assessment.STATUSES[self.manifest["role"]],
            "scope": "model_only_frozen_assessment",
            "allowed_roles": [self.manifest["role"]],
            "bindings": bindings,
            "device": "cpu",
            "batch_size": 8,
            "robustness": self.manifest["robustness"],
            "output_path": str(self.output),
            "methods": copy.deepcopy(self.manifest["methods"]),
            "allowed_loader_ids": ["spy"],
            **self.review_overrides,
        }
        fs.files[self.review_path] = _encoded(self.review)

    def admit(self):
        return assessment.admit(
            self.manifest_path, self.review_path, self.output, loaders={"spy": spy_factory}
        )

    def run(self):
        return assessment.execute_assessment(
            self.manifest_path, self.review_path, self.output, loaders={"spy": spy_factory}
        )


@pytest.fixture
def memory_case(monkeypatch):
    return Case(MemoryFS(monkeypatch))


@pytest.mark.parametrize(
    "change",
    [
        {"status": "APPROVED_DEVELOPMENT_ASSESSMENT_EXECUTION"},
        {"reviewer_session_id": assessment.IMPLEMENTER_SESSION_ID},
        {"reviewer_session_id": "synthetic-root"},
        {"allowed_roles": ["development"]},
        {"scope": "model_selection"},
        {"evidence_kind": "REVIEWED_FROZEN_ASSESSMENT"},
    ],
)
def test_wrong_review_denied_before_decode(memory_case, monkeypatch, change):
    case = memory_case
    case.review_overrides.update(change)
    case.seal()
    calls = []
    monkeypatch.setattr(assessment.np, "load", lambda *a, **k: calls.append("decoded"))
    with pytest.raises(ValueError):
        case.run()
    assert calls == []


@pytest.mark.parametrize(
    "name",
    [
        "model.bin",
        "cohort.json",
        "split.json",
        "config.json",
        "statistics.json",
        "ancestry.json",
        "selection.json",
        "protocol.md",
        "dependencies.lock",
        "corpus.npz",
        "numeric-review.json",
        "manifest.json",
    ],
)
def test_tampered_bytes_fail_before_decode(memory_case, monkeypatch, name):
    case = memory_case
    case.fs.files[case.base / name] += b"tampered"
    monkeypatch.setattr(
        assessment.np, "load", lambda *a, **k: pytest.fail("Numeric decode before immutable gate")
    )
    with pytest.raises((ValueError, json.JSONDecodeError)):
        case.run()


def test_missing_imported_scorer_binding_denied_before_decode(memory_case, monkeypatch):
    case = memory_case
    case.review["bindings"].pop(str(Path(assessment.native_product.__file__).resolve()))
    case.fs.files[case.review_path] = _encoded(case.review)
    monkeypatch.setattr(assessment.np, "load", lambda *a, **k: pytest.fail("Numeric decode"))
    with pytest.raises(ValueError, match="binding"):
        case.run()


@pytest.mark.parametrize("key", ["archive_id", "deployment_id", "site_id", "source_ids"])
def test_recursive_fitted_test_overlap_denied_before_decode(memory_case, monkeypatch, key):
    case = Case(memory_case.fs, learned=True)
    train = case.documents["train-cohort.json"]["identities"][0]
    train[key] = identity("test")[key]
    case.seal()
    monkeypatch.setattr(assessment.np, "load", lambda *a, **k: pytest.fail("Numeric decode"))
    with pytest.raises(ValueError, match="ancestry"):
        case.run()


@pytest.mark.parametrize(
    "change",
    [
        {"completeness": "UNKNOWN"},
        {"historical_initial_weights": True},
        {"locally_fitted": True},
        {"parents": ["ancestry.json"]},
    ],
)
def test_unknown_historical_or_cyclic_ancestry_fails(memory_case, monkeypatch, change):
    case = memory_case
    case.documents["ancestry.json"].update(change)
    case.seal()
    monkeypatch.setattr(assessment.np, "load", lambda *a, **k: pytest.fail("Numeric decode"))
    with pytest.raises(ValueError):
        case.run()


def test_metadata_exposure_is_disclosed_without_fitted_exception(memory_case):
    admitted = memory_case.admit()
    assert admitted.provenance["no_universal_sealed_claim"] is True
    assert "Historical" in admitted.provenance["historical_metadata_exposure_disclosure"]
    assert admitted.provenance["external_ancestry_guarantee"] is False


@pytest.mark.parametrize(
    "kind",
    [
        "native_ssl_selected_encoder_v1",
        "native_downstream_resume_v1",
        "chronos",
        "native_band_downstream_supervised_encoder_v1",
    ],
)
def test_unsupported_kind_before_decode(memory_case, monkeypatch, kind):
    case = memory_case
    case.spec["kind"] = kind
    case.seal()
    monkeypatch.setattr(assessment.np, "load", lambda *a, **k: pytest.fail("Numeric decode"))
    with pytest.raises(ValueError, match="kind"):
        case.run()


@pytest.mark.parametrize("field", ["observed", "target_observed"])
def test_target_mask_aliases_are_equivalent(memory_case, field):
    case = memory_case
    mask = case.arrays.pop("target_observed")
    case.arrays[field] = mask
    case.seal()
    data, provenance = assessment.load_input(case.admit())
    np.testing.assert_array_equal(data["observed"], mask)
    assert provenance["original_fields"] == [field]


@pytest.mark.parametrize("malformed", ["conflict", "shape", "dtype", "missing"])
def test_target_masks_fail_without_context_substitution(memory_case, malformed):
    case = memory_case
    mask = case.arrays["target_observed"]
    if malformed == "missing":
        del case.arrays["target_observed"]
    else:
        case.arrays["observed"] = mask.copy()
        if malformed == "conflict":
            case.arrays["observed"][1, 0] = False
        elif malformed == "shape":
            case.arrays["observed"] = case.arrays["context_observed"]
        else:
            case.arrays["observed"] = mask.astype(int)
    case.seal()
    with pytest.raises(ValueError, match="mask"):
        assessment.load_input(case.admit())


def test_equal_dual_masks_keep_source_trace(memory_case):
    case = memory_case
    case.arrays["observed"] = case.arrays["target_observed"].copy()
    case.seal()
    _, trace = assessment.load_input(case.admit())
    assert trace["equal_dual_masks"] is True
    assert set(trace["original_fields"]) == {"observed", "target_observed"}


def test_native_corpus_schema_keeps_context_and_target_masks_distinct(memory_case):
    case = memory_case
    case.arrays = synthetic_arrays(schema="native_corpus_v1")
    case.manifest["input_schema"] = "native_corpus_v1"
    case.seal()
    data, trace = assessment.load_input(case.admit())
    assert data["context_observed"].shape == (24, 96, 4)
    assert data["observed"].shape == (24, 3)
    assert trace["original_fields"] == ["y_observed"]


@pytest.mark.parametrize(
    "mutation",
    [
        "primary",
        "frequency",
        "geometry",
        "interval",
        "observed_nan",
        "metadata_nan",
        "query",
        "future_suffix",
        "context_dtype",
        "target_nan",
        "duplicate",
        "row_set",
    ],
)
def test_bad_numeric_support_rejected(memory_case, mutation):
    case = memory_case
    a = case.arrays
    if mutation == "primary":
        a["context_observed"][0, 0, 0] = False
    elif mutation == "frequency":
        a["metadata"][..., 0] = 0
    elif mutation == "geometry":
        a["metadata"][..., 4] = 200 / 250
    elif mutation == "interval":
        a["metadata"][..., 1] = 2
    elif mutation == "observed_nan":
        a["x"][0, 0, 0] = np.nan
    elif mutation == "metadata_nan":
        a["metadata"][0, 0, 0] = np.nan
    elif mutation == "query":
        a["query"][..., 9] = 0
    elif mutation == "future_suffix":
        a["x"] = np.concatenate([a["x"], a["x"][:, :1]], axis=1)
    elif mutation == "context_dtype":
        a["context_observed"] = a["context_observed"].astype(int)
    elif mutation == "target_nan":
        a["targets"][1, 1] = np.nan
    elif mutation == "duplicate":
        a["row_id"][1] = a["row_id"][0]
    else:
        a["row_id"][0] = "unknown"
    case.seal()
    with pytest.raises(ValueError):
        assessment.load_input(case.admit())


def test_wrong_dropout_recipe_denied_before_decode(memory_case, monkeypatch):
    case = memory_case
    case.manifest["robustness"] = {**RECIPE, "probability": 0.50}
    case.seal()
    monkeypatch.setattr(assessment.np, "load", lambda *a, **k: pytest.fail("Numeric decode"))
    with pytest.raises(ValueError, match="recipe"):
        case.run()


def test_final_based_selection_is_denied_before_decode(memory_case, monkeypatch):
    case = memory_case
    case.documents["selection.json"]["selection_role"] = "final_test"
    case.seal()
    monkeypatch.setattr(assessment.np, "load", lambda *a, **k: pytest.fail("Numeric decode"))
    with pytest.raises(ValueError, match="selection"):
        case.admit()


@pytest.mark.parametrize(
    "change",
    [
        {"status": "APPROVED_DEVELOPMENT_ASSESSMENT_EXECUTION"},
        {"frozen_model_sha256": {}},
        {"frozen_selection_sha256": "0" * 64},
        {"reviewer_session_id": assessment.IMPLEMENTER_SESSION_ID},
    ],
)
def test_separate_numeric_access_review_cannot_be_bypassed(memory_case, monkeypatch, change):
    case = memory_case
    case.numeric_overrides.update(change)
    case.seal()
    monkeypatch.setattr(assessment.np, "load", lambda *a, **k: pytest.fail("Numeric decode"))
    with pytest.raises(ValueError):
        case.run()


def test_genuinely_clean_learned_ancestry_does_not_decode_train(memory_case, monkeypatch):
    case = Case(memory_case.fs, learned=True)
    calls, original = [], assessment.np.load

    def loaded(stream, **kwargs):
        calls.append(stream.getvalue())
        return original(stream, **kwargs)

    monkeypatch.setattr(assessment.np, "load", loaded)
    result = case.run()
    assert len(calls) == 1
    assert calls[0] == case.fs.files[case.base / "corpus.npz"]
    assert result["methods"]["fixed"]["feature_training_kind"] == "ssl_pretrained"


def test_grandparent_test_source_overlap_is_not_hidden_by_clean_child(memory_case, monkeypatch):
    case = Case(memory_case.fs, learned=True)
    parent = copy.deepcopy(case.documents["ancestry.json"])
    parent["parents"] = []
    parent["fit_inputs"][0]["cohort_path"] = str(case.base / "parent-cohort.json")
    parent["fit_inputs"][0]["manifest_path"] = str(case.base / "parent-manifest.json")
    cohort = copy.deepcopy(case.documents["train-cohort.json"])
    cohort["identities"][0]["site_id"] = identity("test")["site_id"]
    case.documents["ancestry.json"]["parents"] = [str(case.base / "grandparent.json")]
    case.seal()
    # Reuse the exact TRAIN file identities while declaring the parent's true site.
    cohort.update(case.documents["train-cohort.json"])
    cohort["identities"] = [{**identity("train"), "site_id": identity("test")["site_id"]}]
    case.fs.files[case.base / "parent-cohort.json"] = _encoded(cohort)
    fit = copy.deepcopy(case.documents["fit-manifest.json"])
    fit["cohort_sha256"] = assessment.sha(_encoded(cohort))
    case.fs.files[case.base / "parent-manifest.json"] = _encoded(fit)
    parent.update(
        config_sha256=assessment.sha(_encoded(case.config)),
        statistics_sha256=assessment.sha(_encoded(case.statistics)),
    )
    parent["artifacts"][0]["sha256"] = assessment.sha(case.fs.files[case.base / "model.bin"])
    case.fs.files[case.base / "grandparent.json"] = _encoded(parent)
    for name in ("parent-cohort.json", "parent-manifest.json", "grandparent.json"):
        case.review["bindings"][str(case.base / name)] = assessment.sha(
            case.fs.files[case.base / name]
        )
    case.fs.files[case.review_path] = _encoded(case.review)
    monkeypatch.setattr(assessment.np, "load", lambda *a, **k: pytest.fail("Numeric decode"))
    with pytest.raises(ValueError, match="ancestry"):
        case.run()


def test_actual_secondary_native_bounds_retained(memory_case):
    case = memory_case
    case.arrays["metadata"][:, 1, 3:5] = [10 / 250, 120 / 250]
    case.seal()
    data, _ = assessment.load_input(case.admit())
    np.testing.assert_array_equal(data["metadata"], case.arrays["metadata"])


def test_source_header_test_role_only_under_exact_final_execution_review(memory_case):
    case = memory_case
    case.manifest["input_corpus_role"] = "test"
    case.arrays["corpus_role"] = np.asarray("test")
    case.seal()
    _, trace = assessment.load_input(case.admit())
    assert trace["original_corpus_role"] == "test"


def test_source_header_absence_is_explicit_and_fixture_stays_synthetic(memory_case):
    case = memory_case
    case.manifest["input_evidence_kind"] = None
    del case.arrays["evidence_kind"]
    case.seal()
    _, trace = assessment.load_input(case.admit())
    assert trace["original_evidence_header"] is None


def test_development_cannot_admit_test_source_even_with_updated_review(memory_case, monkeypatch):
    case = memory_case
    case.manifest["role"] = "development"
    case.manifest["input_corpus_role"] = "test"
    case.documents["cohort.json"]["role"] = "development"
    case.seal()
    monkeypatch.setattr(
        assessment.np, "load", lambda *a, **k: pytest.fail("Decoded test under development")
    )
    with pytest.raises(ValueError, match="test access"):
        case.run()


def test_synthetic_label_cannot_admit_real_source_header(memory_case, monkeypatch):
    case = memory_case
    case.manifest["input_evidence_kind"] = "REAL_PUBLIC_CORPUS"
    case.seal()
    monkeypatch.setattr(
        assessment.np,
        "load",
        lambda *a, **k: pytest.fail("Decoded real values under synthetic label"),
    )
    with pytest.raises(ValueError, match="Synthetic"):
        case.run()


def test_unmarked_source_cannot_use_synthetic_interface(memory_case):
    case = memory_case
    del case.arrays["fixture_identity"]
    case.seal()
    with pytest.raises(ValueError, match="provenance"):
        assessment.load_input(case.admit())


def test_native_corpus_split_header_must_match_exact_frozen_split(memory_case):
    case = memory_case
    case.arrays = synthetic_arrays(schema="native_corpus_v1")
    case.arrays["split_sha256"] = np.asarray("0" * 64)
    case.manifest["input_schema"] = "native_corpus_v1"
    case.seal()
    with pytest.raises(ValueError, match="role/evidence"):
        assessment.load_input(case.admit())


def native_source_split_case(case):
    """The original reader schema, with synthetic source identities and no payloads."""
    test_identity = {
        "archive_id": "1",
        "deployment_id": "synthetic-test",
        "site_id": "test-site",
        "source_ids": ["f" * 64],
    }
    case.documents["cohort.json"]["identities"] = [test_identity]
    case.documents["split.json"] = {
        "schema_version": "native_acoustic_ssl_v1",
        "sources": [
            {
                "file_id": 1,
                "deployment": "synthetic-test",
                "site": "test-site",
                "archive_sha256": "f" * 64,
                "role": "final_test",
            },
            {
                "file_id": 2,
                "deployment": "synthetic-train",
                "site": "train-site",
                "archive_sha256": "a" * 64,
                "role": "train",
            },
        ],
    }
    case.arrays = synthetic_arrays(schema="native_corpus_v1")
    case.arrays["split_sha256"] = np.asarray(assessment.sha(_encoded(case.documents["split.json"])))
    case.manifest["input_schema"] = "native_corpus_v1"


def test_original_native_source_split_is_admitted_without_rewriting_its_digest(memory_case):
    case = memory_case
    native_source_split_case(case)
    case.seal()
    admitted = case.admit()
    data, _ = assessment.load_input(admitted)
    np.testing.assert_array_equal(data["targets"], case.arrays["y"])
    assert (
        admitted.provenance["bindings"][str(case.base / "split.json")]
        == case.arrays["split_sha256"].item()
    )


@pytest.mark.parametrize(
    "case_kind", ["mixed", "duplicate", "bad_role", "bad_digest", "unknown_schema", "overlap_site"]
)
def test_original_native_split_rejects_conflicting_membership_before_decode(
    memory_case, monkeypatch, case_kind
):
    case = memory_case
    native_source_split_case(case)
    split = case.documents["split.json"]
    if case_kind == "mixed":
        split["reserved_test"] = [identity("test")]
    elif case_kind == "duplicate":
        split["sources"].append(copy.deepcopy(split["sources"][0]))
    elif case_kind == "bad_role":
        split["sources"][0]["role"] = "legacy_test"
    elif case_kind == "bad_digest":
        split["sources"][0]["archive_sha256"] = "unknown"
    elif case_kind == "unknown_schema":
        split["schema_version"] = "unreviewed_native_v2"
    else:
        split["sources"][1]["site"] = "test-site"
    case.seal()
    monkeypatch.setattr(assessment.np, "load", lambda *a, **k: pytest.fail("Numeric decode"))
    with pytest.raises(ValueError):
        case.admit()


@pytest.mark.parametrize("device", ["cuda", "cuda:1"])
def test_real_execution_policy_rejects_unindexed_or_other_cuda_before_decode(
    memory_case, monkeypatch, device
):
    """Synthetic policy bytes cover the real admission branch; no GPU is initialized."""
    case = memory_case
    case.manifest.update(
        device=device, evidence_kind="REVIEWED_FROZEN_ASSESSMENT", input_evidence_kind=None
    )
    case.documents["cohort.json"]["evidence_kind"] = "REVIEWED_FROZEN_ASSESSMENT"
    case.review_overrides["device"] = device
    case.seal()
    monkeypatch.setattr(assessment.np, "load", lambda *a, **k: pytest.fail("Numeric decode"))
    with pytest.raises(ValueError, match="device"):
        case.admit()


def test_indexed_cuda_admission_checks_policy_without_loading_values_or_gpu(
    memory_case, monkeypatch
):
    case = memory_case
    case.manifest.update(
        device="cuda:0", evidence_kind="REVIEWED_FROZEN_ASSESSMENT", input_evidence_kind=None
    )
    case.documents["cohort.json"]["evidence_kind"] = "REVIEWED_FROZEN_ASSESSMENT"
    case.review_overrides["device"] = "cuda:0"
    case.seal()
    monkeypatch.setattr(assessment.np, "load", lambda *a, **k: pytest.fail("Numeric decode"))
    assert case.admit().manifest["device"] == "cuda:0"


@pytest.mark.parametrize("seed", [7, 13, 23])
def test_replication_typed_admission_precedes_decode(monkeypatch, seed):
    fs = MemoryFS(monkeypatch)
    case = Case(fs, learned=True)
    from marine_echo.training.native_band_replication_ssl import Config

    case.config.clear()
    case.config.update(
        Config(seed=seed, width=8, latent=4, blocks=1, heads=2, batch_size=8).to_dict()
    )
    case.spec.update(kind="native_band_replication_ssl_weights_only_inference_v2", seed=seed)
    case.seal()
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Admission decoded values"))
    assert case.admit().manifest["methods"]["fixed"]["seed"] == seed


@pytest.mark.parametrize("damage", ["v1_seed13", "seed33", "architecture", "objective", "lr"])
def test_replication_typed_bad_metadata_before_decode(monkeypatch, damage):
    fs = MemoryFS(monkeypatch)
    case = Case(fs, learned=True)
    from marine_echo.training.native_band_replication_ssl import Config

    case.config.clear()
    case.config.update(Config(width=8, latent=4, blocks=1, heads=2, batch_size=8).to_dict())
    case.spec.update(kind="native_band_replication_ssl_weights_only_inference_v2")
    if damage == "v1_seed13":
        case.spec.update(kind="native_band_ssl_weights_only_inference_v1", seed=13)
        case.config["seed"] = 13
    elif damage == "seed33":
        case.spec["seed"] = case.config["seed"] = 33
    elif damage == "objective":
        case.spec["method"] = case.config["method"] = "cf_jepa"
    elif damage == "lr":
        case.config["lr"] = 0.001
    else:
        case.config["architecture"] = "shared_temporal_v1"
    case.seal()
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Invalid metadata decoded"))
    with pytest.raises(ValueError, match="typed band|CF"):
        case.admit()
