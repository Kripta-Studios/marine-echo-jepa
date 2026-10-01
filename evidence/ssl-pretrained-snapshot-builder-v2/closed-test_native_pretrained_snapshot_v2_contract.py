"""SYNTHETIC_CORRECTNESS_ONLY metadata/opaque-byte packaging policy fixtures.

No fixture decodes pretrained tensors or executes a scientific process. The
supervisor doubles below are virtual policy checks, never process evidence.
"""

from __future__ import annotations

import copy
import json
import sys
import uuid
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
EVIDENCE = BUILDER / "evidence/ssl-pretrained-snapshot-builder-v2"
sys.path.insert(0, str(BUILDER / "tools"))
import check_native_pretrained_snapshot_owned_v3 as owned
import check_native_pretrained_snapshot_worker_v3 as worker
import package_native_pretrained_snapshot_v2 as package


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(
            value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode()
        )
    return path


def replace_json(path, value):
    # Only a new private fixture is mutable; production receipts are exclusive.
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


@pytest.fixture
def fixture(monkeypatch):
    root = EVIDENCE / ("synthetic-Álvaro-" + uuid.uuid4().hex)
    root.mkdir()
    for name in ("outputs", "evidence", "configs", "orchestration", "tools"):
        (root / name).mkdir()
    write(root / "orchestration/native_ssl_run_ledger_v1.json", {"runs": []})
    write(
        root / "tools/native_reference_supervisor.py",
        b'"""SYNTHETIC_CORRECTNESS_ONLY unexecuted supervisor."""\n',
    )
    for name in (
        "package_native_pretrained_snapshot_v1.py",
        "check_native_pretrained_snapshot_worker_v2.py",
        "check_native_pretrained_snapshot_owned_v2.py",
    ):
        write(
            root / "tools" / name, b'"""SYNTHETIC_CORRECTNESS_ONLY unexecuted original tool."""\n'
        )
    for name in package.ENTRIES:
        write(
            root / "src/marine_echo/inference" / name, b"from marine_echo.models.toy import TOKEN\n"
        )
    write(root / "src/marine_echo/models/toy.py", b"TOKEN = 'SYNTHETIC_CORRECTNESS_ONLY'\n")
    write(root / "src/marine_echo/__init__.py", b'"""Synthetic namespace."""\n')
    stats = {
        "channel_mean": [-80.0] * 4,
        "channel_std": [2.0] * 4,
        "target_mean": [-80.0] * 3,
        "target_std": [2.0] * 3,
    }
    members = {
        "train_row_ids": ["synthetic-train-row"],
        "train_deployments": ["synthetic-train-site"],
        "train_archive_sha256": ["a" * 64],
    }
    previous_bindings, replay_records, original_models = {}, [], []
    for name, method, seed, kind in package.SPECS:
        parent = root / "outputs/native_acoustic_ssl_v1" / name
        for leaf in ("selected_encoder.pt", "inference.pt"):
            write(
                parent / leaf,
                ("SYNTHETIC_CORRECTNESS_ONLY_OPAQUE_NOT_A_TENSOR_" + name + leaf).encode(),
            )
        write(parent / "scalers.json", stats)
        write(parent / "membership.json", members)
        config = {
            "method": method,
            "seed": seed,
            "history": 96,
            "pretrain_updates": 6000,
            "readout_updates": 500,
            "width": 192,
            "latent": 64,
            "blocks": 4,
            "heads": 4,
        }
        if kind != package.LEGACY_KIND:
            config["architecture"] = package.ARCHITECTURE
        config_path = write(root / "configs" / (name + ".json"), config)
        report = {
            "status": "COMPLETED",
            "evidence_kind": "REAL_TRAIN_DEVELOPMENT_FIT",
            "test_access": "NOT_RUN",
            "config": config,
            "pretrain_steps": 6000,
            "readout_steps_all_probes": 2000,
            "selected_pretrain_step": 6000,
            "inference_sha256": package.digest(parent / "inference.pt"),
            "membership_sha256": package.digest(parent / "membership.json"),
            "bindings": {str(config_path): package.digest(config_path)},
            "fixture_identity": "SYNTHETIC_CORRECTNESS_ONLY_NOT_PUBLIC_EVIDENCE",
        }
        if kind != package.LEGACY_KIND:
            report["architecture"] = package.ARCHITECTURE
        write(parent / "run.json", report)
        if kind == package.LEGACY_KIND:
            original_models.append({"id": name, "method": method, "seed": seed})
            for leaf in package.LEAVES:
                previous_bindings[str(parent / leaf)] = package.digest(parent / leaf)
            replay_records.append(
                {
                    "run": name,
                    "artifact_sha256": package.digest(parent / "inference.pt"),
                    "selected_encoder_cpu_replay": "BITIDENTICAL",
                    "saved_forward_head_cpu_replay": "BITIDENTICAL",
                    "fixture_identity": "SYNTHETIC_VIRTUAL_METADATA_ONLY",
                }
            )
    replay = write(
        root
        / "evidence/ssl-research-v1/Unicode-actual-latent-Álvaro-cpu-replay-v1/completion.json",
        {"records": replay_records},
    )
    closeout = write(
        root / "evidence/ssl-research-v1/native-latent-actual-cpu-replay-closeout-v1.json",
        {
            "actual_cli_exit_code": 0,
            "completion_sha256": package.digest(replay),
            "fixture_identity": "SYNTHETIC_VIRTUAL_METADATA_ONLY",
        },
    )
    for path in (
        replay,
        closeout,
        write(
            root / "configs/native_ssl_split_v1.json",
            {"fixture_identity": "SYNTHETIC_CORRECTNESS_ONLY"},
        ),
        write(
            root / "external/cf-jepa-vnext/LICENSE",
            b"Synthetic fixture license placeholder; grants nothing.",
        ),
        write(root / "docs/NATIVE_SSL_MODEL_RESEARCH_REPORT_V1.md", b"Synthetic fixture"),
        write(root / "docs/NATIVE_SSL_MODEL_RESEARCH_REPORT_ADDENDUM_V2.md", b"Synthetic fixture"),
    ):
        previous_bindings[str(path)] = package.digest(path)
    v1 = root / "outputs/native_pretrained_model_snapshot_v1"
    write(
        v1 / "manifest.json",
        {
            "kind": "native_pretrained_model_snapshot_v1",
            "status": "INCOMPLETE_RESEARCH_SNAPSHOT",
            "models": original_models,
            "source_bindings": previous_bindings,
        },
    )
    write(v1 / "MODEL_CARD.md", b"Synthetic incomplete snapshot")
    old_zip = write(v1.with_suffix(".zip"), b"SYNTHETIC_OPAQUE_OLD_ZIP_PIN_POLICY_FIXTURE")
    monkeypatch.setattr(package, "V1_ARCHIVE_SHA256", package.digest(old_zip))
    last = root / "outputs/native_acoustic_ssl_v1" / package.SPECS[-1][0] / "run.json"
    monkeypatch.setattr(package, "BAND23_RUN_SHA256", package.digest(last))
    catalogs = []
    for number in range(3):
        archived = write(
            root / f"evidence/ssl-research-v1/original-source-archives-v3/old{number}.py",
            b"# Synthetic preserved historical source\n",
        )
        catalog = write(
            archived.with_suffix(".py.manifest.json"),
            {
                "status": "ORIGINAL_SOURCE_PRESERVED_NOT_COMPATIBILITY_APPROVAL",
                "path": str(archived),
                "sha256": package.digest(archived),
                "original_path": str(root / f"tools/old{number}.py"),
                "methods": ["shared_ssl", "cf_jepa"],
            },
        )
        catalogs.append(str(catalog))
    write(
        root / "evidence/ssl-research-v1/original-source-archives-v3/index.json",
        {"scientific_approval": False, "catalogs": catalogs},
    )
    return SimpleNamespace(
        root=root,
        folder=root / "outputs/snapshot-v2",
        receipt=root / "evidence/snapshot-v2.json",
        old_zip=old_zip,
    )


def plan(f):
    return package.build_plan(f.root, f.folder, f.folder.with_suffix(".zip"), f.receipt)


def snapshot(f):
    return package.package_snapshot(f.root, f.folder, f.folder.with_suffix(".zip"), f.receipt)


def test_band_v1_routes_to_existing_typed_band_encoder():
    assert worker.route({"inference_kind": package.BAND_KIND}) == "band_v1"


@pytest.mark.parametrize(
    "kind,expected", [(package.LEGACY_KIND, "legacy"), (package.REPLICATION_KIND, "band_v2")]
)
def test_remaining_routes(kind, expected):
    assert worker.route({"inference_kind": kind}) == expected


def test_seven_private_unicode_payloads_roundtrip(fixture):
    old_hash = package.digest(fixture.old_zip)
    result = snapshot(fixture)
    manifest = worker.validate_bundle(fixture.folder)
    assert len(manifest["models"]) == 7
    assert {m["id"] for m in manifest["models"]} == {s[0] for s in package.SPECS}
    assert manifest["status"] == "INCOMPLETE_RESEARCH_SNAPSHOT"
    assert result["independent_package_review"] == "NOT_RUN"
    assert package.digest(fixture.old_zip) == old_hash
    with zipfile.ZipFile(result["archive"]) as archive:
        assert set(archive.namelist()) == set(result["files"])
        assert archive.read("models/" + package.SPECS[-1][0] + "/inference.pt").startswith(
            b"SYNTHETIC_CORRECTNESS_ONLY"
        )
        assert not any(name.endswith(".npz") or "latest.pt" in name for name in archive.namelist())
    with pytest.raises(FileExistsError):
        snapshot(fixture)


@pytest.mark.parametrize("leaf", package.LEAVES)
def test_missing_endpoint_never_falls_through(fixture, leaf):
    (fixture.root / "outputs/native_acoustic_ssl_v1" / package.SPECS[-1][0] / leaf).unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        plan(fixture)
    assert not fixture.folder.exists() and not fixture.receipt.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "RUNNING_CUDA"),
        ("evidence_kind", "SYNTHETIC_CORRECTNESS_ONLY"),
        ("pretrain_steps", 5999),
        ("readout_steps_all_probes", 500),
        ("inference_sha256", "0" * 64),
        ("selected_encoder_sha256", "0" * 64),
        ("test_access", "ACCESSED"),
        ("supervised_ancestry", {"mode": "full_finetune"}),
    ],
)
def test_incomplete_or_changed_endpoint_refused(fixture, field, value):
    path = fixture.root / "outputs/native_acoustic_ssl_v1" / package.SPECS[4][0] / "run.json"
    report = package.document(path)
    report[field] = value
    replace_json(path, report)
    with pytest.raises(ValueError):
        plan(fixture)
    assert not fixture.folder.exists()


def test_train_scalers_cannot_change(fixture):
    path = fixture.root / "outputs/native_acoustic_ssl_v1" / package.SPECS[4][0] / "scalers.json"
    stats = package.document(path)
    stats["channel_mean"][0] += 1
    replace_json(path, stats)
    with pytest.raises(ValueError, match="TRAIN scalers"):
        plan(fixture)


def test_membership_identity_cannot_change(fixture):
    path = fixture.root / "outputs/native_acoustic_ssl_v1" / package.SPECS[4][0] / "membership.json"
    members = package.document(path)
    members["train_deployments"] = ["foreign-site"]
    replace_json(path, members)
    report_path = path.with_name("run.json")
    report = package.document(report_path)
    report["membership_sha256"] = package.digest(path)
    replace_json(report_path, report)
    with pytest.raises(ValueError, match="TRAIN membership"):
        plan(fixture)


def test_missing_source_and_wrong_archive_pin_refused(fixture, monkeypatch):
    monkeypatch.setattr(package, "V1_ARCHIVE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="V1 immutable"):
        plan(fixture)
    monkeypatch.setattr(package, "V1_ARCHIVE_SHA256", package.digest(fixture.old_zip))
    (fixture.root / "src/marine_echo/inference/native_band_replication_latent.py").unlink()
    with pytest.raises(FileNotFoundError):
        plan(fixture)
    assert not fixture.folder.exists()


@pytest.mark.parametrize(
    "state",
    [
        "running",
        "reconciliation",
        "owner_lock",
        "all.pending",
        ".prefix-pending",
        ".band-pending",
        ".assessment-pending",
        ".reconciliation-pending",
    ],
)
def test_idle_required_before_copy_or_launch(fixture, state):
    if state in ("running", "reconciliation"):
        replace_json(
            fixture.root / "orchestration/native_ssl_run_ledger_v1.json",
            {
                "runs": [
                    {
                        "status": "RUNNING_CUDA" if state == "running" else "FAILED",
                        "requires_reconciliation": state == "reconciliation",
                    }
                ]
            },
        )
    elif state == "owner_lock":
        write(
            fixture.root / "evidence/ssl-builder-v1/gpu-owner.lock",
            b"Synthetic owner; must remain intact",
        )
    else:
        write(fixture.root / "orchestration" / state, b"Synthetic active journal")
    with pytest.raises(ValueError):
        plan(fixture)
    assert not fixture.folder.exists()


def test_historical_versions_remain_explicit_not_approval(fixture):
    original = fixture.root / "tools/old0.py"
    write(original, b"# Changed current source; never execute\n")
    archived = fixture.root / "evidence/ssl-research-v1/original-source-archives-v3/old0.py"
    report_path = fixture.root / "outputs/native_acoustic_ssl_v1" / package.SPECS[4][0] / "run.json"
    report = package.document(report_path)
    report["bindings"][str(original)] = package.digest(archived)
    report["bindings"][str(fixture.root / "tools/missing-old.py")] = "b" * 64
    replace_json(report_path, report)
    entries = plan(fixture)["historical_sources"]
    assert {e["status"] for e in entries} == {
        "EXACT_HISTORICAL_VERSION_PRESERVED_NOT_COMPATIBILITY_APPROVAL",
        "ORIGINAL_BYTES_UNRECOVERED_NOT_COMPATIBILITY_APPROVAL",
    }
    assert all(not e["compatibility_approval"] and not e["source_executed"] for e in entries)


def test_current_root_source_closure_is_read_only_and_complete():
    closure = package.source_closure(MAIN)
    assert all(p.is_relative_to(MAIN / "src/marine_echo") for p in closure)
    assert {
        "native_band_replication_encoder.py",
        "native_band_replication_latent.py",
        "native_band_replication_ssl.py",
        "native_band_replication_downstream.py",
        "native_band_temporal.py",
        "native_temporal.py",
        "native_resources.py",
    } <= {p.name for p in closure}


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "wrong_kind", "wrong_seed", "wrong_selected"]
)
def test_worker_fixed_manifest_identities(fixture, change):
    models = copy.deepcopy(plan(fixture)["models"])
    if change == "missing":
        models.pop()
    elif change == "duplicate":
        models[-1] = models[0]
    elif change == "wrong_kind":
        models[-1]["inference_kind"] = package.LEGACY_KIND
    elif change == "wrong_seed":
        models[-1]["seed"] = 17
    else:
        models[-1]["selected_kind"] = "native_band_downstream_supervised_encoder_v1"
    with pytest.raises(ValueError):
        worker.validate_models(
            {
                "kind": "native_pretrained_model_snapshot_v2",
                "status": "INCOMPLETE_RESEARCH_SNAPSHOT",
                "models": models,
            }
        )


def test_namespace_path_must_be_copied_even_with_correct_file(fixture):
    module = SimpleNamespace(
        __file__=str(fixture.root / "src/marine_echo/__init__.py"),
        __path__=[str(MAIN / "src/marine_echo")],
    )
    with pytest.raises(ValueError, match="namespace"):
        worker.check_namespace(fixture.root, {"marine_echo": module})


@pytest.mark.parametrize("kind", [package.BAND_KIND, package.REPLICATION_KIND])
def test_actual_native_query_validator_rejects_200_relabel(kind):
    import numpy as np

    sys.path.insert(0, str(MAIN / "src"))
    from marine_echo.inference.native_band_replication_acoustic import _query

    metadata = np.zeros((2, 4, 10), dtype=np.float32)
    metadata[:, :, 0] = np.array([38000, 125000, 200000, 455000]) / 455000
    metadata[:, :, 1:3] = 1
    metadata[:, :, 4] = 230 / 250
    query = np.repeat(metadata[:, :1], 3, axis=1)
    query[:, :, 9] = [1, 3, 6]
    item = next(
        m
        for m in [
            {
                "id": n,
                "method": m,
                "seed": s,
                "inference_kind": k,
                "selected_kind": package.selected_kind(k),
                "latent_semantics": "SYNTHETIC_VIRTUAL_METADATA_ONLY",
            }
            for n, m, s, k in package.SPECS
        ]
        if m["inference_kind"] == kind
    )

    class SyntheticAdapter:
        artifact_kind = kind

        def __init__(self):
            self.config_metadata = {"method": item["method"], "seed": item["seed"], "history": 96}

        def encode(self, x, observed, metadata):
            return np.zeros((len(x), 64), dtype=np.float32)

        def predict_latents(self, x, observed, metadata, query):
            _query(query, metadata)
            return np.zeros((len(x), 3, 64), dtype=np.float32)

    encoder, predictor = SyntheticAdapter(), SyntheticAdapter()
    encoder.artifact_kind = item["selected_kind"]
    record = worker.check_outputs(
        item,
        encoder,
        predictor,
        np,
        np.zeros((2, 96, 4)),
        np.ones((2, 96, 4), dtype=bool),
        metadata,
        query,
    )
    assert record["native230_query_guard"] == "PASSED"
    assert np.allclose(query[:, :, 4], 230 / 250)


def virtual_attempt(f, *, completion=True, resources=None, fail=False):
    result = snapshot(f)
    destination = f.root / "evidence/virtual-QA"

    def launch(command, **kwargs):
        assert command[1:3] == ["-I", "-B"]
        assert kwargs["cwd"] == destination
        if completion:
            write(
                destination / "completion.json",
                {
                    "status": "COPIED_SEVEN_MODEL_PACKAGE_ISOLATED_CPU_CORRECTNESS_PASSED",
                    "manifest_sha256": result["manifest_sha256"],
                    "records": [{"id": m["id"]} for m in result["models"]],
                    "all_imports_from_copied_source": True,
                    "cuda_initialized": False,
                    "rng_unchanged": True,
                    "fixture_identity": "SYNTHETIC_VIRTUAL_NOT_A_REAL_PROCESS",
                },
            )
        return SimpleNamespace(pid=123456789)

    def supervise(child, **kwargs):
        assert kwargs["deadline_seconds"] == 600
        assert kwargs["rss_limit_bytes"] == 22 * 1024**3
        if fail:
            raise RuntimeError("Synthetic supervisor exception")
        return {
            "exit_code": 0,
            "stopped_for": None,
            "peak_process_rss_bytes": 4096,
            "elapsed_full_attempt_seconds": 1.0,
            "owned_tree_cleanup_verified": True,
            **(resources or {}),
        }

    return owned.execute(f.root, f.receipt, destination, launcher=launch, supervisor=supervise)


def test_virtual_owned_success_has_separate_receipt(fixture):
    code, record = virtual_attempt(fixture)
    assert code == 0 and record["device"] == "cpu"
    assert record["child_pid"] == 123456789
    assert (fixture.root / "evidence/virtual-QA/resources.json").exists()
    assert (
        package.document(fixture.folder / "manifest.json")["status"]
        == "INCOMPLETE_RESEARCH_SNAPSHOT"
    )


@pytest.mark.parametrize(
    "resources",
    [
        {"exit_code": 2},
        {"owned_tree_cleanup_verified": False},
        {"peak_process_rss_bytes": 22 * 1024**3},
        {"elapsed_full_attempt_seconds": 600},
        {"peak_process_rss_bytes": True},
        {"stopped_for": "OWNED_PROCESS_MONITOR_DENIED"},
    ],
)
def test_virtual_failures_never_claim_completed(fixture, resources):
    code, record = virtual_attempt(fixture, resources=resources)
    assert code == 1 and record["status"] == "COPIED_PACKAGE_CPU_QA_FAILED"
    assert record["resources"] == {
        "exit_code": 0,
        "stopped_for": None,
        "peak_process_rss_bytes": 4096,
        "elapsed_full_attempt_seconds": 1.0,
        "owned_tree_cleanup_verified": True,
        **resources,
    }


def test_virtual_missing_completion_is_failure(fixture):
    assert virtual_attempt(fixture, completion=False)[0] == 1


def test_virtual_parent_exception_retains_reconciliation(fixture):
    with pytest.raises(RuntimeError, match="Synthetic supervisor"):
        virtual_attempt(fixture, fail=True)
    receipt = package.document(fixture.root / "evidence/virtual-QA/parent-exception.json")
    assert receipt["requires_reconciliation"] is True
    assert receipt["child_pid"] == 123456789


def test_owned_stale_source_before_launcher_and_destination(fixture):
    result = snapshot(fixture)
    result["source_bindings"].pop(str(Path(owned.__file__).resolve()))
    replace_json(fixture.receipt, result)
    destination = fixture.root / "evidence/never-launched"
    with pytest.raises(ValueError, match="source bindings"):
        owned.execute(
            fixture.root,
            fixture.receipt,
            destination,
            launcher=lambda *a, **k: pytest.fail("Must not launch"),
        )
    assert not destination.exists()


def test_load_pair_uses_v1_and_v2_safe_existing_apis(monkeypatch):
    calls = []

    class Loader:
        def __init__(self, path, *, device):
            calls.append((Path(path).name, device))

    for name, symbol in [
        ("native_encoder", "NativeAcousticEncoder"),
        ("native_latent", "NativeLatentPredictor"),
        ("native_band_replication_encoder", "NativeBandReplicationEncoder"),
        ("native_band_replication_latent", "NativeLatentPredictor"),
    ]:
        monkeypatch.setitem(
            sys.modules, "marine_echo.inference." + name, SimpleNamespace(**{symbol: Loader})
        )
    for name, method, seed, kind in package.SPECS:
        worker.load_pair(
            Path("synthetic-relocated"),
            {
                "inference_kind": kind,
                "selected_encoder": "selected_encoder.pt",
                "latent_inference": "inference.pt",
            },
        )
    assert calls == [("selected_encoder.pt", "cpu"), ("inference.pt", "cpu")] * 7


@pytest.mark.parametrize("state", ["running", "owner_lock", "pending"])
def test_owned_idle_gate_remains_after_packaging(fixture, state):
    snapshot(fixture)
    if state == "running":
        replace_json(
            fixture.root / "orchestration/native_ssl_run_ledger_v1.json",
            {"runs": [{"status": "RUNNING_CPU_FIT"}]},
        )
    elif state == "owner_lock":
        write(
            fixture.root / "evidence/ssl-builder-v1/gpu-owner.lock",
            b"SYNTHETIC_OWNER_DO_NOT_REMOVE",
        )
    else:
        write(fixture.root / "orchestration/all.pending", b"SYNTHETIC_PENDING_DO_NOT_REMOVE")
    destination = fixture.root / "evidence/never-started"
    with pytest.raises(ValueError):
        owned.execute(
            fixture.root,
            fixture.receipt,
            destination,
            launcher=lambda *a, **k: pytest.fail("Must not launch"),
        )
    assert not destination.exists()


def test_changed_payload_or_extra_file_denied_before_decode(fixture):
    snapshot(fixture)
    write(fixture.folder / "unexpected.npz", b"Synthetic forbidden extra payload")
    with pytest.raises(ValueError, match="Unexpected"):
        worker.validate_bundle(fixture.folder)
    (fixture.folder / "unexpected.npz").unlink()
    weights = fixture.folder / "models" / package.SPECS[-1][0] / "inference.pt"
    weights.write_bytes(b"SYNTHETIC_TAMPERED_NOT_A_TENSOR")
    with pytest.raises(ValueError, match="hash/path"):
        worker.validate_bundle(fixture.folder)


def test_unknown_native_kind_refused():
    with pytest.raises(ValueError, match="Unknown"):
        worker.route({"inference_kind": "native_ssl_resume_v1"})
