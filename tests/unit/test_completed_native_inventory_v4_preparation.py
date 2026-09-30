"""SYNTHETIC_CORRECTNESS_ONLY metadata guards; no real tensors or processes."""

from __future__ import annotations

import ast
import copy
import importlib
import json
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
HERE = BUILDER / "evidence/ssl-inventory-preparation-builder-v4"
sys.path.insert(0, str(MAIN / "src"))
sys.path.insert(0, str(BUILDER / "tools"))
prep = importlib.import_module("prepare_completed_native_inventory_v4")
owned = importlib.import_module("execute_completed_native_inventory_owned_v4")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False)


def opaque(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value)


def defaults(module, cls):
    tree = ast.parse((MAIN / f"src/marine_echo/training/{module}.py").read_bytes())
    constants = {
        n.targets[0].id: n.value
        for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
    }
    definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    values = {}
    for node in definition.body:
        if not isinstance(node, ast.AnnAssign) or node.value is None:
            continue
        if node.target.id == "cf_source":
            values[node.target.id] = str(MAIN / "external/cf-jepa-vnext")
        elif isinstance(node.value, ast.Name) and node.value.id == "ARCHITECTURE":
            values[node.target.id] = prep.ARCHITECTURE
        else:
            values[node.target.id] = ast.literal_eval(
                constants[node.value.id] if isinstance(node.value, ast.Name) else node.value
            )
    return values


@pytest.fixture(scope="module")
def case():
    root = HERE / ("SYNTHETIC_CORRECTNESS_ONLY-Á-Metadata-" + str(uuid.uuid4()))
    for directory in ("orchestration", "configs", "evidence/ssl-research-v1", "outputs", "tools"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    train = root / "data/processed/native_ssl_v1/train.npz"
    opaque(train, b"SYNTHETIC_CORRECTNESS_ONLY opaque byte-binding; not an NPZ codec")
    split = root / "configs/native_ssl_split_v1.json"
    write(
        split,
        {"schema_version": "native_acoustic_ssl_v1", "synthetic_fixture": True, "sources": []},
    )
    write(
        train.with_suffix(".json"),
        {"role": "train", "npz_sha256": prep.digest(train), "synthetic_fixture": True},
    )
    write(
        root / "orchestration/native_assessment_seed7_metadata_v1/train_cohort.json",
        {"role": "train", "synthetic_fixture": True},
    )
    index_source = MAIN / "evidence/ssl-research-v1/original-source-archives-v3/index.json"
    write(
        root / "evidence/ssl-research-v1/original-source-archives-v3/index.json",
        prep.document(index_source),
    )
    opaque(
        root / "tools/native_reference_supervisor.py",
        b"# private synthetic policy stub; not executed\n",
    )
    opaque(
        root / "tools/prepare_native_research_inventory.py",
        (MAIN / "tools/prepare_native_research_inventory.py").read_bytes(),
    )
    for ssl, downstream, kind in prep.PRODUCERS:
        for module in (ssl, downstream):
            opaque(
                root / f"src/marine_echo/training/{module}.py",
                f"def run():\n    return {{'kind': '{kind}'}}\n".encode(),
            )
    sources = [
        Path(prep.__file__),
        Path(owned.__file__),
        root / "tools/native_reference_supervisor.py",
        root / "tools/prepare_native_research_inventory.py",
    ]
    matrix = {
        "role": "development",
        "methods": {
            name: str(root / "references" / name / "predictions.npz") for name in prep.REFERENCES
        },
    }
    runs, folders = [], []
    for i in range(43):
        name = f"neural{i:02d}"
        folder = root / "outputs/native_acoustic_ssl_v1" / name
        folders.append(folder)
        band = i in (4, 5, 6, 7)
        version = 2 if i in (5, 6, 7) else 1
        ssl, downstream, _ = prep.PRODUCERS[2 if band and version == 2 else 1 if band else 0]
        method = "direct" if i == 3 else "shared_ssl"
        mode = (
            "full_finetune"
            if i == 1
            else "frozen_readout"
            if i in (2, 6)
            else "direct_end_to_end"
            if i == 3
            else "core_frozen_readout"
        )
        seed = 13 if i == 5 else 23 if i in (6, 7) else 7
        config = defaults(ssl, "Config")
        config.update(
            method=method,
            seed=seed,
            pretrain_updates=3000 if method == "direct" else 6000,
            readout_updates=500,
        )
        saved_config = copy.deepcopy(config)
        if mode != "core_frozen_readout":
            saved_config = defaults(downstream, "DownstreamConfig")
            saved_config.update(
                method=method,
                seed=seed,
                mode=mode,
                updates=2000 if mode == "frozen_readout" else 3000,
                cadence=500 if mode == "frozen_readout" else 750,
            )
        config_path = root / "configs" / (name + ".json")
        write(config_path, saved_config)
        producer = (
            root
            / f"src/marine_echo/training/{ssl if mode == 'core_frozen_readout' else downstream}.py"
        )
        bindings = {str(p): prep.digest(p) for p in (producer, config_path, train, split)}
        # V2 preserves V1 as an explicitly bound compatibility dependency.
        if band and version == 2:
            original = (
                root
                / f"src/marine_echo/training/{'native_band_ssl' if mode == 'core_frozen_readout' else 'native_band_downstream'}.py"
            )
            bindings[str(original)] = prep.digest(original)
        review_path = root / "orchestration" / (name + "-original-review.json")
        write(
            review_path,
            {
                "status": "APPROVED_PREFIT"
                if mode == "core_frozen_readout"
                else "APPROVED_DOWNSTREAM_PREFIT",
                "reviewer_session_id": "private-distinct-reviewer",
                "implementer_session_id": "private-implementer",
                "bindings": bindings,
                "synthetic_fixture": True,
            },
        )
        opaque(
            folder / "inference.pt", ("SYNTHETIC_CORRECTNESS_ONLY-not-a-tensor-" + name).encode()
        )
        opaque(
            folder / "selected_encoder.pt",
            ("SYNTHETIC_CORRECTNESS_ONLY-not-an-encoder-" + name).encode(),
        )
        opaque(folder / "predictions.npz", b"SYNTHETIC_CORRECTNESS_ONLY-not-a-forecast")
        write(
            folder / "membership.json", {"role": "train", "synthetic_fixture": True, "method": name}
        )
        if i not in (2, 3):
            write(
                folder / "scalers.json",
                {
                    "channel_mean": [0.0] * 4,
                    "channel_std": [1.0] * 4,
                    "target_mean": [0.0] * 3,
                    "target_std": [1.0] * 3,
                },
            )
        run = {
            "status": "COMPLETED",
            "evidence_kind": prep.REAL,
            "test_access": "NOT_RUN",
            "config": saved_config,
            "bindings": bindings,
            "review_sha256": prep.digest(review_path),
            "inference_sha256": prep.digest(folder / "inference.pt"),
            "membership_sha256": prep.digest(folder / "membership.json"),
            "selected_encoder_sha256": prep.digest(folder / "selected_encoder.pt"),
            "synthetic_fixture": True,
            "fixture_identity": name,
        }
        if band:
            run["architecture"] = prep.ARCHITECTURE
        if mode != "core_frozen_readout":
            run.update(
                mode=mode,
                core_config=config,
                supervised_ancestry={
                    "mode": mode,
                    "ssl_only": False,
                    "ancestor_run_sha256": None,
                    "ancestor_encoder_sha256": None,
                },
            )
        write(folder / "run.json", run)
        matrix["methods"][name] = str(folder / "predictions.npz")
        runs.append({"status": "COMPLETED", "output": str(folder), "review": str(review_path)})
    for child, parent in ((1, 0), (2, 0), (6, 7)):
        path = folders[child] / "run.json"
        value = prep.document(path)
        value["supervised_ancestry"].update(
            ancestor_run_sha256=prep.digest(folders[parent] / "run.json"),
            ancestor_encoder_sha256=prep.digest(folders[parent] / "selected_encoder.pt"),
        )
        path.write_text(json.dumps(value), encoding="utf-8")
    fallback = root / "outputs/native_acoustic_ssl_v1/shared_ssl_seed7_h96_cuda0/scalers.json"
    write(fallback, prep.document(folders[0] / "scalers.json"))
    ledger = root / "orchestration/native_ssl_run_ledger_v1.json"
    write(ledger, {"runs": runs, "synthetic_fixture": True})
    matrix_path = root / "orchestration/native_development_comparison_v4.json"
    write(matrix_path, matrix)
    return SimpleNamespace(
        root=root, matrix=matrix_path, ledger=ledger, folders=folders, sources=sources
    )


@contextmanager
def changed(path, transform):
    original = path.read_bytes()
    value = prep.json_bytes(original)
    transform(value)
    path.write_text(json.dumps(value), encoding="utf-8")
    try:
        yield
    finally:
        path.write_bytes(original)


def destinations(case):
    token = str(uuid.uuid4())
    return SimpleNamespace(
        manifest=case.root / "orchestration" / (token + "-manifest.json"),
        snapshot=case.root / "evidence" / (token + "-ledger.json"),
        output=case.root / "evidence" / (token + "-output"),
        receipt=case.root / "evidence" / (token + "-receipt"),
    )


def prepare(case, paths):
    return prep.prepare(
        case.root,
        case.matrix,
        paths.manifest,
        paths.snapshot,
        paths.output,
        source_paths=case.sources,
    )


def test_snapshot_binding_is_immutable_and_exact43_parented_inventory(case):
    paths = destinations(case)
    original = case.ledger.read_bytes()
    facts = prepare(case, paths)
    manifest = prep.document(paths.manifest)
    assert str(case.ledger) not in manifest["bindings"]
    assert paths.snapshot.read_bytes() == original
    assert manifest["bindings"][str(paths.snapshot)] == prep.digest(paths.snapshot)
    assert set(manifest["endpoints"]) == set(prep.document(case.matrix)["methods"]) - set(
        prep.REFERENCES
    )
    assert manifest["endpoints"]["neural01"]["parent"] == "neural00"
    assert manifest["endpoints"]["neural06"]["parent"] == "neural07"
    assert manifest["endpoints"]["neural04"]["kind"] == prep.PRODUCERS[1][2]
    assert manifest["endpoints"]["neural05"]["kind"] == prep.PRODUCERS[2][2]
    assert (
        Path(manifest["endpoints"]["neural02"]["scalers_path"]).parent.name
        == "shared_ssl_seed7_h96_cuda0"
    )
    assert manifest["references"] == {} and len(facts["references"]) == 4
    assert facts["fit_authority"] is facts["final_selection"] is False
    assert manifest["endpoints"]["neural03"]["mode"] == "direct_end_to_end"
    assert manifest["endpoints"]["neural03"]["parent"] is None
    assert (
        manifest["endpoints"]["neural03"]["scalers_path"]
        == manifest["endpoints"]["neural02"]["scalers_path"]
    )


@pytest.mark.parametrize("phase", [2, 3, 4])
def test_changed_ledger_capture_refused(case, monkeypatch, phase):
    paths = destinations(case)
    original = prep.idle_ledger
    count = 0

    def capture(root):
        nonlocal count
        count += 1
        raw, value = original(root)
        return (raw + b"\n", value) if count == phase else (raw, value)

    monkeypatch.setattr(prep, "idle_ledger", capture)
    with pytest.raises(ValueError, match="changed"):
        prepare(case, paths)
    assert paths.manifest.exists() == (phase == 4)
    assert paths.snapshot.exists() == (phase in (3, 4))
    assert not prep.reservation_path(paths.snapshot).exists()


@pytest.mark.parametrize(
    "blocker",
    [
        "RUNNING_CUDA",
        "RUNNING_CPU_FIT",
        "reconciliation",
        "lock",
        "all.pending",
        ".pending",
        ".prefix-pending",
        ".band-pending",
        ".assessment-pending",
        ".reconciliation-pending",
    ],
)
def test_live_ownership_refuses_preparation_without_artifacts(case, blocker):
    paths = destinations(case)
    if blocker.startswith("RUNNING") or blocker == "reconciliation":

        def alter(value):
            value["runs"].append(
                {"status": blocker}
                if blocker.startswith("RUNNING")
                else {"status": "FAILED", "requires_reconciliation": True}
            )

        with changed(case.ledger, alter), pytest.raises(ValueError):
            prepare(case, paths)
    else:
        path = (
            case.root / "evidence/ssl-builder-v1/gpu-owner.lock"
            if blocker == "lock"
            else case.ledger.parent
            / ("all.pending" if blocker == "all.pending" else "private" + blocker)
        )
        opaque(path, b"SYNTHETIC_CORRECTNESS_ONLY owned marker")
        try:
            with pytest.raises(ValueError):
                prepare(case, paths)
        finally:
            path.unlink()  # Private test-owned marker only; never a scientific lock.
    assert not paths.snapshot.exists() and not paths.manifest.exists()


@pytest.mark.parametrize(
    "damage",
    [
        "46",
        "48",
        "reference",
        "alias",
        "incomplete",
        "missing",
        "parent",
        "encoder",
        "config",
        "review",
        "kind",
    ],
)
def test_no_incomplete_cardinality_or_parent_fallthrough(case, monkeypatch, damage):
    paths = destinations(case)
    target = case.matrix

    def alter(value):
        if damage in ("46", "reference"):
            missing = value["methods"].pop("neural42" if damage == "46" else "chronos2_zero_shot")
            if damage == "reference":
                value["methods"]["unknown_reference"] = missing
        elif damage == "48":
            value["methods"]["unprescribed"] = "new-path"
        else:
            value["methods"]["neural42"] = value["methods"]["neural41"]

    if damage in ("incomplete", "parent", "encoder", "config", "kind"):
        target = case.folders[1 if damage in ("parent", "encoder") else 42] / "run.json"

        def alter(value):
            if damage == "incomplete":
                value["status"] = "RUNNING"
            elif damage == "parent":
                value["supervised_ancestry"]["ancestor_run_sha256"] = "f" * 64
            elif damage == "encoder":
                value["supervised_ancestry"]["ancestor_encoder_sha256"] = "e" * 64
            elif damage == "config":
                value["config"]["seed"] = 99
            else:
                value["architecture"] = "undeclared_architecture"
    elif damage == "review":
        target = case.root / "orchestration/neural42-original-review.json"

        def alter(value):
            value["reviewer_session_id"] = "private-implementer"
    elif damage == "missing":
        original = prep.regular

        def absent(path):
            if Path(path) == case.folders[42] / "selected_encoder.pt":
                raise FileNotFoundError("Private missing endpoint fixture")
            return original(path)

        monkeypatch.setattr(prep, "regular", absent)
        with pytest.raises(FileNotFoundError):
            prepare(case, paths)
        assert not paths.snapshot.exists()
        return
    with changed(target, alter), pytest.raises((ValueError, FileNotFoundError)):
        prepare(case, paths)
    assert not paths.snapshot.exists() and not paths.manifest.exists()


@pytest.mark.parametrize("destination", ["manifest", "snapshot", "output", "receipt"])
def test_protected_attempt_destinations(case, destination):
    paths = destinations(case)
    if destination == "receipt":
        prepare(case, paths)
        paths.receipt.mkdir()
        with pytest.raises(FileExistsError):
            owned.execute(case.root, paths.manifest, paths.snapshot, paths.output, paths.receipt)
    else:
        opaque(getattr(paths, destination), b"SYNTHETIC_CORRECTNESS_ONLY prior attempt")
        with pytest.raises(FileExistsError):
            prepare(case, paths)


def virtual_call(
    case, paths, *, exit_code=0, cleanup=True, stop=None, create_output=True, exception=False
):
    calls = []

    def launch(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=123456)

    def supervise(child, **kwargs):
        assert kwargs["deadline_seconds"] == 600 and kwargs["rss_limit_bytes"] == 22 * 1024**3
        if exception:
            raise RuntimeError("private supervisor failure")
        if create_output:
            paths.output.mkdir()
            write(
                paths.output / "inventory.json",
                {
                    "kind": "native_research_inventory_v1",
                    "status": "DERIVED_METADATA_NOT_FINAL_SELECTION",
                    "manifest_sha256": prep.digest(paths.manifest),
                    "models": {key: {} for key in prep.document(paths.manifest)["endpoints"]},
                    "numeric_corpus_decoded": False,
                    "model_constructed": False,
                    "synthetic_fixture": True,
                },
            )
        return {
            "exit_code": exit_code,
            "stopped_for": stop,
            "owned_tree_cleanup_verified": cleanup,
            "owned_process_pids": [child.pid],
            "elapsed_full_attempt_seconds": 0.01,
            "peak_process_rss_bytes": 1234,
        }

    return owned.execute(
        case.root,
        paths.manifest,
        paths.snapshot,
        paths.output,
        paths.receipt,
        launcher=launch,
        supervisor=supervise,
    ), calls


def test_idle_live_ledger_may_change_after_snapshot_without_staling_audit(case):
    paths = destinations(case)
    prepare(case, paths)
    snapshot = paths.snapshot.read_bytes()
    with changed(
        case.ledger,
        lambda v: v["runs"].append({"status": "COMPLETED", "other_idle_metadata": True}),
    ):
        (code, record), calls = virtual_call(case, paths)
    assert code == 0 and len(calls) == 1 and record["device"] == "cpu"
    assert paths.snapshot.read_bytes() == snapshot
    assert record["ledger_snapshot_sha256"] == prep.digest(paths.snapshot)


@pytest.mark.parametrize("damage", ["running", "snapshot", "source", "pending", "matrix"])
def test_executor_denies_changed_or_active_inputs_before_launch_mutation(case, monkeypatch, damage):
    paths = destinations(case)
    prepare(case, paths)
    calls = []
    if damage == "running":
        target = case.ledger
        transform = lambda value: value["runs"].append({"status": "RUNNING_CUDA"})
    elif damage == "matrix":
        target = case.matrix
        transform = lambda value: value.update(extra="changed reservation")
    elif damage == "snapshot":
        target = paths.snapshot
        transform = lambda value: value.update(extra="changed snapshot")
    elif damage == "source":
        original = owned.digest
        monkeypatch.setattr(
            owned, "digest", lambda p: "0" * 64 if Path(p) == case.sources[2] else original(p)
        )
        with pytest.raises(ValueError):
            owned.execute(
                case.root,
                paths.manifest,
                paths.snapshot,
                paths.output,
                paths.receipt,
                launcher=lambda *a, **k: calls.append(a),
            )
        assert not paths.receipt.exists() and not calls
        return
    else:
        target = case.ledger.parent / "private.assessment-pending"
        opaque(target, b"private pending fixture")
        try:
            with pytest.raises(ValueError):
                owned.execute(
                    case.root, paths.manifest, paths.snapshot, paths.output, paths.receipt
                )
        finally:
            target.unlink()
        assert not paths.receipt.exists()
        return
    with changed(target, transform), pytest.raises(ValueError):
        owned.execute(
            case.root,
            paths.manifest,
            paths.snapshot,
            paths.output,
            paths.receipt,
            launcher=lambda *a, **k: calls.append(a),
        )
    assert not calls and not paths.receipt.exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"exit_code": 2},
        {"cleanup": False},
        {"stop": "FULL_OWNED_ATTEMPT_DEADLINE"},
        {"create_output": False},
    ],
)
def test_virtual_child_failure_never_reports_success(case, kwargs):
    paths = destinations(case)
    prepare(case, paths)
    (code, record), _ = virtual_call(case, paths, **kwargs)
    assert code == 1 and record["status"] == "CPU_ARTIFACT_AUDIT_FAILED"
    assert (paths.receipt / "resources.json").is_file()


def test_virtual_supervisor_exception_preserves_reconciliation_receipt(case):
    paths = destinations(case)
    prepare(case, paths)
    with pytest.raises(RuntimeError, match="private supervisor"):
        virtual_call(case, paths, exception=True)
    assert prep.document(paths.receipt / "parent-exception.json")["requires_reconciliation"] is True
    assert (paths.receipt / "attempt-start.json").is_file()


@pytest.mark.parametrize(
    "index,mode",
    [(index, mode) for index in range(3) for mode in ("core_frozen_readout", "frozen_readout")],
)
def test_actual_immutable_producer_schema_is_derivable_without_tensor_decode(index, mode):
    ssl, downstream, kind = prep.PRODUCERS[index]
    source = (
        MAIN / f"src/marine_echo/training/{ssl if mode == 'core_frozen_readout' else downstream}.py"
    )
    config = defaults(ssl, "Config")
    report = {"mode": mode, "config": config, "bindings": {str(source): prep.digest(source)}}
    if index:
        report["architecture"] = prep.ARCHITECTURE
    assert prep.declared_kind(MAIN, report) == kind


def test_final_test_matrix_cannot_be_promoted_into_development_metadata(case):
    paths = destinations(case)
    with (
        changed(case.matrix, lambda value: value.update(role="final_test")),
        pytest.raises(ValueError, match="development"),
    ):
        prepare(case, paths)
    assert not paths.snapshot.exists()


def test_completion_report_required_despite_virtual_zero_exit(case):
    paths = destinations(case)
    prepare(case, paths)

    def launch(command, **kwargs):
        return SimpleNamespace(pid=123456)

    def supervise(child, **kwargs):
        paths.output.mkdir()
        return {
            "exit_code": 0,
            "stopped_for": None,
            "owned_tree_cleanup_verified": True,
            "elapsed_full_attempt_seconds": 0.1,
            "peak_process_rss_bytes": 1234,
        }

    code, record = owned.execute(
        case.root,
        paths.manifest,
        paths.snapshot,
        paths.output,
        paths.receipt,
        launcher=launch,
        supervisor=supervise,
    )
    assert code == 1 and record["status"] == "CPU_ARTIFACT_AUDIT_FAILED"
