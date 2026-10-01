"""SYNTHETIC_CORRECTNESS_ONLY metadata guards; no real tensors or processes."""

from __future__ import annotations

import ast
import copy
import importlib
import json
import stat
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
HERE = BUILDER / "evidence/ssl-native-scaler-reservation-builder-v1"
sys.path.insert(0, str(MAIN / "src"))
sys.path.insert(0, str(BUILDER / "tools"))
prep = importlib.import_module("prepare_completed_native_inventory_v6")
owned = importlib.import_module("execute_completed_native_inventory_owned_v6")
scaler = importlib.import_module("native_train_scaler_reservation_v1")
dev = importlib.import_module("prepare_native_development_comparison_v5")


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
        if i not in (1, 2, 3, 6):
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
    patcher = pytest.MonkeyPatch()
    patcher.setattr(scaler, "TRAIN_SCALERS_SHA256", prep.digest(fallback))
    yield SimpleNamespace(
        root=root, matrix=matrix_path, ledger=ledger, folders=folders, sources=sources
    )
    patcher.undo()


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


def test_band_missing_duplicate_scalers_is_reserved_not_approved(case):
    paths = destinations(case)
    facts = prepare(case, paths)
    manifest = prep.document(paths.manifest)
    item = manifest["endpoints"]["neural06"]
    assert Path(item["scalers_path"]).parts[-2:] == ("shared_ssl_seed7_h96_cuda0", "scalers.json")
    assert facts["neural_count"] == 43 and facts["method_count"] == 47
    alias = facts["train_scaler_reservations"]["neural06"]
    assert alias["approval"] is alias["fit_authority"] is alias["numeric_access_authority"] is False
    assert alias["scalers_sha256"] == scaler.TRAIN_SCALERS_SHA256
    assert set(alias["saved_tensor_bindings"]) == {
        str(case.folders[6] / leaf) for leaf in ("inference.pt", "selected_encoder.pt")
    }


def endpoint(case, index, mode, *, own=False, method=None, seed=None):
    ssl, downstream, kind = prep.PRODUCERS[index]
    seed = seed if seed is not None else (23 if index == 2 else 7)
    method = (
        method
        if method is not None
        else ("direct" if mode == "direct_end_to_end" else "shared_ssl")
    )
    folder = case.root / "outputs/native_acoustic_ssl_v1" / ("private-endpoint-" + uuid.uuid4().hex)
    core = defaults(ssl, "Config")
    core.update(method=method, seed=seed)
    config = core
    if mode != "core_frozen_readout":
        config = defaults(downstream, "DownstreamConfig")
        config.update(
            method=method,
            seed=seed,
            mode=mode,
            updates=2000 if mode == "frozen_readout" else 3000,
            cadence=500 if mode == "frozen_readout" else 750,
        )
    for leaf in ("inference.pt", "selected_encoder.pt", "predictions.npz"):
        opaque(folder / leaf, ("SYNTHETIC_CORRECTNESS_ONLY_OPAQUE_" + leaf).encode())
    producer = (
        case.root
        / f"src/marine_echo/training/{ssl if mode == 'core_frozen_readout' else downstream}.py"
    )
    report = {
        "status": "COMPLETED",
        "evidence_kind": prep.REAL,
        "test_access": "NOT_RUN",
        "config": config,
        "bindings": {str(producer): prep.digest(producer)},
        "inference_sha256": prep.digest(folder / "inference.pt"),
        "selected_encoder_sha256": prep.digest(folder / "selected_encoder.pt"),
        "synthetic_fixture": True,
    }
    if index:
        report["architecture"] = prep.ARCHITECTURE
    if mode != "core_frozen_readout":
        report.update(
            mode=mode,
            core_config=core,
            supervised_ancestry={
                "mode": mode,
                "ssl_only": False,
                "ancestor_run_sha256": None if mode == "direct_end_to_end" else "a" * 64,
                "ancestor_encoder_sha256": None if mode == "direct_end_to_end" else "b" * 64,
            },
        )
    if own:
        opaque(
            folder / "scalers.json",
            (
                case.root / "outputs/native_acoustic_ssl_v1/shared_ssl_seed7_h96_cuda0/scalers.json"
            ).read_bytes(),
        )
    return SimpleNamespace(
        folder=folder, report=report, kind=kind, method=method, seed=seed, mode=mode
    )


def reserve(case, item):
    return scaler.reserve_train_scalers(
        case.root,
        item.folder,
        item.report,
        item.kind,
        method=item.method,
        seed=item.seed,
        mode=item.mode,
    )


@pytest.mark.parametrize(
    "index,mode",
    [(i, m) for i in range(3) for m in ("direct_end_to_end", "frozen_readout", "full_finetune")],
)
def test_typed_original_band_v1_v2_downstream_aliases(case, index, mode):
    item = endpoint(case, index, mode)
    result = reserve(case, item)
    assert result["producer_kind"] == item.kind and result["mode"] == mode
    assert result["standalone_scalers_present"] is False
    assert result["alias"] == "ORIGINAL_SHARED7_TRAIN_SCALERS_PENDING_EMBEDDED_EQUALITY_AUDIT"
    assert result["embedded_selected_and_inference_scaler_equality"].startswith("NOT_ESTABLISHED")
    assert not (item.folder / "scalers.json").exists()


@pytest.mark.parametrize("index", range(3))
def test_core_requires_own_scalers_even_when_canonical_exists(case, index):
    item = endpoint(case, index, "core_frozen_readout")
    with pytest.raises(ValueError, match="own standalone"):
        reserve(case, item)
    opaque(
        item.folder / "scalers.json",
        (
            case.root / "outputs/native_acoustic_ssl_v1/shared_ssl_seed7_h96_cuda0/scalers.json"
        ).read_bytes(),
    )
    result = reserve(case, item)
    assert result["alias"] is None and result["standalone_scalers_present"] is True


@pytest.mark.parametrize(
    "damage",
    [
        "kind",
        "mode",
        "seed",
        "seed_bool",
        "method",
        "status",
        "synthetic",
        "test",
        "architecture",
        "source",
        "core_identity",
        "saved_identity",
        "saved_mode",
        "ancestry",
        "parent",
        "inference",
        "selected",
        "scaler_declared",
        "historical",
    ],
)
def test_unsafe_alias_rejected_without_model_decode(case, damage):
    item = endpoint(case, 2, "direct_end_to_end")
    report = item.report
    if damage in ("kind", "mode", "seed", "seed_bool", "method"):
        setattr(
            item,
            damage if damage != "seed_bool" else "seed",
            {
                "kind": "native_ssl_resume_v1",
                "mode": "unknown",
                "seed": 99,
                "seed_bool": True,
                "method": "unknown",
            }[damage],
        )
    elif damage == "status":
        report["status"] = "RUNNING_CPU_FIT"
    elif damage == "synthetic":
        report["evidence_kind"] = "SYNTHETIC_CORRECTNESS_ONLY"
    elif damage == "test":
        report["test_access"] = "ACCESSED"
    elif damage == "architecture":
        report["architecture"] = "linear"
    elif damage == "source":
        report["bindings"] = {}
    elif damage == "core_identity":
        report["core_config"]["seed"] = 13
    elif damage == "saved_identity":
        report["config"]["method"] = "masked_ssl"
    elif damage == "saved_mode":
        report["config"]["mode"] = "frozen_readout"
    elif damage == "ancestry":
        report["supervised_ancestry"]["ssl_only"] = True
    elif damage == "parent":
        report["supervised_ancestry"]["ancestor_run_sha256"] = "a" * 64
    elif damage in ("inference", "selected"):
        report["inference_sha256" if damage == "inference" else "selected_encoder_sha256"] = (
            "f" * 64
        )
    elif damage == "scaler_declared":
        report["scalers_sha256"] = "f" * 64
    else:
        report["historical_initial_weights"] = True
    with pytest.raises((ValueError, FileNotFoundError)):
        reserve(case, item)


def test_canonical_wrong_bytes_and_missing_file_refused(case, monkeypatch):
    item = endpoint(case, 2, "direct_end_to_end")
    fallback = case.root / "outputs/native_acoustic_ssl_v1/shared_ssl_seed7_h96_cuda0/scalers.json"
    with (
        changed(fallback, lambda value: value["channel_mean"].__setitem__(0, 5)),
        pytest.raises(ValueError, match="unchanged"),
    ):
        reserve(case, item)
    real = scaler.regular

    def absent(path):
        if Path(path) == fallback:
            raise FileNotFoundError("Private missing canonical fixture")
        return real(path)

    monkeypatch.setattr(scaler, "regular", absent)
    with pytest.raises(FileNotFoundError):
        reserve(case, item)


def test_no_arbitrary_fallback_path_or_foreign_endpoint(case):
    item = endpoint(case, 2, "direct_end_to_end")
    with pytest.raises(TypeError):
        scaler.reserve_train_scalers(
            case.root,
            item.folder,
            item.report,
            item.kind,
            method=item.method,
            seed=item.seed,
            mode=item.mode,
            fallback_path=case.root / "foreign.json",
        )
    item.folder = case.root / "foreign"
    with pytest.raises(ValueError, match="local native"):
        reserve(case, item)


def test_foreign_standalone_scaler_bytes_are_not_silently_aliased(case):
    item = endpoint(case, 1, "core_frozen_readout", own=True)
    with (
        changed(
            item.folder / "scalers.json", lambda value: value["target_mean"].__setitem__(0, 123)
        ),
        pytest.raises(ValueError, match="original pin"),
    ):
        reserve(case, item)


@pytest.mark.parametrize("reparse", [False, True])
def test_reparse_and_symlink_refusal_without_creating_links(case, monkeypatch, reparse):
    item = endpoint(case, 2, "direct_end_to_end", own=True)
    target = item.folder / "scalers.json"
    original = Path.lstat

    def unsafe(path, *args, **kwargs):
        if path == target:
            return SimpleNamespace(
                st_mode=stat.S_IFREG if reparse else stat.S_IFLNK,
                st_file_attributes=0x400 if reparse else 0,
            )
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", unsafe)
    with pytest.raises(ValueError, match="Unsafe"):
        reserve(case, item)


@pytest.mark.parametrize("count", [40, 42, 44])
def test_inventory_still_requires_all_43_neural_endpoints(case, count):
    paths = destinations(case)

    def alter(matrix):
        if count < 43:
            for index in range(count, 43):
                matrix["methods"].pop(f"neural{index:02d}")
        else:
            matrix["methods"]["extra"] = "not-a-completed-endpoint"

    with changed(case.matrix, alter), pytest.raises(ValueError, match="Exactly47"):
        prepare(case, paths)
    assert not paths.manifest.exists() and not paths.snapshot.exists()


def test_inventory_routes_every_endpoint_through_shared_helper(case, monkeypatch):
    calls = []
    original = prep.reserve_train_scalers

    def trace(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(prep, "reserve_train_scalers", trace)
    paths = destinations(case)
    raw = case.ledger.read_bytes()
    facts = prepare(case, paths)
    manifest = prep.document(paths.manifest)
    assert len(calls) == len(facts["train_scaler_reservations"]) == 43
    assert paths.snapshot.read_bytes() == raw
    assert str(case.ledger) not in manifest["bindings"]
    assert manifest["bindings"][str(Path(scaler.__file__).resolve())] == prep.digest(
        Path(scaler.__file__)
    )
    assert manifest["endpoints"]["neural06"]["parent"] == "neural07"
    assert facts["numeric_decoding"] is False


@pytest.mark.parametrize("phase", [2, 3, 4])
def test_changed_live_ledger_capture_keeps_original_fail_closed_behavior(case, monkeypatch, phase):
    original = prep.idle_ledger
    count = 0

    def capture(root):
        nonlocal count
        count += 1
        raw, value = original(root)
        return (raw + b"\n", value) if count == phase else (raw, value)

    monkeypatch.setattr(prep, "idle_ledger", capture)
    paths = destinations(case)
    with pytest.raises(ValueError, match="changed"):
        prepare(case, paths)
    assert not prep.reservation_path(paths.snapshot).exists()


def test_original_guard_rejects_wrong_parent_before_snapshot(case):
    paths = destinations(case)
    with (
        changed(
            case.folders[6] / "run.json",
            lambda value: value["supervised_ancestry"].update(ancestor_run_sha256="c" * 64),
        ),
        pytest.raises(ValueError, match="parent"),
    ):
        prepare(case, paths)
    assert not paths.snapshot.exists()


@pytest.fixture(scope="module")
def comparison_case(case):
    # New private metadata endpoints in the current delivery, never ROOT runs.
    records = []
    for method in ("shared_ssl", "masked_ssl", "permuted_ssl", "random_frozen"):
        records.append(
            (
                f"band_{method}_short_probe",
                f"band_{method}_seed7_h96_reviewed",
                method,
                7,
                "core_frozen_readout",
                1,
            )
        )
    records.append(
        (
            "band_direct_end_to_end",
            "band_direct_end_to_end_seed7_h96_reviewed",
            "direct",
            7,
            "direct_end_to_end",
            1,
        )
    )
    for method, modes in (
        ("shared_ssl", ("frozen_readout", "full_finetune")),
        ("masked_ssl", ("frozen_readout", "full_finetune")),
        ("permuted_ssl", ("frozen_readout",)),
        ("random_frozen", ("frozen_readout",)),
    ):
        for mode in modes:
            name = f"band_{method}_{mode}"
            records.append(
                (
                    name,
                    f"{name}_seed7_h96" + ("_native_retry01" if method == "random_frozen" else ""),
                    method,
                    7,
                    mode,
                    1,
                )
            )
    for seed in (13, 23):
        records.append(
            (
                f"band_shared_ssl_short_probe_seed{seed}",
                f"band_shared_ssl_seed{seed}_h96_replication_v2",
                "shared_ssl",
                seed,
                "core_frozen_readout",
                2,
            )
        )
        records.append(
            (
                f"band_direct_end_to_end_seed{seed}",
                f"band_direct_end_to_end_seed{seed}_h96_replication_v2"
                + ("_ownership_retry01" if seed == 13 else ""),
                "direct",
                seed,
                "direct_end_to_end",
                2,
            )
        )
        for mode in ("frozen_readout", "full_finetune"):
            records.append(
                (
                    f"band_shared_ssl_{mode}_seed{seed}",
                    f"band_shared_ssl_{mode}_seed{seed}_h96_replication_v3",
                    "shared_ssl",
                    seed,
                    mode,
                    2,
                )
            )
    assert len(records) == 19
    folders = {}
    for name, directory, method, seed, mode, index in records:
        item = endpoint(
            case, index, mode, own=mode == "core_frozen_readout", method=method, seed=seed
        )
        destination = case.root / "outputs/native_acoustic_ssl_v1" / directory
        destination.mkdir()
        # Only tiny private opaque/JSON fixtures copied; nothing numerical/public.
        for path in item.folder.iterdir():
            opaque(destination / path.name, path.read_bytes())
        write(destination / "run.json", item.report)
        write(destination / "membership.json", {"fixture": "SYNTHETIC_CORRECTNESS_ONLY"})
        folders[name] = destination
    methods = {f"original{i:02d}": str(case.folders[i] / "predictions.npz") for i in range(24)}
    for name in prep.REFERENCES:
        path = case.root / "references" / name / "predictions.npz"
        opaque(path, b"SYNTHETIC_CORRECTNESS_ONLY_NOT_FORECASTS")
        methods[name] = str(path)
    original = case.root / "orchestration/native_development_comparison_v3.json"
    write(
        original,
        {
            "role": "development",
            "methods": methods,
            "reference": "lightgbm",
            "fixture": "SYNTHETIC_CORRECTNESS_ONLY",
        },
    )
    previous = case.root / "evidence/ssl-research-v1/development-comparison-v3-review-final.json"
    bindings = {p: prep.digest(p) for p in methods.values()}
    write(
        previous,
        {"bindings": bindings, "fixture": "SYNTHETIC_CORRECTNESS_ONLY_VIRTUAL_PRIOR_BINDINGS"},
    )
    for name in (
        "prepare_completed_native_inventory_v3.py",
        "execute_completed_native_inventory_owned_v3.py",
        "prepare_native_development_comparison_v5.py",
        "prepare_native_development_comparison_v4.py",
        "prepare_completed_native_inventory_v4.py",
        "execute_completed_native_inventory_owned_v4.py",
    ):
        opaque(case.root / "tools" / name, b"# SYNTHETIC_CORRECTNESS_ONLY unexecuted dependency\n")
    return SimpleNamespace(
        root=case.root, folders=folders, original=original, previous=previous, bindings=bindings
    )


@pytest.mark.parametrize(
    "damage",
    [
        "missing_three",
        "incomplete",
        "wrong_identity",
        "original_27",
        "wrong_reference",
        "wrong_role",
        "changed_original_binding",
    ],
)
def test_development_no_missing_endpoint_fallthrough_or_28_guard_waiver(
    comparison_case, monkeypatch, damage
):
    target = comparison_case.root / "orchestration/native_development_comparison_v5.json"
    if damage == "missing_three":
        original = dev.document
        blocked = {
            comparison_case.folders[name] / "run.json"
            for name in (
                "band_direct_end_to_end_seed23",
                "band_shared_ssl_frozen_readout_seed23",
                "band_shared_ssl_full_finetune_seed23",
            )
        }

        def absent(path):
            if Path(path) in blocked:
                raise FileNotFoundError("SYNTHETIC_MISSING_THREE_FIXED_ENDPOINTS")
            return original(path)

        monkeypatch.setattr(dev, "document", absent)
        with pytest.raises(FileNotFoundError):
            dev.prepare(comparison_case.root)
    else:
        path = comparison_case.original
        if damage in ("incomplete", "wrong_identity"):
            path = comparison_case.folders["band_direct_end_to_end_seed23"] / "run.json"
        elif damage == "changed_original_binding":
            path = comparison_case.previous

        def alter(value):
            if damage == "incomplete":
                value["status"] = "RUNNING_CUDA"
            elif damage == "wrong_identity":
                value["config"]["method"] = "shared_ssl"
            elif damage == "original_27":
                value["methods"].pop("original23")
            elif damage == "wrong_reference":
                value["methods"]["unknown_external"] = value["methods"].pop("chronos2_zero_shot")
            elif damage == "wrong_role":
                value["role"] = "final_test"
            else:
                value["bindings"][next(iter(value["bindings"]))] = "0" * 64

        with changed(path, alter), pytest.raises(ValueError):
            dev.prepare(comparison_case.root)
    assert not target.exists()
    assert not (
        comparison_case.root / "orchestration/native_development_comparison_admission_v5.json"
    ).exists()


def test_development_preparation_actually_reserves_19_endpoints_and_13_aliases(
    comparison_case, monkeypatch
):
    calls = []
    original = dev.reserve_train_scalers

    def trace(*args, **kwargs):
        receipt = original(*args, **kwargs)
        calls.append(receipt)
        return receipt

    monkeypatch.setattr(dev, "reserve_train_scalers", trace)
    proposal = dev.prepare(comparison_case.root)
    assert proposal["methods"] == 47 and len(calls) == 19
    assert sum(c["alias"] is not None for c in calls) == 13
    assert all(
        proposal["bindings"][path] == expected
        for path, expected in comparison_case.bindings.items()
    )
    assert len(proposal["train_scaler_reservations"]) == 19
    manifest = prep.document(proposal["manifest_path"])
    assert len(manifest["methods"]) - len(prep.REFERENCES) == 43
    assert set(prep.REFERENCES) <= manifest["methods"].keys()
    assert proposal["status"] == "PROPOSED_COMPARISON_RECONSTRUCTION_NOT_APPROVAL"
    assert proposal["fitting"] is proposal["final_numeric_access"] is False
    assert all(
        not (comparison_case.folders[cname] / "scalers.json").exists()
        for cname, record in proposal["train_scaler_reservations"].items()
        if record["alias"] is not None
    )
    with pytest.raises(FileExistsError):
        dev.prepare(comparison_case.root)


@pytest.mark.parametrize("blocker", ["running", "pending", "lock"])
def test_new_inventory_idle_gates_refuse_before_mutation(case, blocker):
    paths = destinations(case)
    if blocker == "running":
        with (
            changed(case.ledger, lambda v: v["runs"].append({"status": "RUNNING_CPU_FIT"})),
            pytest.raises(ValueError),
        ):
            prepare(case, paths)
    else:
        marker = case.root / (
            "orchestration/all.pending"
            if blocker == "pending"
            else "evidence/ssl-builder-v1/gpu-owner.lock"
        )
        opaque(marker, b"SYNTHETIC_CORRECTNESS_ONLY_TEST_OWNED")
        try:
            with pytest.raises(ValueError):
                prepare(case, paths)
        finally:
            marker.unlink()  # Only this new private test marker, never ROOT ownership.
    assert not paths.manifest.exists() and not paths.snapshot.exists()


def test_new_owned_executor_enforces_helper_binding_before_virtual_launch(case):
    paths = destinations(case)
    prepare(case, paths)

    def remove(value):
        value["bindings"].pop(str(Path(scaler.__file__).resolve()))

    # Restage the private reservation hash so the helper gate itself is tested.
    with (
        changed(paths.manifest, remove),
        changed(
            prep.reservation_path(paths.snapshot),
            lambda value: value.update(manifest_sha256=prep.digest(paths.manifest)),
        ),
        pytest.raises(ValueError, match="bindings"),
    ):
        owned.execute(
            case.root,
            paths.manifest,
            paths.snapshot,
            paths.output,
            paths.receipt,
            launcher=lambda *a, **k: pytest.fail("Must not launch"),
        )
    assert not paths.receipt.exists()


def test_owned_virtual_cpu_success_preserves_600s_22gib_and_exact43(case):
    paths = destinations(case)
    prepare(case, paths)

    def launch(command, **kwargs):
        assert "prepare_native_research_inventory.py" in command[2]
        paths.output.mkdir()
        write(
            paths.output / "inventory.json",
            {
                "kind": "native_research_inventory_v1",
                "status": "DERIVED_METADATA_NOT_FINAL_SELECTION",
                "manifest_sha256": prep.digest(paths.manifest),
                "models": {name: {} for name in prep.document(paths.manifest)["endpoints"]},
                "numeric_corpus_decoded": False,
                "model_constructed": False,
                "fixture": "SYNTHETIC_CORRECTNESS_ONLY_VIRTUAL_PROCESS_NOT_ACTUAL_AUDIT",
            },
        )
        return SimpleNamespace(pid=123456789)

    def supervise(child, **kwargs):
        assert kwargs["deadline_seconds"] == 600 and kwargs["rss_limit_bytes"] == 22 * 1024**3
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
    assert code == 0 and record["kind"] == "native_completed_inventory_owned_attempt_v6"
    assert record["device"] == "cpu" and record["fitting"] is False
