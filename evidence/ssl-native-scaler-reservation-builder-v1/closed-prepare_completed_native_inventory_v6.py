"""Reserve all 43 neural endpoints of the fixed 47-method matrix, without decode.

This prepares metadata inputs only. It grants no fit, selection or numeric access
authority. Existing ancestry software owns the subsequent safe tensor audit.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import stat
from pathlib import Path

from native_train_scaler_reservation_v1 import reserve_train_scalers

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ("persistence", "seasonal24", "lightgbm", "chronos2_zero_shot")
REAL = "REAL_TRAIN_DEVELOPMENT_FIT"
ARCHITECTURE = "nonlinear_frequency_conditioned_v1"
PRODUCERS = (
    ("native_ssl", "native_downstream", "native_ssl_weights_only_inference_v1"),
    ("native_band_ssl", "native_band_downstream", "native_band_ssl_weights_only_inference_v1"),
    (
        "native_band_replication_ssl",
        "native_band_replication_downstream",
        "native_band_replication_ssl_weights_only_inference_v2",
    ),
)
JOURNALS = (
    ".pending",
    ".prefix-pending",
    ".band-pending",
    ".assessment-pending",
    ".reconciliation-pending",
)


def digest(path):
    with regular(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def regular(path):
    path = Path(os.path.abspath(path))
    for parent in [path, *path.parents]:
        info = parent.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Unsafe reparse/symlink identity")
    if not path.is_file():
        raise ValueError("Regular immutable input required")
    return path


def json_bytes(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON identity")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Nonfinite JSON metadata")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def document(path):
    return json_bytes(regular(path).read_bytes())


def fresh(path, root, category):
    path = Path(os.path.abspath(path))
    if not path.is_relative_to(root / category) or path.exists() or path.is_symlink():
        raise FileExistsError("Fresh root-owned destination required; never overwrite attempts")
    if not path.parent.is_dir():
        raise FileNotFoundError("Destination parent must already exist")
    for parent in path.parents:
        info = parent.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Unsafe destination parent")
    return path


def idle_ledger(root):
    """Read live ownership metadata only; never remove any lock or journal."""
    path = root / "orchestration/native_ssl_run_ledger_v1.json"
    raw = regular(path).read_bytes()
    value = json_bytes(raw)
    if not isinstance(value, dict) or not isinstance(value.get("runs"), list):
        raise TypeError("Explicit live scientific ledger required")
    if value.get("requires_reconciliation"):
        raise ValueError("Live ledger requires reconciliation")
    for run in value["runs"]:
        if (
            not isinstance(run, dict)
            or str(run.get("status", "")).startswith("RUNNING_")
            or run.get("requires_reconciliation")
        ):
            raise ValueError("Scientific tree must be idle and reconciled")
    lock = root / "evidence/ssl-builder-v1/gpu-owner.lock"
    if lock.exists() or lock.is_symlink():
        raise ValueError("Existing scientific ownership lock")
    journals = {path.parent / "all.pending"}
    for suffix in JOURNALS:
        journals.add(path.with_suffix(suffix))
        journals.add(path.parent / suffix)
        journals.update(path.parent.glob("*" + suffix))
    if any(p.exists() or p.is_symlink() for p in journals):
        raise ValueError("Pending scientific journal")
    return raw, value


def reservation_path(snapshot):
    return snapshot.with_name(snapshot.stem + ".reservation.json")


def dependencies(root):
    from marine_echo.evaluation.native_ancestry_inventory import required_sources

    return [
        *required_sources(),
        Path(__file__).resolve(),
        Path(__file__).with_name("execute_completed_native_inventory_owned_v6.py").resolve(),
        Path(__file__).with_name("native_train_scaler_reservation_v1.py").resolve(),
        *[
            root / "tools" / name
            for name in (
                "prepare_completed_native_inventory_v3.py",
                "execute_completed_native_inventory_owned_v3.py",
                "prepare_native_development_comparison_v5.py",
                "prepare_native_development_comparison_v4.py",
                "prepare_completed_native_inventory_v4.py",
                "execute_completed_native_inventory_owned_v4.py",
                "native_reference_supervisor.py",
            )
        ],
    ]


def declared_kind(root, report, *, cache=None):
    """Read the bound producer's literal saved schema; never inspect tensor codecs."""
    mode = report.get("mode", "core_frozen_readout")
    bound = report["bindings"]
    candidates = []
    for ssl, downstream, kind in PRODUCERS:
        producer = (
            root
            / f"src/marine_echo/training/{ssl if mode == 'core_frozen_readout' else downstream}.py"
        )
        if str(producer) in bound:
            if digest(producer) != bound[str(producer)]:
                raise ValueError("Changed artifact producer source")
            cache_key = (str(producer), bound[str(producer)])
            declared = cache.get(cache_key) if cache is not None else None
            if declared is None:
                tree = ast.parse(producer.read_bytes())
                run = next(
                    (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"),
                    None,
                )
                declared = (
                    {
                        v.value
                        for n in ast.walk(run)
                        if isinstance(n, ast.Dict)
                        for k, v in zip(n.keys, n.values, strict=True)
                        if isinstance(k, ast.Constant)
                        and k.value == "kind"
                        and isinstance(v, ast.Constant)
                    }
                    if run
                    else set()
                )
                if cache is not None:
                    cache[cache_key] = declared
            if kind not in declared:
                raise ValueError("Producer does not declare the expected safe inference schema")
            candidates.append(kind)
    config = report.get("core_config", report["config"])
    band = report.get("architecture") == ARCHITECTURE and config.get("architecture") == ARCHITECTURE
    if band:
        candidates = [k for k in candidates if k != PRODUCERS[0][2]]
    elif report.get("architecture") is not None or config.get("architecture") is not None:
        raise ValueError("Unknown or conflicting architecture")
    else:
        candidates = [k for k in candidates if k == PRODUCERS[0][2]]
    # V2 binds V1 as an immutable compatibility dependency, but V1 cannot emit V2.
    if PRODUCERS[2][2] in candidates:
        candidates = [PRODUCERS[2][2]]
    if len(candidates) != 1:
        raise ValueError("One typed actual producer schema required")
    return candidates[0]


def prepare(root, matrix_path, manifest_path, snapshot_path, output_path, *, source_paths=None):
    """Metadata-only helper; production CLI fixes root to its actual checkout."""
    from marine_echo.evaluation.native_ancestry_inventory import (
        _config,
        resolve_original_source_binding,
        validate_parent_graph,
    )

    root = Path(root).absolute()
    manifest_path = fresh(manifest_path, root, "orchestration")
    snapshot_path = fresh(snapshot_path, root, "evidence")
    output_path = fresh(output_path, root, "evidence")
    reservation = fresh(reservation_path(snapshot_path), root, "evidence")
    if len({manifest_path, snapshot_path, output_path, reservation}) != 4:
        raise ValueError("Separate fresh manifest/snapshot/output/reservation required")
    before, ledger = idle_ledger(root)
    matrix_path = regular(matrix_path)
    matrix = document(matrix_path)
    methods = matrix.get("methods")
    if (
        matrix.get("role") != "development"
        or not isinstance(methods, dict)
        or len(methods) != 47
        or not set(REFERENCES) <= methods.keys()
    ):
        raise ValueError(
            "Exactly47 explicit development methods including four named references required"
        )
    neural = {name: path for name, path in methods.items() if name not in REFERENCES}
    if len(neural) != 43 or any(not isinstance(n, str) or not n for n in neural):
        raise ValueError("Exactly43 explicit neural identities required")
    if any(not isinstance(p, str) for p in methods.values()) or len(set(methods.values())) != 47:
        raise ValueError("Unique exact prediction reservations required")
    bindings = {
        str(regular(p)): digest(p)
        for p in (dependencies(root) if source_paths is None else source_paths)
    }
    helper_source = Path(__file__).with_name("native_train_scaler_reservation_v1.py").resolve()
    bindings[str(helper_source)] = digest(helper_source)
    bindings[str(matrix_path)] = digest(matrix_path)
    ledger_path = root / "orchestration/native_ssl_run_ledger_v1.json"
    index_path = root / "evidence/ssl-research-v1/original-source-archives-v3/index.json"
    index = document(index_path)
    catalogs = index.get("catalogs")
    if (
        index.get("runs_inspected") != 35
        or index.get("scientific_approval") is not False
        or not isinstance(catalogs, list)
        or len(catalogs) != 3
        or len(set(catalogs)) != 3
    ):
        raise ValueError("Three preserved non-approving historical source catalogs required")
    bindings[str(index_path)] = digest(index_path)
    archives = []
    for name in catalogs:
        catalog = regular(name)
        record = document(catalog)
        if record.get("status") != "ORIGINAL_SOURCE_PRESERVED_NOT_COMPATIBILITY_APPROVAL" or digest(
            record["path"]
        ) != record.get("sha256"):
            raise ValueError("Historical source catalog byte identity differs")
        for p in (catalog, regular(record["path"]), regular(record["original_path"])):
            bindings[str(p)] = digest(p)
        archives.append(record)
    entries, reports, scaler_reservations = {}, {}, {}
    producer_cache, config_cache, original_binding_cache = {}, set(), {}
    folders = set()
    for name, prediction in neural.items():
        prediction = regular(prediction)  # Presence/identity only, never decode forecasts.
        folder = prediction.parent
        if (
            not folder.is_relative_to(root / "outputs")
            or folder in folders
            or prediction.name != "predictions.npz"
        ):
            raise ValueError("One distinct explicitly completed endpoint directory required")
        folders.add(folder)
        run_path = folder / "run.json"
        run = document(run_path)
        if (
            run.get("status") != "COMPLETED"
            or run.get("evidence_kind") != REAL
            or run.get("test_access") != "NOT_RUN"
            or run.get("historical_initial_weights", False) is not False
        ):
            raise ValueError(f"Actual complete locally fitted endpoint required: {name}")
        if not isinstance(run.get("bindings"), dict) or not run["bindings"]:
            raise ValueError("Original report bindings required")
        reviews = set()
        for record in ledger["runs"]:
            if (
                record.get("output")
                and record.get("review")
                and (root / record["output"]).absolute() == folder
            ):
                path = regular(root / record["review"])
                if digest(path) == run.get("review_sha256"):
                    reviews.add(path)
        configs = [
            regular(p)
            for p, h in run["bindings"].items()
            if Path(p).is_relative_to(root / "configs")
            and Path(p).suffix == ".json"
            and digest(p) == h
            and document(p) == run["config"]
        ]
        if len(reviews) != 1 or len(configs) != 1:
            raise ValueError(f"One exact original completed review/config must resolve: {name}")
        review_path = next(iter(reviews))
        review = document(review_path)
        mode = run.get("mode", "core_frozen_readout")
        if review.get("status") != (
            "APPROVED_PREFIT" if mode == "core_frozen_readout" else "APPROVED_DOWNSTREAM_PREFIT"
        ):
            raise ValueError("Actual typed original prefit status required")
        reviewer = review.get("reviewer_session_id")
        if not reviewer or reviewer.casefold() in {
            str(review.get("implementer_session_id")).casefold(),
            "01a0ef1a-b166-7f83-91b7-2a2aff7c1b10",
        }:
            raise ValueError("Distinct original reviewer required")
        if not isinstance(review.get("bindings"), dict) or any(
            review["bindings"].get(p) != h for p, h in run["bindings"].items()
        ):
            raise ValueError("Original review/report bindings conflict")
        kind = declared_kind(root, run, cache=producer_cache)
        core = run.get("core_config", run["config"])
        entry = {
            "directory": str(folder),
            "kind": kind,
            "method": core["method"],
            "seed": core["seed"],
            "mode": mode,
            "parent": None,
            "config_path": str(configs[0]),
            "review_path": str(review_path),
        }
        for config, downstream in [(core, False)] + (
            [(run["config"], True)] if mode != "core_frozen_readout" else []
        ):
            config_key = (
                kind,
                core["method"],
                core["seed"],
                mode,
                downstream,
                json.dumps(config, sort_keys=True, allow_nan=False),
            )
            if config_key not in config_cache:
                _config(config, entry, downstream=downstream)
                config_cache.add(config_key)
        for group in (run["bindings"], review["bindings"]):
            for p, expected in group.items():
                if Path(p) == ledger_path:
                    raise ValueError(
                        "Mutable ledger in original immutable review requires an original receipt; no alias rewriting"
                    )
                key = (p, expected, core["method"])
                if key not in original_binding_cache:
                    original_binding_cache[key] = resolve_original_source_binding(
                        p, expected, core["method"], archives
                    )
                resolved = original_binding_cache[key]
                bindings[str(resolved)] = expected
                bindings[str(regular(p))] = digest(p)
        scaler_reservations[name] = reserve_train_scalers(
            root, folder, run, kind, method=core["method"], seed=core["seed"], mode=mode
        )
        scalers = Path(scaler_reservations[name]["scalers_path"])
        entry["scalers_path"] = str(regular(scalers))
        for leaf in ("inference.pt", "membership.json"):
            if digest(folder / leaf) != run.get(leaf.split(".")[0] + "_sha256"):
                raise ValueError("Completed artifact hash differs from report")
        if (
            "selected_encoder_sha256" in run
            and digest(folder / "selected_encoder.pt") != run["selected_encoder_sha256"]
        ):
            raise ValueError("Selected encoder hash differs")
        for p in (
            run_path,
            review_path,
            configs[0],
            scalers,
            folder / "membership.json",
            folder / "inference.pt",
            folder / "selected_encoder.pt",
        ):
            bindings[str(regular(p))] = digest(p)
        entries[name], reports[name] = entry, run
    by_hash = {digest(Path(e["directory"]) / "run.json"): n for n, e in entries.items()}
    if len(by_hash) != 43:
        raise ValueError("Duplicate endpoint report aliases")
    for name, run in reports.items():
        mode = entries[name]["mode"]
        ancestry = run.get("supervised_ancestry")
        if mode == "core_frozen_readout":
            if ancestry is not None:
                raise ValueError("Root SSL/control has unexpected fitted ancestry")
            continue
        if (
            not isinstance(ancestry, dict)
            or ancestry.get("mode") != mode
            or ancestry.get("ssl_only") is not False
        ):
            raise ValueError("Exact supervised ancestry required")
        parent_hash = ancestry.get("ancestor_run_sha256", "MISSING")
        encoder_hash = ancestry.get("ancestor_encoder_sha256", "MISSING")
        if mode == "direct_end_to_end":
            if (
                parent_hash is not None
                or encoder_hash is not None
                or entries[name]["method"] != "direct"
            ):
                raise ValueError("Fresh supervised endpoint must have explicit null parents")
        else:
            parent = by_hash.get(parent_hash)
            if parent is None or encoder_hash != digest(
                Path(entries[parent]["directory"]) / "selected_encoder.pt"
            ):
                raise ValueError("Every exact fitted parent must be explicitly in the43")
            if any(entries[parent][key] != entries[name][key] for key in ("method", "seed")):
                raise ValueError("Fitted parent method/seed identity differs")
            entries[name]["parent"] = parent
    validate_parent_graph(entries)
    train = {
        "input_path": str(root / "data/processed/native_ssl_v1/train.npz"),
        "report_path": str(root / "data/processed/native_ssl_v1/train.json"),
        "identity_cohort_path": str(
            root / "orchestration/native_assessment_seed7_metadata_v1/train_cohort.json"
        ),
    }
    split = root / "configs/native_ssl_split_v1.json"
    for p in (split, *(Path(p) for p in train.values())):
        bindings[str(regular(p))] = digest(p)
    if any(
        run["bindings"].get(train["input_path"]) != bindings[train["input_path"]]
        or run["bindings"].get(str(split)) != bindings[str(split)]
        for run in reports.values()
    ):
        raise ValueError("Every endpoint must bind the declared original TRAIN and split")
    after, _ = idle_ledger(root)
    if after != before:
        raise ValueError("Live ledger changed during metadata capture; no snapshot written")
    with snapshot_path.open("xb") as stream:
        stream.write(before)
    bindings[str(snapshot_path)] = digest(snapshot_path)
    final, _ = idle_ledger(root)
    if final != before:
        raise ValueError(
            "Live ledger changed after capture; preserve snapshot, no completed manifest"
        )
    manifest = {
        "kind": "native_research_inventory_manifest_v1",
        "purpose": "LOCAL_METADATA_DERIVATION_ONLY",
        "execution": "ROOT_LOCAL_COMPLETED_METADATA_AUDIT",
        "evidence_kind": REAL,
        "device": "cpu",
        "owner_session_id": "01a0ef1a-b166-7f83-91b7-2a2aff7c1b10",
        "output_path": str(output_path),
        "split_path": str(split),
        "train": train,
        "endpoints": entries,
        "references": {},
        "bindings": bindings,
        "source_archives": [str(regular(p)) for p in catalogs],
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    closed, _ = idle_ledger(root)
    if closed != before:
        raise ValueError(
            "Live ledger changed at preparation close; preserve partial files without reservation"
        )
    facts = {
        "kind": "native_completed_inventory_preparation_v6",
        "status": "METADATA_RESERVED_NOT_APPROVAL",
        "manifest_path": str(manifest_path),
        "manifest_sha256": digest(manifest_path),
        "ledger_snapshot_path": str(snapshot_path),
        "ledger_snapshot_sha256": digest(snapshot_path),
        "matrix_path": str(matrix_path),
        "matrix_sha256": digest(matrix_path),
        "neural_count": 43,
        "method_count": 47,
        "neural_methods": sorted(entries),
        "references": {name: methods[name] for name in REFERENCES},
        "reference_inventory_status": "NOT_AUDITED_SEPARATE_SCHEMA_REQUIRED",
        "fit_authority": False,
        "final_selection": False,
        "numeric_decoding": False,
        "train_scaler_reservations": scaler_reservations,
    }
    with reservation.open("x", encoding="utf-8") as stream:
        json.dump(facts, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return facts


def main():
    if ROOT.name != "marine-echo-jepa":
        raise RuntimeError(
            "Actual metadata preparation is ROOT-only; builder uses private synthetic function fixtures"
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix", type=Path, default=ROOT / "orchestration/native_development_comparison_v5.json"
    )
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "orchestration/native_completed_inventory_v6.json"
    )
    parser.add_argument(
        "--ledger-snapshot",
        type=Path,
        default=ROOT / "evidence/ssl-research-v1/native-completed-inventory-ledger-v6.json",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "evidence/native-completed-inventory-v6"
    )
    args = parser.parse_args()
    print(json.dumps(prepare(ROOT, args.matrix, args.manifest, args.ledger_snapshot, args.output)))


if __name__ == "__main__":
    main()
