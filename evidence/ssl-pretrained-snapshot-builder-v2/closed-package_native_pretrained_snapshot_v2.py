"""Copy seven fixed pretrained endpoints and inference source, never fit or decode.

Production CLI is ROOT-only. Private metadata/opaque-byte fixtures exercise the
reservation/copy policy without loading public trained tensors.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import stat
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1_ARCHIVE_SHA256 = "ed1ba345a6fe3f04642d5bb2d8c32ef33f67132009295a6bca38cd73d8a3994b"
BAND23_RUN_SHA256 = "46f918e02923b715757fec0e036b622c05d25f4ad5554e69ee2b49e745aabcc5"
ARCHITECTURE = "nonlinear_frequency_conditioned_v1"
LEGACY_KIND = "native_ssl_weights_only_inference_v1"
BAND_KIND = "native_band_ssl_weights_only_inference_v1"
REPLICATION_KIND = "native_band_replication_ssl_weights_only_inference_v2"
SPECS = (
    ("shared_ssl_seed7_h96_cuda0", "shared_ssl", 7, LEGACY_KIND),
    ("cf_jepa_seed7_h96_deterministic", "cf_jepa", 7, LEGACY_KIND),
    ("cf_jepa_seed13_h96_replication", "cf_jepa", 13, LEGACY_KIND),
    ("cf_jepa_seed23_h96_replication", "cf_jepa", 23, LEGACY_KIND),
    ("band_shared_ssl_seed7_h96_reviewed", "shared_ssl", 7, BAND_KIND),
    ("band_shared_ssl_seed13_h96_replication_v2", "shared_ssl", 13, REPLICATION_KIND),
    ("band_shared_ssl_seed23_h96_replication_v2", "shared_ssl", 23, REPLICATION_KIND),
)
LEAVES = ("selected_encoder.pt", "inference.pt", "scalers.json", "membership.json", "run.json")
ENTRIES = (
    "native_encoder.py",
    "native_latent.py",
    "native_acoustic.py",
    "native_band_acoustic.py",
    "native_band_replication_acoustic.py",
    "native_band_replication_encoder.py",
    "native_band_replication_latent.py",
)
JOURNALS = (
    ".pending",
    ".prefix-pending",
    ".band-pending",
    ".assessment-pending",
    ".reconciliation-pending",
)


def regular(path):
    path = Path(os.path.abspath(path))
    for parent in (path, *path.parents):
        info = parent.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Unsafe source identity")
    if not path.is_file():
        raise ValueError("Regular immutable file required")
    return path


def digest(path):
    with regular(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def document(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate metadata key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Nonfinite metadata")

    return json.loads(regular(path).read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)


def fresh(path, root, category):
    path = Path(os.path.abspath(path))
    if not path.is_relative_to(root / category) or path.exists() or path.is_symlink():
        raise FileExistsError("Preserve every historical snapshot and QA attempt")
    if not path.parent.is_dir():
        raise FileNotFoundError("Destination parent must already exist")
    for parent in path.parents:
        info = parent.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Unsafe destination identity")
    return path


def idle(root):
    path = root / "orchestration/native_ssl_run_ledger_v1.json"
    ledger = document(path)
    if not isinstance(ledger.get("runs"), list):
        raise TypeError("Explicit scientific ledger required")
    if ledger.get("requires_reconciliation") or any(
        not isinstance(r, dict)
        or str(r.get("status", "")).startswith("RUNNING_")
        or r.get("requires_reconciliation")
        for r in ledger["runs"]
    ):
        raise ValueError("Scientific ownership must be idle and reconciled")
    lock = root / "evidence/ssl-builder-v1/gpu-owner.lock"
    pending = {path.parent / "all.pending"}
    for suffix in JOURNALS:
        pending.update((path.with_suffix(suffix), path.parent / suffix))
        pending.update(path.parent.glob("*" + suffix))
    if lock.exists() or lock.is_symlink() or any(p.exists() or p.is_symlink() for p in pending):
        raise ValueError("Preserve active owner locks and pending journals")


def source_closure(root):
    """Static local imports only, including namespace packages and V1/V2 Band."""
    package = root / "src/marine_echo"
    pending = [package / "inference" / name for name in ENTRIES]
    found = set()
    while pending:
        path = regular(pending.pop())
        if path in found:
            continue
        found.add(path)
        for parent in (path.parent, *path.parent.parents):
            if parent.is_relative_to(package) and (parent / "__init__.py").is_file():
                pending.append(parent / "__init__.py")
        for node in ast.walk(ast.parse(path.read_bytes())):
            if isinstance(node, ast.Import):
                imports = [(a.name, []) for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = ["marine_echo", *path.parent.relative_to(package).parts]
                    base = ".".join(parts[: len(parts) - node.level + 1] + ([base] if base else []))
                imports = [(base, [a.name for a in node.names])]
            else:
                continue
            for name, members in imports:
                if name != "marine_echo" and not name.startswith("marine_echo."):
                    continue
                candidate = root / "src" / Path(*name.split("."))
                file = candidate.with_suffix(".py")
                if file.is_file():
                    pending.append(file)
                elif candidate.is_dir():
                    initializer = candidate / "__init__.py"
                    if initializer.is_file():
                        pending.append(initializer)
                    for member in members:
                        submodule = candidate / member
                        if submodule.with_suffix(".py").is_file():
                            pending.append(submodule.with_suffix(".py"))
                        elif submodule.is_dir() and (submodule / "__init__.py").is_file():
                            pending.append(submodule / "__init__.py")
                        elif member != "*" and not initializer.is_file():
                            raise ValueError("Missing imported namespace source")
                else:
                    raise ValueError("Missing static local inference dependency")
    return sorted(found)


def scalers(value):
    expected = {"channel_mean": 4, "channel_std": 4, "target_mean": 3, "target_std": 3}
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("Exact original TRAIN scalers required")
    for key, size in expected.items():
        values = value[key]
        if (
            not isinstance(values, list)
            or len(values) != size
            or any(
                type(v) not in (int, float)
                or not math.isfinite(v)
                or (key.endswith("std") and v <= 0)
                for v in values
            )
        ):
            raise ValueError("Finite positive TRAIN scaler identity required")
    return value


def selected_kind(kind):
    return {
        LEGACY_KIND: "native_ssl_selected_encoder_v1",
        BAND_KIND: "native_band_ssl_selected_encoder_v1",
        REPLICATION_KIND: "native_band_replication_ssl_selected_encoder_v2",
    }[kind]


def build_plan(root, folder, archive, receipt_path):
    root = Path(root).absolute()
    folder, archive, receipt_path = (
        fresh(p, root, category)
        for p, category in ((folder, "outputs"), (archive, "outputs"), (receipt_path, "evidence"))
    )
    if archive != folder.with_suffix(".zip"):
        raise ValueError("Versioned archive must match the new snapshot directory")
    idle(root)
    v1 = root / "outputs/native_pretrained_model_snapshot_v1"
    if digest(v1.with_suffix(".zip")) != V1_ARCHIVE_SHA256:
        raise ValueError("Original V1 immutable archive identity changed")
    previous = document(v1 / "manifest.json")
    original = previous.get("models")
    expected = {(name, method, seed) for name, method, seed, _ in SPECS[:4]}
    if (
        previous.get("kind") != "native_pretrained_model_snapshot_v1"
        or previous.get("status") != "INCOMPLETE_RESEARCH_SNAPSHOT"
        or not isinstance(original, list)
        or len(original) != 4
        or {(m.get("id"), m.get("method"), m.get("seed")) for m in original} != expected
    ):
        raise ValueError("Exact four original V1 pretrained identities required")
    replay_path = (
        root / "evidence/ssl-research-v1/Unicode-actual-latent-Álvaro-cpu-replay-v1/completion.json"
    )
    closeout_path = (
        root / "evidence/ssl-research-v1/native-latent-actual-cpu-replay-closeout-v1.json"
    )
    replay, closeout = document(replay_path), document(closeout_path)
    if closeout.get("actual_cli_exit_code") != 0 or closeout.get("completion_sha256") != digest(
        replay_path
    ):
        raise ValueError("Original four-model actual CPU replay provenance required")
    replay_records = replay.get("records", [])
    if len(replay_records) != 4 or {r.get("run") for r in replay_records} != {
        s[0] for s in SPECS[:4]
    }:
        raise ValueError("Exact original four CPU replay records required")
    payloads, bindings, models = {}, {}, []

    def add(destination, source):
        source = regular(source)
        if destination in payloads and payloads[destination] != source:
            raise ValueError("No duplicate payload identities")
        payloads[destination] = source
        bindings[str(source)] = digest(source)

    add("provenance/v1-manifest.json", v1 / "manifest.json")
    add("provenance/v1-model-card.md", v1 / "MODEL_CARD.md")
    bindings[str(v1.with_suffix(".zip"))] = V1_ARCHIVE_SHA256
    stats = None
    training_identity = None
    reports = []
    base = root / "outputs/native_acoustic_ssl_v1"
    for name, method, seed, kind in SPECS:
        parent = base / name
        report = document(parent / "run.json")
        config = report.get("config", {})
        if (
            report.get("status") != "COMPLETED"
            or report.get("evidence_kind") != "REAL_TRAIN_DEVELOPMENT_FIT"
            or report.get("test_access") != "NOT_RUN"
            or report.get("mode") is not None
            or report.get("supervised_ancestry") is not None
            or config.get("method") != method
            or type(config.get("seed")) is not int
            or config["seed"] != seed
            or (
                config.get("history"),
                config.get("pretrain_updates"),
                config.get("readout_updates"),
            )
            != (96, 6000, 500)
            or report.get("pretrain_steps") != 6000
            or report.get("readout_steps_all_probes") != 2000
            or report.get("inference_sha256") != digest(parent / "inference.pt")
            or report.get("membership_sha256") != digest(parent / "membership.json")
        ):
            raise ValueError(f"Exact completed pretrained endpoint required: {name}")
        if kind != LEGACY_KIND and (
            report.get("architecture") != ARCHITECTURE
            or config.get("architecture") != ARCHITECTURE
            or (
                config.get("width"),
                config.get("latent"),
                config.get("blocks"),
                config.get("heads"),
            )
            != (192, 64, 4, 4)
        ):
            raise ValueError("Native Band architecture/version required")
        if (
            seed == 23
            and kind == REPLICATION_KIND
            and (
                digest(parent / "run.json") != BAND23_RUN_SHA256
                or report.get("selected_pretrain_step") != 6000
            )
        ):
            raise ValueError("Owner-pinned actual completed Band23 report required")
        if "selected_encoder_sha256" in report and report["selected_encoder_sha256"] != digest(
            parent / "selected_encoder.pt"
        ):
            raise ValueError("Selected encoder bytes changed")
        if kind == LEGACY_KIND:
            record = next(r for r in replay_records if r["run"] == name)
            if (
                record.get("artifact_sha256") != digest(parent / "inference.pt")
                or record.get("selected_encoder_cpu_replay") != "BITIDENTICAL"
                or record.get("saved_forward_head_cpu_replay") != "BITIDENTICAL"
            ):
                raise ValueError("Original actual replay inference bytes differ")
            for leaf in LEAVES:
                if previous.get("source_bindings", {}).get(str(parent / leaf)) != digest(
                    parent / leaf
                ):
                    raise ValueError("Preserved V1 original model bytes differ")
        current_stats = scalers(document(parent / "scalers.json"))
        if stats is not None and current_stats != stats:
            raise ValueError("All seven must retain the same original TRAIN scalers")
        stats = current_stats
        member = document(parent / "membership.json")
        identity = tuple(
            tuple(member.get(key, []))
            for key in ("train_row_ids", "train_deployments", "train_archive_sha256")
        )
        if (
            not identity[0]
            or len({len(v) for v in identity}) != 1
            or len(set(zip(identity[1], identity[0], strict=True))) != len(identity[0])
            or (training_identity is not None and identity != training_identity)
        ):
            raise ValueError("Identical complete original TRAIN membership required")
        training_identity = identity
        configs = [
            regular(p)
            for p, h in report.get("bindings", {}).items()
            if Path(p).is_relative_to(root / "configs")
            and Path(p).suffix == ".json"
            and digest(p) == h
            and document(p) == config
        ]
        if len(configs) != 1:
            raise ValueError("One exact original full config file required")
        add("configs/" + name + ".json", configs[0])
        for leaf in LEAVES:
            add("models/" + name + "/" + leaf, parent / leaf)
        models.append(
            {
                "id": name,
                "method": method,
                "seed": seed,
                "inference_kind": kind,
                "selected_kind": selected_kind(kind),
                "selected_encoder": "models/" + name + "/selected_encoder.pt",
                "latent_inference": "models/" + name + "/inference.pt",
                "scalers": "models/" + name + "/scalers.json",
                "training_kind": "SSL_PRETRAINED_SHORT_SELECTION_READOUT",
                "latent_semantics": "ONLINE_ORDINAL_ZONES_EMA_ENCODING_FULL_H96_OUTSIDE_CROP_SUPPORT"
                if method == "cf_jepa"
                else "NATIVE_OFFSETS_1_3_6_FOUR_INTERVAL_LATENT_BLOCKS",
            }
        )
        reports.append(report)
    for source in source_closure(root):
        add(source.relative_to(root).as_posix(), source)
    index_path = root / "evidence/ssl-research-v1/original-source-archives-v3/index.json"
    index = document(index_path)
    if (
        index.get("scientific_approval") is not False
        or len(index.get("catalogs", [])) != 3
        or len(set(index["catalogs"])) != 3
    ):
        raise ValueError("Three preserved non-approving historical source catalogs required")
    add("provenance/original-source-index.json", index_path)
    catalogs = []
    for path in index["catalogs"]:
        record = document(path)
        if record.get("status") != "ORIGINAL_SOURCE_PRESERVED_NOT_COMPATIBILITY_APPROVAL" or digest(
            record["path"]
        ) != record.get("sha256"):
            raise ValueError("Historical source archive bytes differ")
        add("provenance/original-source-catalogs/" + Path(path).name, path)
        catalogs.append(record)
    historical = []
    for model, report in zip(models, reports, strict=True):
        for path, expected_sha in report.get("bindings", {}).items():
            original_path = Path(path)
            if original_path.suffix != ".py" or not (
                original_path.is_relative_to(root / "src/marine_echo")
                or original_path.is_relative_to(root / "tools")
            ):
                continue
            current = digest(original_path) if original_path.is_file() else None
            entry = {
                "model": model["id"],
                "original_path": path,
                "original_sha256": expected_sha,
                "current_sha256": current,
                "compatibility_approval": False,
                "source_executed": False,
            }
            if current == expected_sha:
                destination = (
                    "provenance/original-sources/" + expected_sha + "-" + original_path.name
                )
                add(destination, original_path)
                entry.update(status="EXACT_ORIGINAL_BYTES_PRESERVED", copied_path=destination)
            else:
                matches = [
                    r
                    for r in catalogs
                    if r.get("original_path") == path
                    and r.get("sha256") == expected_sha
                    and model["method"] in r.get("methods", [])
                ]
                if len(matches) > 1:
                    raise ValueError("Ambiguous historical source version")
                if matches:
                    destination = (
                        "provenance/original-sources/" + expected_sha + "-" + original_path.name
                    )
                    add(destination, matches[0]["path"])
                    entry.update(
                        status="EXACT_HISTORICAL_VERSION_PRESERVED_NOT_COMPATIBILITY_APPROVAL",
                        copied_path=destination,
                    )
                else:
                    entry.update(
                        status="ORIGINAL_BYTES_UNRECOVERED_NOT_COMPATIBILITY_APPROVAL",
                        copied_path=None,
                    )
            historical.append(entry)
    for destination, source in {
        "provenance/native_split.json": root / "configs/native_ssl_split_v1.json",
        "provenance/original_cpu_replay.json": replay_path,
        "provenance/original_cpu_replay_closeout.json": closeout_path,
        "provenance/CF-JEPA-LICENSE.txt": root / "external/cf-jepa-vnext/LICENSE",
        "provenance/research_report_v1.md": root / "docs/NATIVE_SSL_MODEL_RESEARCH_REPORT_V1.md",
        "provenance/research_report_addendum_v2.md": root
        / "docs/NATIVE_SSL_MODEL_RESEARCH_REPORT_ADDENDUM_V2.md",
    }.items():
        if (
            str(source) not in previous.get("source_bindings", {})
            or digest(source) != previous["source_bindings"][str(source)]
        ):
            raise ValueError("Pinned original license/provenance bytes changed")
        add(destination, source)
    for tool in (
        Path(__file__).resolve(),
        Path(__file__).with_name("check_native_pretrained_snapshot_worker_v3.py"),
        Path(__file__).with_name("check_native_pretrained_snapshot_owned_v3.py"),
        root / "tools/native_reference_supervisor.py",
        root / "tools/package_native_pretrained_snapshot_v1.py",
        root / "tools/check_native_pretrained_snapshot_worker_v2.py",
        root / "tools/check_native_pretrained_snapshot_owned_v2.py",
    ):
        bindings[str(regular(tool))] = digest(tool)
    return {
        "folder": folder,
        "archive": archive,
        "receipt": receipt_path,
        "payloads": payloads,
        "models": models,
        "bindings": bindings,
        "historical_sources": historical,
        "v1_archive_sha256": V1_ARCHIVE_SHA256,
    }


MODEL_CARD = """# Seven-model native acoustic pretrained research snapshot V2

INCOMPLETE_RESEARCH_SNAPSHOT. The fixed endpoints are original Shared7 and
CF7/13/23, plus nonlinear-frequency-conditioned Band Shared7/13/23. These are
pretrained selected encoders and saved latent predictors with original TRAIN
scalers; there is no best-seed selection, new fit or scientific approval here.
Strong readouts, supervised controls, transfer and final scientific evidence
remain separate study artifacts. Short selection probes do not establish transfer.
No SOTA, representation-transfer advantage, sealed-site guarantee or app release
is established. Historical documents retain their original scope and dates.

Use the existing frozen APIs, with src on the Python path and scientific dependency
versions matching the repository: native_encoder.NativeAcousticEncoder and
native_latent.NativeLatentPredictor for original models; native_band_replication_encoder.
NativeBandReplicationEncoder for both Band selected kinds; native_latent.
NativeLatentPredictor for Band V1; native_band_replication_latent.NativeLatentPredictor
for Band V2. These are load-only weights_only CPU/meta interfaces. Absolute
ancestral paths are metadata only and must not be executed or required at relocation.
Copied training modules are static dependencies, not permission to run training.

Inputs are raw-dB H96/four context channels, Boolean observation masks and native
measurement metadata, with issued query offsets1/3/6 for shared latent prediction.
Native0-230m remains0-230m; never relabel it0-200m or invent depth profiles.
The source clock/timezone is not verified UTC. Shared predictions represent
overlapping four-interval future latent blocks, not acoustic dB/species/biomass.
CF encode uses selected EMA features; its ONLINE heads predict three ordinal
zones, not horizons1/3/6. FullH96 CF inference is outside sampled training crops.
No causal biological, intervention, catch or business claim follows from latents.

Package isolated CPU QA and independent package/source/compatibility review remain
pending at creation, with later external receipts. Recovery preserves historical
source bytes/hashes only and grants no compatibility or scientific approval.
CF-JEPA revision5d3d2fd1273c283fbfa03249c078619245e84033 and its pinned license
are preserved. This snapshot grants no new repository license, publication,
held-out numerical access or cloud authority. No raw acoustic arrays are included.
"""


def package_snapshot(root, folder, archive, receipt_path):
    plan = build_plan(root, folder, archive, receipt_path)
    root = Path(root).absolute()
    idle(root)
    manifest = {
        "kind": "native_pretrained_model_snapshot_v2",
        "status": "INCOMPLETE_RESEARCH_SNAPSHOT",
        "models": plan["models"],
        "source_bindings": plan["bindings"],
        "fitting": False,
        "raw_acoustic_data_included": False,
        "held_out_results": "NOT_INCLUDED",
        "package_cpu_relocation_test": "PENDING_EXTERNAL_RECEIPT",
        "independent_package_review": "NOT_RUN",
        "external_qa_receipt": str(
            root / "evidence/ssl-research-v1/native-pretrained-snapshot-isolated-cpu-v3"
        ),
        "sota": "NOT_ESTABLISHED",
        "app_release": False,
        "cf_source_revision": "5d3d2fd1273c283fbfa03249c078619245e84033",
        "preserved_v1_archive_sha256": V1_ARCHIVE_SHA256,
        "historical_sources": plan["historical_sources"],
        "files": {name: digest(source) for name, source in plan["payloads"].items()},
    }
    additions = {
        "manifest.json": (
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode(),
        "MODEL_CARD.md": MODEL_CARD.encode(),
    }
    plan["folder"].mkdir()
    files = {}
    for name in sorted(set(plan["payloads"]) | additions.keys()):
        destination = plan["folder"] / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = additions[name] if name in additions else regular(plan["payloads"][name]).read_bytes()
        expected = hashlib.sha256(raw).hexdigest()
        if name in plan["payloads"] and expected != manifest["files"][name]:
            raise ValueError("Source changed during immutable copy; preserve partial attempt")
        with destination.open("xb") as stream:
            stream.write(raw)
        if digest(destination) != expected:
            raise ValueError("Snapshot copy byte identity failed")
        files[name] = expected
    with zipfile.ZipFile(plan["archive"], "x", compression=zipfile.ZIP_DEFLATED) as output:
        for name in sorted(files):
            output.write(plan["folder"] / name, name)
    receipt = {
        "kind": "native_pretrained_model_snapshot_receipt_v2",
        "status": "SEVEN_FIXED_PRETRAINED_MODELS_COPIED_QA_PENDING",
        "models": plan["models"],
        "directory": str(plan["folder"]),
        "archive": str(plan["archive"]),
        "archive_sha256": digest(plan["archive"]),
        "manifest_sha256": files["manifest.json"],
        "files": files,
        "source_bindings": plan["bindings"],
        "fitting": False,
        "raw_corpus_decoded": False,
        "app_release": False,
        "independent_package_review": "NOT_RUN",
    }
    with plan["receipt"].open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def main():
    if ROOT.name != "marine-echo-jepa":
        raise RuntimeError(
            "Production package is ROOT-only; builder must not load/copy public weights"
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folder", type=Path, default=ROOT / "outputs/native_pretrained_model_snapshot_v2"
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=ROOT / "evidence/ssl-research-v1/native-pretrained-model-snapshot-v2.json",
    )
    args = parser.parse_args()
    result = package_snapshot(ROOT, args.folder, args.folder.with_suffix(".zip"), args.receipt)
    print(
        json.dumps(
            {
                "status": result["status"],
                "models": len(result["models"]),
                "archive_sha256": result["archive_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
