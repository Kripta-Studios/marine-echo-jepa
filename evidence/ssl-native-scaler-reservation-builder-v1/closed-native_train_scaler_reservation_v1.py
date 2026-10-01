"""Reserve TRAIN scaler bytes; an alias is NOT tensor equality or approval.

This metadata-only helper never loads tensors, scaler values or corpora. The
subsequent immutable ancestry audit compares selected/inference embedded scalers.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import stat
from pathlib import Path

TRAIN_SCALERS_SHA256 = "5783edf9c57f0d5b402bcd1ed72317e049e19550beebd38708a554994395d9ed"
ARCHITECTURE = "nonlinear_frequency_conditioned_v1"
PRODUCERS = {
    "native_ssl_weights_only_inference_v1": ("native_ssl", "native_downstream"),
    "native_band_ssl_weights_only_inference_v1": ("native_band_ssl", "native_band_downstream"),
    "native_band_replication_ssl_weights_only_inference_v2": (
        "native_band_replication_ssl",
        "native_band_replication_downstream",
    ),
}
METHODS = {"shared_ssl", "masked_ssl", "permuted_ssl", "random_frozen", "direct", "cf_jepa"}
DOWNSTREAM_MODES = {"frozen_readout", "full_finetune", "direct_end_to_end"}


def regular(path):
    path = Path(os.path.abspath(path))
    for parent in (path, *path.parents):
        info = parent.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Unsafe scaler/artifact reparse or symlink identity")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Existing regular TRAIN scaler/artifact required")
    return path


def digest(path):
    with regular(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reserve_train_scalers(root, folder, report, kind, *, method, seed, mode):
    """Return explicit path/hash/proposed provenance, with no access authority.

    The only missing-file alias is the owner-pinned original Shared7 TRAIN JSON.
    Own core JSON is mandatory. No caller-supplied fallback or path execution.
    """
    root, folder = (Path(os.path.abspath(p)) for p in (root, folder))
    if not folder.is_relative_to(root / "outputs/native_acoustic_ssl_v1"):
        raise ValueError("Exact local native endpoint directory required")
    if (
        kind not in PRODUCERS
        or method not in METHODS
        or type(seed) is not int
        or seed not in (7, 13, 23)
    ):
        raise ValueError("Explicit supported producer/method/fixed seed required")
    if mode not in DOWNSTREAM_MODES | {"core_frozen_readout"}:
        raise ValueError("Unknown saved mode; no implicit fallback")
    if (
        not isinstance(report, dict)
        or (report.get("status"), report.get("evidence_kind"), report.get("test_access"))
        != ("COMPLETED", "REAL_TRAIN_DEVELOPMENT_FIT", "NOT_RUN")
        or report.get("historical_initial_weights", False) is not False
    ):
        raise ValueError("Actual completed local TRAIN/development report required")
    config = report.get("core_config", report.get("config"))
    saved = report.get("config")
    if (
        not isinstance(config, dict)
        or not isinstance(saved, dict)
        or (config.get("method"), config.get("seed"), config.get("history")) != (method, seed, 96)
        or type(config.get("seed")) is not int
        or (saved.get("method"), saved.get("seed"), saved.get("history")) != (method, seed, 96)
        or type(saved.get("seed")) is not int
        or report.get("mode", "core_frozen_readout") != mode
    ):
        raise ValueError("Report/core/saved downstream identities conflict")
    band = kind != "native_ssl_weights_only_inference_v1"
    if band:
        if (
            method == "cf_jepa"
            or any(
                value != ARCHITECTURE
                for value in (
                    report.get("architecture"),
                    config.get("architecture"),
                    saved.get("architecture"),
                )
            )
            or (kind == "native_band_ssl_weights_only_inference_v1" and seed != 7)
        ):
            raise ValueError("Exact Band architecture/version/seed identity required")
    elif any(
        value is not None
        for value in (
            report.get("architecture"),
            config.get("architecture"),
            saved.get("architecture"),
        )
    ):
        raise ValueError("Original producer has conflicting architecture")
    if mode == "core_frozen_readout":
        if report.get("supervised_ancestry") is not None:
            raise ValueError("Core artifact cannot hide downstream ancestry")
    else:
        if (
            "core_config" not in report
            or saved.get("mode") != mode
            or (mode == "direct_end_to_end" and method != "direct")
        ):
            raise ValueError("Explicit exact downstream configuration required")
        ancestry = report.get("supervised_ancestry")
        if (
            not isinstance(ancestry, dict)
            or ancestry.get("mode") != mode
            or ancestry.get("ssl_only") is not False
        ):
            raise ValueError("Explicit typed supervised ancestry required")
        parents = [
            ancestry.get(key, "MISSING")
            for key in ("ancestor_run_sha256", "ancestor_encoder_sha256")
        ]
        if (mode == "direct_end_to_end" and any(value is not None for value in parents)) or (
            mode != "direct_end_to_end"
            and any(
                not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value)
                for value in parents
            )
        ):
            raise ValueError("Accurate fresh/null or transferred parent identities required")
    producer = (
        root
        / "src/marine_echo/training"
        / (PRODUCERS[kind][0 if mode == "core_frozen_readout" else 1] + ".py")
    )
    bindings = report.get("bindings")
    if not isinstance(bindings, dict) or bindings.get(str(producer)) != digest(producer):
        raise ValueError("Actual typed producer source is missing or changed")
    tree = ast.parse(regular(producer).read_bytes())
    run = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run"),
        None,
    )
    kinds = (
        {
            value.value
            for node in ast.walk(run)
            if isinstance(node, ast.Dict)
            for key, value in zip(node.keys, node.values, strict=True)
            if isinstance(key, ast.Constant)
            and key.value == "kind"
            and isinstance(value, ast.Constant)
        }
        if run
        else set()
    )
    if kind not in kinds:
        raise ValueError("Producer does not declare this safe artifact kind")
    tensor_bindings = {
        str(folder / leaf): digest(folder / leaf)
        for leaf in ("inference.pt", "selected_encoder.pt")
    }
    if report.get("inference_sha256") != tensor_bindings[str(folder / "inference.pt")] or (
        "selected_encoder_sha256" in report
        and report["selected_encoder_sha256"]
        != tensor_bindings[str(folder / "selected_encoder.pt")]
    ):
        raise ValueError("Saved downstream/core tensor bytes differ from report")
    own = folder / "scalers.json"
    own_present = own.exists() or own.is_symlink()
    path = own
    if not own_present:
        if mode not in DOWNSTREAM_MODES:
            raise ValueError("Core SSL/control artifact requires its own standalone scalers.json")
        path = root / "outputs/native_acoustic_ssl_v1/shared_ssl_seed7_h96_cuda0/scalers.json"
        if digest(path) != TRAIN_SCALERS_SHA256:
            raise ValueError("Original unchanged Shared7 TRAIN scaler bytes required")
    path = regular(path)
    actual = digest(path)
    if actual != TRAIN_SCALERS_SHA256:
        raise ValueError(
            "Reserved standalone or aliased TRAIN scaler bytes must match the original pin"
        )
    if str(own) in bindings and (path != own or bindings[str(own)] != actual):
        raise ValueError("A declared standalone scaler identity cannot be replaced")
    if "scalers_sha256" in report and report["scalers_sha256"] != actual:
        raise ValueError("Declared TRAIN scaler bytes conflict")
    return {
        "kind": "native_train_scaler_reservation_v1",
        "status": "PROPOSED_METADATA_RESERVATION_NOT_APPROVAL",
        "producer_kind": kind,
        "method": method,
        "seed": seed,
        "mode": mode,
        "endpoint_directory": str(folder),
        "standalone_scalers_present": own_present,
        "scalers_path": str(path),
        "scalers_sha256": actual,
        "alias": None
        if own_present
        else "ORIGINAL_SHARED7_TRAIN_SCALERS_PENDING_EMBEDDED_EQUALITY_AUDIT",
        "saved_tensor_bindings": tensor_bindings,
        "producer_source_bindings": {str(producer): digest(producer)},
        "embedded_selected_and_inference_scaler_equality": "NOT_ESTABLISHED_REQUIRES_EXISTING_BOUNDED_TENSOR_AUDIT",
        "numeric_access_authority": False,
        "fit_authority": False,
        "approval": False,
    }
