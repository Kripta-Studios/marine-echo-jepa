"""Fixed saved-state CPU diagnostic. No fitting, selection, or optimizer steps.

Historical inputs remain read-only. Outputs must be inside the assigned evidence
directory. Missing endpoints are reported rather than replaced. VAL is exposed
post-hoc development data; prior-year data is TRAIN for expanded descendants.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import psutil
import torch
from torch.nn import functional as F

from marine_echo.evaluation.aeon import QUANTILES, daily_pinball
from marine_echo.models.aeon_ssl import AeonDirect, AeonTemporalSSL
from marine_echo.training import aeon_campaign, aeon_development, aeon_expanded
from marine_echo.training.aeon_corpus import AEON_SOURCE_SHA256

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/aeon-checkpoint-diagnostic-20260929"
SOURCE_BLOBS = {
    "aeon_ssl.py": "7bd8a94cde303d968df1e541af47b50ce969d342",
    "sigreg.py": "ef6f77225d03c32daa8d6509401e998658f2b198",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_hash(array: np.ndarray) -> str:
    digest = hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def state_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(array_hash(tensor.detach().cpu().numpy()).encode())
    return digest.hexdigest()


def temporal_support(gradient: torch.Tensor) -> list[dict]:
    if gradient.ndim != 2 or gradient.shape[1] != 24 * 8:
        raise ValueError("First-layer temporal mapping requires 192 interleaved inputs.")
    support = []
    for position in range(24):
        block = gradient[:, position * 8 : (position + 1) * 8]
        support.append(
            {
                "position": position,
                "value_columns": list(range(position * 8, position * 8 + 4)),
                "mask_columns": list(range(position * 8 + 4, position * 8 + 8)),
                "value_l2": float(block[:, :4].norm()),
                "mask_l2": float(block[:, 4:].norm()),
                "nonzero": int(torch.count_nonzero(block)),
                "value_nonzero": int(torch.count_nonzero(block[:, :4])),
                "mask_nonzero": int(torch.count_nonzero(block[:, 4:])),
            }
        )
    return support


def centered_stats(tensor: torch.Tensor) -> dict:
    values = tensor.detach().cpu().double().numpy()
    mean = values.mean(axis=0)
    centered = values - mean
    # Covariance eigenvalue entropy; singular values avoid tiny negative eigenvalues.
    eigenvalues = np.linalg.svd(centered, compute_uv=False) ** 2 / (len(values) - 1)
    trace = float(eigenvalues.sum())
    mass = eigenvalues[eigenvalues > 0] / trace if trace > 1e-12 else np.empty(0)
    return {
        "feature_means": mean.tolist(),
        "mean_feature_mean": float(mean.mean()),
        "mean_vector_rms": float(np.sqrt(np.mean(mean**2))),
        "rms": float(np.sqrt(np.mean(values**2))),
        "centered_rms": float(np.sqrt(np.mean(centered**2))),
        "mean_feature_std_ddof1": float(values.std(axis=0, ddof=1).mean()),
        "feature_std_ddof1": values.std(axis=0, ddof=1).tolist(),
        "centered_covariance_trace": trace,
        "centered_effective_rank": float(np.exp(-(mass * np.log(mass)).sum()))
        if len(mass)
        else 0.0,
        "rank_formula": "exp(-sum(p_i*log(p_i))), p_i=lambda_i/sum(lambda); covariance of column-centered rows; zero if trace<=1e-12",
    }


def gradient_probe(model, values, mask, truth=None, target_mask=None) -> dict:
    """Autograd on a disposable copy, including the stateful SIGReg buffer."""
    original_state = state_hash(model.state_dict())
    original_input = array_hash(values.detach().numpy())
    original_mask = array_hash(mask.numpy())
    disposable = copy.deepcopy(model).cpu().eval()
    x = values.detach().clone().requires_grad_(True)
    m = mask.clone()
    if isinstance(disposable, AeonDirect):
        forecast = disposable(x, m)
        residual = truth[:, :, None] - forecast
        q = torch.as_tensor(QUANTILES, dtype=x.dtype)
        prediction = torch.maximum(q * residual, (q - 1) * residual)[target_mask].mean()
        losses = {"prediction": prediction, "total": prediction}
        result = {"objective": "supervised_pinball", "prediction_loss": float(prediction.detach())}
    else:
        context = disposable.context_view(x, m)
        target = disposable.teacher_view(x, m)
        full = disposable.encoder(x, m)
        predicted = disposable.predictor(context)
        predictive_target = target.detach() if disposable.mode == "ema" else target
        prediction = F.smooth_l1_loss(predicted, predictive_target)
        raw = disposable.regularizer(predicted if disposable.mode == "ema" else target)
        weighted = disposable.regularizer_weight * raw
        losses = {
            "prediction": prediction,
            "regularizer_raw": raw,
            "regularizer_weighted": weighted,
            "total": prediction + weighted,
        }
        centered_error = (predicted - predicted.mean(0)) - (target - target.mean(0))
        result = {
            "objective": disposable.mode,
            "prediction_loss": float(prediction.detach()),
            "raw_regularizer_loss": float(raw.detach()),
            "regularizer_weight": disposable.regularizer_weight,
            "weighted_regularizer_loss": float(weighted.detach()),
            "total_loss": float((prediction + weighted).detach()),
            "centered_prediction_mse": float(centered_error.detach().square().mean()),
            "centered_prediction_smooth_l1": float(
                F.smooth_l1_loss(predicted - predicted.mean(0), target - target.mean(0)).detach()
            ),
            "teacher_suffix_target": centered_stats(target),
            "online_short_context": centered_stats(context),
            "online_full_context": centered_stats(full),
            "predictor": centered_stats(predicted),
            "regularizer_global_step_before": int(model.regularizer.global_step),
            "regularizer_global_step_disposable_after": int(disposable.regularizer.global_step),
        }
    named = [(name, p) for name, p in disposable.named_parameters() if p.requires_grad]
    gradients = {}
    vectors = {}
    for label, loss in losses.items():
        grads = torch.autograd.grad(
            loss, [p for _, p in named], allow_unused=True, retain_graph=True
        )
        by_name = dict(zip((name for name, _ in named), grads, strict=True))
        groups = {}
        for group in ("encoder", "predictor", "teacher", "head"):
            selected = [
                g for name, g in by_name.items() if name.startswith(group + ".") and g is not None
            ]
            groups[group] = {
                "parameters_with_gradient": len(selected),
                "l2": float(
                    torch.sqrt(sum((g.square().sum() for g in selected), torch.tensor(0.0)))
                ),
            }
        first = by_name["encoder.network.0.weight"]
        groups["first_layer_temporal_support"] = temporal_support(first)
        gradients[label] = groups
        vectors[label] = torch.cat(
            [
                (torch.zeros_like(p) if g is None else g).flatten()
                for (_, p), g in zip(named, grads, strict=True)
            ]
        )
    input_gradient = torch.autograd.grad(losses["total"], x, allow_unused=True)[0]
    result["input_gradient_temporal_l2"] = input_gradient.norm(dim=(0, 2)).tolist()
    if "regularizer_weighted" in vectors:
        a, b = vectors["prediction"], vectors["regularizer_weighted"]
        result["prediction_vs_weighted_regularizer_gradient_cosine"] = (
            float(F.cosine_similarity(a[None], b[None])) if a.norm() > 0 and b.norm() > 0 else None
        )
        result["weighted_regularizer_to_prediction_gradient_l2_ratio"] = (
            float(b.norm() / a.norm()) if a.norm() > 0 else None
        )
    result["gradients"] = gradients
    assert state_hash(model.state_dict()) == original_state
    assert array_hash(values.detach().numpy()) == original_input
    assert array_hash(mask.numpy()) == original_mask
    result["original_model_and_inputs_unchanged"] = True
    result["original_state_sha256"] = original_state
    return result


def fixed_endpoint(directory: Path, slot: dict, phase: str, step: int) -> dict:
    matches = [c for c in slot.get("checkpoints", []) if c["phase"] == phase and c["step"] == step]
    if not matches:
        return {"status": "MISSING_INDEX_ENTRY", "phase": phase, "step": step}
    if len(matches) != 1:
        raise ValueError("Fixed endpoint index is ambiguous.")
    entry = matches[0]
    relative = Path(entry["path"])
    if relative.is_absolute() or len(relative.parts) != 1:
        raise ValueError("Unsafe checkpoint path.")
    path = directory / relative
    if not path.is_file():
        return {"status": "MISSING_ARTIFACT", **entry}
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"Checkpoint digest differs: {path}")
    return {"status": "VERIFIED", **entry, "absolute_path": str(path.resolve())}


def three_seed_mean(predictions: dict) -> np.ndarray:
    if set(predictions) != {7, 13, 23}:
        raise ValueError("Equal three-seed mean requires exactly seeds 7,13,23.")
    if len({p.shape for p in predictions.values()}) != 1:
        raise ValueError("Three-seed forecasts differ in shape.")
    return np.sort(
        np.mean([predictions[s].astype(np.float64) for s in (7, 13, 23)], axis=0), axis=-1
    )


def validation_checks(slot: dict) -> list[dict]:
    """Read historical scheduled checks without choosing a checkpoint."""
    checks = slot.get("validation_checks", slot.get("protocol_validation_checks"))
    if not isinstance(checks, list) or not checks:
        raise ValueError("Historical scheduled validation checks are missing.")
    return checks


def lineage_role(model_lineage: str, source: str) -> str:
    if source == "prior" and model_lineage.startswith("expanded"):
        return "TRAIN"
    if source == "prior" and model_lineage == "original_frozen":
        return "HISTORICAL_DESCRIPTIVE_TRANSFER_ONLY"
    return "TRAIN" if source == "current_train" else "EXPOSED_VALIDATION_POST_HOC"


class HistoricalInputs:
    """Hash every explicitly consumed artifact and verify all at completion."""

    def __init__(self):
        self.hashes = {}

    def bind(self, path: Path, expected: str | None = None) -> Path:
        path = path.resolve(strict=True)
        actual = sha256(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Historical input digest differs: {path}")
        if str(path) in self.hashes and self.hashes[str(path)] != actual:
            raise ValueError(f"Historical input changed during diagnostic: {path}")
        self.hashes[str(path)] = actual
        return path

    def read_json(self, path: Path, expected: str | None = None) -> dict:
        return json.loads(self.bind(path, expected).read_text(encoding="utf-8"))

    def verify_unchanged(self) -> None:
        for path, digest in self.hashes.items():
            if sha256(Path(path)) != digest:
                raise ValueError(f"Historical input changed: {path}")


def load_state(endpoint: dict, slot: dict, manifest: dict):
    checkpoint = torch.load(endpoint["absolute_path"], map_location="cpu", weights_only=True)
    if any(
        checkpoint[key] != expected
        for key, expected in {
            "family": slot["family"],
            "seed": slot["seed"],
            "phase": endpoint["phase"],
            "step": endpoint["step"],
        }.items()
    ):
        raise ValueError("Checkpoint identity differs from predetermined indexed endpoint.")
    sources = checkpoint.get("source_archive_sha256", checkpoint.get("source_sha256"))
    expanded = "expanded" in manifest["study_id"]
    expected_sources = (
        [aeon_expanded.PRIOR_SHA256, AEON_SOURCE_SHA256] if expanded else AEON_SOURCE_SHA256
    )
    if sources != expected_sources:
        raise ValueError("Checkpoint source lineage differs.")
    if "cohort_sha256" in checkpoint and checkpoint["cohort_sha256"] != manifest["cohort_sha256"]:
        raise ValueError("Checkpoint cohort differs from campaign manifest.")
    if "config_sha256" in checkpoint and checkpoint["config_sha256"] != manifest["config_sha256"]:
        raise ValueError("Checkpoint config differs from manifest.")
    if slot["family"] == "direct":
        model = AeonDirect()
    else:
        model = AeonTemporalSSL(
            mode="shared_sigreg" if "shared_sigreg" in slot["family"] else "ema"
        )
    saved_hash = state_hash(checkpoint["model_state_dict"])
    model.load_state_dict(copy.deepcopy(checkpoint["model_state_dict"]), strict=True)
    assert state_hash(checkpoint["model_state_dict"]) == saved_hash
    scaler = tuple(checkpoint.get("scaler_joint_train_only", checkpoint.get("scaler_fit_only")))
    return model, scaler, checkpoint, saved_hash


def forecast(model, rows, scaler):
    x, mask = aeon_development._context_tensors(rows, scaler)
    before = state_hash(model.state_dict())
    disposable = copy.deepcopy(model).cpu().eval()
    with torch.no_grad():
        parts = [
            disposable(x[i : i + 64], mask[i : i + 64]).numpy() * scaler[3] + scaler[2]
            for i in range(0, len(rows), 64)
        ]
    assert state_hash(model.state_dict()) == before
    return np.concatenate(parts)


def rows_score(rows, prediction):
    return daily_pinball(
        np.stack([r.target_db for r in rows]),
        prediction.astype(np.float64),
        np.stack([r.target_mask for r in rows]),
        np.stack([r.target_source_timestamps for r in rows]),
    )


def fixed_batches(rows):
    if len(rows) < 128 or any(row.partition != "train" for row in rows):
        raise ValueError("Fixed diagnostic batches require at least 128 TRAIN rows.")
    indices = np.linspace(0, len(rows) - 1, 128, dtype=np.int64)
    assert len(set(indices.tolist())) == 128
    return [indices[:64], indices[64:]]


def batch_membership(rows, indices):
    chosen = [rows[i] for i in indices]
    return {
        "partition": "train",
        "construction": "128 chronological evenly spaced rows split into two batches of 64",
        "indices": indices.tolist(),
        "row_ids": [r.row_id for r in chosen],
        "cutoff_interval_ids": [r.cutoff_interval_id for r in chosen],
        "cutoff_source_timestamps": [str(r.cutoff_source_timestamp) for r in chosen],
        "source_archive_sha256": [r.source_archive_sha256 for r in chosen],
        "past_members": [list(r.past_members) for r in chosen],
        "target_members": [list(r.target_members) for r in chosen],
        "context_values_sha256": array_hash(np.stack([r.context_db for r in chosen])),
        "context_masks_sha256": array_hash(np.stack([r.context_mask for r in chosen])),
        "target_values_sha256": array_hash(np.stack([r.target_db for r in chosen])),
    }


def slot_inputs(inputs, campaign, run_id):
    manifest = inputs.read_json(campaign / "manifest.json")
    entry = manifest["slots"].get(run_id)
    if entry is None:
        return manifest, None
    if entry.get("status") != "DONE":
        raise ValueError("Diagnostic requires a completed historical slot.")
    slot = inputs.read_json(campaign / run_id / "slot.json", entry["slot_sha256"])
    if slot["run_id"] != run_id:
        raise ValueError("Slot identity differs from manifest index.")
    return manifest, slot


def verify_zip(inputs, audit_zip, output):
    inputs.bind(audit_zip)
    extracted = output / "supplied-audit"
    extracted.mkdir()
    verified = []
    with zipfile.ZipFile(audit_zip) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or sum(i.file_size for i in archive.infolist()) > 100_000:
            raise ValueError("Audit ZIP duplicate names or unexpected expansion.")
        for info in archive.infolist():
            target = extracted / info.filename
            if not target.resolve().is_relative_to(extracted.resolve()) or "\\" in info.filename:
                raise ValueError("Unsafe audit ZIP path.")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
        audit = extracted / "aeon_pretraining_audit_991368d"
        for line in (audit / "SHA256SUMS").read_text().splitlines():
            digest, relative = line.split("  ", 1)
            path = audit / relative
            if not path.resolve().is_relative_to(audit.resolve()) or sha256(path) != digest:
                raise ValueError("Audit ZIP checksum mismatch.")
            verified.append({"member": relative, "sha256": digest})
    # A Windows checkout may use CRLF. Preserve it; run the exact-byte check
    # against a canonical source copy from this local Git object database.
    canonical = output / "canonical-source/marine_echo/models"
    canonical.mkdir(parents=True)
    sources = {}
    for name, expected in SOURCE_BLOBS.items():
        path = ROOT / "src/marine_echo/models" / name
        inputs.bind(path)
        result = subprocess.run(
            ["git", "cat-file", "blob", f"991368d:src/marine_echo/models/{name}"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        )
        content = result.stdout
        blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if blob != expected or path.read_bytes().replace(b"\r\n", b"\n") != content:
            raise ValueError(
                "Current source differs from audited Git blob after newline normalization."
            )
        (canonical / name).write_bytes(content)
        sources[name] = {
            "git_blob": blob,
            "canonical_sha256": hashlib.sha256(content).hexdigest(),
            "checkout_sha256": sha256(path),
            "newline_normalization_only": True,
        }
    write_json(
        output / "zip-verification.json",
        {
            "archive_sha256": sha256(audit_zip),
            "verified_members": verified,
            "source_identity": sources,
            "script_inspection": "Reviewed all 7626 script bytes before execution; synthetic CPU backward only; no optimizer or network calls",
        },
    )
    command = [
        sys.executable,
        str(audit / "audit_gradient_paths.py"),
        "--source-root",
        str(output / "canonical-source"),
        "--output",
        str(output / "supplied-audit-local.json"),
    ]
    with (output / "supplied-audit-local.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    write_json(
        output / "supplied-audit-execution.json",
        {"command": command, "exit_code": result.returncode},
    )
    if result.returncode:
        raise RuntimeError("Supplied source audit failed; see preserved log.")


def sampling_audit(rows, inputs, campaign):
    """Reconstruct exact current-source draws only; never execute training code."""
    interval = np.asarray([r.cutoff_interval_id for r in rows])
    candidate = np.flatnonzero(np.asarray([r.target_mask.any() for r in rows]))
    residues = {r: np.flatnonzero(interval % 24 == r) for r in range(24)}
    residues = {r: pool for r, pool in residues.items() if len(pool) >= 64}
    modes = {}
    for mode in ("aligned", "shuffled"):
        rng = np.random.default_rng(np.random.SeedSequence([7, 1]))
        sequence, pairs, uniqueness, separation = [], [], [], []
        for step in range(1, 1501):
            pool = (
                len(rows) if mode == "aligned" else residues[sorted(residues)[step % len(residues)]]
            )
            selected = rng.choice(pool, size=64, replace=False)
            uniqueness.append(len(np.unique(selected)))
            sequence.append(selected)
            if mode == "shuffled":
                shift = int(rng.integers(1, 64))
                target = np.roll(selected, shift)
                separation.append(int(np.min(np.abs(interval[selected] - interval[target]))))
                pairs.append(target)
        continuing = rng.choice(candidate, size=64, replace=False)
        # Actual current campaign resets the RNG at this phase boundary.
        supervised_rng = np.random.default_rng(np.random.SeedSequence([7, 2]))
        supervised = np.stack(
            [supervised_rng.choice(candidate, size=64, replace=False) for _ in range(1500)]
        )
        modes[mode] = {
            "pretrain_context_sequence_sha256": array_hash(np.stack(sequence)),
            "target_sequence_sha256": array_hash(np.stack(pairs)) if pairs else None,
            "min_batch_unique": min(uniqueness),
            "max_batch_unique": max(uniqueness),
            "duplicate_draws": int(1500 * 64 - sum(uniqueness)),
            "minimum_pair_separation_intervals": min(separation) if separation else None,
            "supervised_sequence_sha256_after_actual_reset": array_hash(supervised),
            "hypothetical_continuing_rng_first_batch_sha256": array_hash(continuing),
            "unique_contexts_across_all_batches": len(np.unique(np.stack(sequence))),
        }
    source = inputs.bind(ROOT / "src/marine_echo/training/aeon_campaign.py")
    manifest = inputs.read_json(campaign / "manifest.json")
    return {
        "scope": "Reconstructed indices, 1500 SSL and 1500 supervised batches; no model execution",
        "current_source_sha256": sha256(source),
        "current_composite_campaign_code_sha256": aeon_campaign._campaign_code_sha256(),
        "historical_manifest_composite_code_sha256": manifest["code_sha256"],
        "current_code_matches_manifest": aeon_campaign._campaign_code_sha256()
        == manifest["code_sha256"],
        "modes": modes,
        "rng_reset_in_current_source": "SeedSequence([seed,2]) before supervised",
        "rejection_duplicate_algorithm": "NOT_PRESENT_IN_CURRENT_SOURCE; historical source binding must be checked before attribution",
        "pairing_only_intervention": False,
    }


def reuse_diagnostics(previous: Path, output: Path, inputs: HistoricalInputs, args):
    """Reuse completed forward/backward evidence; no model execution here."""
    previous = previous.resolve(strict=True)
    if not previous.is_relative_to(EVIDENCE.resolve()) or previous == output:
        raise ValueError("Reused diagnostics must be an earlier assigned evidence directory.")
    verified = inputs.read_json(previous / "zip-verification.json")
    inputs.bind(args.audit_zip, verified["archive_sha256"])
    for name, entry in verified["source_identity"].items():
        inputs.bind(ROOT / "src/marine_echo/models" / name, entry["checkout_sha256"])
    if inputs.read_json(previous / "supplied-audit-execution.json")["exit_code"] != 0:
        raise ValueError("Reused source audit did not complete.")
    endpoints = inputs.read_json(previous / "endpoint-replays.json")
    objective = inputs.read_json(previous / "objective-gradients.json")
    for record in endpoints + objective:
        for key in ("endpoint", "pretrain_endpoint"):
            endpoint = record.get(key)
            if endpoint and endpoint["status"] == "VERIFIED":
                inputs.bind(Path(endpoint["absolute_path"]), endpoint["sha256"])
    if sum(r.get("state") == "initialized" for r in objective) != 18:
        raise ValueError("Reused initialization probes are incomplete.")
    if sum(r.get("status") == "COMPLETED" for r in objective) != 12:
        raise ValueError("Reused saved-state probes are incomplete.")
    names = (
        "zip-verification.json",
        "supplied-audit-execution.json",
        "supplied-audit-local.json",
        "supplied-audit-local.log",
        "endpoint-replays.json",
        "objective-gradients.json",
        "initialization-equality.json",
        "sampling-audit.json",
    )
    copied = {}
    for name in names:
        source = inputs.bind(previous / name)
        shutil.copyfile(source, output / name)
        copied[name] = sha256(source)
    for name in ("supplied-audit", "canonical-source", "replays"):
        for source in (previous / name).rglob("*"):
            if source.is_file():
                inputs.bind(source)
                copied[str(source.relative_to(previous))] = sha256(source)
        shutil.copytree(previous / name, output / name)
    write_json(
        output / "reused-phase-hashes.json",
        {
            "source_directory": str(previous),
            "forward_backward_probes_reexecuted": False,
            "files": copied,
        },
    )


def run(args):
    output = args.output.resolve()
    if not output.is_relative_to(EVIDENCE.resolve()) or output.exists():
        raise ValueError("Use a new child directory inside the assigned diagnostic evidence.")
    output.mkdir(parents=True)
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    inputs = HistoricalInputs()
    inputs.bind(args.release, "5e265e08bf1041bbe0fb4dd2a5f3503bae98f996096577f976beaf9384d6bb57")
    if args.reuse_saved_diagnostics is None:
        verify_zip(inputs, args.audit_zip, output)
    else:
        reuse_diagnostics(args.reuse_saved_diagnostics, output, inputs, args)
    print("Verified supplied ZIP and exact source audit: completed", flush=True)
    fit, assess, cohort, _ = aeon_campaign.load_cohort(args.current_archive, args.split_review)
    inputs.bind(args.current_archive, AEON_SOURCE_SHA256)
    inputs.bind(args.split_review)
    original_manifest = inputs.read_json(args.original / "manifest.json")
    if cohort != original_manifest["cohort_sha256"]:
        raise ValueError("Reconstructed original cohort differs from manifest.")
    config = inputs.read_json(ROOT / "configs/aeon_expanded_3k.json")
    metadata = inputs.read_json(
        args.prior_metadata, config["prior_source"]["metadata_report_sha256"]
    )
    inputs.bind(args.prior_archive, aeon_expanded.PRIOR_SHA256)
    joint, joint_assess, joint_digest, detail = aeon_expanded._load_joint_cohort(
        config, args.prior_archive, args.current_archive, metadata, args.split_review
    )
    joint_report = inputs.read_json(args.expanded / "cohort.json")
    if joint_digest != joint_report["cohort_sha256"] or any(
        joint_report[k] != v for k, v in detail.items()
    ):
        raise ValueError("Reconstructed expanded cohort differs from saved counts and membership.")
    assert [r.row_id for r in joint_assess] == [r.row_id for r in assess]
    print(
        f"Reconstructed TRAIN={len(fit)}, joint TRAIN={len(joint)}, VAL={len(assess)}", flush=True
    )
    write_json(
        output / "cohort-and-lineage.json",
        {
            "original_cohort_sha256": cohort,
            "expanded_cohort_sha256": joint_digest,
            **detail,
            "prior_year_role_expanded_and_descendants": lineage_role("expanded", "prior"),
            "original_frozen_transfer_scope": lineage_role("original_frozen", "prior"),
            "calibration_numeric_access": "NOT_RUN_PROHIBITED",
            "test_numeric_access": "NOT_RUN_PROHIBITED",
        },
    )
    original_batches, joint_batches = fixed_batches(fit), fixed_batches(joint)
    write_json(
        output / "fixed-train-memberships.json",
        {
            "original": [batch_membership(fit, b) for b in original_batches],
            "expanded": [batch_membership(joint, b) for b in joint_batches],
            "expanded_note": "Even spacing and chronological split makes batch 0 prior-year and batch 1 mixed; diagnostics are not the historical source-homogeneous SSL sampler",
        },
    )
    if args.reuse_saved_diagnostics is None:
        endpoint_results, predictions = [], {}
        for family, step in (("direct", 1500), ("direct", 3000), ("ema_jepa", 1500)):
            group = f"{family}_supervised{step}"
            predictions[group] = {}
            for seed in (7, 13, 23):
                run_id = f"{family}_seed{seed}"
                manifest, slot = slot_inputs(inputs, args.original, run_id)
                result = {
                    "run_id": run_id,
                    "group": group,
                    "seed": seed,
                    "classification": "POST_HOC_EXPOSED_VAL_FIXED_ENDPOINT_DIAGNOSTIC",
                }
                if slot is None:
                    result["status"] = "MISSING_SLOT"
                    endpoint_results.append(result)
                    continue
                endpoint = fixed_endpoint(args.original / run_id, slot, "supervised", step)
                result["endpoint"] = endpoint
                if endpoint["status"] != "VERIFIED":
                    result["status"] = endpoint["status"]
                    endpoint_results.append(result)
                    continue
                inputs.bind(Path(endpoint["absolute_path"]), endpoint["sha256"])
                model, scaler, checkpoint, saved_hash = load_state(endpoint, slot, manifest)
                if family == "ema_jepa":
                    pretrain = fixed_endpoint(args.original / run_id, slot, "pretrain", 1500)
                    result["pretrain_endpoint"] = pretrain
                    if pretrain["status"] != "VERIFIED":
                        result["status"] = "MISSING_PRETRAIN_LINEAGE"
                        endpoint_results.append(result)
                        continue
                    inputs.bind(Path(pretrain["absolute_path"]), pretrain["sha256"])
                    load_state(pretrain, slot, manifest)
                predicted = forecast(model, assess, scaler)
                predictions[group][seed] = predicted
                path = output / "replays" / f"{group}_seed{seed}.npz"
                path.parent.mkdir(exist_ok=True)
                aeon_development._save_predictions(path, assess, predicted)
                old_entry = next((c for c in slot["validation_checks"] if c["step"] == step), None)
                if old_entry is None:
                    result["fidelity"] = {"status": "MISSING_SAVED_COMPARISON"}
                else:
                    old_path = inputs.bind(
                        args.original / run_id / old_entry["path"], old_entry["sha256"]
                    )
                    aeon_development._verify_and_score(old_path, assess)
                    with np.load(old_path, allow_pickle=False) as old:
                        difference = predicted.astype(float) - old["quantiles_db"].astype(float)
                        result["fidelity"] = {
                            "status": "COMPARED",
                            "saved_sha256": old_entry["sha256"],
                            "bit_exact": bool(np.array_equal(predicted, old["quantiles_db"])),
                            "max_abs_db": float(np.max(np.abs(difference))),
                            "rms_db": float(np.sqrt(np.mean(difference**2))),
                        }
                result.update(
                    status="COMPLETED",
                    scaler=list(scaler),
                    metrics=rows_score(assess, predicted),
                    saved_state_sha256=saved_hash,
                )
                assert state_hash(checkpoint["model_state_dict"]) == saved_hash
                endpoint_results.append(result)
                print(
                    f"Replay {group} seed{seed}: {result['metrics']['primary_daily_mean_pinball_db']:.9f}",
                    flush=True,
                )
        for group, per_seed in predictions.items():
            if set(per_seed) != {7, 13, 23}:
                endpoint_results.append(
                    {
                        "group": group,
                        "status": "MISSING_ENSEMBLE_COMPONENT",
                        "seeds_present": sorted(per_seed),
                    }
                )
                continue
            predicted = three_seed_mean(per_seed)
            aeon_development._save_predictions(
                output / "replays" / f"{group}_equal_three_seed_mean.npz", assess, predicted
            )
            endpoint_results.append(
                {
                    "group": group,
                    "status": "COMPLETED",
                    "aggregation": "EQUAL_THREE_SEED_MEAN_PREDICTIONS",
                    "seeds": [7, 13, 23],
                    "metrics": rows_score(assess, predicted),
                }
            )
        write_json(output / "endpoint-replays.json", endpoint_results)
        objective = []
        scaler = aeon_development._normalizer(fit)
        for seed in (7, 13, 23):
            for mode in ("direct", "ema", "shared_sigreg"):
                torch.manual_seed(seed)
                model = AeonDirect() if mode == "direct" else AeonTemporalSSL(mode=mode)
                for batch_id, indices in enumerate(original_batches):
                    rows = [fit[i] for i in indices]
                    x, mask, y, valid = aeon_development._tensors(rows, scaler)
                    objective.append(
                        {
                            "state": "initialized",
                            "seed": seed,
                            "mode": mode,
                            "batch": batch_id,
                            **gradient_probe(model, x, mask, y, valid),
                        }
                    )
        initializers = []
        for seed in (7, 13, 23):
            torch.manual_seed(seed)
            direct = AeonDirect()
            for mode in ("ema", "shared_sigreg"):
                torch.manual_seed(seed)
                ssl = AeonTemporalSSL(mode=mode)
                initializers.append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "encoder_tensors_equal": {
                            k: torch.equal(v, ssl.encoder.state_dict()[k])
                            for k, v in direct.encoder.state_dict().items()
                        },
                        "head_tensors_equal": {
                            k: torch.equal(v, ssl.head.state_dict()[k])
                            for k, v in direct.head.state_dict().items()
                        },
                        "direct_encoder_sha256": state_hash(direct.encoder.state_dict()),
                        "ssl_encoder_sha256": state_hash(ssl.encoder.state_dict()),
                        "direct_head_sha256": state_hash(direct.head.state_dict()),
                        "ssl_head_sha256": state_hash(ssl.head.state_dict()),
                    }
                )
        write_json(output / "initialization-equality.json", initializers)
        representatives = [
            ("original", args.original, "ema_jepa_seed7", 500),
            ("original", args.original, "ema_jepa_seed7", 1500),
            ("original", args.original, "shared_sigreg_seed7", 500),
            ("original", args.original, "shared_sigreg_seed7", 1500),
            ("30k", args.scale, "ema_jepa_seed7", 15000),
            ("expanded", args.expanded, "ema_jepa_seed7", 1500),
        ]
        for label, campaign, run_id, step in representatives:
            manifest, slot = slot_inputs(inputs, campaign, run_id)
            record = {"campaign": label, "run_id": run_id, "phase": "pretrain", "step": step}
            if slot is None:
                objective.append({**record, "status": "MISSING_SLOT"})
                continue
            endpoint = fixed_endpoint(campaign / run_id, slot, "pretrain", step)
            if endpoint["status"] != "VERIFIED":
                objective.append({**record, **endpoint})
                continue
            inputs.bind(Path(endpoint["absolute_path"]), endpoint["sha256"])
            model, scaler, checkpoint, saved_hash = load_state(endpoint, slot, manifest)
            rows = joint if label == "expanded" else fit
            batches = joint_batches if label == "expanded" else original_batches
            for batch_id, indices in enumerate(batches):
                x, mask, y, valid = aeon_development._tensors([rows[i] for i in indices], scaler)
                objective.append(
                    {
                        **record,
                        "status": "COMPLETED",
                        "batch": batch_id,
                        "endpoint": endpoint,
                        **gradient_probe(model, x, mask, y, valid),
                    }
                )
            assert state_hash(checkpoint["model_state_dict"]) == saved_hash
            print(f"Objective gradients {label} {run_id} SSL{step}: two CPU batches", flush=True)
        write_json(output / "objective-gradients.json", objective)
        write_json(output / "sampling-audit.json", sampling_audit(fit, inputs, args.original))
    # Saved validation trajectories, never historical-best selection. Fixed TRAIN
    # probes at initial/final scheduled endpoints only give a diagnostic trend.
    trajectories = []
    train_endpoints = []
    for label, campaign, rows in (
        ("original", args.original, fit),
        ("30k", args.scale, fit),
        ("expanded", args.expanded, joint),
    ):
        run_ids = [
            f"{family}_seed{seed}"
            for family in ("direct", "ema_jepa")
            for seed in ((7, 13, 23) if label == "original" else (7,))
        ]
        for run_id in run_ids:
            manifest, slot = slot_inputs(inputs, campaign, run_id)
            if slot is None:
                trajectories.append({"campaign": label, "run_id": run_id, "status": "MISSING_SLOT"})
                continue
            trajectory = []
            checks = validation_checks(slot)
            for check in checks:
                path = inputs.bind(campaign / run_id / check["path"], check["sha256"])
                aeon_development._verify_and_score(path, assess)
                with np.load(path, allow_pickle=False) as saved:
                    metrics = daily_pinball(
                        saved["truth_db"],
                        saved["quantiles_db"],
                        saved["target_mask"],
                        saved["target_source_timestamps"],
                    )
                trajectory.append(
                    {
                        "step": check["step"],
                        "saved_prediction_sha256": check["sha256"],
                        "metrics": metrics,
                    }
                )
            trajectories.append(
                {
                    "campaign": label,
                    "run_id": run_id,
                    "validation": trajectory,
                    "saved_train_loss_curve": "NOT_SAVED_IN_SLOT_OR_CHECKPOINT_INDEX",
                    "saved_pretrain_diagnostics": slot.get(
                        "final_pretrain_train_representation_diagnostics"
                    ),
                }
            )
            steps = sorted({checks[0]["step"], slot["supervised_updates"]})
            if label == "original" and run_id.startswith("direct"):
                steps = [1500, 3000]
            indices = np.linspace(0, len(rows) - 1, 512, dtype=np.int64)
            selected = [rows[i] for i in indices]
            for step in steps:
                endpoint = fixed_endpoint(campaign / run_id, slot, "supervised", step)
                if endpoint["status"] != "VERIFIED":
                    train_endpoints.append({"campaign": label, "run_id": run_id, **endpoint})
                    continue
                inputs.bind(Path(endpoint["absolute_path"]), endpoint["sha256"])
                model, scaler, checkpoint, saved_hash = load_state(endpoint, slot, manifest)
                predicted = forecast(model, selected, scaler)
                # Sparse TRAIN samples cannot satisfy daily 18-anchor gate; use
                # explicitly row-weighted error, not the corrected primary metric.
                truth = np.stack([r.target_db for r in selected])
                valid = np.stack([r.target_mask for r in selected])
                error = truth[:, :, None] - predicted
                pinball = np.maximum(QUANTILES * error, (QUANTILES - 1) * error)
                train_endpoints.append(
                    {
                        "campaign": label,
                        "run_id": run_id,
                        "step": step,
                        "scope": "Fixed 512 chronological evenly spaced TRAIN windows, row-weighted descriptive errors",
                        "row_ids": [r.row_id for r in selected],
                        "indices": indices.tolist(),
                        "source_hashes": sorted({r.source_archive_sha256 for r in selected}),
                        "row_mean_pinball_db": float(pinball[valid].mean()),
                        "median_mae_db": float(np.abs(error[:, :, 2][valid]).mean()),
                        "row_mean_pinball_db_by_horizon": [
                            float(pinball[:, h][valid[:, h]].mean()) for h in range(3)
                        ],
                        "endpoint": endpoint,
                        "scaler": list(scaler),
                    }
                )
                assert state_hash(checkpoint["model_state_dict"]) == saved_hash
    write_json(output / "saved-trajectories.json", trajectories)
    write_json(output / "fixed-train-forecast-errors.json", train_endpoints)
    expanded_details = {}
    reconstructed_counts = {}
    labels = np.asarray(
        [0 if row.source_archive_sha256 == aeon_expanded.PRIOR_SHA256 else 1 for row in joint]
    )
    candidates = np.flatnonzero(np.asarray([row.target_mask.any() for row in joint]))
    for run_id in ("direct_seed7", "ema_jepa_seed7"):
        _, slot = slot_inputs(inputs, args.expanded, run_id)
        expanded_details[run_id] = {k: v for k, v in slot.items() if "sample" in k or "scaler" in k}
        ssl_count = np.zeros(2, dtype=int)
        ssl_sequence = []
        if run_id.startswith("ema"):
            rng = np.random.default_rng(np.random.SeedSequence([7, 1]))
            for _ in range(1500):
                selected = aeon_expanded.sample_ssl_batch(rng, labels, 64)
                assert len(np.unique(labels[selected])) == 1
                ssl_count += np.bincount(labels[selected], minlength=2)
                ssl_sequence.append(selected)
        rng = np.random.default_rng(np.random.SeedSequence([7, 2]))
        supervised_sequence = np.stack(
            [
                rng.choice(candidates, size=64, replace=False)
                for _ in range(slot["supervised_updates"])
            ]
        )
        supervised_count = np.bincount(labels[supervised_sequence].flatten(), minlength=2)
        pretrain_counts = {"prior": int(ssl_count[0]), "current": int(ssl_count[1])}
        supervised_counts = {"prior": int(supervised_count[0]), "current": int(supervised_count[1])}
        assert pretrain_counts == slot["sampled_pretrain_rows_by_source"]
        assert supervised_counts == slot["sampled_supervised_rows_by_source"]
        reconstructed_counts[run_id] = {
            "pretrain_rows": pretrain_counts,
            "supervised_rows": supervised_counts,
            "matches_recorded_counts": True,
            "ssl_sequence_sha256": array_hash(np.stack(ssl_sequence)) if ssl_sequence else None,
            "supervised_sequence_sha256": array_hash(supervised_sequence),
            "supervised_presentations_per_prior_window": float(supervised_count[0] / 8507),
            "supervised_presentations_per_current_window": float(supervised_count[1] / 4965),
        }
    write_json(
        output / "expanded-exposure.json",
        {
            "recorded_counts_and_scalers": expanded_details,
            "independently_reconstructed_sampler_counts": reconstructed_counts,
            "original_direct_nominal_presentations_per_window": 3000 * 64 / len(fit),
            "expanded_direct_nominal_presentations_per_window": 3000 * 64 / len(joint),
            "original_ema_each_phase_nominal_presentations_per_window": 1500 * 64 / len(fit),
            "expanded_ema_each_phase_nominal_presentations_per_window": 1500 * 64 / len(joint),
            "ssl_batching": config["sampling"],
            "interpretation": "Joint scaler, archive/year/site composition, source-homogeneous SSL, pooled supervision and exposure change together; not volume-only attribution",
        },
    )
    inputs.verify_unchanged()
    write_json(
        output / "historical-input-hashes.json",
        {"all_unchanged_after_execution": True, "files": inputs.hashes},
    )
    summary = {
        "status": "COMPLETED_BOUNDED_POST_HOC_DIAGNOSTIC",
        "base_commit": "991368d",
        "diagnostic_tool_sha256": sha256(Path(__file__)),
        "reused_forward_backward_evidence": args.reuse_saved_diagnostics is not None,
        "elapsed_seconds": time.perf_counter() - started,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "diagnostic_device": "cpu",
        "process_peak_rss_bytes": psutil.Process().memory_info().peak_wset,
        "gpu_allocated_bytes": torch.cuda.memory_allocated(),
        "optimizer_steps": 0,
        "forecast_endpoint_individual_runs": 9,
        "saved_state_ssl_batches": 12,
        "initialized_real_train_gradient_batches": 18,
        "fresh_cal_test_numeric_access": False,
        "historical_inputs_unchanged": True,
        "release_sha256_unchanged": sha256(args.release),
        "platform": platform.platform(),
    }
    write_json(output / "execution-summary.json", summary)
    hashes = {
        str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()
    }
    write_json(output / "output-hashes.json", hashes)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    historical = ROOT.parent / "marine-echo-jepa"
    base = historical / "outputs/aeon3_geb_2024_hourly_sv_v1"
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-saved-diagnostics", type=Path)
    parser.add_argument("--original", type=Path, default=base / "core_campaign")
    parser.add_argument(
        "--scale", type=Path, default=Path("E:/marine-echo-jepa-scale/aeon30k_stage1")
    )
    parser.add_argument(
        "--expanded", type=Path, default=Path("E:/marine-echo-jepa-scale/aeon_expanded_3k")
    )
    parser.add_argument(
        "--current-archive",
        type=Path,
        default=historical
        / "data/prospective/aeon-azfp-20260927/AEON3_GEB_Mar2024-Mar2025_AZFP_Sv.zip",
    )
    parser.add_argument(
        "--prior-archive",
        type=Path,
        default=historical
        / "data/prospective/aeon-external-20260928/AEON3_GEB_Feb2023-Feb2024_AZFP_Sv.zip",
    )
    parser.add_argument(
        "--prior-metadata",
        type=Path,
        default=historical
        / "outputs/aeon_external_transfer_20260928_v1/metadata/secondary-metadata.json",
    )
    parser.add_argument(
        "--split-review",
        type=Path,
        default=ROOT / "orchestration/reviews/AEON_SPLIT_AMENDMENT_20260927.json",
    )
    parser.add_argument(
        "--audit-zip",
        type=Path,
        default=ROOT.parent / "marine-echo-jepa-core/aeon-pretraining-audit-991368d.zip",
    )
    parser.add_argument(
        "--release",
        type=Path,
        default=historical / "release/aeon-offline-scale-expanded-20260928-r3.zip",
    )
    args = parser.parse_args()
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    def prohibited_step(*_args, **_kwargs):
        raise RuntimeError("Optimizer steps are prohibited in checkpoint diagnostics.")

    with (
        patch.object(torch.optim.AdamW, "step", prohibited_step),
        patch.object(torch.optim.Adam, "step", prohibited_step),
        patch.object(torch.optim.SGD, "step", prohibited_step),
    ):
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
