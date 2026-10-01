"""Execute one exactly reviewed band job under full-owned 12/96-hour limits.

Production ROOT is this script's actual checkout, never a CLI provenance path.
Parent exceptions retain an active ledger record for root reconciliation.
No scientific fitting is authorized merely by config generation or owner budget.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = "nonlinear_frequency_conditioned_v1"
FAMILY = "native_band_v1"
EVIDENCE = "REAL_TRAIN_DEVELOPMENT_FIT"
IMPLEMENTER = "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1"
RECIPES = (
    "band_shared_ssl",
    "band_masked_ssl",
    "band_permuted_ssl",
    "band_random_frozen",
    "band_direct_end_to_end",
)
METHODS = ("shared_ssl", "masked_ssl", "permuted_ssl", "random_frozen", "direct")
ACTIVE = {"RUNNING_CUDA", "RUNNING_CPU_FIT"}
RSS_LIMIT = 22 * 2**30


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pairs(values):
    result = {}
    for k, v in values:
        if k in result:
            raise ValueError("Duplicate JSON identity/key.")
        result[k] = v
    return result


def read_json(path):
    def bad(value):
        raise ValueError("Nonfinite JSON policy/accounting value.")

    result = json.loads(Path(path).read_bytes(), object_pairs_hook=_pairs, parse_constant=bad)
    if not isinstance(result, dict):
        raise TypeError("Explicit JSON object required.")
    return result


def _number(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"Invalid/nonfinite {label}.")
    return float(value)


def validate_budget(resolution):
    if (
        resolution.get("kind") != "native_band_owner_budget_resolution_v1"
        or resolution.get("status") != "ROOT_RESOLVED"
    ):
        raise ValueError("Exact owner ROOT_RESOLVED receipt required.")
    for key, value in {
        "seed7_recipe_limit_including_controls": 11,
        "revision_gpu_hours_limit_full_owned": 12,
        "aggregate_gpu_hours_limit_full_owned": 96,
    }.items():
        if type(resolution.get(key)) is not int or resolution[key] != value:
            raise ValueError("Owner budget must remain exactly11/12/96.")
    if resolution.get("single_revision_recipes") != list(RECIPES) or resolution.get(
        "original_recipes"
    ) != ["shared_ssl", "cf_jepa", "masked_ssl", "permuted_ssl", "direct", "random_frozen"]:
        raise ValueError("Owner's five band plus six original recipes differ.")
    if (
        resolution.get("prefit_source_review") != "REQUIRED_NOT_GRANTED_BY_OWNER_REPLY"
        or resolution.get("cloud_or_paid_resource_approval") is not False
        or resolution.get("whole_site_final_numeric_access") != "NOT_AUTHORIZED_BY_THIS_REPLY"
    ):
        raise ValueError("Budget reply cannot authorize source review/cloud/final access.")


def budget_totals(ledger):
    if ledger.get("gpu_limit_hours") != 96 or isinstance(ledger.get("gpu_limit_hours"), bool):
        raise ValueError("Aggregate ledger limit must equal96.")
    aggregate = _number(ledger.get("gpu_hours_spent_owned_scientific_jobs"), "aggregate hours")
    runs = ledger.get("runs")
    if not isinstance(runs, list):
        raise TypeError("Complete run ledger required.")
    if any(r.get("status") in ACTIVE for r in runs):
        raise ValueError("Another journalled scientific fit is active.")
    band_seconds = 0.0
    for record in runs:
        signature = (
            record.get("architecture") == ARCHITECTURE
            or str(record.get("method", "")).startswith("band_")
            or "native_band" in str(record.get("config", "")).lower()
            or any("native_band" in str(v) for v in record.get("command", []))
        )
        if signature and record.get("budget_family") != FAMILY:
            raise ValueError("Unaccounted historical band record; root reconciliation required.")
        if record.get("budget_family") != FAMILY:
            continue
        if record.get("requires_reconciliation"):
            raise ValueError("Unreconciled band attempt; no second process.")
        full = record.get("resources_full_attempt", {}).get("elapsed_full_attempt_seconds")
        seconds = record.get("elapsed_owned_seconds", full)
        seconds = _number(seconds, "full owned band attempt seconds")
        if full is not None and not math.isclose(
            seconds, _number(full, "supervised full elapsed"), abs_tol=1e-6, rel_tol=1e-9
        ):
            raise ValueError("Conflicting band attempt elapsed counters.")
        band_seconds += seconds
    if aggregate * 3600 + 1e-6 < band_seconds:
        raise ValueError("Aggregate counter undercounts band owned attempts.")
    remaining = min((96 - aggregate) * 3600, 12 * 3600 - band_seconds)
    if remaining <= 0:
        raise ValueError("Band or aggregate full-owned budget exhausted.")
    return aggregate, band_seconds / 3600, remaining


def budget_allowance(ledger):
    return budget_totals(ledger)[2]


def source_paths(root):
    """Fixed local static import closure; no review-controlled imports."""
    package = root / "src/marine_echo"
    pending = [
        root / "tools/execute_native_band_job.py",
        root / "tools/prepare_native_band_configs.py",
        root / "tools/native_reference_supervisor.py",
        package / "training/native_band_ssl.py",
        package / "training/native_band_downstream.py",
    ]
    found = set()
    while pending:
        p = pending.pop().resolve()
        if p in found:
            continue
        if not p.is_file():
            raise ValueError(f"Required immutable source missing: {p}")
        found.add(p)
        for node in ast.walk(ast.parse(p.read_bytes())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = node.module or ""
                names = [name, *(name + "." + a.name for a in node.names)]
            for name in names:
                if name.startswith("marine_echo."):
                    candidate = package.joinpath(*name.split(".")[1:]).with_suffix(".py")
                    if candidate.is_file():
                        pending.append(candidate)
    for name in (
        "__init__.py",
        "models/__init__.py",
        "training/__init__.py",
        "data/__init__.py",
        "data/native_ssl_corpus.py",
        "training/aeon_corpus.py",
    ):
        p = package / name
        if not p.is_file():
            raise ValueError("Required reader/package source missing.")
        found.add(p.resolve())
    return sorted(found)


def _review_identity(review, status):
    author, coordinator, reviewer = (
        review.get(k)
        for k in ("implementer_session_id", "root_coordinator_session_id", "reviewer_session_id")
    )
    if any(
        not isinstance(v, str) or not v or v != v.strip() for v in (author, coordinator, reviewer)
    ):
        raise ValueError("Explicit author/root/reviewer session identities required.")
    if reviewer.casefold() in {IMPLEMENTER.casefold(), author.casefold(), coordinator.casefold()}:
        raise ValueError("Reviewer must be distinct from builder and root.")
    if review.get("status") != status or author != IMPLEMENTER:
        raise ValueError("Exact distinct band prefit status/implementer required.")
    if review.get("architecture") != ARCHITECTURE or review.get("evidence_kind") != EVIDENCE:
        raise ValueError("Exact real TRAIN/development band review required.")
    if sorted(review.get("allowed_roles", [])) != ["development", "train"]:
        raise ValueError("Only exact TRAIN/development roles; final test stays closed.")


def _verify_bindings(review, required, allowed_npz):
    bindings = review.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("Nonempty exact bindings required.")
    verified = {}
    for value, expected in bindings.items():
        p = Path(value)
        if (
            not p.is_absolute()
            or str(p.resolve()) != value
            or not isinstance(expected, str)
            or not re.fullmatch("[0-9a-f]{64}", expected)
        ):
            raise ValueError("Canonical absolute path/SHA256 bindings required.")
        if p.suffix.lower() == ".npz" and p not in allowed_npz:
            raise ValueError("Non-TRAIN/development NPZ binding prohibited.")
        actual = digest(p)
        if actual != expected:
            raise ValueError(f"Stale binding: {p}")
        verified[value] = actual
    if any(str(p.resolve()) not in verified for p in required):
        raise ValueError(
            "Missing required wrapper/supervisor/source/runtime/config/budget binding."
        )
    return verified


@dataclass(frozen=True)
class Job:
    kind: str
    config: Path
    review: Path
    output: Path
    receipt: Path
    encoder: Path | None = None
    ancestor_review: Path | None = None
    ancestor_config: Path | None = None
    resume: Path | None = None


@dataclass
class Plan:
    job: Job
    root: Path
    config: dict
    review: dict
    ledger: dict
    command: list
    bindings: dict
    aggregate_hours: float
    band_hours: float
    deadline_seconds: float
    ledger_hash: str
    trainer_review: Path


def _fixed_inputs(root, kind):
    paths = {
        "train": root / "data/processed/native_ssl_v1/train.npz",
        "dev": root / "data/processed/native_ssl_v1/development.npz",
        "split": root / "configs/native_ssl_split_v1.json",
        "protocol": root / "docs/adr/0016-native-ssl-selection-and-source-baseline.md",
    }
    if kind == "downstream":
        paths.update(
            {
                "train-cohort": root / "data/processed/native_ssl_v1/train.json",
                "dev-cohort": root / "data/processed/native_ssl_v1/development.json",
                "adr0016": paths["protocol"],
                "protocol": root / "docs/adr/0018-native-acoustic-downstream-transfer.md",
            }
        )
    return paths


def _check_config(config, kind, root):
    # Import the fixed config generator only after every source binding verified.
    path = root / "tools/prepare_native_band_configs.py"
    spec = importlib.util.spec_from_file_location("_native_band_fixed_configs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    screens, prospective, _ = module._catalog(*module._definitions(root))
    if kind == "ssl":
        candidates = [c for c in screens.values() if "mode" not in c]
    else:
        candidates = [c for c in [*screens.values(), *prospective.values()] if "mode" in c]
    if config not in candidates:
        raise ValueError("Config differs from an executable fixed band recipe/endpoint.")


def _admit(job, root, *, config_validator=_check_config):
    root = Path(root).resolve()
    job = Job(
        **{
            k: (Path(v).resolve() if v is not None and k != "kind" else v)
            for k, v in vars(job).items()
        }
    )
    if job.kind not in ("ssl", "downstream"):
        raise ValueError("Only SSL or downstream execution.")
    if (
        not job.output.is_relative_to(root)
        or not job.receipt.is_relative_to(root)
        or job.output == job.receipt
        or job.output.is_relative_to(job.receipt)
        or job.receipt.is_relative_to(job.output)
    ):
        raise ValueError("Separate root-owned output/receipt paths required.")
    if (
        job.receipt.exists()
        or (job.output.exists() and job.resume is None)
        or (job.resume is not None and not job.output.is_dir())
    ):
        raise FileExistsError("New receipt/output required; resume explicit only.")
    if job.resume is not None and (
        not job.resume.is_file()
        or not job.resume.is_relative_to(job.output)
        or job.resume.suffix != ".pt"
    ):
        raise ValueError("Existing approved owned resume tensor path required.")
    if (job.output / "run.json").exists() or (job.output / "inference.pt").exists():
        raise FileExistsError("Completed output artifacts protected; no resume overwrite.")
    lock = root / "evidence/ssl-builder-v1/gpu-owner.lock"
    if lock.exists():
        raise ValueError("Existing GPU lock preserved; no launch.")
    config, review = read_json(job.config), read_json(job.review)
    status = "APPROVED_PREFIT" if job.kind == "ssl" else "APPROVED_DOWNSTREAM_PREFIT"
    _review_identity(review, status)
    method, mode, seed = config.get("method"), config.get("mode", "ssl_screen"), config.get("seed")
    if (
        config.get("architecture") != ARCHITECTURE
        or type(seed) is not int
        or seed != 7
        or method not in METHODS
        or config.get("history") != 96
    ):
        raise ValueError("Exact band architecture/method/seed7/H96 required.")
    if (
        method not in review.get("allowed_methods", [])
        or seed not in review.get("allowed_seeds", [])
        or (job.kind == "downstream" and mode not in review.get("allowed_modes", []))
    ):
        raise ValueError("Method/mode/seed outside exact review.")
    if job.kind == "ssl" and (
        method == "direct"
        or any(v is not None for v in (job.encoder, job.ancestor_review, job.ancestor_config))
    ):
        raise ValueError("No direct SSL screen or SSL ancestor arguments.")
    if job.kind == "downstream":
        if mode == "direct_end_to_end":
            if method != "direct" or any(
                v is not None for v in (job.encoder, job.ancestor_review, job.ancestor_config)
            ):
                raise ValueError("Direct endpoint starts random, without selected ancestor.")
        elif mode not in ("frozen_readout", "full_finetune") or any(
            v is None for v in (job.encoder, job.ancestor_review, job.ancestor_config)
        ):
            raise ValueError("Pretrained endpoints need exact encoder/ancestor review/config.")
    budget_path = root / "orchestration/native_band_budget_owner_resolution_v1.json"
    if (
        review.get("budget_resolution_path") != str(budget_path)
        or review.get("band_budget_status") != "ROOT_RESOLVED"
    ):
        raise ValueError("Explicit exact root budget-resolution path/status required.")
    inputs = _fixed_inputs(root, job.kind)
    trainer_review = job.review
    if job.resume is not None:
        value = review.get("trainer_review_path")
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError("Resume approval must bind the unchanged original trainer review.")
        trainer_review = Path(value).resolve()
    elif review.get("trainer_review_path") not in (None, str(job.review)):
        raise ValueError("Fresh fitting uses this exact original trainer review.")
    required = [
        *source_paths(root),
        job.config,
        budget_path,
        root / "docs/adr/0023-owner-resolved-band-budget.md",
        root / "docs/adr/0016-native-ssl-selection-and-source-baseline.md",
        root / "docs/adr/0018-native-acoustic-downstream-transfer.md",
        root / ".venv/Scripts/python.exe",
        *inputs.values(),
    ]
    for name in ("pyproject.toml", "uv.lock"):
        required.append(root / name)
    if job.resume is not None:
        required.extend([job.resume, trainer_review])
    if job.encoder is not None:
        required += [
            job.encoder,
            job.ancestor_review,
            job.ancestor_config,
            *(job.encoder.parent / n for n in ("run.json", "inference.pt", "membership.json")),
        ]
    bindings = _verify_bindings(review, required, {inputs["train"], inputs["dev"]})
    validate_budget(read_json(budget_path))
    if review.get("approved_config") != config:
        raise ValueError("Review exact runtime config mismatch.")
    runtime = {
        **{k: str(v) for k, v in inputs.items()},
        "config": str(job.config),
        "review": str(job.review),
        "trainer-review": str(trainer_review),
        "output": str(job.output),
        "receipt": str(job.receipt),
        "device": "cuda:0",
        "kind": job.kind,
    }
    for name in ("encoder", "ancestor_review", "ancestor_config", "resume"):
        runtime[name.replace("_", "-")] = (
            str(getattr(job, name)) if getattr(job, name) is not None else None
        )
    if review.get("runtime_arguments") != runtime:
        raise ValueError("Exact approved runtime paths/arguments differ.")
    for role, key in (("train", "train_npz_sha256"), ("dev", "dev_npz_sha256")):
        if review.get(key) != bindings[str(inputs[role])]:
            raise ValueError("Review TRAIN/development identity differs.")
    if review.get("split_sha256") != bindings[str(inputs["split"])]:
        raise ValueError("Review split identity differs.")
    config_validator(config, job.kind, root)
    if job.resume is not None:
        original = read_json(trainer_review)
        _review_identity(original, status)
        _verify_bindings(
            original,
            [job.config, budget_path, *source_paths(root), *inputs.values()],
            {inputs["train"], inputs["dev"]},
        )
        if (
            original.get("band_budget_status") != "ROOT_RESOLVED"
            or original.get("budget_resolution_path") != str(budget_path)
            or original.get("approved_config") != config
            or method not in original.get("allowed_methods", [])
            or seed not in original.get("allowed_seeds", [])
            or (job.kind == "downstream" and mode not in original.get("allowed_modes", []))
        ):
            raise ValueError("Original immutable trainer prefit identity differs on resume.")
    if job.encoder is not None:
        ancestor = read_json(job.ancestor_review)
        _review_identity(ancestor, "APPROVED_PREFIT")
        # Every declared ancestor binding must remain exact; deeper ancestry/kind
        # admission stays in the unchanged downstream runner before array parsing.
        _verify_bindings(
            ancestor,
            [
                job.ancestor_config,
                budget_path,
                inputs["train"],
                inputs["dev"],
                inputs["split"],
                root / "docs/adr/0016-native-ssl-selection-and-source-baseline.md",
            ],
            {inputs["train"], inputs["dev"]},
        )
        if (
            ancestor.get("band_budget_status") != "ROOT_RESOLVED"
            or ancestor.get("budget_resolution_path") != str(budget_path)
            or ancestor["bindings"].get(str(budget_path)) != bindings[str(budget_path)]
        ):
            raise ValueError("Original independently reviewed ancestor budget missing.")
    ledger_path = root / "orchestration/native_ssl_run_ledger_v1.json"
    if ledger_path.with_suffix(".band-pending").exists():
        raise ValueError(
            "Existing unfinished band journal preserved; root reconciliation required."
        )
    ledger = read_json(ledger_path)
    aggregate, band, deadline = budget_totals(ledger)
    command = [
        str(root / ".venv/Scripts/python.exe"),
        "-u",
        "-m",
        f"marine_echo.training.native_band_{job.kind}",
    ]
    for name, value in {
        **inputs,
        "config": job.config,
        "review": trainer_review,
        "output": job.output,
        "device": "cuda:0",
    }.items():
        command.extend(["--" + name, str(value)])
    for name in ("encoder", "ancestor_review", "ancestor_config", "resume"):
        value = getattr(job, name)
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    return Plan(
        job,
        root,
        config,
        review,
        ledger,
        command,
        bindings,
        aggregate,
        band,
        deadline,
        digest(ledger_path),
        trainer_review,
    )


def _import_supervisor(root, bindings):
    path = root / "tools/native_reference_supervisor.py"
    if digest(path) != bindings[str(path)]:
        raise ValueError("Supervisor changed after admission.")
    spec = importlib.util.spec_from_file_location("native_reference_supervisor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if digest(path) != bindings[str(path)]:
        raise ValueError("Supervisor changed during import.")
    return module.supervise_owned


def _write_ledger(root, ledger):
    # Operational journal mutable only in root execution; historical artifacts
    # never overwritten. A denied write propagates without fallback/retry.
    target = root / "orchestration/native_ssl_run_ledger_v1.json"
    pending = target.with_suffix(".band-pending")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(ledger, stream, indent=2, allow_nan=False)
        stream.write("\n")
    pending.replace(target)


def _exclusive_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _valid_resources(resources, child):
    child_pid = child.pid
    pids = resources.get("owned_process_pids", [])
    if (
        type(child_pid) is not int
        or child_pid < 1
        or not isinstance(pids, list)
        or any(type(v) is not int or v < 1 for v in pids)
        or len(pids) != len(set(pids))
        or resources.get("owned_tree_cleanup_verified") is not True
        or type(resources.get("exit_code")) is not int
        or child_pid not in pids
        or child.poll() != resources["exit_code"]
    ):
        raise ValueError("Actual exit/owned identity/cleanup proof missing.")
    for key in (
        "elapsed_full_attempt_seconds",
        "peak_process_rss_bytes",
        "deadline_seconds",
        "rss_limit_bytes",
    ):
        _number(resources.get(key), key)


def _completion(plan):
    path = plan.job.output / "run.json"
    if not path.is_file():
        return None, "MISSING_COMPLETED_REPORT"
    report = read_json(path)
    if (
        report.get("status") != "COMPLETED"
        or report.get("architecture") != ARCHITECTURE
        or report.get("evidence_kind") != EVIDENCE
        or report.get("correctness_smoke") is not False
        or report.get("config") != plan.config
        or report.get("test_access") != "NOT_RUN"
        or report.get("review_sha256") != digest(plan.trainer_review)
    ):
        return None, "INVALID_COMPLETED_REPORT"
    if plan.job.kind == "downstream" and report.get("mode") != plan.config["mode"]:
        return None, "WRONG_COMPLETED_MODE"
    resources = report.get(
        "resources" if plan.job.kind == "ssl" else "resources_additional_downstream", {}
    )
    for key in (
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "peak_rss_bytes",
        "elapsed_gpu_seconds",
    ):
        _number(resources.get(key), key)
    if (
        max(resources["peak_allocated_bytes"], resources["peak_reserved_bytes"]) >= 10 * 2**30
        or resources["peak_rss_bytes"] >= RSS_LIMIT
    ):
        return None, "REPORT_RESOURCE_LIMIT"
    for key, file in (
        ("inference_sha256", "inference.pt"),
        ("membership_sha256", "membership.json"),
    ):
        p = plan.job.output / file
        if not p.is_file() or report.get(key) != digest(p):
            return None, "MISSING_OR_CHANGED_COMPLETION_ARTIFACT"
    return report, None


def _execute(
    plan, *, popen, supervisor, clock=time.monotonic, write_ledger=_write_ledger, started=None
):
    job, root = plan.job, plan.root
    ledger_path = root / "orchestration/native_ssl_run_ledger_v1.json"
    lock = root / "evidence/ssl-builder-v1/gpu-owner.lock"
    if lock.exists() or digest(ledger_path) != plan.ledger_hash:
        raise ValueError("GPU/ledger ownership changed after admission.")
    for path, expected in plan.bindings.items():
        if digest(path) != expected:
            raise ValueError("An admitted binding changed before launch/mutation.")
    # Fresh receipt isolates logs on retries; output belongs to the child.
    job.receipt.mkdir(parents=True, exist_ok=False)
    started = clock() if started is None else started
    record = {
        "id": job.receipt.name,
        "attempt_id": str(uuid.uuid4()),
        "status": "RUNNING_CUDA",
        "architecture": ARCHITECTURE,
        "evidence_kind": EVIDENCE,
        "budget_family": FAMILY,
        "kind": job.kind,
        "method": plan.config["method"],
        "mode": plan.config.get("mode", "ssl_screen"),
        "seed": plan.config["seed"],
        "history": plan.config["history"],
        "config": str(job.config),
        "config_sha256": digest(job.config),
        "review": str(job.review),
        "review_sha256": digest(job.review),
        "output": str(job.output),
        "receipt": str(job.receipt),
        "trainer_review": str(plan.trainer_review),
        "trainer_review_sha256": digest(plan.trainer_review),
        "budget_resolution_path": str(
            root / "orchestration/native_band_budget_owner_resolution_v1.json"
        ),
        "budget_resolution_sha256": plan.bindings[
            str(root / "orchestration/native_band_budget_owner_resolution_v1.json")
        ],
        "deadline_seconds": plan.deadline_seconds,
        "command": plan.command,
        "gpu_owner": "root",
        "resume": str(job.resume) if job.resume else None,
        "bindings": plan.bindings,
        "pid": None,
        "requires_reconciliation": True,
    }
    ledger = copy.deepcopy(plan.ledger)
    ledger["runs"].append(record)
    ledger["status"] = "REAL_BAND_SCIENTIFIC_ATTEMPT_RUNNING"
    journal_failed = False

    def journal():
        nonlocal journal_failed
        try:
            write_ledger(root, ledger)
        except BaseException:
            journal_failed = True
            raise

    journal()  # Reservation exists before spawning a child.
    try:
        with (job.receipt / "console.log").open("x", encoding="utf-8") as log:
            child = popen(plan.command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            record["pid"] = child.pid
            journal()
            resources = supervisor(
                child,
                started=started,
                deadline_seconds=plan.deadline_seconds,
                rss_limit_bytes=RSS_LIMIT,
            )
        _valid_resources(resources, child)
        elapsed = max(
            _number(resources["elapsed_full_attempt_seconds"], "full elapsed"), clock() - started
        )
        record.update(
            resources_full_attempt=resources,
            elapsed_owned_seconds=elapsed,
            exit_code=resources["exit_code"],
        )
        # The wrapper adds verification/report time; normalize both accounting
        # fields to that full owned attempt, never just child training GPU time.
        record["resources_full_attempt"]["elapsed_full_attempt_seconds"] = elapsed
        if resources.get("stopped_for") is not None and lock.exists():
            owner_raw = lock.read_bytes()
            try:
                owner = json.loads(owner_raw)
            except (ValueError, UnicodeDecodeError):
                owner = {}
            if (
                resources["owned_tree_cleanup_verified"]
                and type(owner.get("pid")) is int
                and owner.get("pid") in resources["owned_process_pids"]
                and owner.get("output") == str(job.output)
                and lock.read_bytes() == owner_raw
            ):
                lock.unlink()
                record["owned_terminated_tree_lock_cleanup"] = True
            else:
                record["unknown_lock_preserved"] = True
        stopped = (
            resources.get("stopped_for") is not None
            or elapsed >= plan.deadline_seconds
            or resources["peak_process_rss_bytes"] >= RSS_LIMIT
        )
        report, reason = (
            (None, "CHILD_EXIT_OR_SUPERVISOR_LIMIT")
            if resources["exit_code"] != 0 or stopped
            else _completion(plan)
        )
        completed = report is not None and not lock.exists()
        if lock.exists():
            reason = "GPU_LOCK_REMAINS"
        record.update(
            status="COMPLETED_REAL_BAND_CUDA" if completed else "FAILED_REAL_BAND_CUDA_ATTEMPT",
            verified_completion=completed,
            failure_reason=reason,
            owned_tree_cleanup_verified=True,
            gpu_lock_absent=not lock.exists(),
            requires_reconciliation=lock.exists(),
        )
        # Charge startup/evaluation/saves/report validation as full owned time,
        # including unsuccessful attempts and each resumed invocation.
        elapsed = max(elapsed, clock() - started)
        if elapsed >= plan.deadline_seconds and completed:
            completed = False
            record.update(
                status="FAILED_REAL_BAND_CUDA_ATTEMPT",
                verified_completion=False,
                failure_reason="FULL_OWNED_ATTEMPT_DEADLINE_AFTER_VERIFICATION",
            )
        record["elapsed_owned_seconds"] = elapsed
        record["resources_full_attempt"]["elapsed_full_attempt_seconds"] = elapsed
        ledger["gpu_hours_spent_owned_scientific_jobs"] = plan.aggregate_hours + elapsed / 3600
        ledger["native_band_gpu_hours_spent_full_owned"] = plan.band_hours + elapsed / 3600
        if completed:
            record.update(
                run_sha256=digest(job.output / "run.json"),
                resources_report=report.get(
                    "resources", report.get("resources_additional_downstream")
                ),
            )
            ledger["scientific_fits_completed"] = ledger.get("scientific_fits_completed", 0) + 1
            ledger["gpu_hours_completed_scientific_training"] = (
                ledger.get("gpu_hours_completed_scientific_training", 0)
                + record["resources_report"]["elapsed_gpu_seconds"] / 3600
            )
        ledger["status"] = (
            "REAL_BAND_ATTEMPT_COMPLETED" if completed else "REAL_BAND_ATTEMPT_FAILED"
        )
        _exclusive_json(job.receipt / "attempt.json", record)
        journal()
        return 0 if completed else (resources["exit_code"] or 1)
    except BaseException as error:
        # A supervisor/save/verification exception is NOT proof of cleanup.
        # The already journalled active record blocks every later launch.
        record.update(
            status="RUNNING_CUDA",
            requires_reconciliation=True,
            parent_exception_type=type(error).__name__,
            parent_exception=str(error),
            elapsed_owned_seconds=max(0, clock() - started),
            verified_completion=False,
        )
        record.pop("owned_tree_cleanup_verified", None)
        ledger["gpu_hours_spent_owned_scientific_jobs"] = (
            plan.aggregate_hours + record["elapsed_owned_seconds"] / 3600
        )
        ledger["native_band_gpu_hours_spent_full_owned"] = (
            plan.band_hours + record["elapsed_owned_seconds"] / 3600
        )
        ledger["status"] = "BAND_PARENT_EXCEPTION_REQUIRES_ROOT_RECONCILIATION"
        # If journalling itself failed, the prior active reservation remains.
        # No retry of that failed write through another API.
        if not journal_failed:
            journal()
        raise


def execute(job):
    started = time.monotonic()
    if not (ROOT / ".venv/Scripts/python.exe").is_file():
        raise ValueError("Execute only in the integrated scientific root checkout.")
    plan = _admit(job, ROOT)
    supervisor = _import_supervisor(ROOT, plan.bindings)
    return _execute(plan, popen=subprocess.Popen, supervisor=supervisor, started=started)


def _execute_for_test(job, root, *, popen, supervisor, config_validator, clock, writer):
    caller = inspect.stack()[1]
    if (
        Path(caller.filename).name != "test_native_band_execution.py"
        or not caller.function.startswith("test_")
        or "SYNTHETIC_CORRECTNESS_ONLY" not in str(root)
    ):
        raise ValueError("Private injected root only inside synthetic test functions.")
    return _execute(
        _admit(job, root, config_validator=config_validator),
        popen=popen,
        supervisor=supervisor,
        clock=clock,
        write_ledger=writer,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("ssl", "downstream"))
    for flag in ("config", "review", "output", "receipt"):
        parser.add_argument("--" + flag, required=True, type=Path)
    for flag in ("encoder", "ancestor-review", "ancestor-config", "resume"):
        parser.add_argument("--" + flag, type=Path)
    args = parser.parse_args()
    raise SystemExit(execute(Job(**vars(args))))


if __name__ == "__main__":
    main()
