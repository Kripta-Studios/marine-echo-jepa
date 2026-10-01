"""Guard one separately reviewed CF control; preserve original campaign caps.

The worker is the supervised Popen tree, including the Windows Python redirector.
It records its actual interpreter PID. No reviewer operation is launched here.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
import stat
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic as clock

from native_cf_control_budget_v1 import CONTROLS, FAMILY, STUDY, Allowance, limits
from native_reference_supervisor import supervise_owned

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTER = "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1"
REAL = "REAL_TRAIN_DEVELOPMENT_FIT"
PARENT_RAM_RESERVE = 128 * 2**20
CHILD_RAM_LIMIT = 22 * 2**30 - PARENT_RAM_RESERVE


def regular(path):
    path = Path(os.path.abspath(path))
    for p in (path, *path.parents):
        info = p.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Unsafe reparse/symlink path")
    if not path.is_file():
        raise ValueError("Regular input required")
    return path


def digest(path):
    with regular(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("Duplicate JSON key")
            out[key] = value
        return out

    def bad(value):
        raise ValueError("Nonfinite JSON")

    value = json.loads(regular(path).read_bytes(), object_pairs_hook=pairs, parse_constant=bad)
    if not isinstance(value, dict):
        raise TypeError("JSON object required")
    return value


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_ledger(root, value):
    path = root / "orchestration/native_ssl_run_ledger_v1.json"
    temporary = path.with_suffix(".pending")
    write_new(temporary, value)
    temporary.replace(path)


def catalog():
    result = {}
    for method in CONTROLS:
        for seed in (7, 13, 23):
            frozen = method == "cf_random_frozen"
            result[f"{method}_seed{seed}_v1"] = {
                "method": method,
                "seed": seed,
                "history": 96,
                "updates": 2000 if frozen else 3000,
                "cadence": 500 if frozen else 750,
                "batch_size": 64,
                "width": 256,
                "latent": 128,
                "blocks": 5,
                "heads": 4,
                "lr": 0.0003,
                "weight_decay": 0.0001,
                "gradient_clip": 1.0,
                "patience": 4,
                "min_daily_anchors": 18,
                "selection_policy": "scheduled_native_daily_pinball_earliest_strict_improvement_v1",
            }
    return result


def review_identity(review, *, implementer=IMPLEMENTER):
    if review.get("status") != "APPROVED_CF_CONTROL_PREFIT":
        raise ValueError("Genuine distinct CF control prefit required")
    identities = [
        review.get(k)
        for k in ("implementer_session_id", "root_coordinator_session_id", "reviewer_session_id")
    ]
    if (
        any(not isinstance(v, str) or not v or v != v.strip() for v in identities)
        or len({v.casefold() for v in identities}) != 3
        or identities[0] != implementer
    ):
        raise ValueError("Distinct exact author/root/reviewer identities required")
    if review.get("evidence_kind") != REAL or review.get("allowed_roles") != [
        "train",
        "development",
    ]:
        raise ValueError("Only exact real TRAIN/development evidence and roles admitted")


def source_paths(root):
    package = root / "src/marine_echo"
    pending = [
        package / "training/native_cf_controls.py",
        package / "inference/native_cf_controls.py",
        package / "training/native_downstream.py",
        package / "inference/native_acoustic.py",
        package / "data/native_ssl_corpus.py",
        package / "training/aeon_corpus.py",
        package / "evaluation/native_product.py",
    ]
    found = set()
    while pending:
        path = regular(pending.pop())
        if path in found:
            continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_bytes())):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else (
                    [node.module] + [node.module + "." + a.name for a in node.names]
                    if isinstance(node, ast.ImportFrom) and node.module
                    else []
                )
            )
            for name in names:
                if name == "marine_echo" or name.startswith("marine_echo."):
                    parts = name.split(".")[1:]
                    for count in range(len(parts) + 1):
                        base = package / Path(*parts[:count])
                        for file in (base.with_suffix(".py"), base / "__init__.py"):
                            if file.is_file():
                                pending.append(file)
    author = root / "external/cf-jepa-vnext"
    return sorted(
        found
        | {
            root / "tools" / name
            for name in (
                "execute_native_cf_control_job_v1.py",
                "native_cf_control_budget_v1.py",
                "native_reference_supervisor.py",
            )
        }
        | {
            author / "baselines/cf_jepa" / name
            for name in ("encoder.py", "trainer.py", "losses.py", "cf_jepa.py")
        }
        | {author / "LICENSE"}
    )


@dataclass(frozen=True)
class Job:
    config: Path
    review: Path
    output: Path
    receipt: Path
    resume: Path | None = None


@dataclass
class Plan:
    job: Job
    root: Path
    config: dict
    review: dict
    bindings: dict
    ledger: dict
    ledger_hash: str
    allowance: Allowance
    inputs: dict


def admit(job, root=ROOT):
    root = Path(root).resolve()
    job = Job(**{k: Path(v).resolve() if v is not None else None for k, v in asdict(job).items()})
    review = read(job.review)
    review_identity(review)  # No source loading, imports, mkdir or Popen before this.
    config = read(job.config)
    if config not in catalog().values() or any(
        type(config[k]) is not int
        for k in (
            "seed",
            "history",
            "updates",
            "cadence",
            "batch_size",
            "width",
            "latent",
            "blocks",
            "heads",
            "patience",
            "min_daily_anchors",
        )
    ):
        raise ValueError("One of the six fixed exact CF control recipes required")
    identifier = next(k for k, v in catalog().items() if v == config)
    if (
        job.output != root / "outputs/native_cf_matched_controls_v1" / identifier
        or not job.receipt.is_relative_to(root / "evidence/ssl-research-v1")
        or job.receipt.exists()
    ):
        raise ValueError("Fixed new control output and fresh root-owned receipt required")
    for destination in (job.output, job.receipt):
        for p in destination.parents:
            if p.exists() and (
                p.is_symlink() or getattr(p.lstat(), "st_file_attributes", 0) & 0x400
            ):
                raise ValueError("Unsafe destination ancestor")
    if job.output.exists() and job.resume is None or (job.output / "run.json").exists():
        raise FileExistsError("Preserve prior output; resume explicit unfinished latest.pt only")
    if job.resume is not None and (
        job.resume != job.output / "latest.pt" or not job.resume.is_file()
    ):
        raise ValueError("Exact owned latest.pt resume required")
    budget = root / "orchestration/native_cf_control_budget_owner_resolution_v1.json"
    inputs = {
        "train": str(root / "data/processed/native_ssl_v1/train.npz"),
        "dev": str(root / "data/processed/native_ssl_v1/development.npz"),
        "train_cohort": str(root / "data/processed/native_ssl_v1/train.json"),
        "dev_cohort": str(root / "data/processed/native_ssl_v1/development.json"),
        "split": str(root / "configs/native_ssl_split_v1.json"),
        "adr0016": str(root / "docs/adr/0016-native-ssl-selection-and-source-baseline.md"),
        "protocol": str(root / "docs/adr/0025-cf-matched-controls-continuation.md"),
        "config": str(job.config),
        "review": str(Path(review.get("trainer_review_path", job.review)).resolve()),
        "executor": str(root / "tools/execute_native_cf_control_job_v1.py"),
        "budget": str(budget),
    }
    runtime = {
        **inputs,
        "review": str(job.review),
        "trainer_review": inputs["review"],
        "output": str(job.output),
        "receipt": str(job.receipt),
        "resume": str(job.resume) if job.resume else None,
        "device": "cuda:0",
    }
    if review.get("execution_runtime") != runtime or review.get("approved_config") != config:
        raise ValueError("Exact reviewed wrapper arguments and configuration required")
    if review.get("allowed_methods") != [config["method"]] or review.get("allowed_seeds") != [
        config["seed"]
    ]:
        raise ValueError("Only the exact reviewed control cell admitted")
    required = [
        *source_paths(root),
        *[Path(v) for k, v in inputs.items() if k != "review"],
        job.config,
        root / "docs/adr/0022-cf-backbone-matched-controls.md",
        root / "orchestration/ssl_vnext_cf_matched_controls_builder_contract.txt",
        root / "orchestration/native_cf_control_owner_scope_v1.json",
        root / ".venv/Scripts/python.exe",
        root / "pyproject.toml",
        root / "uv.lock",
    ]
    if inputs["review"] != str(job.review):
        required.append(Path(inputs["review"]))
    if job.resume:
        required.append(job.resume)
    bindings = review.get("bindings", {})
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("Nonempty exact source/input bindings required")
    for value, expected in bindings.items():
        p = Path(value)
        if (
            not p.is_absolute()
            or str(p.resolve()) != value
            or not isinstance(expected, str)
            or len(expected) != 64
            or digest(p) != expected
        ):
            raise ValueError("Changed or noncanonical reviewed binding")
        if p.suffix.lower() == ".npz" and value not in (inputs["train"], inputs["dev"]):
            raise ValueError("Reserved/non-TRAIN-development NPZ prohibited")
    if any(str(regular(p)) not in bindings for p in required):
        raise ValueError("Incomplete executor/source/data/budget bindings")
    trainer_review = read(inputs["review"])
    review_identity(trainer_review)
    if trainer_review.get("approved_config") != config or trainer_review.get("runtime") != {
        "output": str(job.output),
        "device": "cuda:0",
        "executor": inputs["executor"],
        "budget": inputs["budget"],
    }:
        raise ValueError("Unchanged trainer review and runtime required")
    for p, expected in trainer_review.get("bindings", {}).items():
        if digest(p) != expected:
            raise ValueError("Original trainer approval binding changed")
    resolution = read(budget)
    for k, v in {
        "status": "ROOT_RESOLVED",
        "study": STUDY,
        "fits": 6,
        "cf_gpu_hours": 12,
        "aggregate_gpu_hours": 96,
        "evaluation_reserve_hours": 12,
        "seeds": [7, 13, 23],
    }.items():
        if resolution.get(k) != v or (type(v) is int and type(resolution.get(k)) is not int):
            raise ValueError("Unchanged owner budget resolution required")
    ledger_path = root / "orchestration/native_ssl_run_ledger_v1.json"
    if (
        (root / "evidence/ssl-builder-v1/gpu-owner.lock").exists()
        or list(ledger_path.parent.glob("*.pending"))
        or list(ledger_path.parent.glob("*pending"))
    ):
        raise ValueError("Existing ownership or pending journal preserved")
    ledger = read(ledger_path)
    allowance = limits(ledger)
    if any(
        r.get("budget_family") == FAMILY
        and r.get("status") == "COMPLETED_REAL_CF_CONTROL"
        and r.get("method") == config["method"]
        and r.get("seed") == config["seed"]
        for r in ledger["runs"]
    ):
        raise ValueError("Completed control cell cannot be retrained")
    return Plan(
        job, root, config, review, dict(bindings), ledger, digest(ledger_path), allowance, inputs
    )


def completed_report(plan):
    report = read(plan.job.output / "run.json")
    if (
        report.get("status") != "COMPLETED"
        or report.get("evidence_kind") != REAL
        or report.get("config") != plan.config
    ):
        raise ValueError("Exact completed real control report required")
    count, selected = report.get("supervised_updates"), report.get("selected_supervised_step")
    if (
        type(count) is not int
        or not 1 <= count <= plan.config["updates"]
        or type(selected) is not int
        or selected not in range(plan.config["cadence"], count + 1, plan.config["cadence"])
    ):
        raise ValueError("Actual scheduled optimizer and selection counts required")
    ancestry = report.get("supervised_ancestry", {})
    if (
        ancestry.get("ssl_updates") != 0
        or ancestry.get("ancestor_encoder_sha256") is not None
        or ancestry.get("readout_initialization_seed") != plan.config["seed"] + 100000
    ):
        raise ValueError("Parent-free zero-SSL control lineage required")
    resources = report.get("resources_additional_downstream", {})
    if (
        max(
            resources.get("peak_allocated_bytes", math.inf),
            resources.get("peak_reserved_bytes", math.inf),
        )
        >= 10 * 2**30
        or resources.get("peak_rss_bytes", math.inf) >= CHILD_RAM_LIMIT
    ):
        raise ValueError("Control report resource ceiling")
    for name, filename in (
        ("inference", "inference.pt"),
        ("membership", "membership.json"),
        ("selected_encoder", "selected_encoder.pt"),
    ):
        if report.get(name + "_sha256") != digest(plan.job.output / filename):
            raise ValueError("Saved control artifacts disagree with completion")
    return report


def execute_plan(plan, *, popen=subprocess.Popen, supervisor=supervise_owned):
    started = clock()
    root, job = plan.root, plan.job
    ledger_path = root / "orchestration/native_ssl_run_ledger_v1.json"
    lock = root / "evidence/ssl-builder-v1/gpu-owner.lock"
    if digest(ledger_path) != plan.ledger_hash or lock.exists():
        raise ValueError("Scientific ownership changed after admission")
    for p, expected in plan.bindings.items():
        if digest(p) != expected:
            raise ValueError("Changed admitted source/input")
    job.receipt.mkdir(parents=True, exist_ok=False)
    ledger = copy.deepcopy(plan.ledger)
    record = {
        "id": job.receipt.name,
        "study": STUDY,
        "budget_family": FAMILY,
        "status": "RUNNING_CUDA",
        "method": plan.config["method"],
        "seed": plan.config["seed"],
        "config": str(job.config),
        "output": str(job.output),
        "receipt": str(job.receipt),
        "review": str(job.review),
        "requires_reconciliation": True,
        "bindings": plan.bindings,
    }
    ledger["runs"].append(record)
    ledger["status"] = "CF_CONTROL_RUNNING"
    write_ledger(root, ledger)  # Must precede Popen.
    journal_failed = False
    try:
        import psutil

        if psutil.Process().memory_info().rss >= PARENT_RAM_RESERVE:
            raise RuntimeError("Executor exceeded its reserved RAM allowance")
        payload = {
            "root": str(root),
            "job": {k: str(v) if v is not None else None for k, v in asdict(job).items()},
            "inputs": plan.inputs,
            "bindings": plan.bindings,
            "config": plan.config,
            "allowance": asdict(plan.allowance),
            "parent_pid": os.getpid(),
            "parent_create_time": psutil.Process().create_time(),
        }
        launch = job.receipt / "launch.json"
        write_new(launch, payload)
        command = [
            str(root / ".venv/Scripts/python.exe"),
            "-B",
            str(root / "tools/execute_native_cf_control_job_v1.py"),
            "--worker",
            str(launch),
            "--launch-sha256",
            digest(launch),
        ]
        record["command"] = command
        with (job.receipt / "console.log").open("x", encoding="utf-8") as log:
            child = popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            record["pid"] = child.pid
            write_ledger(root, ledger)
            resources = supervisor(
                child,
                started=started,
                deadline_seconds=plan.allowance.deadline_seconds,
                rss_limit_bytes=CHILD_RAM_LIMIT,
            )
        if (
            resources.get("owned_tree_cleanup_verified") is not True
            or type(resources.get("exit_code")) is not int
        ):
            raise ValueError("Actual owned cleanup and exit witness required")
        elapsed = max(resources["elapsed_full_attempt_seconds"], clock() - started)
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid full-owned lifetime")
        complete = (
            resources["exit_code"] == 0
            and resources.get("stopped_for") is None
            and not lock.exists()
            and elapsed < plan.allowance.deadline_seconds
            and resources["peak_process_rss_bytes"] < CHILD_RAM_LIMIT
        )
        report, reason = None, None
        if complete:
            try:
                report = completed_report(plan)
            except (ValueError, OSError) as error:
                complete, reason = False, str(error)
        elapsed = max(elapsed, clock() - started)
        complete = complete and elapsed < plan.allowance.deadline_seconds
        resources["elapsed_full_attempt_seconds"] = elapsed
        record.update(
            status="COMPLETED_REAL_CF_CONTROL" if complete else "FAILED_REAL_CF_CONTROL_ATTEMPT",
            elapsed_owned_seconds=elapsed,
            resources_full_attempt=resources,
            verified_completion=complete,
            requires_reconciliation=lock.exists(),
            failure_reason=reason,
            owned_tree_cleanup_verified=True,
        )
        ledger["gpu_hours_spent_owned_scientific_jobs"] = (
            plan.allowance.aggregate_hours + elapsed / 3600
        )
        ledger["native_cf_control_hours_spent_full_owned"] = (
            plan.allowance.extension_hours + elapsed / 3600
        )
        if complete:
            record["run_sha256"] = digest(job.output / "run.json")
            ledger["scientific_fits_completed"] = ledger.get("scientific_fits_completed", 0) + 1
            ledger["gpu_hours_completed_scientific_training"] = (
                ledger.get("gpu_hours_completed_scientific_training", 0)
                + report["resources_additional_downstream"]["elapsed_gpu_seconds"] / 3600
            )
        ledger["status"] = "CF_CONTROL_COMPLETED" if complete else "CF_CONTROL_FAILED"
        write_new(job.receipt / "attempt.json", record)
        try:
            write_ledger(root, ledger)
        except BaseException:
            journal_failed = True
            raise
        return 0 if complete else resources["exit_code"] or 1
    except BaseException as error:
        record.update(
            status="RUNNING_CUDA",
            requires_reconciliation=True,
            parent_exception=type(error).__name__,
            elapsed_owned_seconds=max(0, clock() - started),
        )
        record.pop("owned_tree_cleanup_verified", None)
        ledger["gpu_hours_spent_owned_scientific_jobs"] = (
            plan.allowance.aggregate_hours + record["elapsed_owned_seconds"] / 3600
        )
        ledger["native_cf_control_hours_spent_full_owned"] = (
            plan.allowance.extension_hours + record["elapsed_owned_seconds"] / 3600
        )
        if not journal_failed:
            write_ledger(root, ledger)
        raise


def worker(path, expected):
    if digest(path) != expected:
        raise ValueError("Launch bytes changed")
    payload = read(path)
    if Path(payload["root"]).resolve() != ROOT:
        raise ValueError("Worker must use this actual checkout")
    import psutil

    parents = {p.pid: p for p in psutil.Process().parents()}
    parent = parents.get(payload["parent_pid"])
    if parent is None or parent.create_time() != payload["parent_create_time"]:
        raise ValueError("Original executor is not a live ancestor")
    for p, sha in payload["bindings"].items():
        if digest(p) != sha:
            raise ValueError("Source/input changed before worker admission")
    inputs, config, allowance = payload["inputs"], payload["config"], payload["allowance"]
    execution = path.parent / "execution.json"
    write_new(
        execution,
        {
            "kind": "native_cf_control_owned_execution_v1",
            "status": "RUNNING_CUDA",
            "child_pid": os.getpid(),
            "review_sha256": digest(inputs["review"]),
            "executor_sha256": digest(inputs["executor"]),
            "budget_sha256": digest(inputs["budget"]),
            "config_sha256": digest(inputs["config"]),
            "output": payload["job"]["output"],
            "device": "cuda:0",
            "method": config["method"],
            "seed": config["seed"],
            "remaining_cf_hours": 12 - allowance["extension_hours"],
            "remaining_aggregate_hours": 96 - allowance["aggregate_hours"],
            "deadline_seconds": allowance["deadline_seconds"],
        },
    )
    from marine_echo.training.native_cf_controls import Config, RunInputs, run

    report = run(
        RunInputs(**{k: Path(v) for k, v in inputs.items()}, execution=execution),
        Config(**config),
        output=Path(payload["job"]["output"]),
        device="cuda:0",
        resume=Path(payload["job"]["resume"]) if payload["job"]["resume"] else None,
    )
    print(
        json.dumps({"status": report["status"], "supervised_updates": report["supervised_updates"]})
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--launch-sha256")
    for name in ("config", "review", "output", "receipt", "resume"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.launch_sha256)
    else:
        if any(getattr(args, k) is None for k in ("config", "review", "output", "receipt")):
            parser.error("config/review/output/receipt are required")
        raise SystemExit(
            execute_plan(
                admit(Job(args.config, args.review, args.output, args.receipt, args.resume))
            )
        )


if __name__ == "__main__":
    main()
