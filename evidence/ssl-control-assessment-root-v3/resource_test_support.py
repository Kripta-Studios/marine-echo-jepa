"""SYNTHETIC_CORRECTNESS_ONLY virtual filesystem/process policy transport.

No Popen, GPU, scientific ledger or actual process termination is executed.
"""

import copy
from pathlib import Path
from types import SimpleNamespace


def resource_case(helpers, monkeypatch, *, cuda=False, band=False):
    fs = helpers.MemoryFS(monkeypatch)
    case = helpers.Case(fs, learned=band)
    wrapper = helpers.wrapper
    root = fs.base / "SYNTHETIC_CORRECTNESS_ONLY_root"
    fs.dirs.update(
        {
            root,
            root / "evidence",
            root / "orchestration",
            root / "evidence/ssl-builder-v1",
            root / "receipt-parent",
        }
    )
    case.output = root / "evidence/out"
    actual = helpers.MAIN
    # Explicit private import transport for the one fixed budget-policy source.
    # It exists before first import; no disk error or denial is rerouted.
    import importlib.machinery

    original_get_data = importlib.machinery.SourceFileLoader.get_data

    def get_data(loader, name):
        p = Path(name)
        if p.is_relative_to(fs.base):
            if p not in fs.files:
                raise FileNotFoundError(p)
            return fs.files[p]
        return original_get_data(loader, name)

    monkeypatch.setattr(importlib.machinery.SourceFileLoader, "get_data", get_data)
    sources = helpers.assessment.required_sources()
    for p in sources:
        relative = (
            p.relative_to(actual / "src/marine_echo")
            if p.is_relative_to(actual / "src/marine_echo")
            else Path("evaluation/native_assessment_controls_v3.py")
        )
        fs.files[root / "src/marine_echo" / relative] = p.read_bytes()
    for name in (
        "execute_native_control_assessment_worker_v3.py",
        "execute_native_band_replication_job.py",
        "native_reference_supervisor.py",
    ):
        p = (
            (
                helpers.BUILDER
                if name == "execute_native_control_assessment_worker_v3.py"
                else actual
            )
            / "tools"
            / name
        )
        fs.files[root / "tools" / name] = p.read_bytes()
    fs.files[root / ".venv/Scripts/python.exe"] = (
        b"SYNTHETIC fake child transport; never executable"
    )
    for name in (wrapper.OWNER, "docs/adr/0023-owner-resolved-band-budget.md"):
        fs.files[root / name] = (actual / name).read_bytes()
    fs.files[root / wrapper.LEDGER] = helpers._encoded(
        {"gpu_limit_hours": 96, "gpu_hours_spent_owned_scientific_jobs": 0.0, "runs": []}
    )
    # Extend the explicit virtual FS, never redirect an attempted denied write.
    originals = {name: getattr(Path, name) for name in ("is_dir", "replace", "unlink")}
    monkeypatch.setattr(
        Path,
        "is_dir",
        lambda p: p in fs.dirs if p.is_relative_to(fs.base) else originals["is_dir"](p),
    )

    def replace(p, target):
        target = Path(target)
        if not p.is_relative_to(fs.base) or not target.is_relative_to(fs.base):
            raise AssertionError("Only virtual journal writes are permitted")
        fs.files[target] = fs.files.pop(p)
        return target

    def unlink(p, missing_ok=False):
        if not p.is_relative_to(fs.base):
            raise AssertionError("Only virtual owned-lock cleanup is permitted")
        if p not in fs.files and not missing_ok:
            raise FileNotFoundError(p)
        fs.files.pop(p, None)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(Path, "unlink", unlink)
    case.spec["loader"] = "builtin"
    if band:
        from marine_echo.training.native_band_replication_ssl import Config

        case.config.clear()
        case.config.update(Config(seed=13).to_dict())
        case.spec.update(kind="native_band_replication_ssl_weights_only_inference_v2", seed=13)
        case.manifest["batch_size"] = 64
    if cuda:
        case.manifest.update(
            device="cuda:0", evidence_kind="REVIEWED_FROZEN_ASSESSMENT", input_evidence_kind=None
        )
        case.documents["cohort.json"]["evidence_kind"] = case.manifest["evidence_kind"]
        if band:
            case.documents["train-cohort.json"]["evidence_kind"] = case.manifest["evidence_kind"]
    receipt = root / "evidence/attempt-receipt"
    runtime = {
        "manifest": str(case.manifest_path),
        "review": str(case.review_path),
        "output": str(case.output),
        "receipt": str(receipt),
        "device": case.manifest["device"],
    }
    case.review_overrides.update(
        device=case.manifest["device"],
        batch_size=case.manifest["batch_size"],
        allowed_loader_ids=["builtin"],
        receipt_path=str(receipt),
        runtime_arguments=runtime,
        band_budget_status="ROOT_RESOLVED",
        budget_resolution_path=str(root / wrapper.OWNER),
    )
    case.seal()
    case.review["bindings"].update(
        {
            str(p): wrapper.sha(raw)
            for p, raw in fs.files.items()
            if p.is_relative_to(root) and p != root / wrapper.LEDGER
        }
    )
    case.review["bindings"].update(
        {
            str(p): wrapper.digest(p)
            for p in wrapper.required_sources(package=root / "src/marine_echo")
        }
    )
    fs.files[case.review_path] = helpers._encoded(case.review)
    job = wrapper.Job(case.manifest_path, case.review_path, case.output, receipt)
    return SimpleNamespace(
        case=case, fs=fs, root=root, job=job, plan=lambda: wrapper.prelaunch(job, _root=root)
    )


def add_attempt(
    helpers, fixture, seconds=3600.0, *, status="FAILED_REAL_BAND_CUDA_ATTEMPT", family=True
):
    w = helpers.wrapper
    ledger = w.read_json(fixture.root / w.LEDGER)
    ledger["gpu_hours_spent_owned_scientific_jobs"] += (
        seconds / 3600 if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) else 0
    )
    ledger["runs"].append(
        {
            "status": status,
            "budget_family": w.FAMILY if family else None,
            "architecture": "nonlinear_frequency_conditioned_v1",
            "elapsed_owned_seconds": seconds,
            "resources_full_attempt": {"elapsed_full_attempt_seconds": seconds},
        }
    )
    fixture.fs.files[fixture.root / w.LEDGER] = helpers._encoded(ledger)


def add_v1_method(helpers, fixture):
    """Independent typed selection/config/fit receipt, same fixed support."""
    c, w = fixture.case, helpers.wrapper
    spec = copy.deepcopy(c.spec)
    spec.update(
        kind="native_band_ssl_weights_only_inference_v1",
        seed=7,
        config_path=str(c.base / "v1-config.json"),
        selection_path=str(c.base / "v1-selection.json"),
        ancestry_path=str(c.base / "v1-ancestry.json"),
    )
    config = {**c.config, "seed": 7}
    c.documents["v1-config.json"] = config
    c.documents["v1-fit-manifest.json"] = copy.deepcopy(c.documents["fit-manifest.json"])
    ancestry = copy.deepcopy(c.documents["ancestry.json"])
    ancestry["fit_inputs"][0].update(
        config_path=spec["config_path"], manifest_path=str(c.base / "v1-fit-manifest.json")
    )
    c.documents["v1-ancestry.json"] = ancestry
    c.documents["v1-selection.json"] = copy.deepcopy(c.documents["selection.json"])
    c.manifest["methods"]["v1"] = spec
    c.seal()
    config_hash = w.sha(helpers._encoded(config))
    c.documents["v1-selection.json"]["config_sha256"] = config_hash
    ancestry["config_sha256"] = config_hash
    fit = c.documents["v1-fit-manifest.json"]
    fit["config_sha256"] = config_hash
    for name in ("v1-config.json", "v1-fit-manifest.json", "v1-ancestry.json", "v1-selection.json"):
        c.fs.files[c.base / name] = helpers._encoded(c.documents[name])
        c.review["bindings"][str(c.base / name)] = w.digest(c.base / name)
    c.review["bindings"].update(
        {
            str(p): w.sha(raw)
            for p, raw in c.fs.files.items()
            if p.is_relative_to(fixture.root) and p != fixture.root / w.LEDGER
        }
    )
    c.review["bindings"].update(
        {str(p): w.digest(p) for p in w.required_sources(package=fixture.root / "src/marine_echo")}
    )
    c.fs.files[c.review_path] = helpers._encoded(c.review)


def virtual_execute(
    helpers, fixture, *, exit_code=0, report=True, exception=False, lock="none", extra_seconds=120.0
):
    w = helpers.wrapper
    plan = fixture.plan()
    child = SimpleNamespace(pid=8123, poll=lambda: exit_code)
    ticks = iter([10.0, 10.0 + extra_seconds, 11.0 + extra_seconds, 12.0 + extra_seconds])

    def supervisor(child, **arguments):
        if exception:
            raise RuntimeError("SYNTHETIC supervisor exception")
        if lock != "none":
            fixture.fs.files[fixture.root / w.LOCK] = helpers._encoded(
                {"pid": child.pid if lock == "owned" else 99999, "output": str(plan.job.output)}
            )
        if report:
            fixture.fs.dirs.add(plan.job.output)
            outputs = {}
            for name, spec in plan.admission.manifest["methods"].items():
                artifact = plan.job.output / (name + ".npz")
                fixture.fs.files[artifact] = b"SYNTHETIC prediction bytes; no actual worker"
                outputs[name] = {k: spec[k] for k in ("kind", "method", "seed", "mode")}
                outputs[name].update(
                    prediction_path=str(artifact), prediction_sha256=w.digest(artifact)
                )
            fixture.fs.files[plan.job.output / "completion.json"] = helpers._encoded(
                {
                    "kind": "native_control_frozen_assessment_completion_v3",
                    "status": "COMPLETED_FORECASTS",
                    "role": plan.admission.manifest["role"],
                    "evidence_kind": plan.admission.manifest["evidence_kind"],
                    "provenance": copy.deepcopy(plan.admission.provenance),
                    "methods": outputs,
                    "resources": {
                        "device": plan.admission.manifest["device"],
                        "process_id": child.pid,
                        "sampled_peak_rss_bytes": 100,
                        "process_cuda_peak_allocated_bytes": 0,
                        "process_cuda_peak_reserved_bytes": 0,
                    },
                }
            )
        return {
            "owned_process_pids": [child.pid],
            "owned_tree_cleanup_verified": True,
            "exit_code": exit_code,
            "peak_process_rss_bytes": 100,
            "elapsed_full_attempt_seconds": extra_seconds,
            "deadline_seconds": arguments["deadline_seconds"],
            "rss_limit_bytes": arguments["rss_limit_bytes"],
            "stopped_for": "SYNTHETIC_OWNED_STOP" if lock != "none" else None,
        }

    return w._execute(
        plan, popen=lambda *a, **k: child, supervisor=supervisor, clock=lambda: next(ticks)
    )
