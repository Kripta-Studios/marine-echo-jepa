"""Private SYNTHETIC_CORRECTNESS_ONLY policy transport. No actual OS ledger/process."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def encode(value):
    return json.dumps(value, allow_nan=False, sort_keys=True).encode()


class Memory:
    def __init__(self, monkeypatch, root):
        self.root, self.files, self.dirs = root, {}, {root}
        self.mutations = []
        old_read, old_open = Path.read_bytes, Path.open
        old_exists, old_dir, old_file = Path.exists, Path.is_dir, Path.is_file
        old_mkdir, old_replace, old_unlink = Path.mkdir, Path.replace, Path.unlink

        def owned(p):
            return p == root or p.is_relative_to(root)

        def read(p):
            return self.files[p] if owned(p) else old_read(p)

        def exists(p):
            return p in self.files or p in self.dirs if owned(p) else old_exists(p)

        def directory(p):
            return p in self.dirs if owned(p) else old_dir(p)

        def isfile(p):
            return p in self.files if owned(p) else old_file(p)

        def mkdir(p, *a, **kw):
            if not owned(p):
                return old_mkdir(p, *a, **kw)
            if exists(p) and not kw.get("exist_ok"):
                raise FileExistsError(p)
            self.mutations.append(("mkdir", p))
            self.dirs.add(p)

        def replace(p, target):
            target = Path(target)
            if not owned(p):
                return old_replace(p, target)
            self.mutations.append(("replace", p, target))
            self.files[target] = self.files.pop(p)

        def unlink(p, *a, **kw):
            if not owned(p):
                return old_unlink(p, *a, **kw)
            self.mutations.append(("unlink", p))
            del self.files[p]

        fs = self

        class Sink(io.BytesIO):
            def __init__(self, p):
                super().__init__()
                self.path = p

            def write(self, value):
                return super().write(value.encode() if isinstance(value, str) else value)

            def close(self):
                if not self.closed:
                    fs.files[self.path] = self.getvalue()
                super().close()

        def opened(p, mode="r", *a, **kw):
            if not owned(p):
                return old_open(p, mode, *a, **kw)
            if "x" in mode and exists(p):
                raise FileExistsError(p)
            if "x" in mode or "w" in mode:
                self.mutations.append(("open", p, mode))
                return Sink(p)
            return io.BytesIO(read(p)) if "b" in mode else io.StringIO(read(p).decode())

        for name, fn in (
            ("read_bytes", read),
            ("exists", exists),
            ("is_dir", directory),
            ("is_file", isfile),
            ("mkdir", mkdir),
            ("replace", replace),
            ("unlink", unlink),
            ("open", opened),
        ):
            monkeypatch.setattr(Path, name, fn)


def load_generator(builder):
    spec = importlib.util.spec_from_file_location(
        "band_config_generator_under_test", builder / "tools/prepare_native_band_configs.py"
    )
    gen = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = gen
    spec.loader.exec_module(gen)
    return gen


def policy_case(job, gen, monkeypatch, builder, main, *, kind="ssl", mode=None, method=None):
    # Actual immutable Config classes, no model construction/array loading.
    definitions = gen._definitions(main)
    screens, prospective, unsupported = gen._catalog(*definitions)
    config = copy.deepcopy(
        screens["native_band_shared_ssl_screen_v1.json"]
        if kind == "ssl"
        else screens["native_band_direct_end_to_end_v1.json"]
    )
    if method is not None:
        config = copy.deepcopy(
            next(
                c
                for c in [*screens.values(), *prospective.values()]
                if c["method"] == method and (mode is None or c.get("mode") == mode)
            )
        )
    real_paths = {
        "tools/execute_native_band_job.py": builder / "tools/execute_native_band_job.py",
        "tools/prepare_native_band_configs.py": builder / "tools/prepare_native_band_configs.py",
        "tools/native_reference_supervisor.py": main / "tools/native_reference_supervisor.py",
    }
    # Fixed source-only closure from actual current MAIN, copied into virtual root.
    pending = [
        main / "src/marine_echo/training/native_band_ssl.py",
        main / "src/marine_echo/training/native_band_downstream.py",
    ]
    import ast

    seen = set()
    while pending:
        p = pending.pop()
        if p in seen:
            continue
        seen.add(p)
        real_paths[str(p.relative_to(main)).replace("\\", "/")] = p
        for node in ast.walk(ast.parse(p.read_bytes())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = node.module or ""
                names = [name, *(name + "." + a.name for a in node.names)]
            for name in names:
                if name.startswith("marine_echo."):
                    candidate = main.joinpath("src", *name.split(".")).with_suffix(".py")
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
        real_paths["src/marine_echo/" + name] = main / ("src/marine_echo/" + name)
    raw_sources = {key: p.read_bytes() for key, p in real_paths.items()}
    root = (
        builder / "evidence/ssl-band-execution-builder-v1/SYNTHETIC_CORRECTNESS_ONLY_VIRTUAL_ROOT"
    )
    fs = Memory(monkeypatch, root)
    fs.files.update({root / k: v for k, v in raw_sources.items()})
    for name in (
        "data/processed/native_ssl_v1/train.npz",
        "data/processed/native_ssl_v1/development.npz",
        "data/processed/native_ssl_v1/train.json",
        "data/processed/native_ssl_v1/development.json",
        ".venv/Scripts/python.exe",
        "pyproject.toml",
        "uv.lock",
        "docs/adr/0016-native-ssl-selection-and-source-baseline.md",
        "docs/adr/0018-native-acoustic-downstream-transfer.md",
        "docs/adr/0023-owner-resolved-band-budget.md",
        "configs/native_ssl_split_v1.json",
    ):
        fs.files[root / name] = b"SYNTHETIC_CORRECTNESS_ONLY unparsed bytes"
    budget = {
        "kind": "native_band_owner_budget_resolution_v1",
        "status": "ROOT_RESOLVED",
        "seed7_recipe_limit_including_controls": 11,
        "revision_gpu_hours_limit_full_owned": 12,
        "aggregate_gpu_hours_limit_full_owned": 96,
        "single_revision_recipes": list(job.RECIPES),
        "original_recipes": [
            "shared_ssl",
            "cf_jepa",
            "masked_ssl",
            "permuted_ssl",
            "direct",
            "random_frozen",
        ],
        "prefit_source_review": "REQUIRED_NOT_GRANTED_BY_OWNER_REPLY",
        "cloud_or_paid_resource_approval": False,
        "whole_site_final_numeric_access": "NOT_AUTHORIZED_BY_THIS_REPLY",
    }
    ledger = {
        "gpu_limit_hours": 96,
        "gpu_hours_spent_owned_scientific_jobs": 0.0,
        "scientific_fits_completed": 0,
        "gpu_hours_completed_scientific_training": 0.0,
        "runs": [],
    }
    output, receipt = (
        root / "evidence/SYNTHETIC_CORRECTNESS_ONLY-model",
        root / "evidence/SYNTHETIC_CORRECTNESS_ONLY-receipt",
    )
    request = job.Job(
        kind, root / "configs/config.json", root / "reviews/review.json", output, receipt
    )
    if kind == "downstream" and config["mode"] != "direct_end_to_end":
        parent = root / "evidence/SYNTHETIC_CORRECTNESS_ONLY-parent"
        request = job.Job(
            **{
                **vars(request),
                "encoder": parent / "selected_encoder.pt",
                "ancestor_review": parent / "review.json",
                "ancestor_config": parent / "config.json",
            }
        )
        for name in ("selected_encoder.pt", "inference.pt", "membership.json", "run.json"):
            fs.files[parent / name] = b"SYNTHETIC_CORRECTNESS_ONLY no checkpoint parsing"
        parent_config = screens[f"native_band_{config['method']}_screen_v1.json"]
        fs.files[request.ancestor_config] = encode(parent_config)
    holder = SimpleNamespace(
        root=root,
        fs=fs,
        config=config,
        budget=budget,
        ledger=ledger,
        request=request,
        started=10.0,
        now=10.0,
        calls=[],
        journals=[],
        definitions=definitions,
        screens=screens,
        prospective=prospective,
        unsupported=unsupported,
    )

    def seal():
        request = holder.request
        fs.files[request.config] = encode(config)
        fs.files[root / "orchestration/native_band_budget_owner_resolution_v1.json"] = encode(
            budget
        )
        fs.files[root / "orchestration/native_ssl_run_ledger_v1.json"] = encode(holder.ledger)
        common = {
            "reviewer_session_id": "synthetic-distinct-reviewer",
            "implementer_session_id": job.IMPLEMENTER,
            "root_coordinator_session_id": "synthetic-root",
            "architecture": job.ARCHITECTURE,
            "evidence_kind": job.EVIDENCE,
            "allowed_roles": ["train", "development"],
            "allowed_methods": [config["method"]],
            "allowed_seeds": [7],
            "budget_resolution_path": str(
                root / "orchestration/native_band_budget_owner_resolution_v1.json"
            ),
            "band_budget_status": "ROOT_RESOLVED",
        }
        if request.ancestor_review is not None:
            ancestor_bindings = {str(p): job.digest(p) for p in job.source_paths(root)}
            ancestor_bindings.update(
                {
                    str(p): job.digest(p)
                    for p in fs.files
                    if p != request.ancestor_review
                    and p.name != "review.json"
                    and p.suffix != ".npz"
                }
            )
            for p in (
                root / "data/processed/native_ssl_v1/train.npz",
                root / "data/processed/native_ssl_v1/development.npz",
            ):
                ancestor_bindings[str(p)] = job.digest(p)
            ancestor = {**common, "status": "APPROVED_PREFIT", "bindings": ancestor_bindings}
            fs.files[request.ancestor_review] = encode(ancestor)
        bindings = {
            str(p): job.digest(p)
            for p in fs.files
            if p != request.review and "ledger" not in p.name
        }
        inputs = job._fixed_inputs(root, kind)
        runtime = {
            **{k: str(v) for k, v in inputs.items()},
            "kind": kind,
            "config": str(request.config),
            "review": str(request.review),
            "trainer-review": str(request.review),
            "output": str(request.output),
            "receipt": str(request.receipt),
            "device": "cuda:0",
        }
        for k in ("encoder", "ancestor_review", "ancestor_config", "resume"):
            v = getattr(request, k)
            runtime[k.replace("_", "-")] = str(v) if v is not None else None
        review = {
            **common,
            "status": "APPROVED_PREFIT" if kind == "ssl" else "APPROVED_DOWNSTREAM_PREFIT",
            "allowed_modes": [config.get("mode", "ssl_screen")],
            "bindings": bindings,
            "approved_config": config,
            "runtime_arguments": runtime,
            "train_npz_sha256": job.digest(inputs["train"]),
            "dev_npz_sha256": job.digest(inputs["dev"]),
            "split_sha256": job.digest(inputs["split"]),
        }
        fs.files[request.review] = encode(review)
        holder.review = review

    def validate(c, k, r):
        assert r == root
        expected = [
            v
            for v in [*screens.values(), *prospective.values()]
            if ("mode" in v) == (k == "downstream")
        ]
        if c not in expected:
            raise ValueError("Fixed actual core configuration differs.")

    def writer(root_arg, value):
        assert root_arg == root
        holder.journals.append(copy.deepcopy(value))
        fs.files[root / "orchestration/native_ssl_run_ledger_v1.json"] = encode(value)

    holder.seal, holder.validator, holder.writer = seal, validate, writer
    holder.clock = lambda: holder.now

    def popen(command, **kwargs):
        holder.calls.append(command)
        fs.dirs.add(output)
        return SimpleNamespace(pid=77, poll=lambda: holder.exit_code)

    holder.popen = popen
    holder.exit_code, holder.seconds, holder.reason = 0, 60.0, None
    holder.write_report = True

    def supervisor(child, **kwargs):
        holder.supervisor_args = kwargs
        holder.now += holder.seconds
        if holder.write_report:
            fs.files[output / "inference.pt"] = b"SYNTHETIC_CORRECTNESS_ONLY fake safe output bytes"
            fs.files[output / "membership.json"] = b"SYNTHETIC_CORRECTNESS_ONLY fake membership"
            resources = {
                "peak_allocated_bytes": 100,
                "peak_reserved_bytes": 100,
                "peak_rss_bytes": 100,
                "elapsed_gpu_seconds": holder.seconds / 2,
            }
            report = {
                "status": "COMPLETED",
                "architecture": job.ARCHITECTURE,
                "evidence_kind": job.EVIDENCE,
                "correctness_smoke": False,
                "config": copy.deepcopy(config),
                "test_access": "NOT_RUN",
                "review_sha256": job.digest(request.review),
                "mode": config.get("mode"),
                "resources" if kind == "ssl" else "resources_additional_downstream": resources,
                "inference_sha256": job.digest(output / "inference.pt"),
                "membership_sha256": job.digest(output / "membership.json"),
                "fixture_identity": "SYNTHETIC_CORRECTNESS_ONLY_FAKE_REPORT_NOT_SCIENTIFIC_EVIDENCE",
            }
            fs.files[output / "run.json"] = encode(report)
        return {
            "exit_code": holder.exit_code,
            "stopped_for": holder.reason,
            "peak_process_rss_bytes": 100,
            "elapsed_full_attempt_seconds": holder.seconds,
            "deadline_seconds": kwargs["deadline_seconds"],
            "rss_limit_bytes": kwargs["rss_limit_bytes"],
            "owned_process_pids": [77, 78],
            "owned_tree_cleanup_verified": True,
        }

    holder.supervisor = supervisor
    seal()
    return holder
