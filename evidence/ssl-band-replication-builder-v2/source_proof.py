"""Stdlib-only scientific AST equivalence and immutable source-closure proof."""

import ast
import difflib
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILDER = HERE.parents[1]
MAIN = BUILDER.parent / "marine-echo-jepa"
OLD = json.loads((HERE / "v1-source-snapshot.json").read_bytes())
PAIRS = {
    "src/marine_echo/training/native_band_ssl.py": "src/marine_echo/training/native_band_replication_ssl.py",
    "src/marine_echo/training/native_band_downstream.py": "src/marine_echo/training/native_band_replication_downstream.py",
    "src/marine_echo/inference/native_band_acoustic.py": "src/marine_echo/inference/native_band_replication_acoustic.py",
    "src/marine_echo/inference/native_latent.py": "src/marine_echo/inference/native_band_replication_latent.py",
    "tools/execute_native_band_job.py": "tools/execute_native_band_replication_job.py",
}


def normalized(text):
    for a, b in (
        ("native_band_ssl", "native_band_replication_ssl"),
        ("native_band_downstream", "native_band_replication_downstream"),
        ("native_band_acoustic", "native_band_replication_acoustic"),
        ("execute_native_band_job.py", "execute_native_band_replication_job.py"),
        ("prepare_native_band_configs.py", "prepare_native_band_replication_configs.py"),
        ("test_native_band_execution.py", "test_native_band_replication_execution.py"),
    ):
        text = text.replace(a, b)
    for kind in (
        "native_band_replication_ssl_resume",
        "native_band_replication_ssl_selected_encoder",
        "native_band_replication_ssl_weights_only_inference",
        "native_band_replication_downstream_resume",
        "native_band_replication_downstream_supervised_encoder",
    ):
        text = text.replace(kind + "_v1", kind + "_v2")
    return text


def nodes(text):
    result = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, ast.FunctionDef):
                    result[node.name + "." + member.name] = member
    return result


def dump(node):
    return ast.dump(node, include_attributes=False)


equivalent, changed = {}, {}
for old, new in PAIRS.items():
    a, b = nodes(normalized(OLD[old]["text"])), nodes((BUILDER / new).read_bytes())
    equivalent[new] = [name for name in a if name in b and dump(a[name]) == dump(b[name])]
    changed[new] = [name for name in a if name not in equivalent[new]]

for new, required in {
    "src/marine_echo/training/native_band_replication_ssl.py": [
        "initialize_model",
        "batch_indices",
        "paired_indices",
        "Scalers.fit",
        "schedule",
        "rng_state",
        "restore_rng",
        "tensor_batch",
        "cpu_state",
    ],
    "src/marine_echo/training/native_band_replication_downstream.py": [
        "prepare_model",
        "supervised_indices",
        "_forecast_train",
    ],
    "src/marine_echo/inference/native_band_replication_acoustic.py": [
        "_check_state",
        "_query",
        "NativeBandAcousticPredictor.forecast",
        "NativeBandAcousticPredictor.encode",
        "_ancestry",
    ],
    "src/marine_echo/inference/native_band_replication_latent.py": [
        "NativeLatentPredictor.predict_latents",
        "NativeLatentPredictor.encode",
        "NativeLatentPredictor._batches",
        "NativeLatentPredictor._output",
    ],
    "tools/execute_native_band_replication_job.py": [
        "validate_budget",
        "_execute",
        "_write_ledger",
        "_valid_resources",
        "_completion",
    ],
}.items():
    if not set(required).issubset(equivalent[new]):
        raise ValueError(
            "Scientific/state/accounting AST changed: "
            + new
            + repr(set(required) - set(equivalent[new]))
        )

trajectory_tails = {}
for old in (
    "src/marine_echo/training/native_band_ssl.py",
    "src/marine_echo/training/native_band_downstream.py",
):
    new = PAIRS[old]
    a, b = nodes(normalized(OLD[old]["text"]))["run"], nodes((BUILDER / new).read_bytes())["run"]
    anchor = "train" if old.endswith("ssl.py") else "identities"

    def tail(node, anchor=anchor):
        index = next(
            i
            for i, item in enumerate(node.body)
            if isinstance(item, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == anchor for t in item.targets)
        )
        return [dump(item) for item in node.body[index:]]

    if tail(a) != tail(b):
        raise ValueError("Scientific run trajectory differs after admission: " + new)
    trajectory_tails[new] = "EXACT_NORMALIZED_AST_EQUAL_AFTER_ADMISSION"

# Enumerate source imports only; never inspect a corpus, ledger or tensor.
pending = [MAIN / p for p in OLD]
pending += [
    MAIN / p
    for p in (
        "pyproject.toml",
        "uv.lock",
        "src/marine_echo/__init__.py",
        "src/marine_echo/models/__init__.py",
        "src/marine_echo/training/__init__.py",
        "src/marine_echo/data/__init__.py",
        "src/marine_echo/data/native_ssl_corpus.py",
        "src/marine_echo/training/aeon_corpus.py",
        "src/marine_echo/evaluation/native_product.py",
    )
]
pending += [BUILDER / p for p in PAIRS.values()]
pending += [
    BUILDER / "src/marine_echo/inference/native_band_replication_encoder.py",
    BUILDER / "tools/prepare_native_band_replication_configs.py",
]
closure = {}
while pending:
    path = pending.pop().resolve()
    if str(path) in closure:
        continue
    raw = path.read_bytes()
    closure[str(path)] = hashlib.sha256(raw).hexdigest()
    if path.suffix != ".py":
        continue
    for node in ast.walk(ast.parse(raw)):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            names = [name, *(name + "." + a.name for a in node.names)]
        for name in names:
            if name.startswith("marine_echo."):
                relative = Path("src").joinpath(*name.split(".")).with_suffix(".py")
                root = BUILDER if "native_band_replication" in relative.name else MAIN
                candidate = root / relative
                if candidate.is_file():
                    pending.append(candidate)
for p, record in OLD.items():
    if hashlib.sha256((MAIN / p).read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("Original immutable source changed: " + p)
baseline = HERE / "protected-source-baseline.json"
protected = {p: digest for p, digest in closure.items() if Path(p).is_relative_to(MAIN)}
if baseline.exists() and any(
    protected.get(path) != digest for path, digest in json.loads(baseline.read_bytes()).items()
):
    raise ValueError("Imported protected main closure changed.")

result = {
    "kind": "native_band_replication_scientific_ast_proof_v2",
    "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
    "normalized_equal_functions": equivalent,
    "bounded_changed_functions": changed,
    "scientific_run_tails": trajectory_tails,
    "source_closure_sha256": closure,
    "protected_main_unchanged": True,
    "architecture": "nonlinear_frequency_conditioned_v1",
    "model_source_sha256": OLD["src/marine_echo/models/native_band_temporal.py"]["sha256"],
    "budget_family": "native_band_v1",
    "scientific_approval": False,
}
mode = sys.argv[1] if len(sys.argv) > 1 else "proof"
if mode == "proof":
    print(json.dumps(result, indent=2))
elif mode == "baseline":
    print(json.dumps(protected, indent=2))
elif mode == "diff":
    print(
        "".join(
            "".join(
                difflib.unified_diff(
                    OLD[a]["text"].splitlines(keepends=True),
                    (BUILDER / b).read_text(encoding="utf-8").splitlines(keepends=True),
                    fromfile=a,
                    tofile=b,
                )
            )
            for a, b in PAIRS.items()
        ),
        end="",
    )
else:
    raise ValueError("Explicit bounded read-only proof operation required.")
