"""Stdlib-only immutable source proof and bounded unified diff, no numeric reads."""

import ast
import difflib
import hashlib
import json
from pathlib import Path

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
HERE = Path(__file__).resolve().parent
snapshot = json.loads((HERE / "main-source-snapshot-v1.json").read_bytes())["files"]
pending = [MAIN / p for p in snapshot]
pending.extend(
    MAIN / p
    for p in (
        "src/marine_echo/training/native_downstream.py",
        "src/marine_echo/training/native_band_downstream.py",
        "src/marine_echo/data/native_ssl_corpus.py",
    )
)
found = set()
while pending:
    p = pending.pop()
    if p in found:
        continue
    found.add(p)
    for node in ast.walk(ast.parse(p.read_bytes())):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module, *(node.module + "." + a.name for a in node.names)]
        for name in names:
            if name.startswith("marine_echo."):
                candidate = MAIN / "src" / Path(*name.split(".")).with_suffix(".py")
                if candidate.is_file():
                    pending.append(candidate)
found.update(
    MAIN / p
    for p in (
        "configs/native_ssl_split_v1.json",
        "docs/adr/0021-native-prefix-transfer-assessment.md",
        "pyproject.toml",
        "uv.lock",
        "evidence/ssl-prefix-builder-v1/prefix_test_support.py",
    )
)
protected = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(found)}
older = {}
for delivery in ("ssl-band-execution-builder-v1", "ssl-latent-builder-v1"):
    h = json.loads((BUILDER / "evidence" / delivery / "handoff-v1.json").read_bytes())
    for p, expected in h["authored_sha256"].items():
        actual = hashlib.sha256((BUILDER / p).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("Unassigned closed source changed.")
        older[p] = actual
for name in ("builder", "main"):
    saved = json.loads((HERE / (name + "-source-snapshot-v1.json")).read_bytes())
    for p, record in saved["files"].items():
        if hashlib.sha256(record["text"].encode()).hexdigest() != record["sha256"]:
            raise ValueError("Snapshot bytes corrupted.")
if (HERE / "protected-baseline-v1.json").exists():
    prior = json.loads((HERE / "protected-baseline-v1.json").read_bytes())
    if protected != prior["protected_main_sha256"]:
        raise ValueError("Main source/config/split/protocol changed.")
diff = "".join(
    "".join(
        difflib.unified_diff(
            snapshot[p]["text"].splitlines(keepends=True),
            (BUILDER / p).read_text(encoding="utf-8").splitlines(keepends=True),
            fromfile="main/" + p,
            tofile="builder/" + p,
        )
    )
    for p in snapshot
)
source = "src/marine_echo/training/native_prefix_transfer.py"
before, after = ast.parse(snapshot[source]["text"]), ast.parse((BUILDER / source).read_bytes())


def nodes(tree):
    result = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            result[node.name] = ast.dump(node, include_attributes=False)
        elif isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, ast.FunctionDef):
                    result[node.name + "." + member.name] = ast.dump(
                        member, include_attributes=False
                    )
    return result


old, new = nodes(before), nodes(after)
unchanged = (
    "boundaries",
    "sha",
    "_pairs",
    "_json",
    "_timestamp",
    "_path",
    "_distinct",
    "_strict_state",
    "decode_checkpoint",
    "encode_checkpoint",
    "_batch",
    "sample_indices",
    "predict",
    "PrefixModel.__init__",
    "PrefixModel.forecast",
    "PrefixPredictor.__init__",
    "PrefixPredictor.forecast",
    "_Trajectory.__init__",
    "_Trajectory.advance",
)
for name in unchanged:
    if old[name] != new[name]:
        raise ValueError("Frozen recipe/computation changed: " + name)


def fields(tree):
    return [
        ast.dump(n, include_attributes=False)
        for c in tree.body
        if isinstance(c, ast.ClassDef) and c.name == "PrefixConfig"
        for n in c.body
        if isinstance(n, ast.AnnAssign)
    ]


if fields(before) != fields(after):
    raise ValueError("Frozen configuration fields/defaults changed.")
packages = {}
for name in ("", "training", "models", "evaluation", "inference", "data"):
    p = MAIN / "src/marine_echo" / name / "__init__.py"
    if p.is_file():
        packages[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
if (HERE / "runtime-package-baseline-v1.json").exists() and packages != json.loads(
    (HERE / "runtime-package-baseline-v1.json").read_bytes()
)["runtime_package_sha256"]:
    raise ValueError("Runtime package initializer changed since bound closure inspection.")
print(
    json.dumps(
        {
            "kind": "native_prefix_completion_source_proof_v2",
            "protected_main_sha256": protected,
            "runtime_package_sha256": packages,
            "other_closed_builder_sources_unchanged": older,
            "snapshots_byte_exact": True,
            "authored_sha256": {
                p: hashlib.sha256((BUILDER / p).read_bytes()).hexdigest() for p in snapshot
            },
            "unchanged_recipe_ast": list(unchanged),
            "config_fields_defaults_unchanged": True,
            "bounded_diff": diff,
            "public_numeric_access": "NOT_RUN",
        },
        indent=2,
    )
)
