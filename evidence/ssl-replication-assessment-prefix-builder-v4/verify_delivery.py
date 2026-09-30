"""Read-only stdlib hashes, bounded AST equivalence and integration diff."""

import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILDER = HERE.parents[1]
MAIN = BUILDER.parent / "marine-echo-jepa"
AUTHORED = (
    "src/marine_echo/training/native_prefix_transfer.py",
    "tests/unit/test_native_prefix_transfer.py",
    "tests/integration/test_native_prefix_transfer.py",
    "src/marine_echo/evaluation/native_assessment_replication.py",
    "tools/execute_bounded_native_replication_assessment.py",
    "tools/execute_native_replication_assessment_worker.py",
    "tests/unit/test_native_replication_assessment.py",
    "tests/integration/test_native_replication_assessment.py",
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def nodes(text):
    return {
        n.name: n for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }


def equal(a, b):
    return ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)


class TypedOnly(ast.NodeTransformer):
    def visit_Expr(self, node):
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_typed_band_config"
        ):
            return None
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if node.value == "native_replication_frozen_assessment_completion_v2":
            return ast.Constant("native_frozen_assessment_completion_v1")
        return node

    def visit_If(self, node):
        if (
            isinstance(node.test, ast.Compare)
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value
            == "native_band_replication_ssl_weights_only_inference_v2"
        ):
            return [self.visit(n) for n in node.orelse]
        return self.generic_visit(node)


class PrefixVersionOnly(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in {"_backbone", "PrefixModel"}
            and len(node.args) == 3
        ):
            node.args.pop()
        return node

    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        if node.name == "__init__" and node.args.args[-1].arg == "band_artifact_version":
            node.args.args.pop()
            node.args.defaults.pop()
        return node

    def visit_IfExp(self, node):
        node = self.generic_visit(node)
        if (
            isinstance(node.body, ast.Constant)
            and node.body.value == "native_band_replication_ssl_weights_only_inference_v2"
        ):
            return node.orelse
        return node


def closure():
    package = MAIN / "src/marine_echo"
    replacements = {
        "marine_echo.training.native_prefix_transfer": BUILDER / AUTHORED[0],
        "marine_echo.evaluation.native_assessment_replication": BUILDER / AUTHORED[3],
    }
    pending = [BUILDER / p for p in (AUTHORED[0], AUTHORED[3], AUTHORED[4], AUTHORED[5])]
    pending.extend(
        MAIN / p
        for p in (
            "tools/native_reference_supervisor.py",
            "tools/execute_native_prefix_job.py",
            "tools/execute_native_band_replication_job.py",
            "tools/execute_bounded_native_assessment.py",
            "tools/execute_native_assessment_worker.py",
        )
    )
    # Original v2 required_sources also binds these literal factory/API paths,
    # even though a prefix readout/assessment does not execute latent APIs.
    pending.extend(
        MAIN / "src/marine_echo" / p
        for p in (
            "inference/native_band_replication_encoder.py",
            "inference/native_band_replication_latent.py",
            "inference/native_latent.py",
        )
    )
    found = set()
    while pending:
        p = pending.pop().resolve()
        if p in found:
            continue
        found.add(p)
        for node in ast.walk(ast.parse(p.read_bytes())):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = node.module or ""
                if node.level and p.is_relative_to(package):
                    prefix = ["marine_echo", *p.relative_to(package).parts[:-1]]
                    name = ".".join(
                        [*prefix[: len(prefix) - node.level + 1], *name.split(".")]
                    ).rstrip(".")
                names = [name, *(name + "." + a.name for a in node.names)]
            else:
                continue
            for name in names:
                if name == "marine_echo" or name.startswith("marine_echo."):
                    parts = name.split(".")[1:]
                    candidate = replacements.get(name, package.joinpath(*parts).with_suffix(".py"))
                    if candidate.is_file():
                        pending.append(candidate)
                    for i in range(len(parts) + 1):
                        init = package.joinpath(*parts[:i]) / "__init__.py"
                        if init.is_file():
                            pending.append(init)
    return {str(p): digest(p.read_bytes()) for p in sorted(found)}


def proof():
    before = json.loads((HERE / "baseline-prefix-v4.json").read_bytes())
    protected = json.loads((HERE / "baseline-dependencies-v4.json").read_bytes())
    protected.update(before)
    unchanged, externally_changed = {}, {}
    for p, value in protected.items():
        actual = digest((MAIN / p).read_bytes())
        if actual != value["sha256"]:
            externally_changed[p] = {
                "baseline_sha256": value["sha256"],
                "current_sha256": actual,
                "builder_write": False,
                "cause": "UNKNOWN_EXTERNAL_CHANGE",
                "current_source_text": (MAIN / p).read_text(encoding="utf-8"),
            }
        else:
            unchanged[p] = actual
    prefix_old = nodes(before[AUTHORED[0]]["text"])
    prefix_new = nodes((BUILDER / AUTHORED[0]).read_text(encoding="utf-8"))
    exact = sorted(k for k in prefix_old if equal(prefix_old[k], prefix_new[k]))
    altered = sorted(set(prefix_old) - set(exact))
    required = {
        "boundaries",
        "partitions",
        "_Trajectory",
        "predict",
        "_decode",
        "_strict_state",
        "_supervised_ancestry",
    }
    if not required <= set(exact):
        raise ValueError(
            "Scientific prefix trajectory/partition changed: " + str(required - set(exact))
        )
    prefix_normalized = {
        k: equal(prefix_old[k], PrefixVersionOnly().visit(copy.deepcopy(prefix_new[k])))
        for k in ("PrefixModel", "prepare_model", "PrefixPredictor", "fit")
    }
    if not all(prefix_normalized.values()):
        raise ValueError(
            "Prefix execution math changed outside typed version arguments: "
            + str(prefix_normalized)
        )
    base = nodes(protected["src/marine_echo/evaluation/native_assessment.py"]["text"])
    adapter = nodes((BUILDER / AUTHORED[3]).read_text(encoding="utf-8"))
    adapter_exact = sorted(k for k in base if equal(base[k], adapter[k]))
    normalized = {}
    for name in ("admit", "_builtin", "execute_assessment"):
        normalized[name] = equal(base[name], TypedOnly().visit(copy.deepcopy(adapter[name])))
        if not normalized[name]:
            raise ValueError("Assessment change exceeds typed/kind differences: " + name)
    expected_altered = {"required_sources", "admit", "_builtin", "execute_assessment"}
    if set(base) - set(adapter_exact) != expected_altered:
        raise ValueError("Unexpected assessment AST differences")
    wrapper = nodes((BUILDER / AUTHORED[4]).read_text(encoding="utf-8"))
    metadata_exact = sorted(k for k in adapter if k in wrapper and equal(adapter[k], wrapper[k]))
    hashes = {p: digest((BUILDER / p).read_bytes()) for p in AUTHORED}
    diff = "".join(
        "".join(
            difflib.unified_diff(
                before[p]["text"].splitlines(True),
                (BUILDER / p).read_text(encoding="utf-8").splitlines(True),
                fromfile="main/" + p,
                tofile="builder/" + p,
            )
        )
        for p in before
    )
    return {
        "kind": "replication_assessment_prefix_source_proof_v4",
        "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
        "authored": hashes,
        "main_protected_unchanged": unchanged,
        "main_read_only_observed_external_changes": externally_changed,
        "source_closure": closure(),
        "prefix_exact_ast": exact,
        "prefix_typed_admission_or_constructor_differences": altered,
        "prefix_normalized_version_only_ast": prefix_normalized,
        "assessment_exact_ast": adapter_exact,
        "assessment_normalized_typed_only_ast": normalized,
        "wrapper_exact_metadata_helpers": metadata_exact,
        "prefix_bounded_diff": diff,
        "assessment_typed_changes": [
            "NEURAL_KINDS v2",
            "required source closure",
            "finite band kind/config admission",
            "v2 safe loader",
            "completion kind v2",
        ],
        "not_scientific_review": True,
    }


if __name__ == "__main__":
    print(json.dumps(proof(), indent=2))
