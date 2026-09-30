"""Produce new unified patches against exact main bytes; never apply to main."""

import ast
import difflib
import hashlib
import json
from pathlib import Path

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
OUTPUT = Path(__file__).parent
EXPECTED = {
    "src/marine_echo/models/native_temporal.py": "85c4af6c67acb661359cde8bcf22de479e4da6a499cc7843bce0fee6c155e422",
    "src/marine_echo/training/native_references.py": "b692368053501ca946978a3c583d649356584e91c1e020597397e669d8aa1d7b",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    changes = []
    for name, expected in EXPECTED.items():
        before_path = MAIN / name
        after_path = (
            BUILDER / name
            if name.endswith("native_temporal.py")
            else OUTPUT / "native_references.after.py"
        )
        before, after = before_path.read_bytes(), after_path.read_bytes()
        if digest(before) != expected:
            raise ValueError(f"Main before hash differs; no stale patch generated: {name}")
        patch_path = OUTPUT / f"{Path(name).stem}.patch"
        diff = "".join(
            difflib.unified_diff(
                before.decode("utf-8").splitlines(keepends=True),
                after.decode("utf-8").splitlines(keepends=True),
                fromfile=f"a/{name}",
                tofile=f"b/{name}",
            )
        )
        if not diff:
            raise ValueError("Repair patch unexpectedly empty.")
        with patch_path.open("x", encoding="utf-8", newline="") as stream:
            stream.write(diff)
        if name.endswith("native_temporal.py"):
            # Independent AST substitution check: preserve every original model
            # definition except the three pooling calls; only one utility added.
            original = ast.parse(before.decode())
            candidate = ast.parse(after.decode())
            utility = next(
                n
                for n in candidate.body
                if isinstance(n, ast.FunctionDef) and n.name == "deterministic_adaptive_avg_pool1d"
            )
            candidate.body.remove(utility)

            class RestorePool(ast.NodeTransformer):
                def visit_Call(self, node):
                    if (
                        isinstance(node.func, ast.Name)
                        and node.func.id == "deterministic_adaptive_avg_pool1d"
                    ):
                        node.func = ast.Attribute(
                            value=ast.Name(id="F", ctx=ast.Load()),
                            attr="adaptive_avg_pool1d",
                            ctx=ast.Load(),
                        )
                    return self.generic_visit(node)

            assert ast.dump(original, include_attributes=False) == ast.dump(
                RestorePool().visit(candidate), include_attributes=False
            )
            with (OUTPUT / "native_temporal.after.py").open("xb") as stream:
                stream.write(after)
        changes.append(
            {
                "target": name,
                "main_before_sha256": expected,
                "after_sha256": digest(after),
                "patch": patch_path.name,
                "patch_sha256": digest(patch_path.read_bytes()),
            }
        )
    manifest = {
        "status": "PATCHES_PREPARED_NOT_APPLIED_NOT_APPROVED",
        "changes": changes,
        "integration_owner": "ROOT",
        "renewed_independent_review": "REQUIRED_BEFORE_FITS",
        "scientific_claim": "NONE",
    }
    with (OUTPUT / "patch-manifest.json").open("x", encoding="utf-8", newline="") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(json.dumps(manifest, indent=2))
