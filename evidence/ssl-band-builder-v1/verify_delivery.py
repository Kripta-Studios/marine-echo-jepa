"""Mechanical source/AST/import checks only; not scientific or independent review."""

import ast
import difflib
import hashlib
import json
from pathlib import Path

import torch
from band_test_support import BUILDER, EVIDENCE, MAIN

from marine_echo.inference import native_band_acoustic
from marine_echo.models import native_band_temporal, native_temporal
from marine_echo.training import native_band_downstream, native_band_ssl, native_ssl

AUTHORED = (
    "src/marine_echo/models/native_band_temporal.py",
    "src/marine_echo/training/native_band_ssl.py",
    "src/marine_echo/training/native_band_downstream.py",
    "src/marine_echo/inference/native_band_acoustic.py",
    "tests/unit/test_native_band_temporal.py",
    "tests/unit/test_native_band_inference.py",
    "tests/integration/test_native_band_ssl.py",
    "tests/integration/test_native_band_downstream.py",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(node):
    return ast.dump(node, include_attributes=False)


def tree(path):
    return ast.parse(Path(path).read_text(encoding="utf-8"))


def functions(module):
    result = {}
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            result[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, ast.FunctionDef):
                    result[node.name + "." + method.name] = method
    return result


def class_node(module, name):
    return next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == name
    )


def checked_delta(original, revised, allowed):
    before, after = functions(original), functions(revised)
    topology_delta = set(before) ^ set(after)
    assert topology_delta <= set(allowed), "Runner function/method topology changed unexpectedly."
    changed = {
        name for name in set(before) & set(after) if dump(before[name]) != dump(after[name])
    } | topology_delta
    assert changed <= set(allowed), f"Unexpected runner behavior edits: {changed - set(allowed)}"
    return {
        "changed_functions": sorted(changed),
        "unchanged_functions": sorted(set(before) - changed),
    }


if __name__ == "__main__":
    records = json.loads((EVIDENCE / "before.json").read_text(encoding="utf-8"))
    for filename, record in records["main_sources"].items():
        assert sha(filename) == sha(EVIDENCE / record["snapshot"]) == record["sha256"]
    for filename, expected in records["closed_comparison"].items():
        assert sha(BUILDER / filename) == expected, f"Closed handoff changed: {filename}"
    old_model = tree(EVIDENCE / "models__native_temporal.py.before.txt")
    new_model = tree(BUILDER / AUTHORED[0])
    original = next(
        n
        for n in class_node(old_model, "SharedTemporalEncoder").body
        if isinstance(n, ast.FunctionDef) and n.name == "tokens"
    )
    candidate = next(
        n for n in class_node(new_model, "NativeBandEncoder").body if isinstance(n, ast.FunctionDef)
    )
    activation = ast.parse("tokens = F.gelu(tokens)").body[0]
    matches = [i for i, n in enumerate(candidate.body) if dump(n) == dump(activation)]
    assert len(matches) == 1
    position = matches[0]
    assert dump(candidate.body[position - 1]) == dump(
        ast.parse("tokens = tokens + self.metadata_projection(metadata).unsqueeze(1)").body[0]
    )
    assert dump(candidate.body[position + 1]) == dump(
        ast.parse("counts = masks.sum(-1).to(x.dtype)").body[0]
    )
    candidate.body.pop(position)
    assert dump(original) == dump(candidate), (
        "Encoder tokens differ beyond one parameter-free GELU."
    )
    same_constructor = (
        native_band_temporal.NativeBandEncoder.__init__
        is native_temporal.SharedTemporalEncoder.__init__
    )
    assert same_constructor
    for name in ("encode", "forward"):
        assert getattr(native_band_temporal.NativeBandEncoder, name) is getattr(
            native_temporal.SharedTemporalEncoder, name
        )
    for name in ("forecast", "shared_loss", "masked_loss", "freeze_encoder"):
        assert getattr(native_band_temporal.NativeBandTemporalModel, name) is getattr(
            native_temporal.NativeTemporalModel, name
        )
    old_init = next(
        n
        for n in class_node(old_model, "NativeTemporalModel").body
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    new_init = next(
        n
        for n in class_node(new_model, "NativeBandTemporalModel").body
        if isinstance(n, ast.FunctionDef)
    )
    assert dump(old_init.body[0]) == dump(ast.parse("super().__init__()").body[0])
    assert dump(new_init.body[0]) == dump(ast.parse("nn.Module.__init__(self)").body[0])
    new_init.body[0] = old_init.body[0]
    for node in ast.walk(new_init):
        if isinstance(node, ast.Name) and node.id == "NativeBandEncoder":
            node.id = "SharedTemporalEncoder"
    assert dump(new_init) == dump(old_init), "Parameter construction or ordering changed."
    old_core = tree(EVIDENCE / "training__native_ssl.py.before.txt")
    new_core = tree(BUILDER / AUTHORED[1])
    old_down = tree(EVIDENCE / "training__native_downstream.py.before.txt")
    new_down = tree(BUILDER / AUTHORED[2])
    core_delta = checked_delta(
        old_core,
        new_core,
        (
            "Config.validate",
            "Scalers.from_dict",
            "required_sources",
            "check_prefit",
            "initialize_model",
            "model_dimensions",
            "predict_from_checkpoint",
            "run",
        ),
    )
    down_delta = checked_delta(
        old_down,
        new_down,
        (
            "DownstreamConfig.validate",
            "required_paths",
            "_distinct_review",
            "check_prefit",
            "_core_config",
            "_load_ancestor",
            "run",
            "Inference.__init__",
            "Inference.encode",
            "Inference.forecast",
        ),
    )
    # New downstream Inference inherits safe target-free methods; compare removed
    # legacy methods separately rather than pretending their implementations match.
    for original_module, revised_module in ((old_core, new_core), (old_down, new_down)):
        old_loops = [
            n for n in ast.walk(functions(original_module)["run"]) if isinstance(n, ast.While)
        ]
        new_loops = [
            n for n in ast.walk(functions(revised_module)["run"]) if isinstance(n, ast.While)
        ]
        assert len(old_loops) == len(new_loops) == 1
        assert dump(old_loops[0]) == dump(new_loops[0]), "Training/selection loop changed."
    for module, name in (
        (native_temporal, "models/native_temporal.py"),
        (native_ssl, "training/native_ssl.py"),
    ):
        assert Path(module.__file__).resolve() == MAIN / "src/marine_echo" / name
    for module, name in (
        (native_band_temporal, AUTHORED[0]),
        (native_band_ssl, AUTHORED[1]),
        (native_band_downstream, AUTHORED[2]),
        (native_band_acoustic, AUTHORED[3]),
    ):
        assert Path(module.__file__).resolve() == BUILDER / name
    with torch.device("meta"):
        model = native_band_temporal.NativeBandTemporalModel()
        reference = native_temporal.NativeTemporalModel()
    measured = {
        "total": sum(p.numel() for p in model.parameters()),
        "encoder": sum(p.numel() for p in model.encoder.parameters()),
        "readout": sum(p.numel() for p in model.readout.parameters()),
    }
    assert measured["total"] == sum(p.numel() for p in reference.parameters())
    sources = native_band_ssl.required_sources(native_band_ssl.Config())
    imports = {
        module.__name__: str(Path(module.__file__).resolve())
        for module in (
            native_band_temporal,
            native_band_ssl,
            native_band_downstream,
            native_band_acoustic,
            native_temporal,
            native_ssl,
        )
    }
    for before_name, after_name, output in (
        ("models__native_temporal.py.before.txt", AUTHORED[0], "model-narrow-diff-v1.patch"),
        ("training__native_ssl.py.before.txt", AUTHORED[1], "ssl-runner-narrow-diff-v1.patch"),
        (
            "training__native_downstream.py.before.txt",
            AUTHORED[2],
            "downstream-runner-narrow-diff-v1.patch",
        ),
    ):
        diff = "".join(
            difflib.unified_diff(
                (EVIDENCE / before_name).read_text(encoding="utf-8").splitlines(keepends=True),
                (BUILDER / after_name).read_text(encoding="utf-8").splitlines(keepends=True),
                fromfile="IMMUTABLE_MAIN/" + before_name,
                tofile=after_name,
            )
        )
        destination = EVIDENCE / output
        if destination.exists():
            assert destination.read_text(encoding="utf-8") == diff
        else:
            with destination.open("x", encoding="utf-8") as stream:
                stream.write(diff)
    print(
        json.dumps(
            {
                "operation": "ENGINEERING_AST_AND_BYTE_VERIFICATION_NOT_INDEPENDENT_REVIEW",
                "status": "VERIFIED",
                "scientific_assessment_occurred": False,
                "preserved_main_sources": len(records["main_sources"]),
                "preserved_closed_comparison_files": len(records["closed_comparison"]),
                "encoder_change": "exactly_one_F.gelu_after_channel_value_plus_metadata_before_observed_count_pooling",
                "same_encoder_constructor_identity": same_constructor,
                "same_parameter_construction_order": True,
                "unchanged_training_and_selection_loop_AST": True,
                "ssl_runner_AST_delta": core_delta,
                "downstream_runner_AST_delta": down_delta,
                "parameters": measured,
                "actual_imports": imports,
                "authored_sha256": {name: sha(BUILDER / name) for name in AUTHORED},
                "required_review_source_bindings": {str(p): sha(p) for p in sources},
                "optimizer_resume_validation": "NOT_RUN_BUILDER_PRIOR_CACHE_DENIAL_ROOT_FIXTURES_PROVIDED",
                "GPU_validation": "NOT_RUN_ROOT_OWNER",
            },
            indent=2,
        )
    )
