"""Close byte proof and source-call proof; no acoustic or trained artifact access."""
import ast
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILDER = HERE.parents[1]
sys.path.insert(0, str(BUILDER / "tests/unit"))
from test_native_cf_controls import MAIN, controls, core


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    with (HERE / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


baseline = json.loads((HERE / "protected-baseline-v1.json").read_bytes())["files"]
current = {path: sha(path) for path in baseline}
assert current == baseline, "Immutable ROOT dependency changed"
files = ["src/marine_echo/training/native_cf_controls.py", "src/marine_echo/inference/native_cf_controls.py",
         "tests/unit/test_native_cf_controls.py", "tests/integration/test_native_cf_controls.py"]
authored = {name: sha(BUILDER / name) for name in files}
closure = {str(path): sha(path) for path in controls.required_paths.__globals__["source_paths"](controls.Config())}
assert all(str(MAIN) in path or path in (str(BUILDER / files[0]), str(BUILDER / files[1])) for path in closure)
for name in files:
    destination = HERE / (name.replace("/", "__") + ".closed-v3.txt")
    with destination.open("xb") as stream:
        stream.write((BUILDER / name).read_bytes())
functions = {}
for path in (Path(core.__file__), Path(controls.downstream.__file__)):
    tree = ast.parse(path.read_bytes())
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in {
            "initialize_model", "model_dimensions", "batch_indices", "tensor_batch", "pinball", "daily_metrics",
            "Scalers", "Resources", "schedule", "rng_state", "restore_rng", "predict", "_forecast_train",
            "supervised_indices", "_verify_data", "_verify_continuation", "prepare_model"}:
            functions[str(path) + "::" + node.name] = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
tree = ast.parse((BUILDER / files[0]).read_bytes())
calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
assert not any(call.endswith((".cf_loss", ".update_ema", ".ssl_loss")) for call in calls)
for call in ("core.initialize_model", "downstream._forecast_train", "downstream.supervised_indices", "core.pinball", "core.daily_metrics", "core.schedule", "core.rng_state", "core.restore_rng", "core.Resources"):
    assert call in calls
counts = {}
for method in controls.METHODS:
    model = controls.prepare_model(controls.Config(method=method))
    counts[method] = {"encoder": sum(p.numel() for p in model.encoder.parameters()),
                      "head": sum(p.numel() for p in model.readout.parameters()),
                      "total": sum(p.numel() for p in model.parameters()),
                      "optimized": sum(p.numel() for p in model.parameters() if p.requires_grad)}
write("source-proof-final-v1.json", {"authored": authored, "protected_baseline": baseline,
    "protected_current": current, "protected_sources_unchanged": True,
    "runtime_static_source_closure": closure, "original_science_ast_sha256": functions,
    "parameters": counts, "proof_type": "unchanged authoritative source bytes plus explicit source helper calls",
    "adaptations": ["two-branch encoder/readout container; zero SSL", "distinct control identity/admission/artifacts", "exclusive final weights; original resumable checkpoints"],
    "training_loss": "original equal-observed-horizon TRAIN pinball",
    "selection": "original eligible-source-date then horizon then deployment DEV pinball floor18",
    "source_imports": {"core": core.__file__, "downstream": controls.downstream.__file__},
    "scientific_numerical_access": False})
print(json.dumps({"protected_dependencies": len(current), "source_closure": len(closure), "parameters": counts, "protected_sources_unchanged": True}))
