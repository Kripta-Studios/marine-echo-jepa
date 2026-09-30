"""Source/import proof only; no corpus/checkpoint/tensor parsing or constructor."""

import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILDER = HERE.parents[1]
MAIN = BUILDER.parent / "marine-echo-jepa"
AUTHORED = [
    "src/marine_echo/inference/native_latent.py",
    "tests/unit/test_native_latent.py",
    "tests/integration/test_native_latent.py",
]


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = module_at("_latent_source_probe", HERE / "source_probe.py")
proof = probe.proof()
baseline = json.loads((HERE / "source-baseline-v2.json").read_bytes())
if proof != baseline:
    raise ValueError("Protected source or closed execution delivery changed.")
support = module_at("_latent_proof_support", HERE / "test_support.py")
api = support.load_api()
origins = {
    "native_encoder": Path(api.legacy.__file__).resolve(),
    "input_validator": Path(inspect.getfile(api.legacy._inputs)).resolve(),
    "scaler_validator": Path(inspect.getfile(api.legacy._scalers)).resolve(),
    "query_validator": Path(inspect.getfile(api._query)).resolve(),
    "state_validator": Path(inspect.getfile(api._check_state)).resolve(),
    "NativeTemporalModel": Path(inspect.getfile(api.NativeTemporalModel)).resolve(),
    "NativeBandTemporalModel": Path(inspect.getfile(api.NativeBandTemporalModel)).resolve(),
    "CFTemporalEncoder": Path(inspect.getfile(api.CFTemporalEncoder)).resolve(),
    "QueryHead": Path(inspect.getfile(api.QueryHead)).resolve(),
    "Config": Path(inspect.getfile(api.Config)).resolve(),
    "BandConfig": Path(inspect.getfile(api.BandConfig)).resolve(),
}
if any(not p.is_relative_to(MAIN / "src") for p in origins.values()):
    raise ValueError("Actual immutable imports must come from main source copies.")
tree = ast.parse((BUILDER / AUTHORED[0]).read_bytes())
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        name = getattr(node.func, "attr", getattr(node.func, "id", ""))
        if name in {
            "manual_seed",
            "set_rng_state",
            "set_default_dtype",
            "initialize_model",
            "compile",
            "fit",
            "required_sources",
            "check_prefit",
            "validate",
            "open",
            "read_bytes",
            "read_text",
            "save",
            "system",
            "Popen",
        }:
            raise ValueError("Unexpected fitting/RNG/provenance/cache/process call in API.")
        if name == "load":
            if (
                not isinstance(node.func, ast.Attribute)
                or not isinstance(node.func.value, ast.Name)
                or node.func.value.id != "torch"
            ):
                raise ValueError("Unexpected numerical loader.")
            keywords = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
            if keywords != {"weights_only": True, "map_location": "cpu"}:
                raise ValueError("Weights-only CPU codec flags changed.")
proof.update(
    authored_sha256={p: hashlib.sha256((BUILDER / p).read_bytes()).hexdigest() for p in AUTHORED},
    actual_imports={
        k: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for k, p in origins.items()
    },
    all_protected_sources_unchanged=True,
    ast_no_fit_rng_policy_provenance_process_calls=True,
    root_durable_checkpoint_replay="NOT_RUN",
    scientific_approval=False,
)
print(json.dumps(proof, indent=2))
