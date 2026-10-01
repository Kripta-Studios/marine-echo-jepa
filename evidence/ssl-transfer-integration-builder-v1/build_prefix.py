"""One-time engineering source copy into a NEW module; never runtime rewriting."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILDER = HERE.parents[1]
ROOT = BUILDER.parent / "marine-echo-jepa"
original = ROOT / "src/marine_echo/training/native_prefix_desktop_transfer_v3.py"
source = original.read_text(encoding="utf-8")
before = source


def change(old, new):
    global source
    assert old in source, old[:100]
    source = source.replace(old, new)


change('from marine_echo.training import native_ssl as core', 'from marine_echo.training import native_ssl as core\nfrom marine_echo.training import native_cf_controls as cf_controls\nfrom marine_echo.training import native_references as references\n\nTRAINING_KINDS = {**TRAINING_KINDS, **cf_controls.TRAINING_KINDS, "lightgbm": "prefix_supervised_conventional"}')
change('    "cf": "cf_jepa_multiscale_v1",', '    "cf": "cf_jepa_multiscale_v1",\n    "reference": "native_lightgbm_prefix_fixed_v1",')
change('(self.method == "cf_jepa") != (self.family == "cf")', '(self.method in ("cf_jepa", *cf_controls.METHODS) or (self.method == "direct" and self.mode == "scratch_direct" and self.family == "cf")) != (self.family == "cf")')
change('("frozen_readout", "scratch_direct")', '("frozen_readout", "scratch_direct", "conventional")')
change('(self.method != "direct" or self.family == "cf")', 'self.method != "direct"')
change('            raise ValueError("Unknown native method/family.")', '            raise ValueError("Unknown native method/family.")\n        if (self.method == "lightgbm") != (self.family == "reference" and self.mode == "conventional") or (self.mode == "conventional" and self.method != "lightgbm"):\n            raise ValueError("Typed fixed conventional cell required.")\n        if self.method in cf_controls.METHODS and self.mode != "frozen_readout":\n            raise ValueError("CF control features are frozen; scratch uses direct/cf identity.")')
change('checkout = Path(__file__).resolve().parents[3]', 'checkout = Path(core.__file__).resolve().parents[3]')
change('        root / "training/native_prefix_transfer.py",', '        root / "training/native_prefix_transfer.py",\n        root / "training/native_prefix_desktop_transfer_v3.py",\n        Path(cf_controls.__file__),\n        Path(references.__file__),')
change('    fields = dict(raw)\n    if family == "band":', '    fields = dict(raw)\n    if family == "reference":\n        validate_reference_policy(raw)\n        return core.Config(method="direct", seed=raw["seed"], history=96)\n    if raw.get("method") in cf_controls.METHODS:\n        c = cf_controls.Config(**raw)\n        c.validate(correctness_smoke=(c.width, c.latent, c.blocks, c.batch_size) != (256, 128, 5, 64))\n        return cf_controls.core_config(c)\n    if family == "band":')
change('    if c.method != config.method or c.history != 96:', '    expected_method = "cf_jepa" if config.method in cf_controls.METHODS or (config.family == "cf" and config.mode == "scratch_direct") else "direct" if config.family == "reference" else config.method\n    if c.method != expected_method or c.history != 96:')
change('    if config.family == "band" and c.seed != config.seed:', '    if (config.family == "band" or config.method in cf_controls.METHODS or config.family == "reference") and c.seed != config.seed:')
change('def selected_kind(config):\n', 'def selected_kind(config):\n    if config.method in cf_controls.METHODS:\n        return cf_controls.ENCODER_KINDS[config.method]\n')
change('    if method_cfg.method != config.method or method_cfg.history != 96:', '    if method_cfg.history != 96:')
change('"native_prefix_transfer_manifest_v1"', '"native_prefix_matched_transfer_manifest_v4"')
change('"native_prefix_transfer_resume_v1"', '"native_prefix_matched_transfer_resume_v4"')
change('"native_prefix_transfer_inference_v1"', '"native_prefix_matched_transfer_inference_v4"')
change('"native_prefix_transfer_completion_v1"', '"native_prefix_matched_transfer_completion_v4"')
change('    config.validate(manifest["evidence_kind"])', '    config.validate(manifest["evidence_kind"])\n    _private_manifest(manifest_path, manifest, output)')
change('    partition = partitions(metadata, config.prefix_days)', '    partition = partitions(metadata, config.prefix_days)\n    if len(metadata["sources"]) != 1:\n        raise ValueError("One true deployment per adapted trajectory; no pooling.")')
change('        if config.method == "direct":\n            _, original_config', '        if config.method in cf_controls.METHODS:\n            _admit_cf_parent(config, manifest, parent_run, architecture, raw, parent_inference, parent_membership, artifacts, ancestor_inputs, ancestor_members, split_path, base, bound, doc)\n        elif config.method == "direct":\n            _, original_config')
change('        elif selected.get("supervised_ancestry") is not None:', '        elif config.method in cf_controls.METHODS:\n            validate_control_ancestry(selected.get("supervised_ancestry"), config, backbone)\n        elif selected.get("supervised_ancestry") is not None:')
change('    _, statistics = doc(manifest["scalers"])', '    if config.family == "reference":\n        _, policy = doc(manifest["conventional_policy"])\n        validate_reference_policy(policy)\n        if policy != architecture:\n            raise ValueError("Bound conventional feature/policy identity differs.")\n        if not config.correctness_smoke and (manifest.get("conventional_policy_status") != "ADOPTED_BEFORE_RESERVED_VALUES" or review.get("conventional_policy_sha256") != sha(snapshots[_path(manifest["conventional_policy"], base)])):\n            raise ValueError("Prospective conventional ADR adoption and exact review required.")\n    _, statistics = doc(manifest["scalers"])')
change('        expected_selected_kind = selected_kind(config)', '        if config.method in cf_controls.METHODS:\n            _check_control_artifacts(selected, decode_checkpoint(admission.snapshots[_path(m["parent_inference"], base)]), parent, config, backbone, statistics)\n        expected_selected_kind = selected_kind(config)')
change('            or parent_artifact.get("config") != backbone', '            or parent_artifact.get("config") != backbone')
change('        inference_kind = (\n', '        inference_kind = cf_controls.INFERENCE_KIND if config.method in cf_controls.METHODS else (\n')
change('    with desktop_runtime.Resources(config.device, output) as resources:', '    if config.family == "reference":\n        return _fit_reference(admission, prefix, dev, output, backbone, statistics)\n    with desktop_runtime.Resources(config.device, output) as resources:')
change('    _, numeric = doc(manifest["numeric_access_review"])', '    _, numeric = doc(manifest["numeric_access_review"])')
change('        if config.family == "band" and selected.get("architecture")', '        if config.family == "band" and selected.get("architecture")')

# New pure admission/support helpers are defined before the captured closure.
extra = (HERE / "prefix_helpers.txt").read_text(encoding="utf-8")
source = source.replace('LOADED_SOURCE_HASHES =', extra + '\n\nLOADED_SOURCE_HASHES =', 1)
original_functions = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(before).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
new_functions = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
same = sorted(k for k in original_functions if original_functions[k] == new_functions[k])
with (BUILDER / "src/marine_echo/training/native_prefix_matched_transfer_v4.py").open("x", encoding="utf-8", newline="\n") as stream:
    stream.write(source)
with (HERE / "prefix-baseline-source.txt").open("xb") as stream:
    stream.write(original.read_bytes())
with (HERE / "prefix-copy-proof.json").open("x", encoding="utf-8") as stream:
    json.dump({"main_before_sha256": hashlib.sha256(original.read_bytes()).hexdigest(), "unchanged_ast": same, "changed_ast": sorted(set(original_functions) - set(same)), "runtime_rewriting": False}, stream, indent=2)
print(json.dumps({"unchanged_ast": len(same), "new_module": True}))
