"""Exclusive small engineering closeout; no data/tensor payload reads or fits."""

import hashlib
import json
from pathlib import Path

import torch

from test_support import BUILDER, MAIN, legacy, prefix_module

HERE = Path(__file__).resolve().parent
p = prefix_module()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    with (HERE / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


protected = json.loads((HERE / "protected-source-proof-final-v1.json").read_text())
current = {name: sha(name) for name in protected["current"]}
assert current == protected["current"], "Protected dependency changed after proof"
old_authored = json.loads((HERE / "authored-source-proof-final-v1.json").read_text())
authored = {name: sha(BUILDER / name) for name in old_authored}
assert authored["src/marine_echo/training/native_prefix_matched_transfer_v4.py"] == old_authored["src/marine_echo/training/native_prefix_matched_transfer_v4.py"]
assert authored["src/marine_echo/evaluation/native_prefix_suffix_assessment_v1.py"] == old_authored["src/marine_echo/evaluation/native_prefix_suffix_assessment_v1.py"]
before = torch.get_rng_state().clone()
counts = {}
for family in ("shared", "cf"):
    config = p.core.Config(method="cf_jepa" if family == "cf" else "direct").to_dict()
    with torch.device("meta"):
        original = legacy.PrefixModel(config, family).float()
        new = p.PrefixModel(config, family).float()
    old_shapes = {k: tuple(v.shape) for k, v in original.state_dict().items()}
    new_shapes = {k: tuple(v.shape) for k, v in new.state_dict().items()}
    assert old_shapes == new_shapes
    counts[family] = {"encoder": sum(v.numel() for v in new.encoder.parameters()),
        "head": sum(v.numel() for v in new.head.parameters()),
        "total": sum(v.numel() for v in new.parameters()), "original_tensor_shapes_equal": True}
assert torch.equal(before, torch.get_rng_state()) and not torch.cuda.is_initialized()
save("closed-source-proof-v2.json", {"authored": authored, "dependencies": current,
    "protected_sources_unchanged": True, "meta_only_parameter_counts": counts,
    "rng_unchanged": True, "cuda_initialized": False,
    "earlier_authored_proof": "AST proof production bytes unchanged; later native-geometry and decoder-perturbation test additions are reflected here"})

report = """# Prefix and adapted suffix engineering delivery

Two additive production modules and four tests are closed. ROOT scientific dependencies remain read-only. This is engineering correctness work, not numerical access, independent review, selection, fitting approval, transfer evidence or programme completion.

The V4 prefix runner preserves the original desktop V3 boundaries, raw interval/configuration/gap guards, native230 target geometry, native225 original development selection, TRAIN scalers, sample stream, fresh seed+100000 heads and numerical optimizer/schedule/checkpoint trajectory. It admits truthful CF random and direct-supervised control selected encoders through their exact completed report, original review/config/source/membership/sampler/scaler ancestry and selected-versus-inference tensor identity. CF scratch uses a fresh corresponding encoder and the same prefix labels. No SSL future crops, online latent predictors or EMA updates are introduced. Frozen encoders keep parameters and BatchNorm buffers immutable. CF pooled full-H96 features retain the original prefix behavior; this does not reproduce sampled CF forward-zone training or establish in-view support. CF128 and shared64 readouts are not cross-architecture parameter matched.

The source-pinned fifteen-booster LightGBM implementation and UTF8 JSON replay/whole-booster continuation work on private synthetic fixtures. Its four original-DEV tree choices are a PROPOSAL in CONVENTIONAL_PREFIX_POLICY_PROPOSAL_V1.md. Production use fails without prospective coordinator adoption and exact separate review. It never applies replacement target-site scalers or calls its tree fits neural optimizer updates. Minimum child samples64 can yield constant short-prefix forecasts; the source recipe is retained and diagnostics remain visible. Persistence/seasonal24 use unchanged source formulas.

The suffix evaluator is separately scoped adapted_suffix. It requires distinct software, numeric-access and execution reviews, a prior exact selection freeze, complete prefix admission/recursive TRAIN and prefix ancestry, its own true deployment, all model/scaler/config/source/data hashes, fixed raw suffix identity and interval disjointness BEFORE decode. It calls only enumerated context-only loaders/builtin references, never fits, passes targets into predictors, intersects rows, changes geometry or selects on suffix. Adapted cells are always zero_shot=false; no-fit references remain explicitly zero-shot. It saves immutable compatible NPZ predictions, native geometry/cutoff/support, daily/macro metrics, adaptation labels/ancestry and resource receipts. Paired uncertainty reuses the immutable 2000-draw seed20260929 nominal48-hour/two-source-calendar-day policy; timestamps are not verified UTC. Unsupported daily support yields NOT_ASSESSABLE/null intervals.

Actual combined focused pytest: 51 passed, two ROOT-only skipped, exit0, 243.89 seconds. Added geometry replay: four selected tests passed, exit0. Added future/suffix perturbation decoder check: four selected tests passed, exit0. Actual tiny synthetic LightGBM fifteen-booster fit/replay/interruption/resume passed separately, exit0. The full suite exercised that case again. All results are SYNTHETIC_CORRECTNESS_ONLY. A deliberately serialized object future field is never decoded; corrupting the unadmitted suffix file leaves fitted prefix arrays unchanged. Source proof: original prefix byte identity, 31 exact AST definitions and restore/checkpoint/inference equivalence under only the declared artifact-kind string mapping. Final Ruff check and format check passed. Initial configuration failures, missing-adapter failures, typed synthetic-parent failure and subsequent fixture/format failures are preserved.

Two actual Torch optimizer/exact-continuation cases remain NOT_RUN in builder because the retained optimizer/cache denial must not be retried. ROOT must integrate the six code/test files and the small test_support.py fixture helper, then execute the supplied CPU-only root command. No public corpus, selected public weights, forecasts, GPU, real prefit, fit, access or review ran here. No agent, commit, denied-operation workaround or existing-source edit occurred. ROOT owns new worker routing, owned-tree deadline/RAM/GPU caps, one scientific process, full attempt accounting and actual independently reviewed manifests. Existing desktop workers only appear as immutable ancestry/source dependencies; this delivery does not change their routing or authorize their use with V4.

Worker APIs are prefix.admit/fit (plus CLI --manifest/--review/--output/[--resume]) and suffix.admit/execute_assessment/run. Suffix input_npz is an absolute bound identity. Use a fresh output and one true deployment per invocation; final/site data may only be supplied later under genuine separate approvals. Portable forecast artifacts are neural native_prefix_matched_transfer_inference_v4 or conventional native_prefix_lightgbm_inference_v1; resume kinds are separately typed. The coordinator's separate zero-shot CF-control adapter is outside this delivery.
"""
with (HERE / "IMPLEMENTATION_REPORT_V1.md").open("x", encoding="utf-8") as stream:
    stream.write(report)
checks = {}
for path in sorted(HERE.glob("*.json")):
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and "command" in value and "exit" in value:
        checks[path.name] = value
assert checks["focused-final-02.json"]["exit"] == 0
assert checks["prefix-decode-perturbation-02.json"]["exit"] == 0
assert checks["lint-closed-02.json"]["exit"] == 0
assert checks["format-close-verified-01.json"]["exit"] == 0
paths = sorted(path for path in HERE.iterdir() if path.is_file())
evidence = {path.name: sha(path) for path in paths}
root_test = ".venv\\Scripts\\python.exe -m pytest --import-mode=importlib -q -p no:cacheprovider tests/unit/test_native_prefix_matched_transfer_v4.py tests/integration/test_native_prefix_matched_transfer_v4.py tests/unit/test_native_prefix_suffix_assessment_v1.py tests/integration/test_native_prefix_suffix_assessment_v1.py --tb=short"
handoff = {"status": "CLOSED_ENGINEERING_DELIVERY", "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
    "implementer_session_id": "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1", "authored": authored,
    "dependency_hashes": current, "protected_sources_unchanged": True, "evidence": evidence,
    "actual_checks": checks, "parameter_counts": counts,
    "integration_support": {"test_support.py": evidence["test_support.py"]},
    "handoff_scope": "Only listed top-level files; private SYNTHETIC_CORRECTNESS_ONLY-* fixture trees and bytecode excluded",
    "root_required_commands": ["set NATIVE_PREFIX_MATCHED_ROOT_CHECKS=1", "set PYTHONPATH=src", root_test],
    "root_required_policy": "CPU-only private synthetic fixtures after exact integration; coordinator supervises lifetime600s/full-treeRAM22GiB. No scientific authority inherited.",
    "root_only_not_run": ["actual CF frozen/scratch Torch optimizer moments and exact continuation/inference fixture (two cases)", "production owner wrapper/routing/accounting integration", "actual completed public CF-control/prefix ancestor admission and replay"],
    "production_status": {key: "NOT_RUN" for key in ["scientific_fit", "numeric_access", "public_artifact_decode", "GPU", "independent_review", "held_out_evaluation", "real_source_prefit"]},
    "unresolved_limits": ["Conventional four-tree-check policy is proposed, requires coordinator ADR before reserved outcomes and distinct exact prefit", "New ROOT bounded process routing/accounting is coordinator-owned and not implemented here", "Original CF full-H96 pooled feature use is not claimed equivalent to sampled forward-zone training", "Small site counts/source-clock proxies limit intervals; unsupported support remains null", "Test fixtures use synthetic identities, never reviewer signatures or scientific approval"],
    "scientific_or_improvement_claim": None}
save("handoff-v1.json", handoff)
print(json.dumps({"status": handoff["status"], "authored": authored,
    "report": {"path": str(HERE / "IMPLEMENTATION_REPORT_V1.md"), "sha256": sha(HERE / "IMPLEMENTATION_REPORT_V1.md")},
    "handoff": {"path": str(HERE / "handoff-v1.json"), "sha256": sha(HERE / "handoff-v1.json")}, "parameter_counts": counts}, indent=2))
