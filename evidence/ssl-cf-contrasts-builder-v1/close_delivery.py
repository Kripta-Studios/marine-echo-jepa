"""Source-only immutable engineering closeout; no scientific payload decoding."""

import ast
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
builder = here.parents[1]
files = ["src/marine_echo/evaluation/native_cf_matched_contrasts_v1.py",
         "tests/unit/test_native_cf_matched_contrasts_v1.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    with (here / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


baseline = json.loads((here / "protected-baseline-v1.json").read_text())
current = {path: sha(path) for path in baseline}
assert current == baseline, "Immutable ROOT dependency changed"
source = (builder / files[0]).read_bytes()
tree = ast.parse(source)
forbidden = {"open", "load", "save", "savez", "savez_compressed", "read_bytes", "read_text",
             "write_bytes", "write_text", "reconstruct", "_admit", "compare_saved_predictions",
             "eval", "exec", "__import__", "compile"}
calls = []
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else None
        assert name not in forbidden, "Unexpected loading/gated/code call: " + str(name)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id in ("comparison", "reconstruction"):
            calls.append(node.func.value.id + "." + node.func.attr)
allowed = {"comparison._query", "comparison._common", "comparison._variation", "comparison._blocks",
           "comparison._geometry", "reconstruction.independent_scores", "reconstruction.independent_bootstrap"}
assert set(calls) == allowed
assert not any(isinstance(node, ast.Import) and any(alias.name.startswith("torch") for alias in node.names) for node in ast.walk(tree))
save("source-proof-final-v1.json", {"authored": {name: sha(builder / name) for name in files},
    "protected_baseline": baseline, "protected_current": current, "protected_sources_unchanged": True,
    "pure_math_root_calls": sorted(set(calls)), "no_gated_entry_or_io_call": True,
    "no_cli_model_or_torch_import": True,
    "scope": "Static AST plus executed call-time IO/RNG/mutation guard tests. Immutable ROOT helper imports pin source files at import, not public data or weights."})
report = """# CF matched numerical contrasts V1

Closed engineering delivery: one pure NumPy module and one synthetic test module. No scientific corpus, forecasts, saved weights, model factory, Torch/CUDA, fit, review operation or ROOT job was accessed. ROOT retains all admission, launcher, integration and scientific authority.

Public API: `prespecified_specs()` returns independent exact specs; `compute_cf_matched_contrasts(records, role='development', original_campaign_modified=False)` accepts a list/tuple of exactly thirteen dictionaries `{name, spec, arrays, lineage}`. Twelve are CF learned frozen/full, zero-SSL random frozen/scratch at seeds7/13/23; the reference is the original fixed LightGBM7. CF specs freeze width256/latent128/depth5, H96, original full-H96 forecasting extraction, 2000 frozen or3000 full/scratch supervised exposure, fresh seed+100000 query heads, native horizons1/3/6 and five quantiles. These are caller assertions, not evidence of actual architecture, weights, head states, ancestry or approvals. Shared/Band scratch substitutions, incomplete inventories and changed specs fail.

Arrays must contain floating predictions[N,3,5]/targets[N,3], a Boolean observed or target_observed label mask[N,3], Unicode row_id/deployment/date identities, native0-225m issued DEV queries[N,3,10] and integer cutoff[N]. Equal dual label masks are traced; context masks never replace labels. Optional context-mask/role/evidence fields remain raw provenance. Hidden context/future values, array callbacks, wrong roles and nonfinite observed values/forecasts fail. Every method must have the exact same canonical deployment+row set, label masks, observed truth, dates, queries and cutoff. No intersection, source rescaling, target filling or caller mutation occurs. Raw record arrays/metadata are independently copied in their original order; canonical common support and derived ensembles are separately returned as arrays, not serialized files.

Each seed retains its native daily/horizon/deployment metrics. Four descriptive arm means and sample SD(ddof1) include all three named seeds. Averaged aligned forecasts are separately scored and explicitly named forecast_mean_ensemble; their score is not a seed-score mean. Paired learned-frozen minus random-frozen and learned-full minus CF-scratch preserve seed pairing and report per-seed differences/intervals, mean difference/interval and descriptive seed-difference SD. Each ensemble is also compared against LightGBM with ensemble-minus-reference direction. No best seed/model is chosen.

The immutable ROOT independent_scores/independent_bootstrap numerical functions are reused, never their adapted-suffix admission/entrypoint. One shared resampling sequence covers all thirteen records plus four ensembles:2000 draws/seed20260929, deployments then nonoverlapping two-source-calendar-day blocks anchored at each deployment's earliest target date. Blocks preserve calendar gaps, horizon/method pairing, repeated-block multiplicity and equal date→horizon→deployment weights, floor18. Unsupported scores remain null without dropping a seed; unsupported bootstrap draws are not filtered into an interval, and zero source-calendar span yields null intervals. Dates and48 nominal source-hour blocks are not verified UTC. Raw records, geometry, daily metrics, coverage, block assignments and labels remain visible.

Lineage is always CALLER_AUTHORED_UNVERIFIED; access_authority and independent_scientific_approval are false; scientific_claim is null. Caller metadata claiming APPROVED does not change these fields. Original43-neural/47-total campaign requirements are unchanged; this extension remains CF_MATCHED_CONTROLS. The CF sampled-SSL-crop/full-H96 mismatch and residual head RNG confound are disclosed, not fixed. Neither SSL objective specificity nor SOTA/transfer benefit is established.

Actual red numerical check failed with seed-mean/ensemble confusion; its source and exit1 log are preserved. Final58 synthetic CPU tests passed in1.50s, exit0. They cover nonlinear ensemble vs score mean, independently calculated unequal day/horizon/deployment weights, all seeds/SD/direction, reorder/masked-fill invariance, exact schema/inventory guards, native225, no mutation/global RNG changes/call-time IO/gated-entry calls, read-only arrays, ignored inaccessible ancestry paths, zero contrasts, incomplete support, zero span and unsupported draws. Final Ruff check/format check both exited0. Seven ROOT source/config/dependency hashes match their captured baseline. No denied scratch/cache/index operation was retried; no agents, commits or ROOT edits occurred. Each check command has a300-second bound and small private in-memory fixtures; no virtual process was presented as a scientific process.

ROOT should integrate only the two exact authored files, then run the provided pytest and Ruff commands. Inputs remain the caller's separately admitted responsibility. This delivery includes no production CLI or source/prefit/numeric review, real comparison, public final access, fitting, packaging or scientific approval.
"""
with (here / "IMPLEMENTATION_REPORT_V1.md").open("x", encoding="utf-8") as stream:
    stream.write(report)
checks = {}
for path in here.glob("*.json"):
    value = json.loads(path.read_text())
    if isinstance(value, dict) and "command" in value and "exit" in value:
        checks[path.name] = value
for name in ("green-closed-01.json", "lint-closed-01.json", "format-closed-01.json"):
    assert checks[name]["exit"] == 0
assert checks["red-01.json"]["exit"] == 1
evidence = {path.name: sha(path) for path in here.iterdir() if path.is_file()}
handoff = {"status": "CLOSED_ENGINEERING_DELIVERY", "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
    "implementer_session_id": "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1", "owner_supplied_root_head": "8621b23",
    "authored": {name: sha(builder / name) for name in files}, "dependency_hashes": current,
    "protected_sources_unchanged": True, "actual_checks": checks, "evidence": evidence,
    "root_check_commands": [
        ".venv\\Scripts\\python.exe -m pytest --import-mode=importlib -q -p no:cacheprovider tests/unit/test_native_cf_matched_contrasts_v1.py --tb=short",
        ".venv\\Scripts\\python.exe -m ruff check src/marine_echo/evaluation/native_cf_matched_contrasts_v1.py tests/unit/test_native_cf_matched_contrasts_v1.py",
        ".venv\\Scripts\\python.exe -m ruff format --check src/marine_echo/evaluation/native_cf_matched_contrasts_v1.py tests/unit/test_native_cf_matched_contrasts_v1.py"],
    "api": {"module": "marine_echo.evaluation.native_cf_matched_contrasts_v1", "specs": "prespecified_specs()",
        "function": "compute_cf_matched_contrasts(records, *, role='development', original_campaign_modified=False)",
        "record_fields": ["name", "spec", "arrays", "lineage"], "record_count": 13,
        "raw_output": "independently copied in-memory arrays; no filesystem serialization or CLI"},
    "lineage_status": "CALLER_AUTHORED_UNVERIFIED", "scientific_claim": None,
    "production_status": {k: "NOT_RUN" for k in ("real_saved_prediction_reconstruction", "public_data_or_weights", "fitting", "CUDA", "source_prefit_numeric_approval", "independent_review")},
    "limits": ["Specs cannot authenticate actual model/head/scaler/ancestor/source lineage or approvals", "Pure numeric function expects already-legitimately-admitted data; it creates no access authority", "CF sampled-crop/full-H96 mismatch and residual head RNG confound unchanged", "Fixed-seed SD is descriptive; deployment/date bootstrap is not a seed population interval", "Nominal48 source-hour blocks and limited deployment/site identities constrain interpretation", "Original43/47 campaign remains separately required and unchanged"]}
save("handoff-v1.json", handoff)
print(json.dumps({"status": handoff["status"], "authored": handoff["authored"],
    "report_sha256": sha(here / "IMPLEMENTATION_REPORT_V1.md"), "handoff_sha256": sha(here / "handoff-v1.json"),
    "dependency_hashes": current}, indent=2))
