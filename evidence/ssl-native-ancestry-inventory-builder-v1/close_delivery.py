"""Close exact source proof and actual synthetic-check receipts, without data loads."""

from __future__ import annotations

import ast
import base64
import difflib
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
BUILDER = BASE.parents[1]
AUTHORED = (
    "src/marine_echo/evaluation/native_ancestry_inventory.py",
    "tools/prepare_native_research_inventory.py",
    "tests/unit/test_native_ancestry_inventory.py",
    "tests/integration/test_native_ancestry_inventory.py",
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(name, document):
    with (BASE / name).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return digest((BASE / name).read_bytes())


def snapshot(path):
    raw = path.read_bytes()
    return {"sha256": digest(raw), "bytes_base64": base64.b64encode(raw).decode("ascii")}


def main():
    before = json.loads((BASE / "protected-baseline-v1.json").read_bytes())
    current = {path: snapshot(Path(path)) for path in before}
    changed = [path for path in before if current[path]["sha256"] != before[path]["sha256"]]
    if changed:
        raise RuntimeError(
            "Protected dependencies changed; do not claim a closed unchanged-source proof."
        )
    dump("protected-current-v1.json", current)
    authored = {name: snapshot(BUILDER / name) for name in AUTHORED}
    dump("authored-snapshots-final-v1.json", authored)
    ast_proof = {}
    for path, entry in current.items():
        if not path.endswith(".py"):
            continue
        old = ast.dump(
            ast.parse(base64.b64decode(before[path]["bytes_base64"])), include_attributes=False
        )
        new = ast.dump(ast.parse(base64.b64decode(entry["bytes_base64"])), include_attributes=False)
        assert old == new
        ast_proof[path] = {
            "baseline_ast_sha256": digest(old.encode()),
            "current_ast_sha256": digest(new.encode()),
            "unchanged": True,
        }
    source = BUILDER / AUTHORED[0]
    tree = ast.parse(source.read_bytes())
    calls = sorted({ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)})
    forbidden = {
        "np.load",
        "numpy.load",
        "torch.manual_seed",
        "torch.compile",
        "torch.set_default_dtype",
        "eval",
        "exec",
        "subprocess.Popen",
    }
    assert not forbidden & set(calls)
    load_calls = [
        n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == "torch.load"
    ]
    assert len(load_calls) == 1
    flags = {item.arg: ast.literal_eval(item.value) for item in load_calls[0].keywords}
    assert flags["weights_only"] is True and flags["map_location"] == "cpu"
    proof = {
        "kind": "native_ancestry_inventory_source_proof_v1",
        "evidence_kind": "SOURCE_ONLY",
        "authored": {name: entry["sha256"] for name, entry in authored.items()},
        "protected_count": len(current),
        "protected_changed": changed,
        "protected_baseline_current": {
            path: {"baseline_sha256": before[path]["sha256"], "current_sha256": entry["sha256"]}
            for path, entry in current.items()
        },
        "protected_asts": ast_proof,
        "candidate_call_inventory": calls,
        "safe_load_flags": flags,
        "numerical_corpus_or_public_weights_decoded": False,
        "static_closure_note": "Exact immutable source schemas/import closure read by AST, never provenance-directed imports. Runtime manifest must bind required_sources() and every original embedded review/run binding.",
        "write_scope": [*AUTHORED, "evidence/ssl-native-ancestry-inventory-builder-v1/**"],
        "previous_transfer_corpus_delivery": "Preserved byte-exact among protected current sources; no prior receipt or delivery edited.",
        "proof_limit": "Source inspection and synthetic codec tests do not establish real artifact ancestry, scientific quality, final selection or numerical access.",
    }
    proof_hash = dump("source-proof-final-v1.json", proof)
    with (BASE / "four-new-files-final-v1.patch").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        for name in AUTHORED:
            stream.writelines(
                difflib.unified_diff(
                    [],
                    (BUILDER / name).read_text(encoding="utf-8").splitlines(keepends=True),
                    fromfile="/dev/null",
                    tofile="b/" + name,
                )
            )
    checks = {
        "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
        "python": "../marine-echo-jepa/.venv/Scripts/python.exe",
        "shell": "cmd.exe",
        "red": [
            {
                "command": "python -B -m pytest tests/unit/test_native_ancestry_inventory.py --import-mode=importlib -p no:cacheprovider -q",
                "exit_code": 1,
                "tool_chunk": "53e2dd",
                "log": "red-parent-graph-v1.log",
            },
            {
                "command": "python -B -m pytest tests/unit/test_native_ancestry_inventory.py tests/integration/test_native_ancestry_inventory.py --import-mode=importlib -p no:cacheprovider -q",
                "exit_code": 1,
                "tool_chunk": "233281",
                "log": "red-full-codec-v1.log",
                "note": "Only exact owned pytest session 77159 was interrupted during slow checks. One failure observed; no complete count/result claimed. The code enumerated process children for every tensor; replaced with single-process RSS checks. No foreign process or fit affected.",
            },
            {
                "command": "python -B -m pytest tests/integration/test_native_ancestry_inventory.py -k unicode_durable --import-mode=importlib -p no:cacheprovider -q",
                "exit_code": 1,
                "tool_chunk": "a2a1f5",
                "log": "red-assessment-receipt-v2.log",
                "result": "1 failed, 37 deselected in 28.18s",
            },
        ],
        "green": {
            "command": "python -B -m pytest tests/unit/test_native_ancestry_inventory.py tests/integration/test_native_ancestry_inventory.py --import-mode=importlib -p no:cacheprovider -q",
            "exit_code": 0,
            "result": "48 passed in 308.94s",
            "tool_chunk": "3accce",
            "log": "green-full-v2.log",
        },
        "lint": {
            "initial_exit_code": 1,
            "initial_log": "lint-red-v1.log",
            "corrected_exit_code": 0,
            "corrected_tool_chunk": "e22ae3",
            "corrected_log": "lint-v2.log",
            "final_witness": "final-checks-v1.json",
        },
        "help": {
            "command": "python -B tools/prepare_native_research_inventory.py --help",
            "exit_code": 0,
            "tool_chunk": "1359ad",
            "log": "cli-help-v1.log",
        },
        "real_root_weight_metadata_audit": "NOT_RUN_ROOT_ONLY",
        "public_corpus_weights_gpu_fitting": "NOT_RUN",
        "restricted_cache_scratch_git_operations": "NOT_RUN_NO_RETRY",
    }
    dump("executed-checks-final-v1.json", checks)
    report = """# Native ancestry inventory implementation

The four new files provide explicit completed-endpoint inventory derivation and a CPU CLI. There is no output enumeration, best-seed choice, model construction, optimizer, RNG reset, inference, corpus decoding, fitting or final numerical access. Real completed-weight decoding is ROOT-only after integration; all builder artifacts and checks are private SYNTHETIC_CORRECTNESS_ONLY.

The versioned input manifest names exact completed directories, kinds, methods, seeds, modes, parent links, original configs/reviews, original TRAIN byte/report/cohort identities and every source/artifact binding. Metadata identities and graph closure are checked before safe tensor loads. TRAIN is the original TR18593 reservation; complete unique ordered row/source membership and original split/report counts must agree. Reserved AEON2 and development identities cannot enter fitted membership. Original embedded review bindings and source/config/scaler provenance are preserved; incomplete or stale ancestry fails closed. Original config schemas are read from immutable source AST without importing trainers or constructing models.

The three original/Band V1/replication V2 kinds remain distinct. Band V1 retains seed7; V2 supports only 7/13/23. Controls, CF EMA features, fresh supervised direct and supervised full-finetuning keep accurate labels. Frozen selected artifacts must match the actual inference encoder, parent tensors/buffers and byte-identical selected file. Full-finetuning requires actual tensor differences, not archive-byte inequality. Unchanged parent probe heads are rejected as unable to establish a fresh readout. Declared multi-generation ancestry resolves exact completed run and selected hashes; missing, substituted, foreign, cyclic and unsupported supervised selected parents are rejected.

Outputs contain exact config/statistics, complete TRAIN cohort, original whole-deployment identity transport, fitted-input and native_assessment_ancestry_v1 receipts, parent links and NOT_FINAL_SELECTION metadata. The derived assessment split, cohort and fitted-input hashes agree; original split paths/hashes remain separately recorded. Existing 82 receipts are not overwritten. Proposed freeze inputs contain no owner confirmation or APPROVED status. Persistence/seasonal controls get no-fit ancestry. LightGBM records its original fifteen booster identities and local TRAIN lineage without booster loading/refitting. External Chronos remains UNKNOWN_EXTERNAL reference-only with no clean-local guarantee or neural-kind claim.

Actual checks: 48 synthetic CPU unit/integration cases passed in 308.94 seconds, exit0. They exercise actual safe Torch serialization, durable Unicode outputs, all typed versions/seeds, multi-generation frozen/full ancestry, fresh direct/random/control labels, tensor equality/differences, stale and malformed identities, safe load flags, unsafe pickle rejection, output collisions and reference separation. Focused graph and assessment-receipt red evidence is retained. An earlier owned pytest run was interrupted after a witnessed failure during slow per-tensor process enumeration; its partial log is not counted as a complete run. Single-process RSS checks replace that enumeration. No foreign process action occurred.

Ruff check and format checks have actual final witnesses in final-checks-v1.json. Source-proof-final-v1.json records exact four-file hashes, protected before/current source bytes and unchanged ASTs. All 28 protected baseline files are unchanged, including the closed transfer adapter and original fitted sources/contracts. No prior evidence, scientific ledger, approval, source, dependency or lock was changed.

The inventory checks this single CPU process RSS below22GiB before/after safe mmap loads and tensor checks; ROOT should supervise the actual audit for continuous resource enforcement. Actual completed-weight metadata audit is NOT_RUN. Incomplete raw membership cannot be repaired by reading numerical arrays. Portable inference is unaffected: inventory audit requires original bound paths, while model-only loaders retain their separately reviewed relocation behavior. Metadata receipts establish no scientific quality, programme completion, SOTA, final selection or final access. Distinct ancestry/freeze/assessment review remains mandatory. No self-review or Git operation is delivered.

ROOT manifest and precise CLI contract are in MANIFEST_CONTRACT_V1.md. After exact-byte integration:

```text
python -B -m pytest tests/unit/test_native_ancestry_inventory.py tests/integration/test_native_ancestry_inventory.py --import-mode=importlib -p no:cacheprovider -q
python -B tools/prepare_native_research_inventory.py --manifest ROOT_BOUND_MANIFEST.json --output ROOT_NEW_INVENTORY_DIRECTORY
```

The second command is a ROOT-required actual metadata audit, not executed by builder and not an access approval. ROOT supplies all fixed completed endpoints without model selection, supervises CPU/RSS, then seeks distinct review before any final freeze or held-out assessment. No new scientific task has begun.
"""
    report_path = BASE / "NATIVE_ANCESTRY_INVENTORY_IMPLEMENTATION_V1.md"
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report)
    handoff = {
        "kind": "native_ancestry_inventory_builder_handoff_v1",
        "status": "CLOSED",
        "implementer_session_id": "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1",
        "scope": "ENGINEERING_METADATA_DERIVATION_NO_SCIENTIFIC_APPROVAL",
        "authored": proof["authored"],
        "source_proof": {"file": "source-proof-final-v1.json", "sha256": proof_hash},
        "report": {"file": report_path.name, "sha256": digest(report_path.read_bytes())},
        "protected_count": len(current),
        "protected_changed": [],
        "checks": {
            "synthetic_cpu_passed": 48,
            "pytest_exit_code": 0,
            "lint_witness": "final-checks-v1.json",
        },
        "root_required_commands": [
            "python -B -m pytest tests/unit/test_native_ancestry_inventory.py tests/integration/test_native_ancestry_inventory.py --import-mode=importlib -p no:cacheprovider -q",
            "python -B tools/prepare_native_research_inventory.py --manifest ROOT_BOUND_MANIFEST.json --output ROOT_NEW_INVENTORY_DIRECTORY",
        ],
        "actual_completed_weight_audit": "NOT_RUN_ROOT_ONLY",
        "independent_review": "NOT_RUN_DISTINCT_REQUIRED",
        "final_numeric_access": "NOT_RUN",
        "programme_completion": "NOT_ESTABLISHED",
        "sota": "NOT_ESTABLISHED",
        "evidence": {
            path.name: digest(path.read_bytes())
            for path in BASE.iterdir()
            if path.is_file() and path.name != "handoff-v1.json"
        },
        "private_fixture_directory_note": "Retained SYNTHETIC_CORRECTNESS_ONLY directories contain real tiny Torch codecs and metadata, never public numerical evidence or genuine approvals.",
        "commit": "NOT_RUN_NO_GIT_INDEX_OPERATION",
    }
    handoff_hash = dump("handoff-v1.json", handoff)
    print(
        json.dumps(
            {
                "status": "CLOSED",
                "authored": proof["authored"],
                "protected_count": len(current),
                "source_proof_sha256": proof_hash,
                "report_sha256": handoff["report"]["sha256"],
                "handoff_sha256": handoff_hash,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
