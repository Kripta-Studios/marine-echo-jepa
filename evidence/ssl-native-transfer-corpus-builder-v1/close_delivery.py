"""Close source-only proof and actual check receipts; never decode numerical data."""

from __future__ import annotations

import ast
import base64
import difflib
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
BUILDER = BASE.parents[1]
MAIN = BUILDER.parent / "marine-echo-jepa"
AUTHORED = (
    "src/marine_echo/data/native_transfer_corpus.py",
    "tools/materialize_native_transfer_corpus.py",
    "tests/unit/test_native_transfer_corpus.py",
    "tests/integration/test_native_transfer_corpus.py",
)
SUPPORT = (
    "root_supervised_fixture.py",
    "snapshot_sources.py",
    "inspect_sources.py",
    "run_closeout_checks.py",
    "close_delivery.py",
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def snapshot(path):
    raw = path.read_bytes()
    return {"sha256": digest(raw), "bytes_base64": base64.b64encode(raw).decode("ascii")}


def write_json(name, value):
    with (BASE / name).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return digest((BASE / name).read_bytes())


def ast_digest(raw):
    return digest(ast.dump(ast.parse(raw), include_attributes=False).encode("utf-8"))


def source_closure():
    # Mirrors the fixed local import resolution, without importing numerical code.
    package = MAIN / "src/marine_echo"
    pending = [
        BUILDER / AUTHORED[0],
        BUILDER / AUTHORED[1],
        package / "data/native_ssl_corpus.py",
        package / "training/native_prefix_transfer.py",
        package / "training/native_references.py",
        MAIN / "tools/native_reference_supervisor.py",
    ]
    found = {}
    while pending:
        path = pending.pop().resolve()
        if str(path) in found:
            continue
        found[str(path)] = snapshot(path)
        for node in ast.walk(ast.parse(path.read_bytes())):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] + [node.module + "." + alias.name for alias in node.names]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            for name in names:
                if not name.startswith("marine_echo"):
                    continue
                parts = name.split(".")[1:]
                for n in range(len(parts) + 1):
                    candidate = package.joinpath(*parts[:n])
                    for source in (candidate.with_suffix(".py"), candidate / "__init__.py"):
                        if source.is_file():
                            pending.append(source)
    return found


def main():
    baseline_path = BASE / "baseline-source-snapshots-v1.json"
    baseline = json.loads(baseline_path.read_bytes())
    current = {name: snapshot(Path(name)) for name in baseline}
    changed = [name for name in baseline if baseline[name]["sha256"] != current[name]["sha256"]]
    if changed:
        raise RuntimeError("Protected baseline changed; record unresolved evidence before closing.")
    authored = {name: snapshot(BUILDER / name) for name in AUTHORED}
    closure = source_closure()
    write_json("current-protected-source-snapshots-v1.json", current)
    write_json("authored-source-snapshots-final-v1.json", authored)
    write_json("current-static-source-closure-v1.json", closure)
    # These metadata-only files were inspected later, so no pre-edit claim is made.
    metadata = {
        str(path): snapshot(path)
        for path in (
            MAIN / "configs/native_ssl_split_v1.json",
            MAIN / "docs/adr/0021-native-prefix-transfer-assessment.md",
        )
    }
    write_json("late-current-metadata-snapshots-v1.json", metadata)
    ast_proof = {}
    for relative in (
        "src/marine_echo/data/native_ssl_corpus.py",
        "src/marine_echo/training/native_prefix_transfer.py",
    ):
        path = str((MAIN / relative).resolve())
        before = base64.b64decode(baseline[path]["bytes_base64"])
        after = Path(path).read_bytes()
        tree = ast.parse(after)
        ast_proof[relative] = {
            "before_sha256": baseline[path]["sha256"],
            "current_sha256": current[path]["sha256"],
            "before_ast_sha256": ast_digest(before),
            "current_ast_sha256": ast_digest(after),
            "byte_identical": before == after,
            "named_definitions": {
                node.name: digest(ast.dump(node, include_attributes=False).encode("utf-8"))
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            },
        }
    tree = ast.parse((BUILDER / AUTHORED[0]).read_bytes())
    calls = sorted(
        {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    )
    required_calls = {"reader.read_source", "reader.issue_windows", "reader.check_numeric_review"}
    assert required_calls <= set(calls)
    forbidden = {"torch.load", "torch.manual_seed", "torch.compile", "eval", "exec"}
    assert not forbidden & set(calls)
    proof = {
        "kind": "native_transfer_source_proof_v1",
        "evidence_kind": "SOURCE_ONLY",
        "authored": {name: entry["sha256"] for name, entry in authored.items()},
        "baseline_snapshot_sha256": digest(baseline_path.read_bytes()),
        "protected_files": len(current),
        "protected_changed": changed,
        "protected_baseline_current": {
            name: {"baseline_sha256": baseline[name]["sha256"], "current_sha256": entry["sha256"]}
            for name, entry in current.items()
        },
        "static_source_closure": {name: entry["sha256"] for name, entry in closure.items()},
        "immutable_scientific_ast": ast_proof,
        "direct_original_reader_calls": sorted(required_calls),
        "candidate_call_inventory": calls,
        "proof_limit": "AST absence and byte equality are source inspection, not executed admission or resource lifecycle.",
        "array_value_proof": {
            "kind": "SYNTHETIC_CORRECTNESS_ONLY",
            "test": "test_actual_reader_source_metadata_and_two_source_equivalence",
            "equality": "Original issue_windows arrays checked for dtype, values and tobytes equality; two private source ZIPs. No public archive decoded.",
        },
        "prefix_baseline": "Already-authorized preceding V4 source extension; not a historical V1 baseline.",
        "write_scope": [*AUTHORED, "evidence/ssl-native-transfer-corpus-builder-v1/**"],
        "ownership_fixture": {
            "actual_lock": "evidence/ssl-builder-v1/gpu-owner.lock",
            "actual_ledger": "orchestration/native_ssl_run_ledger_v1.json",
            "pending": [
                "all.pending",
                ".pending",
                ".prefix-pending",
                ".band-pending",
                ".assessment-pending",
                ".reconciliation-pending",
            ],
            "checks_before_mkdir_and_spawn": True,
            "actual_lifecycle": "NOT_RUN_ROOT_ONLY",
        },
    }
    proof_hash = write_json("source-proof-final-v1.json", proof)
    with (BASE / "four-path-new-files-final-v1.patch").open(
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
        "receipt_note": "Historical summaries below retain actual prior tool witnesses; they are not invented complete stdout logs.",
        "actual_checks": [
            {
                "command": "python -B -m pytest tests/unit/test_native_transfer_corpus.py tests/integration/test_native_transfer_corpus.py --import-mode=importlib -p no:cacheprovider -q",
                "exit_code": 0,
                "result": "35 passed in 386.74s",
                "version": "Prior witnessed source version before later guards",
            },
            {
                "command": "python -B -m pytest tests/integration/test_native_transfer_corpus.py --import-mode=importlib -p no:cacheprovider -q",
                "exit_code": 0,
                "result": "24 passed in 173.98s",
                "tool_chunk": "a16da6",
            },
            {
                "command": "python -B -m pytest tests/integration/test_native_transfer_corpus.py -k wrong_review --import-mode=importlib -p no:cacheprovider -q",
                "exit_code": 0,
                "result": "10 passed, 14 deselected in 44.05s",
            },
            {
                "command": "python -B -m pytest tests/integration/test_native_transfer_corpus.py -k actual_ --import-mode=importlib -p no:cacheprovider -q",
                "exit_code": 0,
                "result": "2 passed, 21 deselected in 52.24s",
                "version": "Prior source version",
            },
            {
                "command": "python -B -m pytest tests/unit/test_native_transfer_corpus.py -k root_only_fixture --import-mode=importlib -p no:cacheprovider -q",
                "exit_code": 0,
                "log": "green-actual-ownership-closeout-v2.log",
                "result": "12 passed, 16 deselected in 1.77s",
            },
            {
                "command": "python -B -m pytest tests/unit/test_native_transfer_corpus.py -k production_transport_calls --import-mode=importlib -p no:cacheprovider -q",
                "exit_code": 1,
                "log": "red-production-wiring-closeout-v1.log",
                "tool_chunk": "0f9b73",
            },
            {
                "command": "python -B -m pytest tests/unit/test_native_transfer_corpus.py -k fixture_or_production --import-mode=importlib -p no:cacheprovider -q",
                "exit_code": 5,
                "log": "green-wiring-closeout-v4.log",
                "note": "No tests selected; not a passing check.",
            },
            {
                "command": "python -B -m pytest tests/unit/test_native_transfer_corpus.py -k production_transport_calls --import-mode=importlib -p no:cacheprovider -q",
                "exit_code": 0,
                "log": "green-production-wiring-closeout-v5.log",
                "tool_chunk": "4cdfdd",
            },
            {
                "command": "python -B tools/materialize_native_transfer_corpus.py --help",
                "exit_code": 0,
                "tool_chunk": "845f44",
            },
        ],
        "retained_red_receipts": [
            "red-initial-v1.json",
            "red-semantic-freeze-v1.json",
            "red-future-approval-v1.json",
            "red-actual-ownership-closeout-v1.json",
        ],
        "prior_lint": {"ruff_check_exit": 0, "tool_chunk": "50e958"},
        "interrupted_event": {
            "cli": 95119,
            "format_check": "NO_ACTUAL_EXIT_WITNESS",
            "cause": "UNKNOWN",
            "not_counted_as_pass": True,
        },
        "inaccessible_or_unexecuted": {
            "actual_owned_supervisor_lifecycle": "NOT_RUN_ROOT_ONLY_DURING_SCIENTIFIC_QUEUE",
            "public_materialization": "NOT_RUN_NO_DISTINCT_FREEZE_OR_NUMERIC_ACCESS_REVIEW",
            "public_arrays_weights_fitting_gpu": "NOT_RUN",
            "restricted_scratch_cache_git_operations": "NOT_RUN_NO_RETRY",
        },
    }
    write_json("executed-check-history-final-v1.json", checks)
    report = """# Native transfer corpus engineering delivery

The four new files implement additive native held-out and prefix materialization. Original reader functions are called directly. The protected reader, preceding authorized V4 prefix validator, scientific helpers and supervisor remain byte-identical to the 25-file task baseline. Current-only split/ADR snapshots are explicitly separate from that baseline. The final proof records exact bytes, static source closure and unchanged scientific ASTs.

Admission requires a distinct APPROVED_NATIVE_NUMERIC_ACCESS final_test review, exact source/runtime/output bindings and an independently reviewed earlier owner selection freeze. Source-role, completed endpoint, chronology, archive and output identities fail closed before archive parsing. There is no fitting, inference, synthetic production-admission shortcut or authority to grant numerical access.

Native 0–230 m geometry and actual reader arrays remain unchanged. Separate metadata receipts contain source ordinal IDs, original Date_M/Time_M clocks, configuration segments, QC and structural-gap explanations without acoustic values. Actual missing or invalid rows differ from structurally absent requested future IDs. True deployment identities are retained across 150/180-ping configurations; 165 remains quarantined. Private real-codec fixtures compare original arrays by dtype, values and bytes. Source clock is UNKNOWN UTC; centre ±30 minutes is a proxy. Richer profiles are unnecessary. Exact secondary clocks that cannot satisfy the immutable prefix registry fail closed instead of being fabricated.

Outputs are exclusive NPZ/cohort/configuration/raw-interval/gap/hash/QC products. Prefix 1/7/30 cohorts reuse unchanged partitions with four-day warmup and common source-start +41-day suffix; fitting inputs contain no future-crop arrays or suffix numeric path. Original DEV, if admitted, is transported without refit. Insufficient suffix support remains NOT_ASSESSABLE.

Actual prior combined checks: 35 passed (386.74 s). Latest integration checks: 24 passed (173.98 s). These runs overlap and must not be added. Closeout ownership checks: 12 passed; two new production-call signature checks passed after meaningful red failures exposed missing/extra transport arguments. The 16 existing core unit cases were previously executed; no claim of one final combined 54-case run is made. Final witnessed Ruff receipts and exact commands are in final-lint-witness-v1.json. Historical failures, partial synthetic output and the interrupted format event remain preserved; that event is NO_ACTUAL_EXIT_WITNESS, cause UNKNOWN.

The ROOT-only lifecycle fixture now checks the actual gpu-owner.lock/native_ssl_run_ledger_v1.json, active CUDA/CPU fit records, reconciliation flags and all pending-journal families before mkdir or spawn. It never removes locks or reconciles records. Its policy checks used only private fakes, not an actual process or scientific ledger. Actual lifecycle is NOT_RUN while ROOT has a scientific GPU queue.

Worker RSS sampling is additional evidence, not continuous peak enforcement. ROOT must use the immutable owned-process supervisor with 22 GiB tree RSS and a root-approved deadline. Actual supervised lifecycle, public archives/arrays/weights, scientific materialization, fitting and GPU execution are NOT_RUN. Restricted scratch/cache/Git operations were not retried. No scientific result, quality claim, access/prefit approval, self-review, commit or publication is delivered.

ROOT instructions (after integrating exact bytes):

```text
python -B -m pytest tests/unit/test_native_transfer_corpus.py tests/integration/test_native_transfer_corpus.py --import-mode=importlib -p no:cacheprovider -q
python -B evidence/ssl-native-transfer-corpus-builder-v1/root_supervised_fixture.py
```

Run the second command only when the actual scientific ledger/lock/journals are idle. Separately reviewed real worker invocation, under supervise_owned(child, started=started, deadline_seconds=ROOT_APPROVED_DEADLINE, rss_limit_bytes=22*1024**3):

```text
python -B tools/materialize_native_transfer_corpus.py --manifest FROZEN_MANIFEST.json --review DISTINCT_NUMERIC_ACCESS.json --output NEW_ROOT_OUTPUT --receipt NEW_ROOT_RECEIPT.json
```

The placeholder command is NOT_RUN and grants no approval. ROOT owns the actual freeze/access receipt, ledger, supervisor, review and execution. The immutable handoff binds the four authored files, support, evidence and final proof. No further task has begun.
"""
    report_path = BASE / "NATIVE_TRANSFER_CORPUS_IMPLEMENTATION_V1.md"
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report)
    handoff = {
        "kind": "native_transfer_corpus_builder_handoff_v1",
        "status": "CLOSED",
        "delivery_scope": "ENGINEERING_ONLY_NO_NUMERIC_ACCESS_OR_SCIENTIFIC_RESULT",
        "implementer_session_id": "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1",
        "authored": proof["authored"],
        "source_proof": {"path": str(BASE / "source-proof-final-v1.json"), "sha256": proof_hash},
        "report": {"path": str(report_path), "sha256": digest(report_path.read_bytes())},
        "protected_files": len(current),
        "protected_changed": [],
        "support": {name: digest((BASE / name).read_bytes()) for name in SUPPORT},
        "evidence": {
            path.name: digest(path.read_bytes())
            for path in BASE.iterdir()
            if path.is_file() and path.name != "handoff-v1.json"
        },
        "synthetic_fixture_tree_note": "Private fixture directories retained intact; no acoustic fixture array dump in handoff. Selected real-codec proof remains SYNTHETIC_CORRECTNESS_ONLY.",
        "root_required_commands": [
            "python -B -m pytest tests/unit/test_native_transfer_corpus.py tests/integration/test_native_transfer_corpus.py --import-mode=importlib -p no:cacheprovider -q",
            "python -B evidence/ssl-native-transfer-corpus-builder-v1/root_supervised_fixture.py",
        ],
        "root_required_status": "NOT_RUN_ACTUAL_LIFECYCLE_AND_REAL_ACCESS",
        "approval": "NONE_IMPLEMENTER_CANNOT_SELF_APPROVE",
        "final_test_numeric_access": "NOT_RUN",
        "git_commit": "NOT_RUN_NO_INDEX_RETRY",
    }
    handoff_hash = write_json("handoff-v1.json", handoff)
    print(
        json.dumps(
            {
                "status": "CLOSED",
                "authored": proof["authored"],
                "source_proof_sha256": proof_hash,
                "report_sha256": handoff["report"]["sha256"],
                "handoff_sha256": handoff_hash,
                "protected_files": len(current),
                "protected_changed": [],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
