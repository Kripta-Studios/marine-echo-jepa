# T08 read-only real-campaign preflight

- Status: REVIEW / BLOCKED_REAL_EXECUTION. No real benchmark run was started or recorded.
- Dependencies: approved physical R0 corpus, independent R1, full canonical real-corpus manifest and eligible-day audit, real train/validation executors, then separate R2 before final test.
- Session: `/root/continuation_builder`, configured native `sol_builder` `gpt-6-sol`/`high`; provider-side attestation not exposed. Isolated worktree `../marine-echo-jepa-core`, branch `impl/core`.
- Owned paths: `src/marine_echo/training/campaign_preflight.py`, `tests/integration/test_campaign_preflight.py`, `evidence/continuation-builder/preflight-*.txt`, this report. The main CLI is owned by the coordinator because this bounded core branch does not contain `src/marine_echo/serving/cli.py`.

`inspect_campaign_preflight(root)` reads only fixed local JSON evidence, hashes each file it opens, and reports protocol, R0/R1, TRAIN census, canonical-corpus and executor blockers. It does not open NPZ acoustic arrays, alter the production training registry, call a trainer, or read protected test outcomes. It has no `READY` return path. Even caller-authored approval flags or a fixture processing manifest cannot authorize real execution; the real promotion contract and executors are absent. A missing or unrecognized protocol status blocks promotion.

The first independent review found that the strict census's per-context-bin 80% rule is not frozen in Docs03/06, so its global D1 classification cannot be used. The revised preflight only reports `D1_MINIMUM_IMPOSSIBLE` from the prospective target-only necessary-bound report with exact `D1_INELIGIBLE_TARGET_SUPPORT_UPPER_BOUND` status, verified contract/code/execution/strict-report/protocol hashes, exact integer counts, zero benchmark runs and false exposure flags. A missing or inconsistent target-only report yields `TARGET_ONLY_BOUND_MISSING/UNVERIFIED` and no D1 conclusion. The target-only bound is still a necessary-condition diagnostic, not corpus approval.

The read-only invocation on the current main metadata returned `BLOCKED` with `PROTOCOL_NOT_REVIEWED`, `R0_NOT_APPROVED`, `R1_NOT_APPROVED`, `CENSUS_INCOMPLETE`, `CENSUS_ELIGIBILITY_MISSING`, `TARGET_ONLY_BOUND_MISSING`, `REAL_CORPUS_INTERFACE_MISSING`, and `REAL_EXECUTOR_MISSING`. This is a checkpoint, not R0/R1 approval or a benchmark result. CLI integration can call `inspect_campaign_preflight(root_path())` and emit its result while preserving the existing registry. It must continue exiting nonzero and must not imply that changing one review flag enables training.

Actual missing contracts before real campaign execution: independently reviewed real canonical schema with source/config/calibration/split/protocol bindings and protected metadata exposure policy; dataset-wide eligible-day audit; streaming train/validation rows carrying scalar and profile targets plus train-only scaler provenance; concrete bounded B0–B3 and direct/JEPA/hybrid/control executors with checkpoint/validation/resume artifacts under one GPU owner. Existing `run_steps` takes in-memory `TrainBatch` objects, and the baseline adapter/runner and finite scheduler only accept synthetic fixtures. Their fixture success count remains separate from production completed benchmark runs.

## Evidence and resources

- RED `preflight-red.txt`: import collection failed before the new module existed, exit 1.
- GREEN `preflight-green.txt`: 4 integration tests passed in 3.65 seconds, exit 0, with core `module.__file__` asserted. They cover current blockers and unchanged registry bytes, census binding/upper-bound integrity, forged approvals plus fixture refusal, and linked evidence without NPZ access.
- `preflight-local.txt`: actual main metadata-only diagnostic with evidence SHA-256 values; exit 0, blocked, no acoustic array access.
- Main-config Ruff format/check, mypy for the new module and `git diff --check` exited 0. GPU, network, installations and paid resources: none. Process peak RAM was not measured.
- Reviewer-change RED `preflight-review-red.txt`: 3 expected failures for strict-census D1 inference, missing target-only binding, and unrecognized protocol/counter handling. `preflight-malformed-red.txt` exposed a malformed census binding exception instead of a blocked diagnostic. GREEN `preflight-review-green.txt`: 7 tests passed in 0.17 seconds with core import path asserted. `preflight-review-mypy.txt`: mypy passed. The corrected fixture preflight was independently approved narrowly at commit `989f030`.

No real-corpus promotion or execution path is implemented by it.

## Versioned TRAIN census evidence update

The prospective singleton repair and full TRAIN rerun use `train_census_v2_execution.json` and `train_census_v2_eligibility.json`. This preflight now reads those paths exclusively; preserved v1 files cannot satisfy completion or enter its evidence hashes. A completed v2 strict report must carry `TRAIN_CENSUS_COMPLETE_STRICT_CONTEXT_CANDIDATE_ONLY`, `NOT_ESTABLISHED_TARGET_ONLY_BOUND_REQUIRED`, exact execution/method identity, 100 resolved TRAIN days, the 67 unmeasured non-TRAIN day ceiling, and `strict_context_policy_day_upper_bound = eligible_train_cutoff_days + 67`. The strict report cannot itself establish D1. The separately hash-bound target-only report remains the sole source of a D1 blocker. It must also bind the exact `target_only_bound_v2_input_contract.json` bytes and declare `census_generation = census-v2`. The preflight remains permanently blocked for real execution and opens no acoustic NPZ.

Focused RED `preflight-v2-red.txt`: six expected failures against the previous v1 reader, including v1 substitution and v2 target-bound linkage. `preflight-v2-input-red.txt`: expected failure when the old preflight accepted a forged input-contract hash and census generation. Final GREEN `preflight-v2-green.txt`: nine tests passed with the core module path printed. Main-config Ruff format/check, mypy, and `git diff --check` passed after a format correction. These are synthetic evidence fixtures, not a completed real census or approval. No GPU, network, installations, or paid resources were used; process peak RAM was not measured. Independent review is requested on this v2 path and strict-report delta.


## Coordinator integration update

Independent continuation reviewer accepted the read-only v2 evidence-binding delta, integrated as1917e78. Main CLI wiring preserves registry bytes and returned explicit exit2; integrated preflight/CLI tests16passed. This remains a permanently blocked diagnostic, not real finite campaign execution.
