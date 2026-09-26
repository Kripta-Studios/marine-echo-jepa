# T08 continuation: finite pre-test campaign scheduler

- Status: REVIEW / FIXTURE_EXECUTION_ONLY. No public acoustic benchmark run was scheduled or completed.
- Dependencies: T02/R0 physical calibration, full canonical corpus and frozen 60/15/10/15 split, independent R1, real train/validation executors and R2 before final test. This scheduler cannot authorize those gates.
- Session: `/root/continuation_builder`, configured native `sol_builder` role `gpt-6-sol`/`high`; provider-side model attestation not exposed. Worktree `../marine-echo-jepa-core`, branch `impl/core`.
- Owned paths: `src/marine_echo/training/campaign.py`, `tests/integration/test_campaign_scheduler.py`, `evidence/continuation-builder/campaign-*.txt`, this report.

## Actual behavior and limits

The state machine constructs exactly 25 declared slots: four B0-B3, six two-configuration seed-7 learned development runs, nine selected learned seed results, two hybrid heads and four JEPA controls. It executes only an explicitly synthetic fixture identity. Every result requires an injected executor, finite nonnegative validation metric, update count at most 3,000 and an artifact under its unique attempt directory. Missing neural executors remain `BLOCKED_EXECUTOR`; dependent slots remain `WAITING_DEPENDENCY`. No callback for an absent family is invented and no slot includes test outcomes.

Both development configurations must complete before a learned family's validation configuration is selected. Selected seed 7 may reuse the chosen development artifact only when that executor explicitly marks it reusable; the ledger records `REUSED_FIXTURE` and the exact source run ID rather than counting another independent execution. Seeds 13 and 23, hybrids and controls run only after their declared predecessors. The scheduler takes one exclusive trainer lock, publishes the ledger atomically, retains failed/interrupted attempts, allows at most two attempts per slot, verifies successful artifact hashes on resume and rejects changed configuration, code/plan or identity hashes. `fixture_completed_slots` may reach 25 in a synthetic harness; `completed_benchmark_runs` remains zero and `test_opened` remains false.

The existing `run_steps` function is an in-memory bounded trainer, not a complete executor. Real direct/JEPA adapters still need train-only day-shard batches and future/profile targets, optimizer/model/checkpoint creation, validation checkpoints and selection, three-seed retraining, hybrid train-only embeddings, random-encoder and shuffled-future control execution, budgets/resources telemetry and immutable prediction artifacts. B0-B3 have a fixture-only runner but no R0/R1-promoted real-corpus adapter. This scheduler has no CLI wiring; the existing `complete-p0` command must continue to report `NOT_IMPLEMENTED` for real campaign execution. The scheduler's local fixture ledger is separate from the production run registry.

After the independent review's `REQUEST_CHANGES`, resume validation now checks exact top-level and per-run schemas, fixed family/phase/seed/dependencies/configuration fields, attempt sequence/status/schema/metric/artifact hashes, final attempt versus run fields, explicit seed-7 reuse, and recomputed selected configurations. Successful fixture resumption cannot accept a changed winning configuration or a completed-to-pending edit. Executor metrics reject Booleans, and the reuse flag must be a Boolean. The lock contains PID plus a unique token; the scheduler checks ownership during slot execution and only unlinks its own regular, unlinked lock. A replacement is preserved and reported as ownership loss. These changes require a new independent review verdict; the earlier veto is not treated as approval.

## RED/GREEN and resources

- RED `evidence/continuation-builder/campaign-red.txt`: module collection failed before `campaign.py` existed, exit 2.
- GREEN `evidence/continuation-builder/campaign-green.txt`: 4 fixture integration tests passed in 3.48 seconds, exit 0, with imported core module path asserted. They cover exact slot count/order, validation selection, seed-7 reuse, verified resume, missing-executor behavior, preserved failure/retry attempts, one-trainer lock, artifact tamper, changed config and real-scope refusal.
- Ruff format/check, mypy for the new module, and `git diff --check` passed in the isolated core worktree. Main integration must rerun standard package checks and independent review.
- Reviewer-fix RED `campaign-review-red.txt`: 8 expected failures exposed ledger tampering, lock replacement deletion, and loose result types. `campaign-config-red.txt`: a selected-run configuration mutation was accepted. Reviewer-fix GREEN `campaign-review-green.txt`: 14 fixture integration tests passed in 14.78 seconds with core `module.__file__` asserted. `campaign-review-mypy.txt`: the module passed mypy. Main-config Ruff and diff checks exited 0.
- GPU/network: unused. Paid/committed infrastructure: USD 0. Process peak RAM was not measured for the small JSON fixture harness.

## Review request

Check slot accounting, dependencies, failure/resume histories, lock behavior and the hard fixture-only/test-closed boundary. Fixture completion is a software test, not a scientific campaign result.
