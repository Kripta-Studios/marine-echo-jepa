# T06 continuation: canonical day shards and eligibility audit

- Status: REVIEW. The code is synthetic-tested; no calibrated real corpus or independent R0/R1 approval exists.
- Parent dependencies: T02 physical calibration and R0; T05 canonical upstream parser/regridding; T06 frozen split; R1 before claim-bearing benchmark execution.
- Session: `/root/continuation_builder`, native `sol_builder` role configured as `gpt-6-sol`/`high` in `.codex/agents/sol_builder.toml`. No separate provider-side model attestation was exposed.
- Worktree and branch: `../marine-echo-jepa-core`, `impl/core`. Source/test/evidence commit: `1fdce6c` (on top of prior core commit `1905adf`).
- Owned files: `src/marine_echo/data/{preprocessing,canonical_store}.py`, `tests/scientific/test_{preprocessing,canonical_store}.py`, `evidence/continuation-builder/store-*.txt`, this report.

## Behavior and reviewer disposition

The store requires a physical-calibration manifest file whose bytes match the declared SHA-256 and whose dataset, deployment, instrument, source inventory and config identities agree with the store identity. This verifies provenance binding; it does not grant scientific R0 approval. Day NPZ shards contain only non-object arrays, are durably staged before publication, and are registered by an atomic manifest update. Resume verifies both shard checksum and exact input digest. A changed identity, input or calibration manifest, a corrupted registered shard, and an unregistered orphan shard fail closed without overwriting it. A crash between shard publication and manifest publication leaves an explicit orphan requiring manual reconciliation.

The metadata audit checks chronological split boundaries, every shard's day/split containment, unique raw-ping and bin timestamps, disjoint source-file identities across partitions, consistent frequency/range grids, complete past context bins, configuration continuity, issue-time availability, and 80% weighted target support over the original full physical analysis band. It reports before/after QC day counts and holds the original minimums at 90 overall, 12 calibration and 20 test days. It opens timestamp, mask and provenance arrays for test eligibility but never loads test `sv_linear` or `sv_db`; ordinary `load_day(test)` and `build_window(partition='test')` reject access. The audit result is only `ELIGIBLE_FOR_REVIEW` or `INELIGIBLE`, never independent approval.

The initial core R1 review's material findings were handled in owned files: the claim-path window geometry is fixed at 96 context bins, horizons (1,3,6), and 0.8 target support; no test truth is materialized; exact analysis-band edges prevent a silent shorter denominator; SHA-256 source/config/calibration/split/processing identities propagate into deterministic row IDs; and every primary context bin must have available support. This conservative context rule may reject otherwise usable periods until a prospective reviewed gap rule exists.

The upstream adapter must still perform actual Echopype calibration, retain original channel-specific geometry, and regrid with physical justification before these functions are called. The synthetic test calibration report is explicitly marked fixture-only. No real eligible day count is inferred from it.

## RED/GREEN and resources

- RED: `evidence/continuation-builder/store-red.txt` records the executed 4 failed / 8 passed run exposing tuple-versus-JSON-list manifest identity comparison, exit 1.
- GREEN: `evidence/continuation-builder/store-green.txt` records the exact isolated-worktree command with `module.__file__` assertions; 12 tests passed in 3.36 seconds, exit 0. Tests cover restart, checksum tampering, incompatible input and provenance, orphan protection, no pickle, corrupted series, protected test-value access, source overlap, minimum day counts, fractional support and exact cutoff behavior.
- Ruff check, main-config format check, two-module mypy with explicit namespace path, and `git diff --check` all exited 0. Main integration must rerun normal pytest and mypy after cherry-pick because the older core worktree lacks the package initializer already present on main.
- GPU: not used. Network/installation: none. Infrastructure paid/committed: USD 0. Process-tree peak RAM was not measured for these small synthetic fixtures.

## Finite scheduler assessment and next action

A non-placeholder `experiment complete-p0` can be built as a bounded run-state machine over the frozen family/configuration/seed/control matrix, with R0/R1 and canonical-store eligibility gates, one-trainer lock, immutable run IDs and input/code hashes, exact checkpoint-compatible resume, failure/attempt history and a hard stop before final test. It must invoke real baseline and neural executors, never just write a blocked registry. The current CLI still blocks baseline/training commands; `run_steps` takes in-memory `TrainBatch` objects and has no day-shard batch adapter, validation selection, artifact writer or full scheduler-state checkpoint. Those concrete executor adapters are the prerequisite implementation, followed by the finite scheduler and synthetic failure/resume tests. No actual benchmark is eligible under the unresolved real G0/R0 calibration and environmental-matching blocker.

Next integration command from main: `git cherry-pick 1fdce6c`. Then run normal main-worktree scientific tests, Ruff and mypy against the integrated package. Independent R1 review should recheck this delta after the actual calibrated adapter and frozen split manifest exist; R0 remains blocked by the separate physical data findings.
