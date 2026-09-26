# T06 continuation: bounded canonical adapter and executable B0-B3 fixture path

- Status: REVIEW / NONPROMOTABLE_ENGINEERING_FIXTURE. This is functioning software exercised on declared synthetic day shards, not a completed public-data benchmark.
- Dependencies: T02/R0 physical calibration, T05 full canonical schema, T06 exact approved catalog split and R1 still block real-corpus promotion. R2/freeze gates still block final test outcomes.
- Session: `/root/continuation_builder`, configured native `sol_builder` role `gpt-6-sol`/`high`; provider-side attestation not exposed. Worktree `../marine-echo-jepa-core`, branch `impl/core`.
- Owned paths: `src/marine_echo/training/data_adapter.py`, `src/marine_echo/evaluation/baseline_runner.py`, `tests/integration/test_canonical_baseline.py`, `evidence/continuation-builder/baseline-*.txt`, this report.

## Executed behavior

`CanonicalWindowAdapter` checks the synthetic-only store boundary and split digest, refuses calibration/test value materialization, rejects train/validation source overlap and cross-day grid/provenance drift, constructs hourly 96-bin `[4,64]` contexts with fixed +1/+3/+6-hour outcomes, and caps materialization at 1,024 windows and a three-day cache. The past context remains identical when a future synthetic value is changed; target truth changes as expected. The adapter requires named observed or simulated availability, and unknown observed latency yields no eligible rows. `TrainOnlyScaler` fits masked statistics only from train and binds split, source and processing provenance before validation transform.

`run_baselines` actually fits the existing persistence, daily seasonal, ridge and histogram-boosted quantile models on train and evaluates validation. Each family writes five quantiles, truth, row IDs, cutoffs, target support and a day-weighted primary score to checksum-bound artifacts. Runner, model, metric and adapter source digests, input arrays and canonical provenance are bound to the run JSON. A rerun verifies prediction checksum and exact provenance; tampering or partial artifacts fail closed. Artifacts are created without replacing an existing file. The status is `COMPLETED_SYNTHETIC_FIXTURE` and cannot satisfy the benchmark ledger.

This adapter is deliberately capped and fixture-only. The real executor needs an independently approved R0/R1 promotion bundle bound to exact calibration, source inventory, XML/config, full catalog split, processing manifest and complete canonical metadata fields. It also needs a streaming train/validation path for deployments beyond the finite in-memory cap. The finite neural/JEPA/hybrid/control campaign, calibration adjustment, pre-test R2 freeze and final evaluation remain NOT_RUN.

## RED/GREEN and resources

- `evidence/continuation-builder/baseline-red.txt` records observed initial collection/fixture/Windows descriptor failures; the initial stdout streams were not saved.
- `evidence/continuation-builder/baseline-green.txt` records the executed isolated-worktree pytest command with both imported module paths asserted: 5 integration tests passed in 16.22 seconds, exit 0. Tests cover fixed geometry, future mutation, protected partitions, bounded cache, train-only scaler, measured-latency refusal, cross-day grid and train/validation source refusal, four fitted model artifacts, verified resume and checksum tamper refusal.
- Ruff check/format, two-source mypy with explicit namespace path and `git diff --check` are required before commit; main integration must rerun against the actual main package.
- GPU and network: unused. Paid/committed infrastructure: USD 0. Peak cached canonical days: at most 3 by invariant. Process-tree peak RAM was not measured.

## Review request

Please verify the fixture-only boundary, no protected value access, row construction and target geometry, train-only fitting, artifact provenance and resume behavior. This report does not request R0/R1 or scientific run approval from synthetic fixtures.
