# T06 continuation: calibrated-array aggregation and window core

- Status: REVIEW (synthetic contract implementation only; G0/R0 remain blocked).
- Parent dependencies: T05 canonical calibrated shards incomplete; T02 physical calibration unresolved; no claim-bearing run is eligible.
- Session: `/root/continuation_builder`, native `sol_builder` role configured as `gpt-6-sol` at `high` in `.codex/agents/sol_builder.toml`. The client did not expose separate provider-side model attestation.
- Branch/worktree: `impl/core`, `../marine-echo-jepa-core`; source/test/evidence commit `1905adf`.
- Owned paths: `src/marine_echo/data/preprocessing.py`, `tests/scientific/test_preprocessing.py`, `evidence/continuation-builder/`, this report.

## Implemented contract

`aggregate_calibrated_pings` accepts only caller-declared verified physical linear sv with a calibration report SHA-256 identity, exact frequency values, range edges, frequency-specific support, configuration and source identities, and an explicit availability policy. It averages sv in linear units over fixed trailing 15-minute bins, logs after aggregation, keeps missing bins masked, invalidates a bin containing two configurations, and limits a batch to 32 hours. It does not calibrate counts or prove the supplied calibration report is scientifically approved.

`build_window` enforces a declared partition/source/deployment/instrument identity, full context and target containment, exact frequency and supported physical range, a single configuration across the full extent, past-only availability at the issue time, and weighted target support. The target periods follow the protocol's hour-ending rule. It rejects incomplete or unsupported windows. Its default context is 96 bins and default horizons are 1, 3 and 6 hours.

This core expects already aligned physical range bins. The future day-shard adapter must preserve original channel geometry and reviewed regridding provenance before calling it. It must also prove split-manifest disjointness and dataset-wide eligible day counts; those are not established by synthetic unit tests.

## RED/GREEN and resources

- RED: `..\marine-echo-jepa\.venv\Scripts\python.exe -m pytest tests/scientific/test_preprocessing.py -q` exited 1 at collection with `ModuleNotFoundError` before the module was created.
- GREEN: focused isolated-worktree command and output are in `evidence/continuation-builder/green.txt`; 6 tests passed in 0.67 seconds, exit 0. The existing main Python 3.12 virtual environment was reused; the core package path was inserted explicitly because this older core worktree lacks the package initializer present on main.
- Ruff check and format check passed; mypy passed for the module; `git diff --check` passed. No GPU, training, network, installation, paid infrastructure or held-out acoustic arrays were used. Infrastructure spend: USD 0. Process-tree peak RAM was not measured for this synthetic test.

## Review request and next action

Request independent R1 review after integration with the calibrated day-shard adapter and frozen split/availability manifest. R0 data/physical calibration review remains prerequisite to claim-bearing benchmark execution. The next unblocked implementation is a resumable day-shard processor and artifact emitter that requires a verified calibration manifest and fails closed otherwise; it should include split-day and raw-timestamp disjointness tests and use this core only after upstream physical calibration and range mapping are reviewed.
