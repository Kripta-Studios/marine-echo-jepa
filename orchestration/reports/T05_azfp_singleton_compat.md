# T05 pinned AZFP singleton compatibility

- Status: REVIEW / TRAIN_ONLY_COMPATIBILITY. The historical census failure and artifacts remain preserved; no failed file was skipped or overwritten.
- Dependencies: preserved TRAIN source inventory/XML, pinned Echopype 0.11.1, fixed calibration/QC method, independently reviewed new census version before a real eligibility claim.
- Session: `/root/continuation_builder`, native `sol_builder` configured as `gpt-6-sol`/`high`; provider-side attestation not exposed. Isolated worktree `../marine-echo-jepa-core`, branch `impl/core`.
- Owned paths: `src/marine_echo/data/azfp_compat.py`, `tests/integration/test_azfp_compat.py`, `evidence/continuation-builder/azfp-*.txt`, this report. Coordinator owns the prospective census driver and contracts.

The pinned Echopype 0.11.1 AZFP parser source SHA-256 is `d5ac606f3372b36e940f32df2255d23fdb8848e87d14f8794cface0050275b45`. Its `_check_uniqueness` skips a one-ping file because `profile_flag.size == 1`, leaving frequency as a Python list before `parse_raw` calls `.astype`. The compatibility subclass applies upstream's exact unique-row reduction to the same nine frequency-vector and nine scalar fields for one ping. Multi-ping files use the original method unchanged. The wrapper verifies version/source hash, holds a process-local lock, temporarily binds only the AZFP parser in the normal `ep.open_raw` registry, and restores the original binding in `finally`. It does not alter the installed package, raw bytes, ping count, timestamps, configuration, calibration coefficients or acoustic values. All AZFP calls in the consuming census process must use this wrapper; unrelated unwrapped calls do not share the lock.

The fixed TRAIN singleton `20030416.01A` is 13,244 bytes, SHA-256 `aa83fe505443235d70d846b898a331c799be9e2fedd7c9fa23f476415993b48a`. The test opens it as exactly one ping and four original channels. It also passes the original fixed environmental inputs through Echopype `compute_Sv`, the candidate `clean_per_ping`, and the same range-grid shaping without adding a ping; raw timestamps/counts/configuration remain unchanged. A multi-ping TRAIN file `20030300.01A` matches normal Echopype `ping_time`, frequency, counts and vendor configuration arrays exactly. A forced `ep.open_raw` error still restores the original parser binding.

## Executed evidence

- RED `azfp-red.txt`: import collection failed before the helper existed, exit 2.
- GREEN `azfp-green.txt`: 4 fixed TRAIN-source integration tests passed in 1.44 seconds, exit 0, with core `module.__file__` asserted. `azfp-smoke.txt` separately records the singleton calibration/clean/regrid smoke, 1 passed, exit 0.
- Main-config Ruff format/check, module mypy (`azfp-mypy.txt`) and `git diff --check` exited 0. No GPU training, protected data, download, environment modification or paid resource was used. Process-tree peak RAM was not measured.

Independent review is required before coordinator integration. Any resumed full TRAIN census needs a new prospective version and must retain the prior failed census record; this helper alone is not R0/R1 calibration or corpus approval.


## Coordinator integration update

Independent continuation reviewer accepted this narrow compatibility repair. Main integration40a22eb retains the helper unchanged; its test source-location lookup was adjusted to the existing sibling extraction and verified with22 integrated tests. V2 launch approval is recorded in orchestration/reviews/TRAIN_CENSUS_V2_LAUNCH_20260927.json. All100TRAINdays are being rerun, with exact comparisons against16 completed v1 days. No corpus or benchmark approval follows.
