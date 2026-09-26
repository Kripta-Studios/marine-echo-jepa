# T06 fractional ping support correction

- Status: REVIEW; synthetic engineering fixtures only. No real R0/R1 promotion or completed benchmark is claimed.
- Parent dependencies: T02 verified physical calibration, T05 calibrated day production, T06 split and eligibility, independent R0/R1 review.
- Session: `/root/continuation_builder`, native `sol_builder` role configured as `gpt-6-sol` / `high`; provider-side model attestation was not exposed.
- Worktree/branch: `../marine-echo-jepa-core`, `impl/core`. Commit: recorded in Git history accompanying this report.
- Owned paths: `src/marine_echo/data/{preprocessing,canonical_store}.py`, `src/marine_echo/training/data_adapter.py`, corresponding scientific/integration tests, `evidence/continuation-builder/support-*.txt`, this report.

The 15-minute canonical cell now preserves nominal expected ping count from configuration, observed ping count, effective support denominator `max(nominal, observed)`, excess observed count, and valid count per channel/range. Clock jitter can yield 61 observed pings against a nominal 60; all 61 remain represented. Duplicate ping timestamps are rejected. The primary context band must reach 80% physical range-width and effective-ping weighted support in every quarter. Each future one-hour target uses the same full-band denominator over four bins. Its linear Sv mean weights each cell mean by its valid ping count and original range width before conversion to dB. The effective support examples are 59/59 observed/valid = 59/60, 61/61 = 1, and 61/48 = 48/61.

The day NPZ, metadata-only read, digest, dataset eligibility audit, cross-day adapter and protected test-metadata exposure record now include the support counts. Processing manifest schema is 2.0; older 1.0 stores fail closed on resume rather than silently changing QC semantics. The test metadata exposure record is durable before opening any protected NPZ and identifies ping support counts as inspected. It does not open protected acoustic arrays. The core remains `NONPROMOTABLE_ENGINEERING_FIXTURE`; its caller-supplied cadence is not proof of a real configuration. Real corpus promotion needs independent calibration/config provenance and R0/R1.

## Executed evidence

- `support-red.txt`: four expected focused failures before count-aware aggregation. `support-schema-red.txt`: negative valid count passed and failed the expected exception. `support-corrupt-red.txt`: direct window accepted a tampered effective denominator. Exit 1 in each saved RED run. An initial `support-schema-red.txt` invocation imported the installed main module and was immediately replaced with an isolated-core `module.__file__`-asserted run; the final saved file contains only the valid core failure.
- `support-green.txt`: 41 scientific/integration tests passed in 10.65 seconds, exit 0. The executed command asserted the core `preprocessing`, `canonical_store`, and `data_adapter` module paths. It includes 1/60, 47/60, 48/60, unequal counts, clock-jitter, corruption, store resume/schema, metadata exposure and baseline adapter cases.
- Main-config Ruff format/check and `git diff --check` exited 0. `support-mypy.txt`: three source files passed, exit 0. No GPU, real acoustic arrays, network, installation, or paid resources were used; process peak RAM was not measured.

Independent R1 delta review is requested. Parent integration must rerun tests and the two actual environment type checks. The scheduler's separate reviewer veto remains unresolved; this correction does not approve it. Real physical calibration, complete real canonical schema and eligible day counts remain unresolved upstream.
