# Test strategy and proof obligations

A green software test suite is not evidence of predictive value. Keep engineering, data,
experiment and commercial statuses separate. Synthetic fixtures are essential for unit tests
but never substitute for real-data ingestion or held-out model evaluation.

| ID | Required test | Level / owner |
|---|---|---|
| D01 | Reject duplicate/decreasing timestamps, incompatible units and undocumented channel remap | unit / Sol |
| D02 | Convert/average dB through linear sv; hand-calculated two-value fixture | unit / Sol |
| D03 | Raw/calibration/config identity and orientation/range metadata survive round-trip | integration / Sol |
| D04 | Bad ZIP paths, symlinks, Windows drive names, duplicates and expansion limits rejected | security / Sol |
| D05 | Partial downloads and failed shards never publish as complete | integration / Sol |
| D06 | An actual AZFP/OOI file calibrates, plots and round-trips with finite supported values | real integration / Sol |
| L01 | Appending or perturbing future observations cannot change features at cutoff t | property / reviewer |
| L02 | No raw timestamps or windows cross train/validation/calibration/test membership | scientific / reviewer |
| L03 | Train scaler is unchanged after arbitrary test-label mutation | scientific / reviewer |
| L04 | Future GPS, centered filters, backward fill and interpolation across cutoff rejected | scientific / reviewer |
| L05 | SSL pretraining rejects validation/calibration/test file IDs | scientific / reviewer |
| L06 | Exact cutoff ping and bin-end boundary: no duplicated last context bin | scientific / reviewer |
| M01 | Shapes, quantile monotonicity, finitude and all-missing abstention | unit / Sol |
| M02 | EMA teacher receives no gradient; shared SIGReg target encoder does receive gradient | autograd / Sol |
| M03 | Target contains only future interval; context self-reconstruction cannot pass as forecasting | unit / reviewer |
| M04 | Tiny deterministic overfit reduces error; shuffled/null control does not prove value | diagnostic / Sol |
| M05 | Collapse probes distinguish constant, rank-one and diverse planted embeddings | scientific / Sol |
| M06 | Resume N updates equals K+resume+(N-K) under declared deterministic CPU fixture | integration / Sol |
| M07 | CUDA matmul, conv, attention, backward and checkpoint pass on the actual laptop | hardware / Sol |
| E01 | Pinball, weighted daily metrics and interval widening match hand calculations | unit / reviewer |
| E02 | Bootstrap resamples contiguous paired blocks and preserves model pairing | scientific / reviewer |
| E03 | Test evaluator rejects a changed config/checkpoint/split after freeze | scientific / reviewer |
| E04 | Missing/abstained predictions affect coverage and cannot disappear from denominators | scientific / reviewer |
| E05 | Every reported value recomputes from immutable prediction rows | integration / reviewer |
| E06 | Future reveal cannot alter the cached/input forecast request | API+E2E / reviewer |
| A01 | Strict schema, UTC, finite numbers, bounded requests and safe ID lookup | API / Sol |
| A02 | Artifact path traversal, untrusted checkpoint and secret leakage rejected | security / reviewer |
| W01 | All five screens, back navigation, real loading/error/empty states | E2E / Sol |
| W02 | Offline replay -> forecast -> reveal -> evidence -> export | E2E / reviewer |
| W03 | 1440px and390px layouts; keyboard focus; chart/table accessibility | E2E / Luna+Sol review |
| R01 | Clean Windows checkout bootstraps and serves the production frontend | release / reviewer |
| R02 | Release hashes, licenses, demo artifacts and CPU outputs verify after extraction | release / reviewer |

## TDD and quality commands

Add the minimal failing test before implementation. Record the red command/result and the
subsequent green result in the task report. A failing import for an unrelated missing package
is not meaningful RED evidence. Do not write tests that reproduce implementation formulas
without an independent expected result. Regression tests accompany bug fixes.

Required repository checks after implementation:
```
uv run ruff format --check .
uv run ruff check .
uv run mypy src/marine_echo
uv run pytest tests/unit tests/integration tests/scientific tests/api tests/security
npm --prefix web run format:check
npm --prefix web run lint
npm --prefix web run typecheck
npm --prefix web run test:run
npm --prefix web run build
npm --prefix web run test:e2e
```

Use strict typing for public contracts; no blanket ignore-errors or lint suppression to make
CI green. Target >=85% line coverage for contracts/split/metrics/serving and100% of enumerated
critical invariants, not inflated global coverage. Mark real-data/GPU tests explicitly; report
skips with reasons and required environment. GPU and external-network tests need not execute
in ordinary hosted CI, but their local logs are required before claiming a local trained MVP.
No xfail/skip may conceal a previously failing P0 requirement.

CI: Python3.12 on Windows and Linux for CPU tests; frontend on one supported platform; artifact
schema/link checks; secret scanning and dependency/license review with false positives handled
explicitly. Use a tiny redistributable derived real slice only after license review. Synthetic
fixtures remain named synthetic. No remote dataset download in every unit-test job.

## Independent review

Reviewer authors adversarial tests from this contract before seeing selected model scores;
reviews code, split and selection before final test; then recomputes the final report from
immutable predictions. Review must come from the bound independent model/session. A leader's
self-review cannot be labelled independent. Preserve disagreements and fixes in review reports.
