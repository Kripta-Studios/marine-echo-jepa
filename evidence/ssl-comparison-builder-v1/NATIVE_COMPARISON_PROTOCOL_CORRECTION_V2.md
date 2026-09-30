The comparison implementation now conforms to the frozen recipe in main's
`docs/adr/0015-native-acoustic-ssl-research.md`, lines 90–94. ADR0015 takes
precedence over the conflicting initial comparison task contract. This is a
protocol-conformance correction before any scientific comparison, not an
adjustment based on comparison outcomes. The coordinator states that the first
delivery was neither integrated nor run and no actual comparative or final-test
assessment occurred. This builder has accessed no real numerical predictions
or acoustic data in either delivery.

| Field | Closed V1 task/delivery (superseded) | Corrected delivery |
| --- | --- | --- |
| Bootstrap draws | 2000 | 2000 |
| Seed | 1729 | 20260929 |
| Source-calendar block days | 7 | 2 |
| Explicit block_hours field | Absent | 48 |
| Daily eligible-row floor | 18 | 18 |
| Primary hierarchy | Equal day → horizon → deployment | Unchanged |

Both `block_hours=48` and `block_days=2`, integer seed 20260929, integer draws
2000 and floor 18 are mandatory in every manifest and appear in the result.
Wrong or missing fields are rejected by the existing admission gate before
`np.load`; this includes the former seven-day/1729 recipe. There is no legacy
fallback, alternative seed or retained seven-day runtime default.

For each deployment, the earliest nonempty issued target source-calendar date
across all horizons anchors the calendar grid. Block IDs are the elapsed whole
source-calendar days divided by two, rounded down. Boundaries are start
inclusive and end exclusive. The result records each observed block's ID and
boundaries, plus 48 nominal hourly source-calendar intervals per block. Gaps
retain their elapsed calendar positions. In the synthetic fixture anchored on
2024-01-01, dates at offsets 0/1/7/8/14/15 yield block IDs 0/3/4/7. Empty grid
blocks 1/2/5/6 are not squeezed away and are not sampled; the previously defined
sampling population remains the observed calendar blocks.

All horizons and methods share the same deployment calendar assignment and
each bootstrap schedule. Repeated sampled blocks retain all repeated eligible
daily records. The existing scorer, daily floor, exact common-support checks,
native geometry, observed-target finiteness, forecast finiteness, constant
diagnostics, unsupported-replicate/null-interval policy and strict
review/identity/source-binding gates are preserved. Each replicate still
averages eligible dates equally within each horizon, horizons equally within
each sampled deployment, and sampled deployments equally. Original incomplete
support remains NOT_ASSESSABLE, with bootstrap work explicitly NOT_RUN.

The date artifacts are source-calendar proxies, not verified UTC timestamps.
Two day-boundary blocks represent 48 **nominal** hourly intervals and do not
establish a verified 48-hour absolute-time duration. The result explicitly
sets `verified_absolute_time=false`, describes source-clock uncertainty and
states that this is a nominal source-calendar implementation of the frozen
48-hour recipe. Within-day timing, clock drift, gaps and absolute-time/UTC
alignment cannot be recovered from date proxies alone. The limited deployment
and correlation assumptions remain disclosed. No native depth/frequency/query
geometry or dB values are rescaled or relabeled.

Corrected manifest recipe fragment, used alongside the unchanged required
role/method/reference/identity/evidence fields:

```json
{
  "bootstrap_seed": 20260929,
  "bootstrap_replicates": 2000,
  "block_hours": 48,
  "block_days": 2,
  "floor": 18
}
```

The public function and CLI remain:

```text
compare_saved_predictions(manifest_path, review_path, output_path)
python -m marine_echo.evaluation.native_comparison --manifest manifest.json --review distinct_review.json --output new_report.json
```

Root must bind the corrected integrated source bytes, the exact current main
native evaluator/helper sources, manifest and every prediction NPZ in a genuine
distinct reconstruction review. Development still requires
APPROVED_COMPARISON_RECONSTRUCTION and explicit development admission; final
assessment requires APPROVED_FINAL_ASSESSMENT and explicit final_test admission.
Source snapshots and exact hashes remain prerequisites before numeric parsing.
These fixture tests and this implementation report do not provide that review
or self-approval.

The original closed delivery is preserved. Its source and tests are archived
byte-for-byte in `native_comparison.py.original-v1.txt` and
`test_native_comparison.py.original-v1.txt`. The original owner contract is
recorded as historical text in `ORIGINAL_COMPARISON_TASK_CONTRACT_V1.txt`.
The original V1 report, every original red/green/lint log, original executed
checks, original provenance and original handoff verification remain unchanged.
`before-protocol-v2.json` captures the old evidence and source hashes before
implementation edits. Its self-output was precreated by shell redirection and
appears with the empty-file digest in that capture; it is a new V2 artifact,
not part of the closed V1 evidence, and is excluded from historical-preservation
checking. The newly created snapshot script is likewise not a V1 artifact.
All genuine closed V1 evidence entries are checked by the V2 handoff verifier.

Focused TDD red evidence `red-protocol-v2.txt` exited **1**, with **3 failed**
and 60 deselected. It demonstrated the wrong recipe, acceptance of a legacy
manifest up to the numeric-load spy, and incorrect calendar block IDs. The
first corrected full run `green-attempt-protocol-v2.txt` exited **0**, with
**68 passed in 25.62 seconds**. The formatted/refactored run
`green-refactor-protocol-v2.txt` exited **0**, with **68 passed in 26.94 seconds**.
The final full run on delivered source/tests `green-final-protocol-v2.txt`
exited **0**, with **68 passed in 27.90 seconds**. One unused test variable was
fixed after the initial lint attempt; its original failure log is retained.
Final Ruff check and format check both exited **0**.
The corrected handoff verifier exited **0** and confirmed **33 closed V1
evidence files** unchanged, plus both original source snapshots, the unchanged
ADR and current source hashes. Whole-handoff Ruff check/format exited **0**
(`lint-handoff-protocol-v2.txt`, `format-check-handoff-protocol-v2.txt`). Its
`handoff-verification-protocol-v2.json` is byte/preservation verification, not
an independent reconstruction review.

Tests use CPU NumPy and real NPZ codecs in memory under visibly
SYNTHETIC_CORRECTNESS_ONLY fixtures, with no scientific artifact access.
The independent bootstrap oracle directly concatenates daily records for
two-calendar-day block draws, retains duplicate dates, uses seed 20260929,
reconstructs hierarchical means and checks all 2000 draws' sequence SHA256 and
paired interval against the implementation's separate sum/count calculation.
Existing independent row-loss/daily metric, identical-method zero interval,
deterministic replay, unequal counts, common support, geometry, strict hash
gates, constants, zero span and unsupported-horizon tests all remain active.
New gate cases separately reject legacy seed, legacy days, wrong hours, missing
hours and noninteger hours before numeric loading. Gap and exact-anchor-boundary
assertions cover all three horizons and paired methods.

Actual verification commands used the existing environment and ordinary
`cmd.exe`, without permission changes or installs:

```text
..\marine-echo-jepa\.venv\Scripts\python.exe -B -m pytest -q -s -p no:cacheprovider tests\unit\test_native_comparison.py -k protocol_v2
..\marine-echo-jepa\.venv\Scripts\python.exe -B -m pytest -q -s -p no:cacheprovider tests\unit\test_native_comparison.py
..\marine-echo-jepa\.venv\Scripts\python.exe -B -m ruff check --no-cache src\marine_echo\evaluation\native_comparison.py tests\unit\test_native_comparison.py evidence\ssl-comparison-builder-v1\snapshot_before_protocol_v2.py
..\marine-echo-jepa\.venv\Scripts\python.exe -B -m ruff format --check --no-cache src\marine_echo\evaluation\native_comparison.py tests\unit\test_native_comparison.py evidence\ssl-comparison-builder-v1\snapshot_before_protocol_v2.py
..\marine-echo-jepa\.venv\Scripts\python.exe -B evidence\ssl-comparison-builder-v1\inspect_sources.py
```

Only corrected source/test bytes and new suffixed V2 evidence are the current
delivery; V1 evidence documents the superseded implementation. Delivered source
SHA256 is
`218987c6afedadef88223366d22fb61cb430c6082bc8a8518fec1d983d26cdfb`.
Delivered tests SHA256 is
`a822cb7d2ab8324ebdf60a4505b7fce3d9e7412d29d08d4227cf1b44db38940f`.
The unchanged main native evaluator SHA256 is
`fd1c643fde93494a21d560794f886056e58031f1ef6dd41ebe53662750bab314`.
The protocol ADR SHA256 read at correction start is
`82deb36e88df2a6660a2389a653d077a186e5072fbb4ede11d9fbc06c9816148`.
`final-source-provenance-protocol-v2.json` records actual imported main paths,
current source hashes, actual builder identity and `torch_imported=false`.

No main core source, dependencies, contracts, STATUS, ledger or past
implementation files outside the allowed comparison paths were edited.
No real prediction/acoustic access, fitting, inference, optimizer, GPU, cloud,
credentials, permission changes, junction operations, Git index retry or agents
occurred. No scientific assessment or scientific metric is claimed. Root owns
integration, durable disk checks and the distinct reconstruction review before
any actual development or final assessment. No commit or self-review is claimed.
