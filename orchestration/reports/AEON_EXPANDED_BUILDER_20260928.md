# AEON expanded TRAIN 3k builder handoff

Task ID: `AEON-EXPANDED-TRAIN-VAL-3K-RUNNER-20260928`. Dependencies: ADR
0008/0009/0012, original 2024 TRAIN/validation split review, prior-year
metadata inventory and distinct outcome review, corrected original 3k
validation rescore and its outcome review. Branch `impl/aeon-scale`, worktree
`C:/Users/Álvaro Schwiedop/Desktop/KriptaStudios/marine-echo-aeon-scale`.
Writer `/root/scale_builder`, configured GPT-6 Sol high builder; an actual
runtime model identifier is not independently exposed in this session.

Owned files: `src/marine_echo/training/aeon_expanded.py`,
`tests/integration/test_aeon_expanded.py`, `configs/aeon_expanded_3k.json`,
`docs/adr/0014-aeon-expanded-data-development.md`, and this report. The
approved/running 30k runner, config, ADR and outputs were not edited. No
historical v1/v2 source, ledger or release was edited.

The new study reclassifies the already-inspected prior-year AEON3 archive as
TRAIN and joins it to only the original 2024 TRAIN dates. Its original 2024
validation is unchanged and remains the only assessment. The prior archive
numeric reader reconstructs the independently reviewed metadata candidate
inventory before issuing windows. Every TRAIN row ID binds archive hash,
publisher file ID, filename serial identifier, license, source role and
cutoff interval. Cohort validation rejects cross-deployment members,
context/target times or raw interval overlap with validation. No CAL/TEST
partition is loaded. The cohort report preserves per-source issued and
observed-target counts, prior-year candidate rejection reasons/count, and
validation row/raw-interval hashes. The original reader lacks a comparable
candidate inventory, so its rejected-candidate count is explicitly
`NOT_AVAILABLE_FROM_ORIGINAL_READER_NO_COMPARABLE_CANDIDATE_UNIVERSE`.

The fit runner has two separate gates. A distinct preaccess review binds
source, metadata, code, config and ADR before new numeric cohort
construction. A second distinct prefit review binds the measured cohort
digest, report, corrected 3k baseline and its independent outcome review
before training. The fit reopens and recomputes the exact cohort, normalizes
only from joint TRAIN, trains direct seed 7 for 3,000 updates and EMA-JEPA
seed 7 for 1,500 source-homogeneous SSL plus 1,500 supervised updates.
Expected SSL batch source frequencies follow natural TRAIN window
proportions; actual source sample counts are saved. The source-conditioned
validation metric excludes target-source dates with fewer than 18 scored
anchors. Every 500-step checkpoint is diagnostic; final endpoints alone are
compared with same-family original-only seed-7 corrected 3k values. A
development difference cannot be attributed causally to data volume alone,
because instrument identifier, calendar period and potentially processing
also differ. Checkpoints carry both source archive hashes, cohort/config/ADR
hashes and TRAIN-only scaler. The single-process RAM/GPU caps remain 22/10
GiB.

RED: `python -m pytest tests/integration/test_aeon_expanded.py -q` failed
collection because `aeon_expanded` did not exist. GREEN: that focused command
passed 5 tests in 26.99 seconds, exit 0, after final changes. The broader
`python -m pytest tests/integration/test_aeon_expanded.py
tests/integration/test_aeon_scale.py tests/integration/test_aeon_campaign.py
tests/unit/test_aeon_evaluation.py -q` passed 24 tests in 49.79 seconds,
exit 0, before a final defensive array shape/dtype check; the five focused
tests were rerun after that check. Both default Ruff and `ruff check
--select I` passed on the final expanded source/tests, exit 0. `compileall`
passed before the final defensive check; Python import/pytest passed after.

Contract digests: ADR SHA-256
`9ab1f0df35e102385f2ea7d5fbf3dae23d3dd02edf428e11ce637e502c0ce2fa`,
config SHA-256
`50b8e017502dd312028eaa759db6508409f345c71f6b93cf6c99d1ce0c318abd`,
combined expanded/model/reader/evaluator code digest
`34cbfafdaae0bf13f873a9d6f9d91ccf5698dfbdb4f15db92288985a335576f2`.
The final commit ID is supplied in the parent handoff after commit.

Real expanded cohort QC/count, GPU fit, memory peak, predictions and
validation metrics are **NOT_RUN** by this builder: neither new preaccess
nor prefit review exists yet. The prior separate external report's 8,507
issued windows and original 2024 TRAIN's 4,965 windows sum to 13,472
historical candidates (2.71x), not a new measured expanded cohort or 3x/5x
result. Actual counts must be recorded by the reviewed cohort phase.
No paid/cloud cost was incurred. Independent reviewer action is needed on
the exact committed files before numeric cohort construction, then on the
measured cohort before leader-owned single-GPU training; a separate outcome
review is required before scientific interpretation.
