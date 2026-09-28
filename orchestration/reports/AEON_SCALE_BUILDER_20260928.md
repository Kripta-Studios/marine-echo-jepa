# AEON scale 30k builder handoff

Task ID: `AEON-SCALE-30K-TRAIN-VAL-RUNNER-20260928`.
Parent dependencies: ADR 0008, ADR 0009, corrected 3k validation rescore
and its independent outcome review. Branch: `impl/aeon-scale`. Worktree:
`C:/Users/Álvaro Schwiedop/Desktop/KriptaStudios/marine-echo-aeon-scale`.
Writer: `/root/scale_builder`, Sol builder role. Configured model binding is
GPT-6 Sol high per repository instructions; the client does not expose a
separate machine-verifiable runtime model identifier in this task report.

Owned files: `src/marine_echo/training/aeon_scale.py`,
`tests/integration/test_aeon_scale.py`, `configs/aeon_scale_30k.json`,
`docs/adr/0013-aeon-scaling-development.md`, and this report. No shared
lock, status, ledger, historical campaign code, output or frozen v1/v2 file
was modified.

The new runner is a post-hoc TRAIN/validation development study. Stage 1 is
direct seed 7 for 30,000 supervised updates and EMA-JEPA seed 7 for 15,000
TRAIN-only pretraining plus 15,000 supervised updates. Stage 2 has seeds 13
and 23 for both families only if either corrected final endpoint improves
at least 1% over its independently reviewed corrected 3k seed-7 validation
score. Intermediate checkpoints are retained as diagnostics every 2,500
supervised updates; the final endpoint alone is compared. Corrected
`daily_pinball` excludes target-source dates with fewer than 18 scored
anchors and persists exact eligible-date hashes. It rejects support changes
across checkpoints or versus the reviewed 3k baseline. The original raw
all-scored-day metric is explicitly marked non-protocol. The prefit approval
must bind source, split review, cohort, code, config, ADR, original 3k
manifest, corrected rescore and rescore outcome review digests. CAL and TEST
are not reader partitions. Each completed slot is hash-verified on resume.

The prior-year same-site archive is a candidate for a separate expanded-data
study, not an input to this runner. The previously observed 8,507 prior-year
issued rows plus 4,965 original TRAIN rows total 13,472 candidates (2.71x)
before new TRAIN QC. No 5x support has been established. Reusing prior-year
external-evaluation data for training forfeits its external-evaluation status
for the derived model. A separate reader, cohort and review are required.

RED: `python -m pytest tests/integration/test_aeon_scale.py -q` failed
collection with `ImportError: cannot import name 'aeon_scale'` before the
module existed. GREEN: the same command passed 5 tests in 25.03 seconds,
exit 0, including a test that excludes a one-anchor date from the corrected
metric. `python -m pytest tests/integration/test_aeon_campaign.py
tests/unit/test_aeon_evaluation.py -q` passed 14 tests in 49.50 seconds,
exit 0. `python -m ruff check src/marine_echo/training/aeon_scale.py
tests/integration/test_aeon_scale.py` passed, exit 0. `git diff --check`
passed, exit 0. The config/code digest command passed after adding `src`
to the ad hoc Python import path; the first invocation lacked that path and
failed `ModuleNotFoundError`, which was an invocation issue, not a product
test failure.

Contract digests at this handoff: ADR SHA-256
`f86f9f4b27eb13d2221a225cb9647a0d99f9bc43d45693db4130828f61e5096b`,
config SHA-256
`81baa528607570ef3d0b251fdfb2d8d591c8ae7580ba9b727f1debebfcaa837e`,
combined scale/campaign/evaluation/rescore code digest
`7ab20744d39d66543eb559db0eb595ee47c39ba7a6ee1d84492896a3cc0636d4`.
The commit ID is supplied in the parent handoff message after commit.

Real TRAIN/validation cohort loading, GPU training, runtime/VRAM/RSS
measurement, predictions and metric outcomes: **NOT_RUN by this builder**.
The coordinator owns the single GPU process. An independent reviewer must
approve the exact committed code/config/ADR/cohort and old corrected baseline
binding before Stage 1. The builder cannot approve its own changes. Any
further edit to `aeon_scale.py` invalidates this code digest and requires a
new prefit review. Stage 2 and 50k are not authorized by tests or this
report alone.
