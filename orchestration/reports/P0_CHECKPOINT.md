# P0 execution checkpoint — incomplete

Native model selections matched requested bindings; actual tool sessions responded. A provider-side
independent model attestation is not exposed. All three native sessions then returned service
usage-limit errors, with availability reported for September 30. No substitute agent was used.
This prevents the required independent review and Sol review of Luna documentation.

Implemented and executed: immutable local five-file registration (including missing manual),
SHA256/CRC checks, safe ZIP extraction and idempotence, preservation of XML/DPL, real AZFP
parsing, first-day raw replay, read-only hash-verified API, five-screen English app, offline
browser tests, bounded model software and GPU tests, checkpoint resume and gradient controls.
Core code was preserved in commit 2af31dd; later integration fixes are recorded in git history.

The source ZIP is 4,676,515,350 bytes with 3744 members. Diagnostic replay is 5564 pings,
96 quarter-hour bins,4 channels and 64 sample-index groups from 24 preselected stableBchunks.
It excludes other chunks explicitly and is not the complete quality-controlled pipeline.

Test evidence:54 app/data/baseline tests passed with no skips (3warnings), selected-module
coverage 80%;17 model tests passed in the core CUDA environment including the 100-update profile.
The integrated rerun passed 16 tests with 1 explicit skip of the already completed 100-update profile. Frontend 2 unit tests and 4 fresh-server browser tests
passed. Ruff, two-environment mypy, frontend format/lint/typecheck/build passed. Earlier failed
checks remain under evidence/tests. Type checks use the original CUDA interpreter for torch
stubs and the app interpreter for FastAPI stubs; neither is silently replaced with CPU PyTorch.

Actual profile:16.8735 seconds for 100 direct updates at B16,width 96,layers 3, peak reserved 100 MiB,
peak allocated 81.86 MiB, sampled process-treeRSS1.884 GiB. All fixtures are synthetic software
checks and supply no forecasting score. There is no paid cloud resource; USD 0spent/committed.

NOT_RUN: all 25 required benchmark entries, selected seeds and controls on real data,
uncertainty/freshness-error analyses, paired inference, scientific freeze and sealed final test.
No negative or positive JEPA hypothesis result exists. Commercial validation isNOT_EVALUATED.

Open implementation requirements: full calibrated chunk merge/QC/eligibility/windows; immutable
source identities enforced end-to-end; actual benchmark/train/freeze/evaluate orchestration;
calibration estimators and shared sealed evaluation support; complete R0/R1/R2/R3approvals.
CLI blocked branches are explicit unavailable operations, not completed implementations.

Calibration research found real environmental profiles; see ADR0003 and its actual download,
metadata and GPS applicability audit. Correct units/geometry, matching, sensitivity and review
remain unresolved. Do not call this an absolute data-access failure. Do not infer that supplying
salinity alone would complete P0. OOI code-path feasibility is separate; no dataset switch made.

ReviewerR1requested changes; some helper/API findings fixed and tested, but real pipeline
identity/evaluation requirements remain. R3interim findings were acted on; final approval is
unavailable due to service quota. Coordinator verification is not independent review.

Portable packaging and relocated checks are recorded in release validation sidecars. The
engineering diagnostic is not the requested completed forecasting MVP. Continue from STATUS.md.
