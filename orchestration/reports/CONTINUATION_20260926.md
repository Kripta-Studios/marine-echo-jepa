Current outcome: [27 September report](CONTINUATION_OUTCOME_20260927.md).

# Continuation checkpoint — P0 incomplete, data ineligibility established

## Current execution checkpoint (27 September)

Main contains the preserved implementation plus reviewed calibration/QC, singleton
parser compatibility, exact census resume/equivalence checks, truthful registry
integration, and a reviewed full TRAIN count-map method. Latest source checks:
229 app/data/scientific/security/API tests passed; formatting, lint and types pass.
The unchanged model suite has 11 passes and two opt-in GPU skips, not real runs.

Full 100-day TRAIN census completed with exact equivalence for all 16 prior successful days. Original 38 kHz 10-100 m target has 0 supported TRAIN anchor dates; granting all 67 non-TRAIN days eligibility yields an overall upper bound of 67 below 90. The distinct reviewer accepted the full support map and both necessary bounds by exact independent recomputation. Any fixed 38 kHz band has a generous overall upper bound of 73 days, below 90. No held-out acoustic processing or benchmarks.

Actual command exits were 0 for census, original-target bound and support map.
Census worker time totaled 4648.436 seconds; sampled worker-tree RAM peaked at
0.319389 GiB. Full pipeline wall time was 4667.160 seconds. No GPU trainer ran.
The approved plot and any-band bound both executed successfully. The any-band result is independently accepted; no replacement target was selected.

D2 bounded discovery is complete: official geometry and embedded raw acquisition
coefficients were found, but external/environmental/FullNC lineage and an approved
bounded acquisition plan remain unresolved. Do not resume broad D2 discovery.
The historical r2 archive/extraction and all original sources remain preserved.

The recovered workspace is `main` at `036af24` before continuation edits. The existing
core/UI worktrees are retained. No reset, remote overwrite, raw-data deletion or public
publication occurred. The owner's continuation file was already untracked on entry.

## Reconciliation and active work

- Historical r2 ZIP SHA-256 rechecked: `df02ff06cbbb05d967f8c157423c0c76b2acf66ddb16f1a1fcfbe36a710a5cc3`.
  It and its extracted release remain untouched. Prior test measurements remain historical.
- Both registries contain 25 BLOCKED entries and no completed benchmark. The physical
  preprocess/train/evaluate commands are still unavailable, not completed implementations.
- Both native roles are now available: `/root/continuation_builder` (configured GPT-6 Sol,
  high) and `/root/continuation_review` (GPT-5.6 Sol, high). File/tool role bindings agree;
  no independent provider attestation is claimed. The old quota blocker is historical.
- Builder owns only assigned preprocessing paths in the clean `impl/core` worktree.
  Coordinator owns calibration/audit/integration paths in main. Reviewer is read-only.
  No trainer owns the GPU and no marine-echo job was running at inspection. Unrelated
  Python jobs were identified and left alone. Paid/committed infrastructure: USD 0.
- Runtime and handoff checks ran. Original hash validation fails at the intentionally
  changed `.gitignore`; 50/51 handoff helper tests pass, with the remaining assertion
  requiring all original task statuses to be TODO. These failures are preserved rather
  than erased by resetting the implementation. See `evidence/continuation/`.

## Earlier numerical assay: applicability assessment superseded by later reviews

GSW 3.6.23 was added to the existing data environment without dependency upgrades or any
PyTorch change, and pinned in the optional data dependency/lock. New tests cover published
TEOS pressure values, invalid environmental inputs, future availability, matching, and
signed Iridium positions. Red and green logs are retained.

The fixed training hour `20030300.01A` was parsed with Echopype 0.11.1. The numerical assay
checks XML frequency/coefficient mapping, scalar range/Sv recomputation and three measured
profile-depth scenarios. The scenarios assume a homogeneous column only to measure numerical
sensitivity. They are NOT an approved calibration. Full physical applicability remains
blocked by geometry, instrument provenance, environmental matching/availability and QC rules.
No daily averaging result becomes an issue-time predictor input.

Iridium metadata addresses the early XEOS gap: the 17 February profile has a same-time
Iridium position about 2.075 km away. This improves location evidence but does not prove
water-column equivalence. The fixed 5 km/24 h screen remains illustrative, not eligibility.

## Exposure and review gates

The audited local and packaged replays contain 17 February 2020 only, inside training.
No known test acoustic outcome exposure was found. This does NOT establish a sealed test:
human browser history and discarded agent contexts are unavailable. The candidate test
remains 7 July–2 August, with no R2 approval. The explicit exposure record is
`evidence/continuation/state_audit.json`; prior `test_opened=false` is not proof of a seal.

Critical path: independent calibration/matching decision -> approved physical preprocessing
and eligibility counts -> R0/R1 -> finite campaign -> R2/freeze -> eligible final evaluation
-> R3/new release. All data-dependent runs remain blocked; missing runs are not negative results.

## Corrections and additional executed evidence

The initial missing-certificate finding is resolved. The distinct reviewer located the ASL
26 March 2019 serial55170 certificate behind swapped O2A attachments and checked its printed
coefficients against XML. Exact bytes, URL and hashes are in `factory_certificate.json`.
Deployment metadata and raw TRAIN metadata establish serial55170; the manual's55169 is stale.
The source manual remains local and is excluded from distribution because it contains
operational credential-like content. Certificate redistribution rights are not established.

The initial Iridium-only late-period screen was incomplete. Both preserved trackers now give
99/100 train,25/25 validation,16/16 calibration and12/24 available test profile-day matches
within the illustrative 5 km/24 h screen. Two test-calendar days lack composite profiles.
Four recovered SIT sources (488,127,156 bytes total) were downloaded from the same documented
MOSAiC ancillary series. They have no co-located matches under the screen. Their QC-qualified
records can inform an explicitly regional physical envelope, not pretend to be local water.
See `sit_inventory.json`, `sit_applicability.json` and `trajectory_matches.json`.

A prospective fixed regional sensitivity assay retained all accepted records, full min/max,
fixed allowances and unchanged 1 dB/2 m gates. It failed: conditional-path maximum at38kHz
1.080721198 dB, displacement1.688427 m. Homogeneous0.905364 dB does not rescue it. The upstream
real TRAIN corner calculations agreed with scalar recomputation within6.21e-14 dB.

A separately preregistered depth-resolved follow-up uses fixed2m depth bins, own-coordinate
seawater conversions, full per-bin envelopes and fallback for missing depths. Its121 sampled
starting depths0–30m passed the primary numerical limits: worst38kHz0.230835841 dB, range
0.825417771 m. Other-frequency maxima are0.6270/0.6019/1.2732 dB. Runtime136.14s. This is still
regional diagnostic evidence: exact depth/tilt, local conditions, continuous bounds and actual
QC support remain unverified. The first attempt's upstream vector incompatibility traceback
is retained; the vectorized pinned equation was tested and independently reviewed before rerun.

## Earlier software checkpoint and review boundaries

Integrated code includes strict environmental availability and unit conversion, conservative
linear-range integration, trailing aggregation and exact windows, immutable day shards,
pre-access exposure logging, chronological source checks, train-only scaling and an actual
B0–B3 baseline fitter/validation artifact writer. The canonical adapter and executor accept
only synthetic engineering fixtures; they cannot authorize a real corpus or fabricate runs.
Independent findings about availability, source identities, grids, exposure durability and
resume validation were fixed before narrow engineering approval. Integrated baseline/report
checks:18 passed in49.33s. Finite scheduler implementation is still in progress.

`experiment complete-p0` was actually invoked through the PowerShell wrapper and returned2
with an explicit prerequisite/implementation block. It preserved existing registry bytes,
including attempted runs and test exposure; the reviewer independently checked that behavior.
`report` now preserves FAILED/RUNNING/attempt evidence rather than reconstructing an empty
registry or mislabelling attempts NOT_RUN. No benchmark run has executed.

Existing full continuation check logs are in `checks-20260926T202901Z`:83 app/data passes,
11 model-software passes/2 explicit GPU skips, frontend2 passes plus formatting/lint/types/build,
and historical release integrity. Later focused checks cover the subsequent changes. The first
format failure is retained and corrected; broader final integration follows the current edits.
Fresh relocated browser:5 passes, navigation p95190.143ms; no new cold-start timing is claimed.

## Reproduction and gates

From the repository, the existing wrapper can run the audit, checks or blocked campaign:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\Continue-Local.ps1 -Stage Audit
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\Continue-Local.ps1 -Stage Checks
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\Continue-Local.ps1 -Stage Campaign
```

Audit produces evidence and never approves calibration. Campaign exits2 while physical
prerequisites and real executors remain missing; it does not open the final test. The depth
assay is separately reproducible using the existing data environment:

```powershell
$env:PYTHONPATH='src'
$env:PYTHONUTF8='1'
.venv/Scripts/python.exe tools/depth_environment_sensitivity.py
```

No change to the25-slot budget, seeds7/13/23, fixed split or eligibility thresholds occurred.
G0 BLOCKED; G1 historical diagnostic tested and new components narrowly reviewed; G2 zero
benchmark runs; G3 NOT_EVALUATED; commercial validation NOT_EVALUATED. Physically approved
days:0. Actual post-QC eligibility counts:NOT_ESTABLISHED, not an assertion that data contain
zero usable days. R0/R1 remain incomplete, R2 ineligible, R3 not approved. Infrastructure USD0.
No new release ZIP has been produced or relabelled as full P0.


## Earlier 27 September TRAIN QC checkpoint (historical; superseded above)

The factory certificate/XML mapping, depth-stratified environmental bound, and slant
sensitivity method have separate independent acceptance for TRAIN candidate processing.
March3 plus the four prospectively fixed monthly TRAIN dates were processed and visually
reviewed. All five have0/96 primary quarter-hours meeting80% support under the unchanged
noise/QC method. Sparse low-SNR cells and genuine acquisition gaps remain masked.

The fractional-support implementation was corrected and independently reviewed. Synthetic
scheduler integrity fixes through main58235d6 were independently accepted; earlier failed
reviews and the caught indentation regression remain in evidence. None is a real benchmark.

At this earlier checkpoint, the independently approved100-day TRAIN census was running with daily immutable aggregates,
source/configuration hashes, expected/observed/effective/valid counts, exact raw timestamps,
one process and a22GiB process-tree limit. Its code and contract are frozen. No held-out
acoustic payload is needed. Historical v1 execution: evidence/continuation/train_census_execution.json.
The completed v2 ledger and accepted result reviews now supersede this execution state.

The original protocol explicitly freezes80% target support, not the implementation's
stricter every-context-bin80% rule. The separate prospective target_only_bound_contract.json
therefore requires a generous target-only upper bound before a definitive global eligibility
conclusion. That calculation ignores all context/configuration restrictions. The strict
census classification alone must not be used to declare the90-day gate impossible.

The census is not a promoted canonical corpus, R0/R1 approval, a model run or a final test.
All25 required real campaign slots remain blocked; JEPA value remains unevaluated.
