"""Update only owner-assigned entrypoints and prepend STATUS without rewriting history."""
from pathlib import Path

R = Path(__file__).resolve().parents[2]
E = Path(__file__).resolve().parent
readme = '''# Marine Echo JEPA

**Approved AEON r3 offline research software, completed mixed public-data studies and a completed independently reviewed checkpoint diagnostic. Marine operational value is unvalidated.**

Start with the [meeting brief](meeting/20260929/MEETING_BRIEF.md), [tested runbook](meeting/20260929/DEMO_RUNBOOK.md), [pilot data request](meeting/20260929/PILOT_DATA_REQUEST.md), [research decision](meeting/20260929/NEXT_RESEARCH_DECISION.md) and [actual readiness checklist](meeting/20260929/READINESS_CHECKLIST.json). The [owner meeting task](CODEX_MEETING_READY_AFTER_DIAGNOSTIC.md) is the current lane; **NO_FURTHER_FITTING_BEFORE_MEETING**. Fresh distinct review of this documentation/evidence increment is pending. Fresh browser smoke and speaking rehearsal are NOT_RUN; launcher/API/static traversal passed. See the checklist for gaps rather than treating this handoff as self-approved.

## Approved runnable artifact

Use [release/aeon-offline-scale-expanded-20260928-r3.zip](release/aeon-offline-scale-expanded-20260928-r3.zip), SHA-256 `5e265e08bf1041bbe0fb4dd2a5f3503bae98f996096577f976beaf9384d6bb57`, source `f975d2b0a425a9cbdadbacf9f50d0cad216d93c1`. The [exact-byte external review](orchestration/reviews/AEON_SCALE_EXPANDED_OFFLINE_R3_REVIEW_20260929.json) approves offline research use; its immutable internal review-pending marker records earlier build-time state. [Release validation](orchestration/reports/AEON_SCALE_EXPANDED_OFFLINE_R3_VALIDATION_20260929.md) records original offline and browser execution.

The new meeting scratch is `outputs/meeting-readiness-20260929-r3/aeon-offline-scale-expanded-20260928-r3/`. Its launcher was tested on `http://127.0.0.1:8784/`, then the owned server was stopped while existing r3/r2 servers were preserved. Windows PowerShell, uv and already available offline Python 3.12 are prerequisites. `Run-AEON-Research.ps1 -Port 8784` verifies 107 hashed assets and installs bundled wheels offline. Node is not needed. Consult the runbook for exact commands, paths, port checks and process ownership.

AEON replay shows saved forecasts with five quantiles at +1/+3/+6 **source intervals**. It is truth-free historical replay, not checkpoint inference or a later-observation reveal. A [separate static fallback](evidence/meeting-readiness-20260929/static-fallback.html) shows predetermined served examples and released aggregate evidence. Public AEON publisher-conditioned 38 kHz FullDepth `Sv_mean` is a source product, not a Marine depth-resolved, biomass or species target. Source clock timezone, availability latency and processing/calibration provenance remain unresolved.

## Results in their original scopes

| Evidence | Direct | EMA-JEPA | Interpretation |
| --- | ---: | ---: | --- |
| Original retrospective daily pinball, dB | 0.563387 | 0.533954 | Positive frozen within-study comparison; one deployment, non-sealed, not external replication. |
| Original frozen-model prior-year same-site transfer, dB | 0.670178 | 0.678922 | Negative descriptive secondary comparison; cross-site primary metadata-ineligible. |
| Exposed-VAL diagnostic ensemble, direct 1500 supervised versus EMA 1500 SSL + 1500 supervised | 0.639577 | 0.642170 | Direct1500 better per seed and ensemble; unequal total update count. |
| Exposed-VAL diagnostic ensemble, direct 3000 supervised versus EMA | 0.639062 | 0.642170 | Direct3000 ensemble slightly better than direct1500 while each direct seed worsens. |

Ensembles score **means of predictions**, not means of scores. The [completed diagnostic](orchestration/reports/AEON_CHECKPOINT_OBJECTIVE_DIAGNOSTIC_20260929.md) and [distinct review](orchestration/reviews/AEON_CHECKPOINT_OBJECTIVE_DIAGNOSTIC_REVIEW_20260929.json) concern repeatedly exposed development data, not a prospective winner or new significance. Established gradient restrictions and centered scale mismatch do not isolate causal attribution. The diagnostic is later evidence outside immutable r3.

The [historical retrospective report](orchestration/reports/AEON_RETROSPECTIVE_OUTCOME_20260927.md) and [negative transfer report](orchestration/reports/AEON_EXTERNAL_SECONDARY_NUMERIC_OUTCOME_20260928.md) retain their conclusions. Prior-year data subsequently became TRAIN for expanded models and all descendants affected by their fitted weights or preprocessing, and cannot be their external test. The negative 30k and mixed expanded-TRAIN development findings remain visible in r3. JEPA is a tested candidate, not a guaranteed winning method.

Software-release evidence, bounded experiment evidence, scoped JEPA-value findings and business validation are separate. There is no established Marine production integration, biological causal model, tuna biomass/species/catch, fuel saving or commercial value. A future private read-only pilot must define its own target, timing, comparator and evaluation. No new research command is queued.

## Labelled historical handoff context

The original 26 September handoff was a specification and helper package without a completed application or trained models. Its MOSAiC unchanged-target work later found at most 67 usable days (73 for a generous fixed-band bound), below the required 90; 25 historical slots remained blocked. The [engineering diagnostic ZIP](release/meeting-20260926-r2.zip), [original start prompt](CODEX_START_PROMPT.md), [task graph](orchestration/tasks.json), [source ledger](references/SOURCES.md), [handoff QA](evidence/HANDOFF_QA.md) and [continuation report](orchestration/reports/CONTINUATION_OUTCOME_20260927.md) remain historical evidence. Their older role pins, setup/push instructions and research queues are not active instructions for this meeting lane. Original entrypoint bytes remain in Git at assignment commit `2455fd39d54017bfad0208884b830ad51de4fa20`.

The existing [meeting playbook](docs/16_MEETING_PLAYBOOK.md) and [transfer request](docs/17_MARINE_TRANSFER_DATA_REQUEST.md) informed the current materials without replacing their history. [Index](INDEX.md), [start here](START_HERE.md) and [STATUS](orchestration/STATUS.md) provide navigation. Local-first resource/security limits remain in AGENTS; no cloud, purchases, public deployment or contact is authorized.
'''
start = '''# Start here — meeting readiness

Use the existing **approved AEON r3 offline research package** and the **completed independently reviewed diagnostic**. The active task is [CODEX_MEETING_READY_AFTER_DIAGNOSTIC.md](CODEX_MEETING_READY_AFTER_DIAGNOSTIC.md); no further fitting before the meeting, no recursive agents, and a fresh distinct reviewer pending.

1. Read the [meeting brief](meeting/20260929/MEETING_BRIEF.md).
2. Follow the [demo runbook](meeting/20260929/DEMO_RUNBOOK.md) for the exact tested scratch launcher, offline prerequisites, port, static fallback and owned-process shutdown.
3. Bring the [pilot data request](meeting/20260929/PILOT_DATA_REQUEST.md) and [conditional research decision](meeting/20260929/NEXT_RESEARCH_DECISION.md).
4. Inspect the [readiness checklist](meeting/20260929/READINESS_CHECKLIST.json) for actual PASSED/FAILED/NOT_RUN/NOT_APPLICABLE checks and evidence. Fresh browser/screenshot checks and a spoken eight-minute rehearsal are NOT_RUN; launcher/API/static checks passed. Meeting handoff approval is pending.

The runnable [r3 archive](release/aeon-offline-scale-expanded-20260928-r3.zip) retains SHA-256 `5e265e08bf1041bbe0fb4dd2a5f3503bae98f996096577f976beaf9384d6bb57`. Its [external exact-byte approval](orchestration/reviews/AEON_SCALE_EXPANDED_OFFLINE_R3_REVIEW_20260929.json) governs unchanged bytes despite the internal build-time review-pending marker. Offline Python 3.12 and uv serve the built app; Node is not required. The meeting scratch on port 8784 was tested and stopped; existing 8782/8783 servers were preserved.

[README](README.md) separates historical positive retrospective, negative transfer and exposed development/diagnostic outcomes with accurate scores. Saved truth-free replay is not checkpoint inference. The target is publisher-conditioned Sv_mean and horizons are source intervals; clock and availability are unresolved. Prior-year TRAIN affects expanded descendants. Completed research software does not establish Marine operational value; JEPA is a tested candidate. Biology and commercial validation remain outside scope.

## Labelled history

The 26 September distribution originally supplied implementation specifications, native role configurations, a task graph and helpers. Its instructions to initialize a repository, publish/push, install/probe data or start an older leader are historical, not actions for today. Original START_HERE.md is retained in Git at `2455fd39d54017bfad0208884b830ad51de4fa20`. The [old start prompt](CODEX_START_PROMPT.md), [historical handoff QA](evidence/HANDOFF_QA.md) and numbered [specifications](INDEX.md) remain available as evidence. Do not resume their research queues. Current owner scope overrides older model pins; root captures actual runtime routing and a separate session reviews.
'''
index_old = (R/'INDEX.md').read_text(encoding='utf-8')
spec_start = index_old.index('- [Repository instructions]')
spec_end = index_old.index('## Machine-readable')
specs = index_old[spec_start:spec_end]
specs = '\n'.join(line for line in specs.splitlines() if not any(s in line for s in ('Resume `marine-echo-jepa`','You are the end-to-end leader')))
index = '''# Package index

## Current meeting handoff

- [Current start](START_HERE.md) and [project status/results](README.md)
- [Owner meeting task; no scientific execution permission](CODEX_MEETING_READY_AFTER_DIAGNOSTIC.md)
- [Meeting brief](meeting/20260929/MEETING_BRIEF.md)
- [Tested demo runbook and explicit browser/rehearsal gaps](meeting/20260929/DEMO_RUNBOOK.md)
- [Proposed pilot data request](meeting/20260929/PILOT_DATA_REQUEST.md)
- [No further fitting; conditional draft decision](meeting/20260929/NEXT_RESEARCH_DECISION.md)
- [Machine-readable actual readiness](meeting/20260929/READINESS_CHECKLIST.json)
- [Static fallback: fixed served examples and packaged evidence](evidence/meeting-readiness-20260929/static-fallback.html)

## Approved research evidence

- [Immutable approved r3 ZIP](release/aeon-offline-scale-expanded-20260928-r3.zip)
- [R3 exact-byte approval](orchestration/reviews/AEON_SCALE_EXPANDED_OFFLINE_R3_REVIEW_20260929.json) and [validation report](orchestration/reports/AEON_SCALE_EXPANDED_OFFLINE_R3_VALIDATION_20260929.md)
- [Completed checkpoint/objective diagnostic](orchestration/reports/AEON_CHECKPOINT_OBJECTIVE_DIAGNOSTIC_20260929.md), [technical report](orchestration/reports/AEON_CHECKPOINT_DIAGNOSTIC_TECHNICAL_20260929.md) and [distinct diagnostic review](orchestration/reviews/AEON_CHECKPOINT_OBJECTIVE_DIAGNOSTIC_REVIEW_20260929.json)
- [Historical positive retrospective](orchestration/reports/AEON_RETROSPECTIVE_OUTCOME_20260927.md) and [negative secondary transfer](orchestration/reports/AEON_EXTERNAL_SECONDARY_NUMERIC_OUTCOME_20260928.md)

Approved r3 plus the completed diagnostic are the meeting's evidence base. Diagnostic numbers remain outside r3. Scopes, accurate prediction-mean ensemble scores, source-interval timing, prior-year TRAIN lineage and the unvalidated Marine value boundary are explicit in README and the brief. Fresh meeting review is pending; current browser smoke and speaking rehearsal are NOT_RUN. No research is queued.

## Labelled historical specifications and handoff

These retained documents describe earlier contracts, studies and roles. They are reference/history, not permission to reopen tasks or use superseded role pins. The original [start prompt](CODEX_START_PROMPT.md) and [resume prompt](CODEX_RESUME_PROMPT.md) remain historical. Original INDEX.md bytes remain in Git at `2455fd39d54017bfad0208884b830ad51de4fa20`.

'''+specs+'''

Historical machine assets include `configs/`, `schemas/`, `.codex/`, `orchestration/tasks.json`, `tools/`, `scripts/`, `tests_handoff/` and `HANDOFF_SHA256SUMS`. Their original distribution state is not the current AEON release state. Use the current checklist and reviewed reports for executed results; preserve the historical task graph and scientific ledgers.
'''
for name, content in [('README.md',readme),('START_HERE.md',start),('INDEX.md',index)]:
    (R/name).write_text(content,encoding='utf-8',newline='\n')
original = (E/'status-original.txt').read_bytes()
assert (R/'orchestration/STATUS.md').read_bytes() == original
prepend = '''## Meeting readiness implementation — 29 September 2026 (fresh review pending)

Five meeting deliverables are in `meeting/20260929/`; current entrypoints now
lead to approved immutable AEON r3 plus the completed diagnostic. New scratch
verified 107 assets, installed bundled wheels offline, served loopback 8784,
passed exact-byte/API/ordered-quantile checks for fixed indices 0 and 100, and
was shut down preserving existing 8782/8783 processes. API/static traversal
took 1.973620 seconds; no spoken eight-minute rehearsal is claimed. Fresh MCP
browser smoke/screenshots are NOT_RUN (no available browser); released static
evidence and historical reviewed screenshots are linked. No fitting or new
scientific outcome occurred. NO_FURTHER_FITTING_BEFORE_MEETING; any conditional
intervention remains DRAFT / NOT_AUTHORIZED / NOT_QUEUED. Fresh distinct review
is pending. Actual evidence/checks: `evidence/meeting-readiness-20260929/` and
`meeting/20260929/READINESS_CHECKLIST.json`. All prior STATUS content follows intact.

'''
(R/'orchestration/STATUS.md').write_bytes(prepend.encode('utf-8')+original)
print('Updated README.md, INDEX.md, START_HERE.md; prepended STATUS with original bytes intact.')
