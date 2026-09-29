# Marine Echo JEPA

**Approved AEON r3 offline research software, completed mixed public-data studies and a completed independently reviewed checkpoint diagnostic. Marine operational value is unvalidated.**

Start with the [meeting brief](meeting/20260929/MEETING_BRIEF.md), [tested runbook](meeting/20260929/DEMO_RUNBOOK.md), [pilot data request](meeting/20260929/PILOT_DATA_REQUEST.md), [research decision](meeting/20260929/NEXT_RESEARCH_DECISION.md) and [actual readiness checklist](meeting/20260929/READINESS_CHECKLIST.json). The [owner meeting task](CODEX_MEETING_READY_AFTER_DIAGNOSTIC.md) is the current lane; **NO_FURTHER_FITTING_BEFORE_MEETING**. For this handoff's review disposition, consult the checklist and its current separate review record. Fresh coordinator MCP Playwright smoke passed; the unavailable in-app browser attempt and spoken rehearsal remain NOT_RUN. Launcher/API/static checks passed. A separately labelled packaged MOSAiC response-code example supplies later observations. See the checklist for gaps rather than treating this handoff as self-approved.

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
