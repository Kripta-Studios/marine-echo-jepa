# AEON r3 meeting runbook

Use only the approved immutable r3 scratch package. **Fresh coordinator MCP Playwright browser smoke passed; the unavailable in-app browser attempt and a spoken eight-minute rehearsal remain NOT_RUN.** The separate coordinator connector supplied actual scratch navigation and screenshots. Launcher/API/static checks passed. The timed automated traversal took **1.973620 seconds**; startup readiness took **22.227760 seconds**, including first offline installation. These are measured checks, not invented speaking time. Fresh distinct review of this handoff is pending.

## Exact package and prerequisites

- Archive: [r3 ZIP](../../release/aeon-offline-scale-expanded-20260928-r3.zip), SHA-256 `5e265e08bf1041bbe0fb4dd2a5f3503bae98f996096577f976beaf9384d6bb57`.
- Reviewed source: `f975d2b0a425a9cbdadbacf9f50d0cad216d93c1`; [external exact-byte approval](../../orchestration/reviews/AEON_SCALE_EXPANDED_OFFLINE_R3_REVIEW_20260929.json). The internal review-pending marker is immutable build-time state and does not supersede that approval.
- Repository: `C:/Users/Álvaro Schwiedop/Desktop/KriptaStudios/marine-echo-jepa`.
- New scratch: `outputs/meeting-readiness-20260929-r3/aeon-offline-scale-expanded-20260928-r3/`. Original release/extraction is untouched. [Preparation record](../../evidence/meeting-readiness-20260929/preparation.json) binds the absolute paths and all source hashes.
- Windows PowerShell, existing `uv` (`C:/Users/Álvaro Schwiedop/.local/bin/uv.exe`) and an already cached offline Python 3.12 are required. Actual Python was 3.12.13. No Node, browser installation, training dependencies, network install or credentials are required to serve the package.

The launcher and both packaged Python entry scripts were inspected before execution. `Run-AEON-Research.ps1` accepts `-Port`, defaults to 8765, creates its `.venv` offline, verifies 107 hashed assets and installs 13 application distributions from bundled wheels with `--offline --no-index`. Do not modify or rebuild it.

## Tested preparation and launch

Executed from the repository root; commands below describe this completed run. Evidence scripts are retained for inspection; do not overwrite the recorded run when rehearsing again.

```powershell
.venv/Scripts/python.exe -B evidence/meeting-readiness-20260929/prepare_scratch.py
powershell.exe -NoProfile -ExecutionPolicy Bypass -File evidence/meeting-readiness-20260929/start_scratch.ps1
.venv/Scripts/python.exe -B evidence/meeting-readiness-20260929/rehearse_static.py
```

Each exited 0. Safe extraction rejected unsafe paths, links, collisions and excess size; manifest coverage was exact. Port **8784 was free** before launch. The actual package command launched by the wrapper was:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:/Users/Álvaro Schwiedop/Desktop/KriptaStudios/marine-echo-jepa/outputs/meeting-readiness-20260929-r3/aeon-offline-scale-expanded-20260928-r3/Run-AEON-Research.ps1" -Port 8784
```

The loopback address was **http://127.0.0.1:8784/**. For another meeting startup, check this port first; select the next free port if occupied and record the new process identities. Use the existing scratch directly without another extraction. For an interactive foreground launch, run the package command above and close that owned launch with Ctrl+C, then verify its port closed. For a tracked background rehearsal, the start/stop evidence scripts capture and validate ownership; preserve the current logs before creating a separate run record.

## Planned approximately eight-minute traversal

This is the proposed speaking schedule, adapted from the [existing playbook](../../docs/16_MEETING_PLAYBOOK.md); it is not a claim that a speaking rehearsal occurred.

| Time | Action and wording |
| --- | --- |
| 0:00–1:00 | State the operational question from the brief. Explain public AEON research data, publisher-conditioned `Sv_mean`, four-frequency histories and masks. Source timestamps are not verified UTC; +1/+3/+6 are source intervals. |
| 1:00–3:00 | Navigate to **AEON study**. Identify historical saved replay, switch direct/EMA/LightGBM, inspect q05/q25/q50/q75/q95 at all horizons for index 0, then fixed index 100. These cutoffs were chosen before values, not because a model looked good. Use the static table if browser navigation is unavailable. |
| 3:00–4:00 | Explain that AEON replay has no later truth and performs no checkpoint inference. Navigate to **#experiment-lab / Raw response-code TRAIN development**: the separately predetermined first chronological MOSAiC row shows saved quantiles and later observations. State **UNCALIBRATED RESPONSE CODE, NOT Sv_mean, NOT AN AEON OR JEPA RESULT**; +1 is missing/ineligible, +3/+6 are observed. Use the separate static fallback table if needed. No raw archive or new CAL/TEST outcome is opened. |
| 4:00–5:30 | Show all retrospective, transfer, scaling and expanded-TRAIN comparisons and limits. Retrospective EMA 0.533954 versus direct 0.563387 is positive in its original non-sealed study; prior-year frozen-model transfer EMA 0.678922 versus direct 0.670178 is negative. Cross-site primary was metadata-ineligible. Prior-year data became TRAIN for expanded models/descendants. |
| 5:30–6:30 | Open the completed diagnostic document outside r3. Direct 1500/3000 and EMA ensembles are 0.639577/0.639062/0.642170 on exposed VAL. Direct 1500 beats EMA per seed/ensemble; each direct seed worsens at 3000 while its ensemble improves. Do not imply these later diagnostic numbers are in r3. |
| 6:30–7:30 | Open the pilot data request: complete low/zero/invalid sequences, masks, units, geometry, identities, acquisition/availability times, versions and permissions; nominate data and decision owners. Replication of supplied products differs from a future raw/depth-resolved study. |
| 7:30–8:00 | State NO_FURTHER_FITTING_BEFORE_MEETING; the later intervention is DRAFT / NOT_AUTHORIZED / NOT_QUEUED. Completed research software does not establish Marine operational, biological or commercial value. |

## Fixed served examples

[Selection](../../evidence/meeting-readiness-20260929/example-selection.json) was recorded before opening forecasts. Zero-based chronological indices are **0** (interval **482280**, `2025-01-06T23:51:58.910000`) and **100** (interval **482380**, `2025-01-11T03:51:58.030000`). Do not label either a representative or difficult case based on unseen error.

These are actual +1-source-interval API values, rounded to six decimals, in dB re 1 m^-1:

| Index | Model | q05 | q50 | q95 |
| --- | --- | ---: | ---: | ---: |
| 0 | Direct ensemble | -92.022537 | -91.437767 | -87.851578 |
| 0 | EMA ensemble | -92.084984 | -91.509204 | -87.532918 |
| 0 | LightGBM | -91.967968 | -91.742660 | -87.817496 |
| 100 | Direct ensemble | -91.635071 | -89.712753 | -85.177361 |
| 100 | EMA ensemble | -91.754974 | -89.749863 | -85.750254 |
| 100 | LightGBM | -91.413158 | -89.670449 | -85.732404 |

All five quantiles at +1/+3/+6 for all models, row identities, requests, response hashes and timings are in [rehearsal.json](../../evidence/meeting-readiness-20260929/rehearsal.json) and the [static tables](../../evidence/meeting-readiness-20260929/static-fallback.html). The study response equals the hash-bound package artifact; HTML/JS/CSS served bytes equal the scratch package files. Quantiles were ordered for both examples in all models. No model fitting, new inference or scoring occurred.

## Fallback and browser evidence

Open [static-fallback.html](../../evidence/meeting-readiness-20260929/static-fallback.html) locally. It contains served forecasts, the exact released retrospective report, package limits and relative links to REPORT, MODEL_CARD and MARINE_DATA_REQUEST in the scratch package. Keep the [brief](MEETING_BRIEF.md), [pilot request](PILOT_DATA_REQUEST.md) and [research decision](NEXT_RESEARCH_DECISION.md) alongside it.

The implementation session's in-app browser setup returned `No browser is available`; discovery returned `[]`, so that attempt remains **NOT_RUN**. Separately, the coordinator's available **MCP Playwright** connector completed real scratch navigation, direct/EMA quantile rendering, fixed-index-100 pagination, comparisons/limits, raw engineering fallback navigation and desktop/mobile screenshots. [Coordinator browser evidence](../../evidence/meeting-readiness-20260929/coordinator-browser.json) and [verification](../../evidence/meeting-readiness-20260929/coordinator-browser-verification.json) identify the executor and limits. Viewports were 1440/390 pixels with document widths 1425/375 (scrollbars), fitting both. A favicon 404 was nonblocking; no JavaScript console error was observed. The initial network inspection observed 20 requests, all loopback; the final saved network buffer was empty after navigation, so a complete persisted network census is **NOT_RUN**. Separate generated static HTML was checked for data/links locally, not browser-clicked. No browser dependency installation occurred. Existing [desktop](../../evidence/browser/aeon-scale-expanded-r3-desktop.png), [mobile](../../evidence/browser/aeon-scale-expanded-r3-mobile.png) and [browser JSON](../../evidence/browser/aeon-scale-expanded-r3-browser-smoke.json) are **historical r3 validation evidence**, not this scratch run. They record 1440/1440 and 390/390 document widths, no page errors and no external requests. Review the [approved validation report](../../orchestration/reports/AEON_SCALE_EXPANDED_OFFLINE_R3_VALIDATION_20260929.md) for their original scope.

## Exact owned shutdown

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File evidence/meeting-readiness-20260929/stop_scratch.ps1
```

The final attempt exited 0. Ownership was launcher **59556**, console child **32096**, venv Python **28976**, serving Python **32600**. The script validated creation times, command lines, parentage and listener ownership, explicitly stopped **32600**, and the remaining owned chain exited. It verified **8784 closed** and protected process identities/listeners unchanged: r3 **8782**, PIDs **55668/46932**; r2 **8783**, PIDs **5620/11048**. Never copy these old PIDs into a future shutdown command; the script uses that run's recorded identities. No natural server exit code is claimed after deliberate termination.

Two earlier shutdown attempts exited 1 before mutation: a decoded-DateTime guard error and an empty child-frontier CIM query. Both failures and corrections are retained in evidence. They were implementation mistakes, not access denials. [Shutdown record](../../evidence/meeting-readiness-20260929/shutdown.json), launcher stdout/stderr and [checklist](READINESS_CHECKLIST.json) retain the actual result. The scratch server is stopped after rehearsal; existing servers remain running.

### Fresh coordinator screenshots and permission boundary

The MCP tool denied snapshot/log writes into the main checkout. The coordinator retained artifacts only in its explicitly allowed `C:/Users/Álvaro Schwiedop/Desktop/KriptaStudios/marine-echo-jepa-core/.playwright-mcp/` directory. No denied main write was retried through another channel and no images were copied into main. Exact external paths/hashes are in coordinator-browser-verification.json. Fresh files include `meeting-ready-20260929-1440.png`, `meeting-ready-20260929-390.png`, `meeting-ready-20260929-raw-observation.png`, initial/raw snapshots and console/network logs. They are coordinator execution evidence, not an independent final review.

### Packaged later-observation example, separate scope

The coordinator fixed the **first chronological raw row** before inspecting values: cutoff `2020-04-02T00:00:00Z` in packaged `artifacts/raw-development.json`. Its SHA-256 is `ad3b69a29395fb83528cd2e015c86e6255a2e3be42b6b7a450ca0f37db310de1`; source result `4237f375ba129fce95be952325e56cf5bc9615a5d89b2ed9ba91dfc8771cb49d`; review record `18d8c890dfef8aacc274308d94251b91059c76fc743071d1cc1bcd61cc584b15`. [Selection](../../evidence/meeting-readiness-20260929/coordinator-raw-example-selection.json), [full quantiles/observation excerpt](../../evidence/meeting-readiness-20260929/coordinator-raw-example.json) and [implementation verification](../../evidence/meeting-readiness-20260929/supplement-verification.json) bind this choice to the package without new evaluation.

| Horizon hours | Target start UTC | Later response-code observation | Eligibility |
| --- | --- | ---: | --- |
| 1 | 2020-04-02T00:00:00Z | Missing / not scored | Ineligible |
| 3 | 2020-04-02T02:00:00Z | 11218.386223 | Eligible |
| 6 | 2020-04-02T05:00:00Z | 11229.620260 | Eligible |

The complete ridge/direct five-quantile values are in the static table and excerpt. This is **UNCALIBRATED RESPONSE CODE, NOT Sv_mean, NOT AN AEON OR JEPA RESULT**. It is TRAIN-development engineering evidence without final evaluation. Keep the missing +1 row visible; do not filter for favorable/error-free cases. The static table supplements the AEON aggregate fallback while AEON row-level truth remains absent.
