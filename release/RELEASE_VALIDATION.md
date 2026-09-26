# Tested diagnostic release — P0 remains incomplete

Launch from the repository:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\release\meeting-20260926-r2\Run-Demo.ps1 -Port 8765
```

Open http://127.0.0.1:8765. Python 3.12 and uv must already be installed/cached.
Dependencies are supplied in the offline wheelhouse. Ctrl+C stops the loopback service.

Distributable: `meeting-20260926-r2.zip` (18,697,810 bytes; 146 payload files).
SHA-256: `df02ff06cbbb05d967f8c157423c0c76b2acf66ddb16f1a1fcfbe36a710a5cc3`.
Code snapshot: `b45b7ec808e9d189e0047dc9fadd94900a3ecf73`, clean tracked state when packaged.
The subsequent validation sidecar binds checks to this exact ZIP hash without modifying it.

Verified after extraction into a different directory: all file hashes, corruption rejection,
restoration, offline environment creation/install, fresh startup and 5 browser tests. Navigation
p95 was 185 ms over 20 changes; cold startup was 7.55 s (target 30 s), excluding initial
installation. Initial offline setup plus startup took 25.69 s. Earlier package navigation
failed at 433 ms; its logs are preserved. The fix renders accessible table cells on expansion.
Live forecast latency is NOT_RUN because no benchmark model is promoted.

Other evidence: 54 app/data tests passed without skips (3 warnings); 17 model tests passed
in the original CUDA environment. Integrated model rerun: 16 passed, 1 explicit skip of the
already completed 100-update profile. Two frontend unit tests passed. Ruff, mypy against the
two actual environments, frontend formatting, linting, type checks and production build passed.
Coverage was 80% for the selected contract/evaluation/API modules, not the whole project.

The 100-update GPU software profile used synthetic fixtures: 16.87 s, 100 MiB peak reserved,
about 1.88 GiB observed process-tree RAM. It is not forecasting evidence. Infrastructure
paid/committed: USD 0. No cloud or public deployment was performed.

Real data: original five files were inventoried and the archive CRC/hash verified and safely
extracted. XML/DPL remain preserved. The app displays one predetermined day of raw counts,
not calibrated Sv. Public environmental profiles were found and downloaded, but instrument
geometry, time/location matching and calibration sensitivity still need validation and review.

All 25 benchmark runs remain blocked; no scientific negative or positive JEPA conclusion
exists. Full canonical preprocessing and training/evaluation orchestration are unfinished.
All three required native agents returned a service usage-limit error. R0/R1 findings remain
open; no R2 test seal or final independent R3 approval exists. The package is an engineering
diagnostic checkpoint, not completion of the requested P0 forecasting MVP.

Durable continuation: `../orchestration/STATUS.md`, per-task checkpoint, run ledger, reviews,
ADR0003 and actual logs. Do not substitute models, relax the protocol or relabel raw counts.

## Continuation evidence — historical payload unchanged

The statements above describe r2 at packaging time. In the continuation the native builder
and distinct independent reviewer became available. They have provided narrow engineering
and factory-certificate reviews, but no R0/R1 corpus approval, R2 approval or R3 release approval.
No new release is represented by these reviews; r2 remains the historical diagnostic.

Its ZIP hash and payload integrity were rechecked unchanged. A fresh relocated browser run
passed all five tests in 34.5 s, with navigation p95 190.143 ms over 20 changes. This does not
replace the historical 185 ms result. See `evidence/continuation/browser-relocated.log` and
`navigation-latency-relocated.json`. Existing and current serializations of the raw replay
have different byte hashes but equal parsed content; both are preserved.

The serial55170 factory certificate was recovered from swapped registry attachments, and
the reviewer accepted its XML coefficient mapping. Environmental matching across both
trackers yields 12/24 candidate test profile days within the illustrative 5 km/24 h screen;
that is neither calibration approval nor the required 20 eligible test days. The prospective
fixed regional sensitivity assay failed its primary bound (1.080721 dB > 1 dB). Its failure
and the subsequent depth-resolved investigation remain in `evidence/continuation/`.

All 25 required benchmark runs remain BLOCKED, with zero completed runs. Approved physical
days are zero; actual post-QC eligible-day counts are NOT_ESTABLISHED. Replay exposure audit
found only known training-period acoustic artifacts, but incomplete historical viewing records
prevent a claim that the candidate holdout is sealed. No new diagnostic ZIP was produced.


Continuation integration on27 September (local time), before final census/CLI changes:
`evidence/continuation/checks-20260926T222119Z/checks.json` records all12 check commands
exiting0 without resource stops.177 app/data/scientific/security/API tests passed; model
software11 passed/2 opt-in GPU checks skipped; frontend2 and browser5 passed. Formatting,
lint, app/model types, frontend build and unchanged historicalr2 integrity passed.
Peak sampled check process-tree RAM0.8313GiB. These are software checks, not benchmarks,
new physical corpus approval, a new release, or a new full-training memory measurement.


Later source integration: `evidence/continuation/checks-20260926T225032Z/checks.json` records
all11 commands exiting0.200 app/data/scientific/security/API tests passed; model software
11 passed/2 explicit opt-in GPU skips; frontend2 passed. Formatting, lint, types, frontend
build and unchanged r2 integrity passed. The current working-preview browser check will use
its own evidence output after the actual census summary is independently reviewed.


## Final continuation source checks (27 September, before data decision)

After registry hardening and the complete count-map/plot safeguards, all 229
app/data/integration/scientific/API/security tests passed in 132.68 seconds.
Formatting, lint and both type-check environments passed; sampled test process-tree
RAM peaked at 0.301 GiB. Logs retain 707 upstream warnings (704 scikit-learn
parallel-configuration warnings plus deprecation warnings). No test was skipped
from this app/data invocation. See
`evidence/continuation/final-app-checks-20260926T233155Z/checks.json`.

The unchanged model suite's preceding 11 passes and two opt-in GPU skips remain
separate evidence in `checks-20260926T225032Z`; they are not fresh training runs.
Current working-preview artifact and changed-browser acceptance remain pending.
No new normal release or diagnostic ZIP was produced; historical r2 is preserved.
