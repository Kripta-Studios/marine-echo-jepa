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
