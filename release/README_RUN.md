# Run the offline engineering diagnostic

This page documents the historical r2 package. Native builder/reviewer access has since
returned; current scientific and release gates are recorded in `orchestration/STATUS.md`
and `release/RELEASE_VALIDATION.md`. The preserved r2 archive is unchanged.

This is an incomplete P0 engineering package. It replays real Arctic raw counts.
No physical-unit forecast, trained benchmark or JEPA value result is included.
At r2 packaging, independent final approval was blocked by the native-agent service usage limit.

On this Windows 11 machine, extract the ZIP to a new folder and run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Run-Demo.ps1 -Port 8765
```

Open http://127.0.0.1:8765. Stop with Ctrl+C. The application binds to loopback only.
There are no external assets, login, telemetry or hosted services. Forecast requests return
an explicit unavailable status; revealing later measurements does not change model inputs.

Prerequisites: Windows x64, `uv`, and an already installed or uv-cached Python 3.12. The
launcher uses `--offline` and the packaged wheelhouse to create its local environment. It
does not download Python or packages. First dependency installation is separate from normal
cold startup. If another process owns the port, choose a different port. Do not stop unrelated
processes. Restore a fresh archive if hash verification fails.

The package includes source snapshots, dependency locks, reviewed and pending review records,
actual test logs, data hashes, a 25-run blocked registry and limitations. Raw archives/manuals
are retained only in the research workspace. `SHA256SUMS` covers payload files. It establishes
local integrity, not publisher authenticity or a cryptographic author signature.

For research continuation, use `orchestration/STATUS.md` in the full repository. The app-only
wheelhouse does not include scientific preprocessing or GPU dependencies. The documented
benchmark CLI branches are blocked and full training/evaluation orchestration remains unfinished.
Reading those commands is not evidence that the full P0 workflow is executable.
