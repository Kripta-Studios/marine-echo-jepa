# Command contract for the implementation agent

## Commands already supplied in this handoff

These use standard-library Python3.11+; Python3.12 is recommended:
```
python tools/verify_handoff.py
python -m unittest discover -s tests_handoff -v
python tools/probe_data.py
python tools/runtime_preflight.py
python tools/runtime_preflight.py --torch
```
`--torch` imports an already installed PyTorch and tests the local device; it does not install
or change packages. The probe fetches bounded metadata/first bytes, never the full archive.
Runtime/access reports go to ignored `outputs/preflight/`. Bootstrap helpers are not the model.
The handoff manifest is for checking the original package; after intentional modifications,
its original hashes are expected to differ. Do not treat that as an app regression or regenerate
it to hide changes. Build a separate release manifest for the finished application.

## Commands the agent must implement and execute

The following interface is a specification. It does not exist until tasks implement it.
Every subcommand must have `--help`, structured output, nonzero failure exit and resumable
idempotent operation where meaningful. Config files in this package are authoritative defaults.
```
uv sync --all-extras --locked
uv run marine-echo doctor --config configs/mvp.json
uv run marine-echo data probe --config configs/datasets.json
uv run marine-echo data download --dataset mosaic_azfp_down_2020 --resume
uv run marine-echo data inventory --dataset mosaic_azfp_down_2020
uv run marine-echo data preprocess --dataset mosaic_azfp_down_2020 --resume
uv run marine-echo data validate --dataset mosaic_azfp_down_2020
uv run marine-echo protocol create --config configs/experiments.json
uv run marine-echo benchmark baselines --protocol active
uv run marine-echo train --family direct --seed 7 --protocol active
uv run marine-echo train --family ema_jepa --seed 7 --protocol active
uv run marine-echo train --family shared_sigreg --seed 7 --protocol active
uv run marine-echo experiment complete-p0 --protocol active --resume
uv run marine-echo protocol freeze --protocol active --review-id R2
uv run marine-echo evaluate --partition test --protocol active
uv run marine-echo report --protocol active
npm --prefix web ci
npm --prefix web run build
uv run marine-echo release build --protocol active --output release/demo
uv run marine-echo serve --host 127.0.0.1 --port 8765 --artifact-root release/demo/artifacts
```

`complete-p0` is a finite orchestrator over declared runs, not an open-ended optimizer. It must
not purchase cloud, access a sealed test, relax gates or stop other user processes. It skips
only verified successful run IDs or documented ineligible branches, never failed runs by name.
Final freeze requires a real completed pre-test review. If `active` is ambiguous, fail rather
than choosing the newest directory. Provide scripts/Run-Demo.ps1 and scripts/Reproduce.ps1
wrapping tested commands after those commands exist. Do not ship an empty stub as implemented.

Frontend package scripts must expose format:check, lint, typecheck, test:run, build and test:e2e.
Release README must include the actual supported lock/env and startup commands, not a generic
Docker instruction that was never executed on Windows.
