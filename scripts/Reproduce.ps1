[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot
& uv sync --extra data --locked
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& uv run marine-echo doctor --config configs/mvp.json
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& uv run marine-echo data inventory --dataset mosaic_azfp_down_2020
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& uv run pytest tests/unit tests/integration tests/scientific tests/api tests/security
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& uv run marine-echo experiment complete-p0 --protocol active --resume
# Exit 2 means the documented physical calibration gate remains blocked.
exit $LASTEXITCODE
