[CmdletBinding()]
param(
    [ValidateSet('Audit', 'Checks', 'TrainCensus', 'Eligibility', 'SupportMap', 'Campaign')]
    [string]$Stage = 'Audit'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot
$Python = Join-Path $RepoRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'Existing project data environment is missing. No automatic environment replacement is allowed.'
}

if ($Stage -eq 'Audit') {
    & $Python tools/continuation_audit.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python tools/metadata_inventory.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python tools/calibration_continuation.py
    exit $LASTEXITCODE
}

if ($Stage -eq 'Checks') {
    & $Python tools/check_continuation.py --browser
    exit $LASTEXITCODE
}

if ($Stage -eq 'TrainCensus') {
    & $Python tools/train_census_v2.py
    exit $LASTEXITCODE
}

if ($Stage -eq 'Eligibility') {
    & $Python tools/target_only_bound.py
    exit $LASTEXITCODE
}

if ($Stage -eq 'SupportMap') {
    & $Python tools/train_support_map.py
    exit $LASTEXITCODE
}

# This wrapper cannot approve a corpus or open a final test. The current campaign
# command must return its explicit prerequisite blocker until an eligible corpus
# and independent reviews exist. A successful future campaign still stops at R2.
& $Python -m marine_echo.serving.cli experiment complete-p0 --protocol active --resume
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Output 'STOP_BEFORE_FINAL_TEST: independent R2 approval and a verified freeze are required.'
exit 3
