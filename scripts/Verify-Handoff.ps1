[CmdletBinding()]
param([string]$Repo = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Set-Location -LiteralPath (Resolve-Path -LiteralPath $Repo).Path
$Uv = Get-Command uv -ErrorAction Stop
& $Uv.Source run --no-project --python 3.12 python tools/verify_handoff.py
if ($LASTEXITCODE -ne 0) { throw 'Original handoff verification failed.' }
& $Uv.Source run --no-project --python 3.12 python -m unittest discover -s tests_handoff -v
if ($LASTEXITCODE -ne 0) { throw 'Handoff helper tests failed.' }
