[CmdletBinding()]
param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot
$PythonExe = Join-Path $RepoRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw 'Create the Python 3.12 app environment first with uv sync --extra data --locked. The portable release has its own offline Run-Demo.ps1.'
}
$env:PYTHONPATH = Join-Path $RepoRoot 'src'
& $PythonExe -m marine_echo.serving.cli serve --host 127.0.0.1 --port $Port --artifact-root (Join-Path $RepoRoot 'release/demo-20260926/artifacts')
exit $LASTEXITCODE
