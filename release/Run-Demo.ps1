[CmdletBinding()]
param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$ReleaseRoot = $PSScriptRoot
Set-Location -LiteralPath $ReleaseRoot
$PythonExe = Join-Path $ReleaseRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $PythonExe)) {
    & uv venv --python 3.12 --offline (Join-Path $ReleaseRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Offline Python 3.12 was not found. Install Python 3.12 and uv before the meeting.' }
}
$env:PYTHONPATH = Join-Path $ReleaseRoot 'src'
& $PythonExe (Join-Path $ReleaseRoot 'Verify-Release.py')
if ($LASTEXITCODE -ne 0) { throw 'Release verification failed. Restore a verified archive.' }
& uv pip install --python $PythonExe --offline --no-index --find-links (Join-Path $ReleaseRoot 'wheelhouse') -r (Join-Path $ReleaseRoot 'requirements-app.txt')
if ($LASTEXITCODE -ne 0) { throw 'Offline dependency installation failed. Check release hashes and wheelhouse.' }
Write-Output "Marine Echo JEPA: http://127.0.0.1:$Port (engineering diagnostic; forecasts unavailable)"
& $PythonExe -m marine_echo.serving.cli serve --host 127.0.0.1 --port $Port --artifact-root (Join-Path $ReleaseRoot 'demo/artifacts')
exit $LASTEXITCODE
