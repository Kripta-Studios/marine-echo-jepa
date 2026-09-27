[CmdletBinding()]
param([ValidateRange(1,65535)][int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$ReleaseRoot = $PSScriptRoot
Set-Location -LiteralPath $ReleaseRoot
$PythonExe = Join-Path $ReleaseRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $PythonExe)) {
    & uv venv --python 3.12 --offline (Join-Path $ReleaseRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Offline Python 3.12 is required.' }
}
& $PythonExe (Join-Path $ReleaseRoot 'Verify-Release.py')
if ($LASTEXITCODE -ne 0) { throw 'Release integrity verification failed.' }
& uv pip install --python $PythonExe --offline --no-index --find-links (Join-Path $ReleaseRoot 'wheelhouse') -r (Join-Path $ReleaseRoot 'requirements-app.txt')
if ($LASTEXITCODE -ne 0) { throw 'Offline wheel installation failed.' }
$env:PYTHONPATH = Join-Path $ReleaseRoot 'src'
Write-Output "AEON offline research: http://127.0.0.1:$Port"
& $PythonExe (Join-Path $ReleaseRoot 'Serve-AEON-Research.py') --port $Port
exit $LASTEXITCODE
