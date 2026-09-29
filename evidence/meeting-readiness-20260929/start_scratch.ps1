$ErrorActionPreference = 'Stop'
$EvidenceRoot = $PSScriptRoot
$Preparation = Get-Content (Join-Path $EvidenceRoot 'preparation.json') -Raw | ConvertFrom-Json
$Port = 8784
while (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) { $Port++ }
$Existing = Get-Process -Id 55668,46932,5620,11048 | Select-Object Id,ProcessName,StartTime
$env:UV_PYTHON_DOWNLOADS = 'never'
$Launcher = Join-Path $Preparation.package 'Run-AEON-Research.ps1'
$Arguments = @('-NoProfile','-ExecutionPolicy','Bypass','-File',('"' + $Launcher + '"'),'-Port',"$Port")
$LaunchTime = Get-Date
$Process = Start-Process -FilePath powershell.exe -ArgumentList $Arguments -WorkingDirectory $Preparation.package -WindowStyle Hidden -RedirectStandardOutput (Join-Path $EvidenceRoot 'launcher.stdout.log') -RedirectStandardError (Join-Path $EvidenceRoot 'launcher.stderr.log') -PassThru
$Record = [ordered]@{ launcher_pid=$Process.Id; launcher_start_time=$Process.StartTime; port=$Port; address="http://127.0.0.1:$Port"; launcher=$Launcher; arguments=$Arguments; package=$Preparation.package; existing=$Existing; launch_time=$LaunchTime; uv=(Get-Command uv).Source; status='STARTING' }
$Record | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $EvidenceRoot 'launch.json') -Encoding UTF8
$Ready = $false
for ($Attempt=0; $Attempt -lt 45; $Attempt++) {
    if ($Process.HasExited) { throw "Launcher exited early: $($Process.ExitCode)" }
    try { $Health=Invoke-RestMethod "$($Record.address)/health"; if ($Health.ready) { $Ready=$true; break } } catch { }
    Start-Sleep -Seconds 1
}
if (-not $Ready) { throw 'Scratch readiness timeout' }
$Connections = @(Get-NetTCPConnection -State Listen -LocalPort $Port)
if (@($Connections | Where-Object LocalAddress -ne '127.0.0.1').Count) { throw 'Non-loopback listener' }
$Record.status='READY'
$Record.health=$Health
$Record.listeners=$Connections | Select-Object LocalAddress,LocalPort,OwningProcess
$Record.ready_seconds=((Get-Date)-$LaunchTime).TotalSeconds
$Record | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $EvidenceRoot 'launch.json') -Encoding UTF8
$Record | ConvertTo-Json -Depth 8
