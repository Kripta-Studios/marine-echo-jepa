$ErrorActionPreference = 'Stop'
$EvidenceRoot = $PSScriptRoot
$Launch = Get-Content (Join-Path $EvidenceRoot 'launch.json') -Raw | ConvertFrom-Json
$OwnedRoot = Get-CimInstance Win32_Process -Filter "ProcessId=$($Launch.launcher_pid)"
if (-not $OwnedRoot -or -not $OwnedRoot.CommandLine.Contains($Launch.launcher)) { throw 'Launcher identity mismatch; stop refused' }
$ExpectedStart = ([datetime]$Launch.launcher_start_time).ToLocalTime()
if ([math]::Abs(($OwnedRoot.CreationDate-$ExpectedStart).TotalSeconds) -gt 1) { throw 'Launcher PID reused; stop refused' }
$Owned = @($OwnedRoot)
$Frontier = @($OwnedRoot.ProcessId)
while ($Frontier.Count) {
    $Next = @()
    foreach ($Parent in $Frontier) {
        $Children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$Parent")
        $Owned += $Children
        foreach ($Child in $Children) { $Next += $Child.ProcessId }
    }
    $Frontier = $Next
}
$Protected = @(55668,46932,5620,11048)
if (@($Owned | Where-Object ProcessId -in $Protected).Count) { throw 'Protected process in ownership set; stop refused' }
$Listener = Get-NetTCPConnection -State Listen -LocalPort $Launch.port
if ($Listener.LocalAddress -ne '127.0.0.1' -or $Listener.OwningProcess -notin $Owned.ProcessId) { throw 'Listener ownership mismatch; stop refused' }
$Before = $Owned | Select-Object ProcessId,ParentProcessId,CreationDate,ExecutablePath,CommandLine
$Targets = @($Owned)
[array]::Reverse($Targets)
$Stopped = @()
foreach ($Target in $Targets) {
    $Current = Get-CimInstance Win32_Process -Filter "ProcessId=$($Target.ProcessId)"
    if ($Current) {
        if ($Current.CreationDate -ne $Target.CreationDate -or $Current.CommandLine -ne $Target.CommandLine) { throw 'PID identity changed; stop refused' }
        Stop-Process -Id $Target.ProcessId -Force -ErrorAction Stop
        $Stopped += $Target.ProcessId
    }
}
Start-Sleep -Seconds 2
if (Get-NetTCPConnection -State Listen -LocalPort $Launch.port -ErrorAction SilentlyContinue) { throw 'Owned port remains open' }
$ExistingAfter = @(Get-Process -Id $Protected | Select-Object Id,ProcessName,StartTime)
foreach ($Original in $Launch.existing) {
    $Now = $ExistingAfter | Where-Object Id -eq $Original.Id
    $OriginalStart = ([datetime]$Original.StartTime).ToLocalTime()
    if ([math]::Abs(($Now.StartTime-$OriginalStart).TotalSeconds) -gt 1) { throw 'Existing process identity changed' }
}
$ExistingListeners = @(Get-NetTCPConnection -State Listen | Where-Object LocalPort -in 8782,8783 | Select-Object LocalAddress,LocalPort,OwningProcess)
if (($ExistingListeners | Where-Object LocalPort -eq 8782).OwningProcess -ne 46932 -or ($ExistingListeners | Where-Object LocalPort -eq 8783).OwningProcess -ne 11048) { throw 'Existing listeners changed' }
$Record = [ordered]@{ status='PASSED'; command='powershell.exe -NoProfile -ExecutionPolicy Bypass -File evidence/meeting-readiness-20260929/stop_scratch.ps1'; owned_before=$Before; explicitly_stopped=$Stopped; owned_port_closed=$true; protected_processes_after=$ExistingAfter; protected_listeners_after=$ExistingListeners; server_exit='Deliberately terminated owned process tree; no natural server exit code claimed'; recorded_at=(Get-Date).ToString('o') }
$Record | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $EvidenceRoot 'shutdown.json') -Encoding UTF8
$Record | ConvertTo-Json -Depth 8
