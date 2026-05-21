[CmdletBinding()]
param(
    [int]$PollSeconds = 60,
    [string]$PythonPath = "C:\Users\Lenovo\anaconda3\envs\lge_cmr\python.exe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($ScriptRoot)) {
    $ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
}

$QueueScript = Join-Path $ScriptRoot "run_remaining_paper_queue.ps1"
if (-not (Test-Path -LiteralPath $QueueScript)) {
    throw "Queue script not found: $QueueScript"
}

function Get-ActiveTrainMpe {
    return @(Get-CimInstance Win32_Process |
        Where-Object { $_.Name -eq "python.exe" -and $_.CommandLine -match "train_mpe.py" })
}

function Wait-ForIdleTrainMpe {
    param([Parameter(Mandatory = $true)][string]$Reason)

    while ($true) {
        $active = Get-ActiveTrainMpe
        if ($active.Count -eq 0) {
            Write-Host ("[{0}] No active train_mpe.py jobs; starting {1}." -f (Get-Date), $Reason)
            return
        }

        $ids = ($active | ForEach-Object { $_.ProcessId }) -join ","
        Write-Host ("[{0}] Waiting for {1}; active train_mpe.py jobs: {2}" -f (Get-Date), $Reason, $ids)
        Start-Sleep -Seconds $PollSeconds
    }
}

Wait-ForIdleTrainMpe -Reason "7-agent ANN queue"
& $QueueScript `
    -PythonPath $PythonPath `
    -ParallelJobs 1 `
    -RolloutThreads 4 `
    -AgentCounts @(7) `
    -ActorArchitectures @("ann") `
    -Seeds @(1, 2, 3)

Wait-ForIdleTrainMpe -Reason "5-agent SNN queue"
& $QueueScript `
    -PythonPath $PythonPath `
    -ParallelJobs 1 `
    -RolloutThreads 4 `
    -AgentCounts @(5) `
    -ActorArchitectures @("snn_lif", "snn_at") `
    -Seeds @(1, 2, 3)

Wait-ForIdleTrainMpe -Reason "7-agent SNN queue"
& $QueueScript `
    -PythonPath $PythonPath `
    -ParallelJobs 1 `
    -RolloutThreads 4 `
    -AgentCounts @(7) `
    -ActorArchitectures @("snn_lif", "snn_at") `
    -Seeds @(1, 2, 3)

Write-Host ("[{0}] Safe remaining scheduler completed." -f (Get-Date))
