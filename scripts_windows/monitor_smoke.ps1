[CmdletBinding()]
param(
    [int]$Tail = 20
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($ScriptRoot)) {
    $ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
}

$LogsRoot = Join-Path $ScriptRoot "logs"

Write-Host "=== Active training processes ==="
$processes = Get-CimInstance Win32_Process |
    Where-Object { $_.Name -match 'powershell|python' -and $_.CommandLine -match 'run_mpe_smoke|run_mpe_paper_queue|train_mpe' } |
    Select-Object ProcessId, Name, CommandLine
if ($processes) {
    $processes | Format-List
}
else {
    Write-Host "No active MPE training queue or train_mpe process found."
}

Write-Host ""
Write-Host "=== Latest run directories ==="
$runs = Get-ChildItem $LogsRoot -Directory -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 6

foreach ($run in $runs) {
    $exitPath = Join-Path $run.FullName "exit_code.txt"
    $stdoutPath = Join-Path $run.FullName "stdout.log"
    $stderrPath = Join-Path $run.FullName "stderr.log"
    if (Test-Path $exitPath) {
        $rawExit = Get-Content $exitPath -Raw
        $exitCode = if ([string]::IsNullOrWhiteSpace($rawExit)) { "EMPTY_EXIT" } else { $rawExit.Trim() }
    }
    else {
        $exitCode = "RUNNING/NO_EXIT"
    }
    $stdoutBytes = if (Test-Path $stdoutPath) { (Get-Item $stdoutPath).Length } else { 0 }
    $stderrBytes = if (Test-Path $stderrPath) { (Get-Item $stderrPath).Length } else { 0 }
    [pscustomobject]@{
        Run = $run.Name
        Exit = $exitCode
        StdoutBytes = $stdoutBytes
        StderrBytes = $stderrBytes
        LastWrite = $run.LastWriteTime
    }
}

$latest = $runs | Select-Object -First 1
if ($latest) {
    $latestStdout = Join-Path $latest.FullName "stdout.log"
    if (Test-Path $latestStdout) {
        Write-Host ""
        Write-Host "=== Tail: $latestStdout ==="
        Get-Content -Tail $Tail -Encoding UTF8 $latestStdout
    }
}
