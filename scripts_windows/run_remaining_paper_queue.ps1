[CmdletBinding()]
param(
    [string]$PythonPath = "C:\Users\Lenovo\anaconda3\envs\lge_cmr\python.exe",
    [ValidateRange(1, 8)]
    [int]$ParallelJobs = 1,
    [int[]]$AgentCounts = @(5, 7),
    [string[]]$ActorArchitectures = @("ann", "snn_lif", "snn_at"),
    [int[]]$Seeds = @(1, 2, 3),
    [int]$NumEnvSteps = 1000000,
    [int]$EpisodeLength = 25,
    [int]$RolloutThreads = 1
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($ScriptRoot)) {
    $ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
}

$RepoRoot = Split-Path -Parent $ScriptRoot
$TrainScript = Join-Path $RepoRoot "onpolicy\scripts\train\train_mpe.py"
$LogsRoot = Join-Path $ScriptRoot "logs"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

New-Item -ItemType Directory -Path $LogsRoot -Force | Out-Null
$env:WANDB_DISABLED = "true"
$env:PYTHONUNBUFFERED = "1"

function ConvertTo-QuotedArgument {
    param([Parameter(Mandatory = $true)][string]$Argument)
    $escaped = $Argument.Replace('"', '\"')
    if ($escaped -match '[\s"]') {
        return '"' + $escaped + '"'
    }
    return $escaped
}

function Join-ArgumentLine {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    return ($Arguments | ForEach-Object { ConvertTo-QuotedArgument $_ }) -join " "
}

function Test-ExperimentActive {
    param([Parameter(Mandatory = $true)][string]$ExperimentName)
    $active = Get-CimInstance Win32_Process |
        Where-Object { $_.Name -eq "python.exe" -and $_.CommandLine -match [regex]::Escape($ExperimentName) }
    return $null -ne $active
}

function Test-ExperimentCompleted {
    param([Parameter(Mandatory = $true)][string]$ExperimentName)

    $summaryPath = Join-Path $RepoRoot "onpolicy\scripts\results\MPE\simple_spread\mappo\$ExperimentName\run1\logs\summary.json"
    if (Test-Path -LiteralPath $summaryPath) {
        try {
            $summaryText = Get-Content -Raw -Encoding UTF8 $summaryPath
            $steps = [regex]::Matches($summaryText, '"step"\s*:\s*(\d+)|\[\s*[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*,\s*(\d+)\s*,')
            foreach ($match in $steps) {
                $candidate = if ($match.Groups[1].Success) { [int]$match.Groups[1].Value } else { [int]$match.Groups[2].Value }
                if ($candidate -ge 990000) {
                    return $true
                }
            }
        }
        catch {
            Write-Warning ("Could not inspect summary for {0}: {1}" -f $ExperimentName, $_.Exception.Message)
        }
    }

    $completionMarker = Join-Path $RepoRoot "onpolicy\scripts\results\MPE\simple_spread\mappo\$ExperimentName\completed.ok"
    if (Test-Path -LiteralPath $completionMarker) {
        return $true
    }

    $logDirs = Get-ChildItem -Path $LogsRoot -Directory -ErrorAction SilentlyContinue
    foreach ($logDir in $logDirs) {
        $metadataPath = Join-Path $logDir.FullName "metadata.json"
        if (-not (Test-Path -LiteralPath $metadataPath)) {
            continue
        }

        try {
            $metadata = Get-Content -Raw -Encoding UTF8 $metadataPath | ConvertFrom-Json
        }
        catch {
            continue
        }

        if ($metadata.PSObject.Properties.Name -notcontains "ExperimentName") {
            continue
        }

        if ($metadata.ExperimentName -ne $ExperimentName) {
            continue
        }

        $exitCodePath = Join-Path $logDir.FullName "exit_code.txt"
        if (Test-Path -LiteralPath $exitCodePath) {
            $exitContent = Get-Content -Raw -Encoding UTF8 $exitCodePath -ErrorAction SilentlyContinue
            if ($null -ne $exitContent) {
                $exitText = $exitContent.Trim()
                if ($exitText -eq "0") {
                    return $true
                }
            }
        }

        $stdoutPath = Join-Path $logDir.FullName "stdout.log"
        if (Test-Path -LiteralPath $stdoutPath) {
            $stdoutTail = Get-Content -Tail 200 -Encoding UTF8 $stdoutPath
            $escapedSteps = [regex]::Escape([string]$NumEnvSteps)
            if (($stdoutTail -join "`n") -match "total num timesteps\s+$escapedSteps/$escapedSteps") {
                return $true
            }
        }
    }

    return $false
}

function New-TrainingRun {
    param(
        [Parameter(Mandatory = $true)][int]$Index,
        [Parameter(Mandatory = $true)][int]$AgentCount,
        [Parameter(Mandatory = $true)][string]$ActorArch,
        [Parameter(Mandatory = $true)][int]$Seed
    )

    $experimentName = "mpe_paper_a${AgentCount}_${ActorArch}_seed${Seed}"
    $runId = "{0}_remaining_a{1}_{2}_s{3}_{4:00}" -f $Timestamp, $AgentCount, $ActorArch, $Seed, $Index
    $logDir = Join-Path $LogsRoot $runId
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null

    $arguments = @(
        "-u",
        $TrainScript,
        "--env_name", "MPE",
        "--algorithm_name", "mappo",
        "--experiment_name", $experimentName,
        "--scenario_name", "simple_spread",
        "--num_agents", "$AgentCount",
        "--num_landmarks", "$AgentCount",
        "--seed", "$Seed",
        "--n_training_threads", "1",
        "--n_rollout_threads", "$RolloutThreads",
        "--num_mini_batch", "1",
        "--episode_length", "$EpisodeLength",
        "--num_env_steps", "$NumEnvSteps",
        "--ppo_epoch", "10",
        "--use_ReLU",
        "--gain", "0.01",
        "--lr", "7e-4",
        "--critic_lr", "7e-4",
        "--actor_arch", $ActorArch,
        "--use_wandb",
        "--log_spike_stats"
    )

    $run = [PSCustomObject]@{
        Index = $Index
        ExperimentName = $experimentName
        RunId = $runId
        LogDir = $logDir
        StdoutLog = Join-Path $logDir "stdout.log"
        StderrLog = Join-Path $logDir "stderr.log"
        Arguments = $arguments
    }

    $commandLine = (ConvertTo-QuotedArgument $PythonPath) + " " + (Join-ArgumentLine $arguments)
    Set-Content -LiteralPath (Join-Path $logDir "command.txt") -Value $commandLine -Encoding UTF8
    $run | Select-Object Index, ExperimentName, RunId, LogDir, StdoutLog, StderrLog |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logDir "metadata.json") -Encoding UTF8
    return $run
}

function Complete-FinishedRuns {
    param([Parameter(Mandatory = $true)][array]$Running)
    $remaining = @()
    foreach ($item in $Running) {
        if ($item.Process.HasExited) {
            $exitCode = $item.Process.ExitCode
            Set-Content -LiteralPath (Join-Path $item.Run.LogDir "exit_code.txt") -Value $exitCode -Encoding UTF8
            if ($exitCode -eq 0) {
                $completionMarker = Join-Path $RepoRoot "onpolicy\scripts\results\MPE\simple_spread\mappo\$($item.Run.ExperimentName)\completed.ok"
                New-Item -ItemType File -Path $completionMarker -Force | Out-Null
                Write-Host ("Completed {0}" -f $item.Run.ExperimentName)
            }
            else {
                Write-Warning ("Failed {0} with exit code {1}. Logs: {2}" -f $item.Run.ExperimentName, $exitCode, $item.Run.LogDir)
            }
        }
        else {
            $remaining += $item
        }
    }
    return $remaining
}

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Python executable not found: $PythonPath"
}
if (-not (Test-Path -LiteralPath $TrainScript)) {
    throw "Training entrypoint not found: $TrainScript"
}

$runs = @()
$index = 1
foreach ($agentCount in $AgentCounts) {
    foreach ($actorArch in $ActorArchitectures) {
        foreach ($seed in $Seeds) {
            $experimentName = "mpe_paper_a${agentCount}_${actorArch}_seed${seed}"
            if (Test-ExperimentCompleted $experimentName) {
                Write-Host "Skipping completed $experimentName"
                continue
            }
            if (Test-ExperimentActive $experimentName) {
                Write-Host "Skipping active $experimentName"
                continue
            }
            $runs += New-TrainingRun -Index $index -AgentCount $agentCount -ActorArch $actorArch -Seed $seed
            $index += 1
        }
    }
}

Write-Host ("Prepared {0} remaining paper runs with ParallelJobs={1}, RolloutThreads={2}." -f $runs.Count, $ParallelJobs, $RolloutThreads)

$running = @()
foreach ($run in $runs) {
    while ($running.Count -ge $ParallelJobs) {
        Start-Sleep -Seconds 10
        $running = @(Complete-FinishedRuns -Running $running)
    }

    Write-Host ("Starting {0}. Logs: {1}" -f $run.ExperimentName, $run.LogDir)
    $argumentLine = Join-ArgumentLine $run.Arguments
    $process = Start-Process -FilePath $PythonPath -ArgumentList $argumentLine -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput $run.StdoutLog -RedirectStandardError $run.StderrLog -WindowStyle Hidden -PassThru
    try {
        $process.PriorityClass = "AboveNormal"
    }
    catch {
        Write-Warning ("Could not raise priority for {0}: {1}" -f $run.ExperimentName, $_.Exception.Message)
    }
    $running = @($running + [PSCustomObject]@{ Run = $run; Process = $process })
}

while ($running.Count -gt 0) {
    Start-Sleep -Seconds 10
    $running = @(Complete-FinishedRuns -Running $running)
}

Write-Host "Remaining paper queue completed."
