[CmdletBinding()]
param(
    [string]$PythonPath = "C:\Users\Lenovo\anaconda3\envs\lge_cmr\python.exe",
    [ValidateRange(1, 64)]
    [int]$ParallelJobs = 1,
    [int[]]$AgentCounts = @(3, 5, 7),
    [string[]]$ActorArchitectures = @("ann", "snn_lif", "snn_at"),
    [int[]]$Seeds = @(1, 2, 3),
    [int]$NumEnvSteps = 1000000,
    [int]$EpisodeLength = 25,
    [int]$RolloutThreads = 4
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

function Test-ExecutableAvailable {
    param([Parameter(Mandatory = $true)][string]$PathOrCommand)

    if (Test-Path -LiteralPath $PathOrCommand) {
        return
    }

    if ($null -ne (Get-Command $PathOrCommand -ErrorAction SilentlyContinue)) {
        return
    }

    throw "Python executable not found: $PathOrCommand"
}

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

function New-UniqueLogDirectory {
    param([Parameter(Mandatory = $true)][string]$RunId)

    $candidate = Join-Path $LogsRoot $RunId
    $suffix = 1
    while (Test-Path -LiteralPath $candidate) {
        $candidate = Join-Path $LogsRoot ("{0}_{1:00}" -f $RunId, $suffix)
        $suffix += 1
    }

    New-Item -ItemType Directory -Path $candidate -Force | Out-Null
    return $candidate
}

function New-TrainingRun {
    param(
        [Parameter(Mandatory = $true)][int]$Index,
        [Parameter(Mandatory = $true)][int]$AgentCount,
        [Parameter(Mandatory = $true)][string]$ActorArch,
        [Parameter(Mandatory = $true)][int]$Seed
    )

    $runId = "{0}_mpe_paper_a{1}_{2}_s{3}_{4:00}" -f $Timestamp, $AgentCount, $ActorArch, $Seed, $Index
    $logDir = New-UniqueLogDirectory $runId
    $arguments = @(
        "-u",
        $TrainScript,
        "--env_name", "MPE",
        "--algorithm_name", "mappo",
        "--experiment_name", "mpe_paper_a${AgentCount}_${ActorArch}_seed${Seed}",
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
        Index     = $Index
        RunId     = $runId
        Agents    = $AgentCount
        Landmarks = $AgentCount
        ActorArch = $ActorArch
        Seed      = $Seed
        LogDir    = $logDir
        StdoutLog = Join-Path $logDir "stdout.log"
        StderrLog = Join-Path $logDir "stderr.log"
        Arguments = $arguments
    }

    $commandLine = (ConvertTo-QuotedArgument $PythonPath) + " " + (Join-ArgumentLine $arguments)
    Set-Content -LiteralPath (Join-Path $logDir "command.txt") -Value $commandLine -Encoding UTF8
    $run | Select-Object Index, RunId, Agents, Landmarks, ActorArch, Seed, LogDir, StdoutLog, StderrLog |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logDir "metadata.json") -Encoding UTF8

    return $run
}

function Complete-FinishedRuns {
    param(
        [Parameter(Mandatory = $true)][array]$Running,
        [Parameter(Mandatory = $true)][System.Collections.ArrayList]$Failures
    )

    $remaining = @()
    foreach ($item in $Running) {
        if ($item.Process.HasExited) {
            $exitCode = $item.Process.ExitCode
            Set-Content -LiteralPath (Join-Path $item.Run.LogDir "exit_code.txt") -Value $exitCode -Encoding UTF8
            if ($exitCode -eq 0) {
                Write-Host ("Completed {0}" -f $item.Run.RunId)
            }
            else {
                Write-Warning ("Failed {0} with exit code {1}. Logs: {2}" -f $item.Run.RunId, $exitCode, $item.Run.LogDir)
                [void]$Failures.Add($item)
            }
        }
        else {
            $remaining += $item
        }
    }

    return $remaining
}

Test-ExecutableAvailable $PythonPath
if (-not (Test-Path -LiteralPath $TrainScript)) {
    throw "Training entrypoint not found: $TrainScript"
}

$env:WANDB_DISABLED = "true"
$env:PYTHONUNBUFFERED = "1"

$runs = @()
$runIndex = 1
foreach ($agentCount in $AgentCounts) {
    foreach ($actorArch in $ActorArchitectures) {
        foreach ($seed in $Seeds) {
            $runs += New-TrainingRun -Index $runIndex -AgentCount $agentCount -ActorArch $actorArch -Seed $seed
            $runIndex += 1
        }
    }
}

Write-Host ("Prepared {0} MPE paper queue runs. Logs root: {1}" -f $runs.Count, $LogsRoot)

if ($ParallelJobs -eq 1) {
    foreach ($run in $runs) {
        Write-Host ("[{0}/{1}] Starting {2}" -f $run.Index, $runs.Count, $run.RunId)
        Write-Host ("Logs: {0}" -f $run.LogDir)
        & $PythonPath @($run.Arguments) > $run.StdoutLog 2> $run.StderrLog
        $exitCode = $LASTEXITCODE
        Set-Content -LiteralPath (Join-Path $run.LogDir "exit_code.txt") -Value $exitCode -Encoding UTF8

        if ($exitCode -ne 0) {
            Write-Error ("Run failed with exit code {0}: {1}. Logs: {2}" -f $exitCode, $run.RunId, $run.LogDir) -ErrorAction Continue
            exit $exitCode
        }

        Write-Host ("Completed {0}" -f $run.RunId)
    }
}
else {
    Write-Host ("Parallel mode enabled with at most {0} Start-Process jobs." -f $ParallelJobs)
    $running = @()
    $failures = New-Object System.Collections.ArrayList

    foreach ($run in $runs) {
        while ($running.Count -ge $ParallelJobs) {
            Start-Sleep -Seconds 5
            $running = Complete-FinishedRuns -Running $running -Failures $failures
        }

        Write-Host ("Starting {0}. Logs: {1}" -f $run.RunId, $run.LogDir)
        $argumentLine = Join-ArgumentLine $run.Arguments
        $process = Start-Process -FilePath $PythonPath -ArgumentList $argumentLine -WorkingDirectory $RepoRoot `
            -RedirectStandardOutput $run.StdoutLog -RedirectStandardError $run.StderrLog -WindowStyle Hidden -PassThru
        $running += [PSCustomObject]@{
            Run     = $run
            Process = $process
        }
    }

    while ($running.Count -gt 0) {
        Start-Sleep -Seconds 5
        $running = Complete-FinishedRuns -Running $running -Failures $failures
    }

    if ($failures.Count -gt 0) {
        Write-Error ("{0} MPE paper queue run(s) failed." -f $failures.Count) -ErrorAction Continue
        exit 1
    }
}

Write-Host ("All MPE paper queue runs completed. Logs root: {0}" -f $LogsRoot)
