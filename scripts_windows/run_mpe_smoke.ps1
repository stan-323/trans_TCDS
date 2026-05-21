[CmdletBinding()]
param(
    [string]$PythonPath = "C:\Users\Lenovo\anaconda3\envs\lge_cmr\python.exe",
    [ValidateRange(1, 64)]
    [int]$ParallelJobs = 1,
    [int]$NumEnvSteps = 50000
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
        [Parameter(Mandatory = $true)][string]$ActorArch
    )

    $runId = "{0}_mpe_smoke_a3_{1}_s1_{2:00}" -f $Timestamp, $ActorArch, $Index
    $logDir = New-UniqueLogDirectory $runId
    $arguments = @(
        "-u",
        $TrainScript,
        "--env_name", "MPE",
        "--algorithm_name", "mappo",
        "--experiment_name", "mpe_smoke_a3_${ActorArch}_seed1",
        "--scenario_name", "simple_spread",
        "--num_agents", "3",
        "--num_landmarks", "3",
        "--seed", "1",
        "--n_training_threads", "1",
        "--n_rollout_threads", "4",
        "--num_mini_batch", "1",
        "--episode_length", "25",
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
        ActorArch = $ActorArch
        LogDir    = $logDir
        StdoutLog = Join-Path $logDir "stdout.log"
        StderrLog = Join-Path $logDir "stderr.log"
        Arguments = $arguments
    }

    $commandLine = (ConvertTo-QuotedArgument $PythonPath) + " " + (Join-ArgumentLine $arguments)
    Set-Content -LiteralPath (Join-Path $logDir "command.txt") -Value $commandLine -Encoding UTF8
    $run | Select-Object Index, RunId, ActorArch, LogDir, StdoutLog, StderrLog |
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

$actorArchitectures = @("ann", "snn_lif", "snn_at")
$runs = @()
for ($i = 0; $i -lt $actorArchitectures.Count; $i++) {
    $runs += New-TrainingRun -Index ($i + 1) -ActorArch $actorArchitectures[$i]
}

Write-Host ("Prepared {0} MPE smoke runs. Logs root: {1}" -f $runs.Count, $LogsRoot)

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
        Write-Error ("{0} MPE smoke run(s) failed." -f $failures.Count) -ErrorAction Continue
        exit 1
    }
}

Write-Host ("All MPE smoke runs completed. Logs root: {0}" -f $LogsRoot)
