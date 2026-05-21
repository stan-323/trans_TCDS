[CmdletBinding()]
param(
    [string]$PythonPath = "C:\Users\Lenovo\anaconda3\envs\lge_cmr\python.exe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

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

Write-Host "=== nvidia-smi ==="
if ($null -ne (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue)) {
    & nvidia-smi
}
else {
    Write-Warning "nvidia-smi was not found on PATH."
}

Write-Host ""
Write-Host "=== torch CUDA probe ==="
Test-ExecutableAvailable $PythonPath

$probe = @'
import sys

print("python:", sys.executable)

try:
    import torch
except Exception as exc:
    print("torch_import_error:", repr(exc))
    raise

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("torch_cuda_version:", torch.version.cuda)
print("device_count:", torch.cuda.device_count())

for index in range(torch.cuda.device_count()):
    print(f"device_{index}:", torch.cuda.get_device_name(index))
'@

& $PythonPath -c $probe
exit $LASTEXITCODE
