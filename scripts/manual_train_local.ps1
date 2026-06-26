param(
  [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location (Join-Path $RepoRoot "backend")

$env:PYTORCH_CUDA_ALLOC_CONF = if ($env:PYTORCH_CUDA_ALLOC_CONF) { $env:PYTORCH_CUDA_ALLOC_CONF } else { "expandable_segments:True" }
$env:CUDA_VISIBLE_DEVICES = "0"
if ($Force) { $env:FORCE_TRAIN = "1" }

python -m app.training.worker
