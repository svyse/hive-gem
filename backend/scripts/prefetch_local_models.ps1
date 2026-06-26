# Run from backend venv:
#   .\.my_rlhf_v6\Scripts\Activate.ps1
#   .\scripts\prefetch_local_models.ps1
#
# Or copy this file into backend\scripts first.

$ErrorActionPreference = "Stop"

Write-Host "Installing faster Hugging Face download helper if available..."
python -m pip install -U "huggingface_hub[hf_xet]" hf_xet

$env:HF_HOME = "$HOME\.cache\huggingface"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

$scriptPath = Join-Path $PSScriptRoot "prefetch_local_models.py"
if (!(Test-Path $scriptPath)) {
  $scriptPath = Join-Path (Get-Location) "prefetch_local_models.py"
}

python $scriptPath
