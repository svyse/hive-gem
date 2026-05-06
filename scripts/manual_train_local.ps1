param(
  [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$argsList = @()
if ($Force) { $argsList += "--force" }
python .\scripts\manual_train_local.py @argsList
