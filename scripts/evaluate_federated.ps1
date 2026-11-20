# evaluate_federated.ps1 - Evaluate Saved Federated Policy on Held-Out Scenarios
[CmdletBinding()]
param(
    [string]$CheckpointPath = "",
    [int]$NumSteps = 50,
    [int]$Seed = 999,
    [string]$Preset = "balanced",
    [int]$NumAgents = 10,
    [string]$Solver = "CLARABEL"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Evaluating Saved Policy Checkpoint on Held-Out Scenario" -ForegroundColor Cyan
Write-Host " Steps: $NumSteps | Held-Out Seed: $Seed | Preset: $Preset" -ForegroundColor DarkGray
Write-Host "============================================================" -ForegroundColor Cyan

Set-Location -Path $ProjectRoot

# If no checkpoint specified, discover the latest global actor checkpoint
if (-not $CheckpointPath) {
    $Candidates = Get-ChildItem -Path (Join-Path $ProjectRoot "experiments_storage\federated_runs") -Filter "global_actor_round_*.pt" -Recurse |
        Sort-Object LastWriteTime -Descending
    if ($Candidates.Count -gt 0) {
        $CheckpointPath = $Candidates[0].FullName
        Write-Host "Discovered latest checkpoint: $CheckpointPath" -ForegroundColor Yellow
    } else {
        Write-Host "ERROR: No checkpoint found in experiments_storage/federated_runs." -ForegroundColor Red
        Write-Host "Please pass an explicit checkpoint via -CheckpointPath <path>." -ForegroundColor Red
        exit 1
    }
}

if (-not (Test-Path $CheckpointPath)) {
    Write-Host "ERROR: Specified checkpoint does not exist: $CheckpointPath" -ForegroundColor Red
    exit 1
}

Write-Host "Running evaluation CLI on held-out seed $Seed..." -ForegroundColor Cyan

conda run --no-capture-output -n ucmec python evaluate.py `
    --checkpoint "$CheckpointPath" `
    --seeds $Seed `
    --num_steps $NumSteps `
    --preset $Preset `
    --num_agents $NumAgents `
    --solver_name $Solver

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nPolicy Checkpoint Evaluation FAILED with code $LASTEXITCODE." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`nEvaluation Completed Successfully." -ForegroundColor Green
exit 0
