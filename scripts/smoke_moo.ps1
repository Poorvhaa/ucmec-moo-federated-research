# smoke_moo.ps1 - Execute Multi-Objective Optimization Smoke Training Run
[CmdletBinding()]
param(
    [string]$Preset = "balanced",
    [int]$NumSteps = 200,
    [int]$NumAgents = 10,
    [int]$Seed = 42,
    [string]$Solver = "CLARABEL"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Executing Multi-Objective Optimization Smoke Training Run" -ForegroundColor Cyan
Write-Host " Preset: $Preset | Steps: $NumSteps | Agents: $NumAgents | Solver: $Solver" -ForegroundColor DarkGray
Write-Host "============================================================" -ForegroundColor Cyan

Set-Location -Path $ProjectRoot

$ExpName = "smoke_moo_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

conda run --no-capture-output -n ucmec python train/train.py `
    --algorithm_name mappo `
    --experiment_name $ExpName `
    --num_agents $NumAgents `
    --num_env_steps $NumSteps `
    --episode_length 200 `
    --seed $Seed `
    --moo_preset $Preset `
    --solver_name $Solver `
    --channel_mode legacy `
    --log_interval 1 `
    --n_rollout_threads 1 `
    --n_training_threads 1 `
    --cuda

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nMOO Smoke Training FAILED with exit code $LASTEXITCODE." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`nMOO Smoke Training COMPLETED SUCCESSFULLY." -ForegroundColor Green
Write-Host "Artifacts written to results/MyEnv/MyEnv/mappo/$ExpName" -ForegroundColor DarkGray
exit 0
