# smoke_federated.ps1 - Execute Federated Multi-Agent Learning Smoke Run
[CmdletBinding()]
param(
    [int]$NumClients = 3,
    [int]$NumRounds = 2,
    [string]$ScenarioType = "iid",
    [string]$Preset = "balanced",
    [int]$Seed = 42,
    [string]$Solver = "CLARABEL"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Executing Federated Multi-Agent Learning Smoke Run" -ForegroundColor Cyan
Write-Host " Clients: $NumClients | Rounds: $NumRounds | Scenario: $ScenarioType | Solver: $Solver" -ForegroundColor DarkGray
Write-Host "============================================================" -ForegroundColor Cyan

Set-Location -Path $ProjectRoot

$ExpName = "smoke_fed_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$OutputDir = Join-Path $ProjectRoot "experiments_storage\federated_runs\$ExpName"

conda run --no-capture-output -n ucmec python federated/runner.py `
    --experiment_name $ExpName `
    --fed_num_clients $NumClients `
    --fed_num_rounds $NumRounds `
    --fed_min_clients 1 `
    --fed_participation_rate 1.0 `
    --fed_dropout_prob 0.0 `
    --fed_scenario_type $ScenarioType `
    --moo_preset $Preset `
    --seed $Seed `
    --solver_name $Solver `
    --channel_mode legacy `
    --fed_output_dir $OutputDir `
    --cuda

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nFederated Smoke Run FAILED with exit code $LASTEXITCODE." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`nFederated Smoke Run COMPLETED SUCCESSFULLY." -ForegroundColor Green
Write-Host "Checkpoints & manifests written to: $OutputDir" -ForegroundColor DarkGray
exit 0
