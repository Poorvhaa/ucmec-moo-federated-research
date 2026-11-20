# doctor.ps1 - UCMEC Platform Environment & Preflight Health Verifier
[CmdletBinding()]
param()

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " UCMEC Environment & Preflight Diagnostic Doctor" -ForegroundColor Cyan
Write-Host " Project Root: $ProjectRoot" -ForegroundColor DarkGray
Write-Host "============================================================" -ForegroundColor Cyan

$FailedChecks = 0

# Helper to execute python inside conda ucmec
function Invoke-UcmecPython([string]$code) {
    $wrappedCode = "import sys; sys.path.insert(0, r'$ProjectRoot'); " + $code
    $tempFile = [System.IO.Path]::GetTempFileName() + ".py"
    try {
        [System.IO.File]::WriteAllText($tempFile, $wrappedCode)
        $out = conda run --no-capture-output --cwd $ProjectRoot -n ucmec python $tempFile 2>&1
        $codeExit = $LASTEXITCODE
        return @{ ExitCode = $codeExit; Output = $out }
    } finally {
        if (Test-Path $tempFile) { Remove-Item -Path $tempFile -Force }
    }
}

# 1. Check Python executable in conda env 'ucmec'
Write-Host "`n[1/6] Checking Conda Environment 'ucmec'..." -NoNewline
$PyVerRes = Invoke-UcmecPython "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
if ($PyVerRes.ExitCode -eq 0) {
    Write-Host " OK ($($PyVerRes.Output))" -ForegroundColor Green
} else {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Host "  Could not execute python in conda environment 'ucmec'." -ForegroundColor Yellow
    $FailedChecks++
}

# 2. Check Core Python Imports
Write-Host "[2/6] Checking Critical Dependencies (torch, cvxpy, clarabel, fastapi)..." -NoNewline
$ImportRes = Invoke-UcmecPython "import torch, cvxpy, clarabel, fastapi, pydantic, numpy; print('OK')"
if ($ImportRes.ExitCode -eq 0) {
    Write-Host " OK" -ForegroundColor Green
} else {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Host "  $($ImportRes.Output)" -ForegroundColor Yellow
    $FailedChecks++
}

# 3. Check CLARABEL Conic Solver in CVXPY
Write-Host "[3/6] Verifying CLARABEL Conic Solver in CVXPY..." -NoNewline
$SolverRes = Invoke-UcmecPython "import cvxpy as cp; assert 'CLARABEL' in cp.installed_solvers(), 'CLARABEL solver missing'; print('OK')"
if ($SolverRes.ExitCode -eq 0) {
    Write-Host " OK" -ForegroundColor Green
} else {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Host "  $($SolverRes.Output)" -ForegroundColor Yellow
    $FailedChecks++
}

# 4. Check Frontend Build Existence
Write-Host "[4/6] Checking Frontend Static Build (frontend/dist/index.html)..." -NoNewline
$DistIndex = Join-Path $ProjectRoot "frontend\dist\index.html"
if (Test-Path $DistIndex) {
    Write-Host " OK (Found)" -ForegroundColor Green
} else {
    Write-Host " WARNING (Missing)" -ForegroundColor Yellow
    Write-Host "  frontend/dist/index.html not found. Run 'npm --prefix frontend run build' before serving UI." -ForegroundColor Yellow
}

# 5. Check Writable Storage Directories
Write-Host "[5/6] Verifying Storage Directories..." -NoNewline
$StorageDirs = @(
    (Join-Path $ProjectRoot "experiments_storage"),
    (Join-Path $ProjectRoot "experiments_storage\runs"),
    (Join-Path $ProjectRoot "experiments_storage\federated_runs"),
    (Join-Path $ProjectRoot "results")
)
$StorageOk = $true
foreach ($dir in $StorageDirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    $testFile = Join-Path $dir ".doctor_test_write.tmp"
    try {
        [System.IO.File]::WriteAllText($testFile, "test")
        Remove-Item -Path $testFile -Force
    } catch {
        $StorageOk = $false
    }
}
if ($StorageOk) {
    Write-Host " OK (Writable)" -ForegroundColor Green
} else {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Host "  One or more storage directories are not writable." -ForegroundColor Yellow
    $FailedChecks++
}

# 6. Check Core Configuration & Presets
Write-Host "[6/6] Verifying Multi-Objective Presets & Federated Scenarios..." -NoNewline
$CfgRes = Invoke-UcmecPython "import warnings; warnings.filterwarnings('ignore'); from moo.config import PRESETS; from federated.scenarios import generate_federation_scenarios; assert len(PRESETS) >= 4; print(f'{len(PRESETS)} presets verified')"
if ($CfgRes.ExitCode -eq 0) {
    Write-Host " OK" -ForegroundColor Green
} else {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Host "  $($CfgRes.Output)" -ForegroundColor Yellow
    $FailedChecks++
}

Write-Host "------------------------------------------------------------" -ForegroundColor DarkGray
if ($FailedChecks -eq 0) {
    Write-Host " Doctor Diagnosis: ALL SYSTEMS HEALTHY & OPERATIONAL." -ForegroundColor Green
    exit 0
} else {
    Write-Host " Doctor Diagnosis: $FailedChecks CHECK(S) FAILED. Please review issues above." -ForegroundColor Red
    exit 1
}
