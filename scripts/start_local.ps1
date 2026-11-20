# start_local.ps1 - Launch UCMEC Research Platform Local Server
[CmdletBinding()]
param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$BuildFrontend = $false
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Starting UCMEC Research Platform (Backend + Dashboard)" -ForegroundColor Cyan
Write-Host " Location: $ProjectRoot" -ForegroundColor DarkGray
Write-Host " Binding: http://$HostAddress`:$Port" -ForegroundColor DarkGray
Write-Host "============================================================" -ForegroundColor Cyan

# Set working directory to project root
Set-Location -Path $ProjectRoot

# 1. Verify / Build Frontend
$DistIndex = Join-Path $ProjectRoot "frontend\dist\index.html"
if ($BuildFrontend -or (-not (Test-Path $DistIndex))) {
    Write-Host "[1/3] Building Frontend Production Bundle..." -ForegroundColor Yellow
    npm --prefix frontend run build
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Frontend build failed with exit code $LASTEXITCODE." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "Frontend build completed successfully." -ForegroundColor Green
} else {
    Write-Host "[1/3] Frontend Production Bundle verified." -ForegroundColor Green
}

# 2. Check Port Availability
$PortOccupied = $false
try {
    $TcpListener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Parse($HostAddress), $Port)
    $TcpListener.Start()
    $TcpListener.Stop()
} catch {
    $PortOccupied = $true
}

if ($PortOccupied) {
    Write-Host "WARNING: Port $Port on $HostAddress is already in use." -ForegroundColor Yellow
    Write-Host "Please terminate the occupying process or pass a different port via -Port <number>." -ForegroundColor Yellow
}

# 3. Launch Backend with Integrated Durable Worker
Write-Host "[2/3] Initializing SQLite database and durable experiment worker..." -ForegroundColor Cyan
Write-Host "Architecture Note: The durable experiment queue worker runs inside the application process." -ForegroundColor DarkGray
Write-Host "[3/3] Starting Uvicorn Server at http://$HostAddress`:$Port" -ForegroundColor Green
Write-Host "Press Ctrl+C to terminate server.`n" -ForegroundColor DarkGray

conda run --no-capture-output -n ucmec python -m uvicorn backend.app:app --host $HostAddress --port $Port
exit $LASTEXITCODE
