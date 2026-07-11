# One-time Windows setup: venv, dependencies, desktop shortcut
# Usage: .\scripts\setup_windows.ps1

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

Write-Host ""
Write-Host "Synapse Windows setup" -ForegroundColor Cyan
Write-Host "=====================" -ForegroundColor Cyan
Write-Host ""

& (Join-Path $PSScriptRoot "check_python.ps1")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
$venvPip = Join-Path $Root ".venv\Scripts\pip.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment (.venv)..." -ForegroundColor Cyan
    $pyCmd = "python"
    $py311 = Get-Command "py" -ErrorAction SilentlyContinue
    if ($py311) {
        try {
            & py -3.11 -c "import sys" 2>$null
            if ($LASTEXITCODE -eq 0) { $pyCmd = "py -3.11" }
        } catch { }
    }
    if ($pyCmd -eq "python") {
        & python -m venv .venv
    } else {
        Invoke-Expression "$pyCmd -m venv .venv"
    }
    if (-not (Test-Path $venvPython)) {
        Write-Host "Failed to create .venv" -ForegroundColor Red
        exit 1
    }
    Write-Host "[ok] Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "[ok] Using existing .venv" -ForegroundColor Green
}

Write-Host "Installing dependencies (this may take a few minutes)..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip
& $venvPip install -r (Join-Path $Root "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    Write-Host "pip install failed." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "[ok] Dependencies installed" -ForegroundColor Green

& $venvPython (Join-Path $Root "scripts\build_icon.py")
if ($LASTEXITCODE -ne 0) {
    Write-Host "Icon build failed (non-fatal)." -ForegroundColor Yellow
}

& (Join-Path $PSScriptRoot "install_desktop_shortcut.ps1")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Double-click the Synapse icon on your desktop, or run:" -ForegroundColor White
Write-Host "  .\Launch Synapse.bat" -ForegroundColor White
Write-Host ""
Write-Host "First time? Choose First Run in the Synapse window." -ForegroundColor DarkGray
Write-Host ""
