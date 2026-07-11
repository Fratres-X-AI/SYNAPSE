# Synapse one-file Windows build (PyInstaller)
# Usage: .\build.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Synapse build - PyInstaller one-file" -ForegroundColor Cyan

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

$pyinstaller = & $python -m PyInstaller --version 2>$null
if (-not $pyinstaller) {
    Write-Host "PyInstaller not found. Install dev requirements first:" -ForegroundColor Yellow
    Write-Host "  $python -m pip install -r requirements-dev.txt"
    exit 1
}

$distDir = Join-Path $PSScriptRoot "dist"
$buildDir = Join-Path $PSScriptRoot "build"

if (Test-Path $distDir) { Remove-Item $distDir -Recurse -Force }
if (Test-Path $buildDir) { Remove-Item $buildDir -Recurse -Force }

& $python -m PyInstaller --clean Synapse.spec

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

$exe = Join-Path $distDir "Synapse.exe"
Write-Host ""
Write-Host "Build complete: $exe" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "  .\scripts\verify_release.ps1"
Write-Host "  .\dist\Synapse.exe first-run"
Write-Host ""
Write-Host "Usage:"
Write-Host "  .\dist\Synapse.exe first-run"
Write-Host "  .\dist\Synapse.exe onboard"
Write-Host "  .\dist\Synapse.exe monitor"
Write-Host "  .\dist\Synapse.exe showcase"
Write-Host "  .\dist\Synapse.exe replay"
Write-Host "  .\dist\Synapse.exe fusion"
Write-Host "  .\dist\Synapse.exe data"
Write-Host "  .\dist\Synapse.exe pilot-summary"
Write-Host '  .\dist\Synapse.exe --tray'
