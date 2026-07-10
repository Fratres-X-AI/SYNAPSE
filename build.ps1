# Synapse one-file Windows build (PyInstaller)
# Usage: .\build.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Synapse build — PyInstaller one-file" -ForegroundColor Cyan

if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
    Write-Host "PyInstaller not found. Install dev requirements first:" -ForegroundColor Yellow
    Write-Host "  pip install -r requirements-dev.txt"
    exit 1
}

$distDir = Join-Path $PSScriptRoot "dist"
$buildDir = Join-Path $PSScriptRoot "build"
$specDir = Join-Path $PSScriptRoot "*.spec"

if (Test-Path $distDir) { Remove-Item $distDir -Recurse -Force }
if (Test-Path $buildDir) { Remove-Item $buildDir -Recurse -Force }
Get-ChildItem $specDir -ErrorAction SilentlyContinue | Remove-Item -Force

pyinstaller `
    --onefile `
    --name Synapse `
    --console `
    --clean `
    --collect-all mediapipe `
    --hidden-import cv2 `
    --hidden-import numpy `
    --hidden-import PIL `
    --hidden-import pystray `
    --add-data "src;src" `
    --add-data "utils;utils" `
    synapse_launcher.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

$exe = Join-Path $distDir "Synapse.exe"
Write-Host ""
Write-Host "Build complete: $exe" -ForegroundColor Green
Write-Host ""
Write-Host "Usage:"
Write-Host "  .\dist\Synapse.exe onboard"
Write-Host "  .\dist\Synapse.exe monitor"
Write-Host "  .\dist\Synapse.exe replay"
Write-Host "  .\dist\Synapse.exe fusion"
Write-Host "  .\dist\Synapse.exe --tray"
