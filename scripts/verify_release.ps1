# Synapse release smoke verification (non-interactive checks)
# Usage: .\scripts\verify_release.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$exe = Join-Path $PSScriptRoot "..\dist\Synapse.exe"
if (-not (Test-Path $exe)) {
    Write-Host "Synapse.exe not found. Run .\build.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host "Verifying $exe" -ForegroundColor Cyan

function Invoke-Synapse {
    param([string[]]$CommandArgs)
    $output = & $exe @CommandArgs 2>&1
    $code = $LASTEXITCODE
    return @{ Output = ($output | Out-String); Code = $code }
}

$help = Invoke-Synapse -CommandArgs @()
if ($help.Code -ne 1 -and $help.Output -notmatch "Synapse cognitive monitoring launcher") {
    Write-Host "Help output check failed." -ForegroundColor Red
    Write-Host $help.Output
    exit 1
}
if ($help.Output -notmatch "showcase") {
    Write-Host "Help output missing showcase command." -ForegroundColor Red
    Write-Host $help.Output
    exit 1
}
Write-Host "[ok] --help" -ForegroundColor Green

$data = Invoke-Synapse -CommandArgs @("data")
if ($data.Code -ne 0 -or $data.Output -notmatch "app_data_dir") {
    Write-Host "data command failed." -ForegroundColor Red
    Write-Host $data.Output
    exit 1
}
Write-Host "[ok] data" -ForegroundColor Green

$settings = Invoke-Synapse -CommandArgs @("settings")
if ($settings.Code -ne 0 -or $settings.Output -notmatch "camera_index") {
    Write-Host "settings command failed." -ForegroundColor Red
    Write-Host $settings.Output
    exit 1
}
Write-Host "[ok] settings" -ForegroundColor Green

Write-Host ""
Write-Host "Automated smoke checks passed." -ForegroundColor Green
Write-Host "Manual pilot checks still required: onboard, monitor, replay, 30-minute soak." -ForegroundColor Yellow
