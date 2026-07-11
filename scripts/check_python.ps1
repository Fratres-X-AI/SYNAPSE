# Synapse Python pre-flight check (run before pip install)
# Usage: .\scripts\check_python.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$MinMajor = 3
$MinMinor = 11
$MaxMajor = 3
$MaxMinor = 12

function Get-PythonVersion {
    param([string]$Command)
    try {
        $raw = & $Command -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $raw) {
            return $null
        }
        return [version]$raw.Trim()
    } catch {
        return $null
    }
}

function Test-VersionInRange {
    param([version]$Version)
    $min = [version]"$MinMajor.$MinMinor.0"
    $max = [version]"$MaxMajor.$MaxMinor.99"
    return ($Version -ge $min -and $Version -le $max)
}

Write-Host "Synapse Python pre-flight check" -ForegroundColor Cyan
Write-Host "Supported range: Python $MinMajor.$MinMinor - $MaxMajor.$MaxMinor (MediaPipe 0.10.21 wheels)" -ForegroundColor DarkGray
Write-Host ""

$candidates = @(
    @{ Label = "active python"; Command = "python" },
    @{ Label = "py -3.11"; Command = "py -3.11" },
    @{ Label = "py -3.12"; Command = "py -3.12" }
)

$usable = @()
foreach ($candidate in $candidates) {
    $version = Get-PythonVersion -Command $candidate.Command
    if ($null -eq $version) {
        Write-Host "[skip] $($candidate.Label) not found" -ForegroundColor DarkGray
        continue
    }

    $versionText = $version.ToString()
    if (Test-VersionInRange -Version $version) {
        Write-Host "[ok]   $($candidate.Label) -> $versionText" -ForegroundColor Green
        $usable += [pscustomobject]@{
            Label = $candidate.Label
            Command = $candidate.Command
            Version = $version
        }
    } elseif ($version.Major -eq 3 -and $version.Minor -ge 13) {
        Write-Host "[fail] $($candidate.Label) -> $versionText (too new; use 3.11 or 3.12)" -ForegroundColor Red
    } elseif ($version.Major -lt $MinMajor -or ($version.Major -eq $MinMajor -and $version.Minor -lt $MinMinor)) {
        Write-Host "[fail] $($candidate.Label) -> $versionText (too old; do not use Python 3.7-3.10)" -ForegroundColor Red
    } else {
        Write-Host "[fail] $($candidate.Label) -> $versionText (outside supported range)" -ForegroundColor Red
    }
}

Write-Host ""

if ($usable.Count -eq 0) {
    Write-Host "No supported Python found." -ForegroundColor Red
    Write-Host ""
    Write-Host "Install Python 3.11 from https://www.python.org/downloads/release/python-3119/" -ForegroundColor Yellow
    Write-Host "Then create a venv and install dependencies:" -ForegroundColor Yellow
    Write-Host "  py -3.11 -m venv .venv" -ForegroundColor Yellow
    Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "  python -m pip install --upgrade pip" -ForegroundColor Yellow
    Write-Host "  pip install -r requirements.txt" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Non-developers can skip Python and use Synapse.exe from docs/windows_install.md" -ForegroundColor Yellow
    exit 1
}

$preferred = $usable | Where-Object { $_.Version.Major -eq 3 -and $_.Version.Minor -eq 11 } | Select-Object -First 1
if (-not $preferred) {
    $preferred = $usable | Select-Object -First 1
}

Write-Host "Recommended interpreter: $($preferred.Label) ($($preferred.Version))" -ForegroundColor Green
Write-Host ""
Write-Host "Suggested setup:" -ForegroundColor Cyan

if ($preferred.Command -eq "python") {
    Write-Host "  python -m venv .venv" -ForegroundColor White
} else {
    Write-Host "  $($preferred.Command) -m venv .venv" -ForegroundColor White
}

Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  python -m pip install --upgrade pip" -ForegroundColor White
Write-Host "  pip install -r requirements.txt" -ForegroundColor White
Write-Host ""
Write-Host "Pre-flight check passed." -ForegroundColor Green
