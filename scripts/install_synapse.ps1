# Install Synapse from a built executable (or dev launcher)
# Usage: .\scripts\install_synapse.ps1

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$ExeDist = Join-Path $Root "dist\Synapse.exe"
$ExeRoot = Join-Path $Root "Synapse.exe"
$TargetExe = if (Test-Path $ExeDist) { $ExeDist } elseif (Test-Path $ExeRoot) { $ExeRoot } else { $null }

Write-Host ""
Write-Host "Synapse installer" -ForegroundColor Cyan
Write-Host "=================" -ForegroundColor Cyan
Write-Host ""

if ($null -eq $TargetExe) {
    Write-Host "Synapse.exe not found. Build first:" -ForegroundColor Yellow
    Write-Host "  .\build.ps1" -ForegroundColor White
    Write-Host ""
    Write-Host "Or run from source after setup:" -ForegroundColor Yellow
    Write-Host "  .\scripts\setup_windows.ps1" -ForegroundColor White
    exit 1
}

Write-Host "Using executable:" -ForegroundColor DarkGray
Write-Host "  $TargetExe" -ForegroundColor White

& (Join-Path $PSScriptRoot "install_desktop_shortcut.ps1")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$StartMenu = [Environment]::GetFolderPath("StartMenu")
$Programs = Join-Path $StartMenu "Programs"
$StartShortcut = Join-Path $Programs "Synapse.lnk"

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($StartShortcut)
$Shortcut.TargetPath = $TargetExe
$Shortcut.Arguments = "home"
$Shortcut.WorkingDirectory = $Root
$Shortcut.WindowStyle = 1
$Shortcut.Description = "Synapse cognitive presence monitoring"
$IconFile = Join-Path $Root "assets\synapse.ico"
if (Test-Path $IconFile) {
    $Shortcut.IconLocation = "$IconFile,0"
}
$Shortcut.Save()

Write-Host "[ok] Start menu shortcut created:" -ForegroundColor Green
Write-Host "     $StartShortcut" -ForegroundColor White

Write-Host ""
Write-Host "Installation complete." -ForegroundColor Green
Write-Host "Launch Synapse from the desktop or Start menu." -ForegroundColor White
Write-Host "First time: choose Get Started, complete onboarding, then Start Monitor." -ForegroundColor DarkGray
Write-Host ""
