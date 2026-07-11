# Force-refresh the Synapse desktop icon (Windows caches aggressively)
# Usage: .\scripts\refresh_desktop_icon.ps1

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

Write-Host "Rebuilding icon..." -ForegroundColor Cyan
$python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }
& $python (Join-Path $Root "scripts\build_icon.py")

$Desktop = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $Desktop "Synapse.lnk"
if (Test-Path $ShortcutPath) {
    Remove-Item $ShortcutPath -Force
    Write-Host "Removed old desktop shortcut." -ForegroundColor DarkGray
}

& (Join-Path $PSScriptRoot "install_desktop_shortcut.ps1")

$iconCache = Join-Path $env:LOCALAPPDATA "IconCache.db"
if (Test-Path $iconCache) {
    try {
        Remove-Item $iconCache -Force
        Write-Host "Cleared IconCache.db" -ForegroundColor DarkGray
    } catch {
        Write-Host "Could not delete IconCache.db (Explorer may be locking it)." -ForegroundColor Yellow
    }
}

$explorerCache = Join-Path $env:LOCALAPPDATA "Microsoft\Windows\Explorer"
if (Test-Path $explorerCache) {
    Get-ChildItem $explorerCache -Filter "iconcache*" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
}

$ie4u = Join-Path $env:WINDIR "System32\ie4uinit.exe"
if (Test-Path $ie4u) {
    & $ie4u -show | Out-Null
}

Write-Host "Restarting Explorer to apply icon..." -ForegroundColor Cyan
Stop-Process -Name explorer -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 800
Start-Process explorer

Write-Host ""
Write-Host "Done. Check the Synapse icon on your desktop." -ForegroundColor Green
Write-Host "It should be a dark rounded square with an orange-to-purple gradient S." -ForegroundColor DarkGray
