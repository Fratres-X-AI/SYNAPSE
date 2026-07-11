# Create a Synapse desktop shortcut
# Usage: .\scripts\install_desktop_shortcut.ps1

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$Desktop = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $Desktop "Synapse.lnk"

$IconFile = Join-Path $Root "assets\synapse.ico"
$BatFile = Join-Path $Root "Launch Synapse.bat"
$ExeRoot = Join-Path $Root "Synapse.exe"
$ExeDist = Join-Path $Root "dist\Synapse.exe"

if (Test-Path $IconFile) {
    $IconLocation = "$IconFile,0"
} elseif (Test-Path $ExeRoot) {
    $IconLocation = "$ExeRoot,0"
} elseif (Test-Path $ExeDist) {
    $IconLocation = "$ExeDist,0"
} else {
    $IconLocation = "$env:SystemRoot\System32\imageres.dll,109"
}

# Prefer the .bat launcher when developing from source so Windows uses synapse.ico,
# not an old icon baked into dist\Synapse.exe.
if ((Test-Path $BatFile) -and (Test-Path $IconFile)) {
    $TargetPath = $BatFile
    $Arguments = ""
} elseif (Test-Path $ExeRoot) {
    $TargetPath = $ExeRoot
    $Arguments = "home"
} elseif (Test-Path $ExeDist) {
    $TargetPath = $ExeDist
    $Arguments = "home"
} elseif (Test-Path $BatFile) {
    $TargetPath = $BatFile
    $Arguments = ""
} else {
    Write-Host "Nothing to shortcut: expected Synapse.exe or Launch Synapse.bat in $Root" -ForegroundColor Red
    exit 1
}

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $TargetPath
$Shortcut.Arguments = $Arguments
$Shortcut.WorkingDirectory = $Root
$Shortcut.WindowStyle = 1
$Shortcut.Description = "Synapse cognitive presence monitoring"
$Shortcut.IconLocation = $IconLocation
$Shortcut.Save()

Write-Host "[ok] Desktop shortcut created:" -ForegroundColor Green
Write-Host "     $ShortcutPath" -ForegroundColor White
Write-Host "     -> $TargetPath $Arguments" -ForegroundColor DarkGray
