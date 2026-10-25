<#
.SYNOPSIS
  Install Sia's Windows integration for the current user (no admin rights needed).

.DESCRIPTION
  - Desktop and Start-menu shortcut "Sia" (pythonw: no console window).
  - Startup entry: Sia starts quietly with Windows (no window, no model loaded); it costs
    almost nothing until summoned with Ctrl+Alt+Space or the icon.
  - File Explorer right-click entry "Ask Sia" on every file: opens Sia with the
    file attached (registered under HKCU\Software\Classes\*\shell\AskSia).
  Global hotkey Ctrl+Alt+Space works whenever Sia is running (no install needed).
  Remove everything with app\uninstall.ps1.
#>
$ErrorActionPreference = "Stop"
$app = $PSScriptRoot
$python = (& python -c "import sys; print(sys.executable)").Trim()
$pythonw = Join-Path (Split-Path $python) "pythonw.exe"
& $python -c "import psutil, numpy, pypdf, PIL" 2>$null
if ($LASTEXITCODE -ne 0) { throw "$python lacks packages: pip install psutil numpy pypdf pillow" }
if (-not (Test-Path $pythonw)) { throw "pythonw.exe not found next to $python" }
$icon = "$app\static\logo.ico"

$shell = New-Object -ComObject WScript.Shell
# the app was called Sprout before: remove its old shortcuts (only ones that start this server.py) and menu entry
foreach ($dir in @("Desktop", "Programs", "Startup")) {
    $old = Join-Path ([Environment]::GetFolderPath($dir)) "Sprout.lnk"
    if ((Test-Path $old) -and ($shell.CreateShortcut($old).Arguments -like "*$app\server.py*")) {
        Remove-Item $old; Write-Host "removed old shortcut: $old"
    }
}
$shellKey = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Software\Classes\*\shell', $true)
if ($shellKey -and ($shellKey.GetSubKeyNames() -contains "AskSprout")) {
    $shellKey.DeleteSubKeyTree("AskSprout"); Write-Host "removed old Explorer menu entry 'Ask Sprout'"
}
if ($shellKey) { $shellKey.Close() }
$links = @(
    @{ path = (Join-Path ([Environment]::GetFolderPath("Desktop")) "Sia.lnk"); args = "" },
    @{ path = (Join-Path ([Environment]::GetFolderPath("Programs")) "Sia.lnk"); args = "" },
    # start quietly with Windows: no window and no model until Ctrl+Alt+Space or the icon
    @{ path = (Join-Path ([Environment]::GetFolderPath("Startup")) "Sia.lnk"); args = " --background" }
)
foreach ($l in $links) {
    $lnk = $shell.CreateShortcut($l.path)
    $lnk.TargetPath = $pythonw
    $lnk.Arguments = "`"$app\server.py`"" + $l.args
    $lnk.WorkingDirectory = Split-Path $app
    $lnk.IconLocation = $icon
    $lnk.Description = "Sia: private offline AI assistant"
    $lnk.Save()
    Write-Host "shortcut: $($l.path)$($l.args)"
}

# .NET registry API: "*" is the literal all-files class key here, never a wildcard
$hk = [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey('Software\Classes\*\shell\AskSia')
$hk.SetValue("", "Ask Sia"); $hk.SetValue("Icon", $icon); $hk.Close()
$hk = [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey('Software\Classes\*\shell\AskSia\command')
$hk.SetValue("", "`"$pythonw`" `"$app\server.py`" --open-file `"%1`""); $hk.Close()
Write-Host "Explorer menu: right-click a file > Ask Sia (Windows 11: under 'Show more options')"
