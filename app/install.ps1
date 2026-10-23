<#
.SYNOPSIS
  Install Sprout's Windows integration for the current user (no admin rights needed).

.DESCRIPTION
  - Desktop and Start-menu shortcut "Sprout" (pythonw: no console window).
  - Startup entry: Sprout starts quietly with Windows (no window, no model loaded); it costs
    almost nothing until summoned with Ctrl+Alt+Space or the icon.
  - File Explorer right-click entry "Ask Sprout" on every file: opens Sprout with the
    file attached (registered under HKCU\Software\Classes\*\shell\AskSprout).
  Global hotkey Ctrl+Alt+Space works whenever Sprout is running (no install needed).
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
$links = @(
    @{ path = (Join-Path ([Environment]::GetFolderPath("Desktop")) "Sprout.lnk"); args = "" },
    @{ path = (Join-Path ([Environment]::GetFolderPath("Programs")) "Sprout.lnk"); args = "" },
    # start quietly with Windows: no window and no model until Ctrl+Alt+Space or the icon
    @{ path = (Join-Path ([Environment]::GetFolderPath("Startup")) "Sprout.lnk"); args = " --background" }
)
foreach ($l in $links) {
    $lnk = $shell.CreateShortcut($l.path)
    $lnk.TargetPath = $pythonw
    $lnk.Arguments = "`"$app\server.py`"" + $l.args
    $lnk.WorkingDirectory = Split-Path $app
    $lnk.IconLocation = $icon
    $lnk.Description = "Sprout: private offline AI assistant"
    $lnk.Save()
    Write-Host "shortcut: $($l.path)$($l.args)"
}

# .NET registry API: "*" is the literal all-files class key here, never a wildcard
$hk = [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey('Software\Classes\*\shell\AskSprout')
$hk.SetValue("", "Ask Sprout"); $hk.SetValue("Icon", $icon); $hk.Close()
$hk = [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey('Software\Classes\*\shell\AskSprout\command')
$hk.SetValue("", "`"$pythonw`" `"$app\server.py`" --open-file `"%1`""); $hk.Close()
Write-Host "Explorer menu: right-click a file > Ask Sprout (Windows 11: under 'Show more options')"
