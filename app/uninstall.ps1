<#
.SYNOPSIS
  Remove Sia's shortcuts and Explorer menu entry (models, chats and Library data stay;
  delete %USERPROFILE%\llm\app-data (or $env:LLM_HOME\app-data) to remove those too).
#>
foreach ($name in @("Sia", "Sprout")) {  # Sprout: the app's earlier name
    foreach ($dir in @("Desktop", "Programs", "Startup")) {
        $path = Join-Path ([Environment]::GetFolderPath($dir)) "$name.lnk"
        if (Test-Path $path) { Remove-Item $path; Write-Host "removed $path" }
    }
}
# .NET registry API: "*" is the literal all-files class key here, never a wildcard
$shellKey = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Software\Classes\*\shell', $true)
foreach ($key in @("AskSia", "AskSprout")) {
    if ($shellKey -and ($shellKey.GetSubKeyNames() -contains $key)) {
        $shellKey.DeleteSubKeyTree($key); Write-Host "removed Explorer menu entry $key"
    }
}
if ($shellKey) { $shellKey.Close() }
