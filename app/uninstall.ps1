<#
.SYNOPSIS
  Remove Sprout's shortcuts and Explorer menu entry (models, chats and Library data stay;
  delete %USERPROFILE%\llm\app-data (or $env:LLM_HOME\app-data) to remove those too).
#>
foreach ($path in @((Join-Path ([Environment]::GetFolderPath("Desktop")) "Sprout.lnk"),
                    (Join-Path ([Environment]::GetFolderPath("Programs")) "Sprout.lnk"),
                    (Join-Path ([Environment]::GetFolderPath("Startup")) "Sprout.lnk"))) {
    if (Test-Path $path) { Remove-Item $path; Write-Host "removed $path" }
}
# .NET registry API: "*" is the literal all-files class key here, never a wildcard
$shellKey = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Software\Classes\*\shell', $true)
if ($shellKey -and ($shellKey.GetSubKeyNames() -contains "AskSprout")) {
    $shellKey.DeleteSubKeyTree("AskSprout"); Write-Host "removed Explorer menu entry"
}
if ($shellKey) { $shellKey.Close() }
