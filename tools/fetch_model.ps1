<#
.SYNOPSIS
  Download one file from a Hugging Face repo and verify it against the published SHA-256.

.EXAMPLE
  tools\fetch_model.ps1 ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-MXFP4.gguf
#>
param(
    [Parameter(Mandatory)][string]$Repo,
    [Parameter(Mandatory)][string]$File,
    [string]$Dest = $(if ($env:LLM_HOME) { "$env:LLM_HOME\models" } else { "$env:USERPROFILE\llm\models" }),
    [string]$Revision = "main",
    [ValidateSet("model", "dataset")][string]$Type = "model"
)
$ErrorActionPreference = "Stop"

$api = if ($Type -eq "dataset") { "datasets" } else { "models" }
$web = if ($Type -eq "dataset") { "https://huggingface.co/datasets" } else { "https://huggingface.co" }
$dir = Split-Path $File
$tree = $null
for ($try = 1; -not $tree; $try++) {  # the API call also fails during short network outages
    try { $tree = Invoke-RestMethod "https://huggingface.co/api/$api/$Repo/tree/$Revision/$dir" }
    catch { if ($try -ge 10) { throw }; Write-Host "API attempt $try failed: $($_.Exception.Message); retrying"; Start-Sleep 30 }
}
$entry = $tree | Where-Object { $_.path -eq $File }
if (-not $entry -or -not $entry.lfs) { throw "$File not found (or not an LFS file) in $Repo@$Revision" }
$size, $sha = [int64]$entry.lfs.size, $entry.lfs.oid

$out = Join-Path $Dest ($Repo -replace '/', '__') | Join-Path -ChildPath $File
New-Item -ItemType Directory -Force (Split-Path $out) | Out-Null
$have = if (Test-Path $out) { (Get-Item $out).Length } else { 0 }
$freeB = (Get-PSDrive ((Resolve-Path $Dest).Drive.Name)).Free
if ($size - $have + 5GB -gt $freeB) { throw ("not enough disk: need {0:N1} GB more, {1:N1} GB free" -f (($size - $have) / 1GB), ($freeB / 1GB)) }
Write-Host ("{0}  {1:N2} GB  sha256 {2}" -f $File, ($size / 1GB), $sha)

for ($try = 1; (Get-Item $out -ErrorAction SilentlyContinue).Length -ne $size; $try++) {
    # --retry alone does not cover a connection reset mid-transfer (curl exit 56);
    # --retry-all-errors does, and the outer loop resumes (-C -) after longer outages.
    curl.exe -L --fail --retry 5 --retry-all-errors --retry-delay 10 -C - -o $out "$web/$Repo/resolve/$Revision/$File"
    if ($LASTEXITCODE -ne 0) {
        if ($try -ge 20) { throw "download failed (curl exit $LASTEXITCODE) after $try attempts; rerun to resume" }
        Write-Host "curl exit $LASTEXITCODE (attempt $try); resuming in 30 s"
        Start-Sleep 30
    }
}
$got = (Get-FileHash -Algorithm SHA256 $out).Hash.ToLower()
if ($got -ne $sha) { throw "SHA-256 MISMATCH for $out`n expected $sha`n got      $got" }
Write-Host "verified: $out"
