<#
.SYNOPSIS
  Daily-use RAM-pressure test (plan item 7): decode speed while other apps hold RAM.

.DESCRIPTION
  For each hold level, tools/ram_hold.py takes that much RAM and keeps touching it (like
  active browser tabs; floor: never below 2 GiB available), then bench.py measures
  decode (-p 0 -n 128, 5 reps: rep 1 = cold, last reps = warm).
  - mmap targets run with --allow-ssd ON PURPOSE: when apps take RAM, the OS evicts
    model pages and re-reads them from the SSD, and that slowdown is what this test
    measures. bench.py still records disk_share and marks such runs valid=false. Report
    them as "SSD-streamed under pressure", never as clean speed results.
  - gpt-oss -lm none (the tuned config) runs WITHOUT --allow-ssd: preflight skips it at
    hold levels where its private allocation would not fit.
  The amount actually held is appended to results/rampressure.jsonl.
  Tags: rampressure-<target>-hold<GiB>. -DryRun: holder + preflight only, no model runs.
#>
param([string]$Holds = "0,2,4,6", [string]$Only = "gptoss-mmap,gptoss-none,qat", [switch]$DryRun)
# Lists are comma strings: 'powershell -File' passes "0,2,4,6" as ONE string (once parsed as 246).
$holdList = $Holds -split "[,\s]+" | Where-Object { $_ } | ForEach-Object { [double]::Parse($_, [Globalization.CultureInfo]::InvariantCulture) }
$onlyList = $Only -split "[,\s]+" | Where-Object { $_ }
foreach ($h in $holdList) { if ($h -lt 0 -or $h -gt 8) { throw "hold $h GiB is outside 0-8 GiB; refusing" } }
Write-Host "holds: $($holdList -join ', ') GiB; targets: $($onlyList -join ', ')"
$llm = if ($env:LLM_HOME) { $env:LLM_HOME } else { Join-Path $env:USERPROFILE "llm" }
$m = "$llm\models"
$gptoss = "$m\ggml-org__gpt-oss-20b-GGUF\gpt-oss-20b-MXFP4.gguf"
$targets = @(
    @{ key = "gptoss-mmap"; model = $gptoss; lm = "auto"; fitt = 256; extra = @("--allow-ssd") },
    @{ key = "gptoss-none"; model = $gptoss; lm = "none"; fitt = 256; extra = @() },
    @{ key = "qat"; model = "$m\google__gemma-4-26B-A4B-it-qat-q4_0-gguf\gemma-4-26B_q4_0-it.gguf"; lm = "auto"; fitt = 128; extra = @("--allow-ssd") }
)
$log = Join-Path $env:TEMP "ram_hold_out.txt"
foreach ($h in $holdList) {
    $holder = $null
    $held = 0.0
    if ($h -gt 0) {
        Remove-Item $log -ErrorAction SilentlyContinue
        $holder = Start-Process python -ArgumentList "tools\ram_hold.py", $h, "--keep-free", "2" `
            -RedirectStandardOutput $log -NoNewWindow -PassThru
        $t0 = Get-Date
        while (-not ((Test-Path $log) -and (Select-String -Path $log -Pattern "READY" -Quiet))) {
            if (((Get-Date) - $t0).TotalSeconds -gt 180 -or $holder.HasExited) { break }
            Start-Sleep -Milliseconds 500
        }
        $ready = Select-String -Path $log -Pattern "READY held=([\d.]+)" -ErrorAction SilentlyContinue
        if (-not $ready) { Write-Host "holder failed to start; skipping hold $h"; if (-not $holder.HasExited) { Stop-Process -Id $holder.Id -Force }; continue }
        $held = [double]$ready.Matches[0].Groups[1].Value
        Write-Host "=== holding $held GiB (requested $h)"
    }
    try {
        foreach ($t in $targets | Where-Object { $onlyList -contains $_.key }) {
            $tag = "rampressure-$($t.key)-hold$h"
            Write-Host "=== $tag"
            python tools\bench.py $t.model --place auto --fitt $t.fitt --lm $t.lm -ub 512 -t 8 -p 0 -n 128 -r 5 `
                --delay 10 --cool-to 62 --tag $tag @($t.extra) @(if ($DryRun) { "--dry-run" })
            if ($DryRun) { continue }
            $rec = @{ time = (Get-Date -Format s); tag = $tag; requested_gib = $h; held_gib = $held } | ConvertTo-Json -Compress
            Add-Content -Path results\rampressure.jsonl -Value $rec -Encoding ascii
        }
    } finally {
        if ($holder -and -not $holder.HasExited) { Stop-Process -Id $holder.Id -Force }
    }
}
