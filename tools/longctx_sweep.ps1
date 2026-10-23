<#
.SYNOPSIS
  Long-context behaviour of the goal models (full-precision f16 KV: no quality cost).

.DESCRIPTION
  Decode at filled context (llama-bench -d) and prompt processing of long inputs
  (time to first token = n_prompt / pp speed). Every configuration goes through
  tools/bench.py's preflight gate (KV grows with context; skipped if it won't fit).
  Tags: longctx-<model>-v1.
#>
param([string]$Only = "gptoss,qat")
# Lists are comma strings: 'powershell -File' passes "0,2,4,6" as ONE string (once parsed as 246).
$onlyList = $Only -split "[,\s]+" | Where-Object { $_ }
$llm = if ($env:LLM_HOME) { $env:LLM_HOME } else { Join-Path $env:USERPROFILE "llm" }
$m = "$llm\models"
$targets = @(
    @{ key = "gptoss"; model = "$m\ggml-org__gpt-oss-20b-GGUF\gpt-oss-20b-MXFP4.gguf"; lm = "none"; fitt = 256; extra = @() },
    @{ key = "qat"; model = "$m\google__gemma-4-26B-A4B-it-qat-q4_0-gguf\gemma-4-26B_q4_0-it.gguf"; lm = "auto"; fitt = 128; extra = @("--allow-ssd") }
)
foreach ($t in $targets | Where-Object { $onlyList -contains $_.key }) {
    $tag = "longctx-$($t.key)-v1"
    Write-Host "=== $tag : decode at depth 0 / 8k / 16k / 32k"
    python tools\bench.py $t.model --place auto --fitt $t.fitt --lm $t.lm -ub 512 -t 8 -p 0 -n 64 -d "0,8192,16384,32768" `
        -r 3 --delay 10 --cool-to 62 --tag $tag @($t.extra)
    foreach ($p in 2048, 8192, 16384) {
        Write-Host "=== $tag : prompt processing of $p tokens"
        python tools\bench.py $t.model --place auto --fitt $t.fitt --lm $t.lm -ub 512 -t 8 -p $p -n 0 `
            -r 3 --delay 10 --cool-to 62 --tag $tag @($t.extra)
    }
}
