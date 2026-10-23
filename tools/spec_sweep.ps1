<#
.SYNOPSIS
  Speculative-decoding sweeps (lossless at temperature 0) for the goal models.

.DESCRIPTION
  For each target: baseline, then drafts with --spec-draft-n-max 2/4/8, then the
  baseline again. The repeated baseline (A-B-A) cancels the page-cache warming that
  favours later runs of mmap'd, partially SSD-streamed models. Results go to
  results/spec.jsonl; compare with tools/spec_report.py --tag <tag>.

  Draft models:
    gpt-oss-20b  : EAGLE-3 (ggml-org eagle3-gpt-oss-20b-Q8_0.gguf), --spec-type draft-eagle3
    Gemma 4 QAT  : QAT MTP head (unsloth QAT repo), --spec-type draft-mtp
                   (Google: a QAT target needs the QAT assistant)
    Gemma 4 Q8_0 : MTP head (unsloth non-QAT repo), streamed from SSD
#>
param([string]$Only = "gptoss,qat,q8")
# Lists are comma strings: 'powershell -File' passes "0,2,4,6" as ONE string (once parsed as 246).
$onlyList = $Only -split "[,\s]+" | Where-Object { $_ }
$ErrorActionPreference = "Continue"
$llm = if ($env:LLM_HOME) { $env:LLM_HOME } else { Join-Path $env:USERPROFILE "llm" }
$m = "$llm\models"
$targets = @(
    @{ key = "gptoss"; tag = "spec-gptoss-v2"; model = "$m\ggml-org__gpt-oss-20b-GGUF\gpt-oss-20b-MXFP4.gguf"; lm = "none"; fitt = 256;
       spec = "--spec-type draft-eagle3 -md $m\ggml-org__gpt-oss-20b-GGUF\eagle3-gpt-oss-20b-Q8_0.gguf"; n = @(2, 4, 8) },
    @{ key = "qat"; tag = "spec-qat-v2"; model = "$m\google__gemma-4-26B-A4B-it-qat-q4_0-gguf\gemma-4-26B_q4_0-it.gguf"; lm = "auto"; fitt = 128;
       spec = "--spec-type draft-mtp -md $m\unsloth__gemma-4-26B-A4B-it-qat-GGUF\mtp-gemma-4-26B-A4B-it.gguf"; n = @(2, 4, 8) },
    @{ key = "q8"; tag = "spec-q8-v2"; model = "$m\unsloth__gemma-4-26B-A4B-it-GGUF\gemma-4-26B-A4B-it-Q8_0.gguf"; lm = "auto"; fitt = 256;
       spec = "--spec-type draft-mtp -md $m\unsloth__gemma-4-26B-A4B-it-GGUF\mtp-gemma-4-26B-A4B-it.gguf"; n = @(3, 6) }
)
foreach ($t in $targets | Where-Object { $onlyList -contains $_.key }) {
    $common = @($t.model, "--lm", $t.lm, "--fitt", $t.fitt, "--tag", $t.tag, "--max-tokens", 256)
    Write-Host "=== $($t.tag): baseline (A)"
    python tools\spec_bench.py @common --label baseline
    foreach ($n in $t.n) {
        Write-Host "=== $($t.tag): draft n-max $n"
        python tools\spec_bench.py @common --label "n$n" --spec "$($t.spec) --spec-draft-n-max $n"
    }
    Write-Host "=== $($t.tag): baseline (A again)"
    python tools\spec_bench.py @common --label baseline2
    python tools\spec_report.py --tag $t.tag
}
