<#
.SYNOPSIS
  Extend MMLU-Pro from 70 to 210 questions (15 per subject) for the QAT verdict.

.DESCRIPTION
  groupby(category).sample(n, random_state=0) takes a prefix of one random permutation,
  so the 15-per-subject sample contains the 70 questions already answered (verified).
  Only the 140 new ids run (--skip-from), at 4096 max tokens. That cap matches the
  merged 70q results: answers that finished under 1536 are unchanged at temperature 0,
  and the truncated ones were rerun at 4096. Tag gemma4-mmlupro-v2. Report:
    python tools\task_eval.py --report --tag gemma4-mmlupro-v1,gemma4-mmlupro-v1-rerun4k,gemma4-mmlupro-v2 --ref Q8_0
  Time: QAT ~1.5 h, Q8_0 (partly SSD-streamed, ~9 tok/s) ~4-5 h, IQ2_M ~1 h.
#>
param([string]$Only = "QAT,Q8_0")
if (Test-Path "results\PAUSE") { Write-Host "=== PAUSED: results\PAUSE exists ($(Get-Content results\PAUSE -Raw)). Delete it to run."; exit 0 }
# Lists are comma strings: 'powershell -File' passes "0,2,4,6" as ONE string (once parsed as 246).
$onlyList = $Only -split "[,\s]+" | Where-Object { $_ }
$llm = if ($env:LLM_HOME) { $env:LLM_HOME } else { Join-Path $env:USERPROFILE "llm" }
$d = "$llm\models\unsloth__gemma-4-26B-A4B-it-GGUF"
$targets = @(
    @{ label = "QAT"; model = "$llm\models\google__gemma-4-26B-A4B-it-qat-q4_0-gguf\gemma-4-26B_q4_0-it.gguf"; lm = "auto"; extra = "-fitt 128" },
    @{ label = "Q8_0"; model = "$d\gemma-4-26B-A4B-it-Q8_0.gguf"; lm = "auto"; extra = "" },
    @{ label = "IQ2_M"; model = "$d\gemma-4-26B-A4B-it-UD-IQ2_M.gguf"; lm = "none"; extra = "" }
)
foreach ($t in $targets | Where-Object { $onlyList -contains $_.label }) {
    Write-Host "=== MMLU-Pro 210q extension: $($t.label)"
    python tools\task_eval.py $t.model --bench mmlupro --per-category 15 --no-think --label $t.label --lm $t.lm `
        --max-tokens 4096 --tag gemma4-mmlupro-v2 --skip-from "gemma4-mmlupro-v1" "--extra=$($t.extra)"
}
python tools\task_eval.py --report --tag "gemma4-mmlupro-v1,gemma4-mmlupro-v1-rerun4k,gemma4-mmlupro-v2" --ref Q8_0
