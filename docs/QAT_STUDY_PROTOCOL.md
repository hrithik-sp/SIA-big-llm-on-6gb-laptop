# Study protocol: how lossless are Gemma 4's QAT releases?

This protocol was written and committed **before any of its data was collected**. It fixes the questions,
measurements, sample sizes and analysis in advance, so the results can't shape the method. Any deviation is
reported with its reason in the results.

## Question

Google releases each Gemma 4 model in 4-bit form trained with quantization-aware training (QAT) and says it keeps
"similar quality to bfloat16". On this laptop, the 26B-A4B QAT release scored 4.8 MMLU-Pro points below the
near-lossless Q8_0 (p = 0.041, lab notebook finding 48). Its next-token predictions were as far from the original
on average as a 2-bit quantization's, yet its worst cases were half as far (finding 31). Two different things could
cause that:

- **Retraining drift:** QAT trains the model further, so it becomes a slightly different model before any
  rounding.
- **Rounding loss:** the final conversion to 4 bits loses information.

Google publishes, for every Gemma 4 size (E2B, E4B, 12B, 26B-A4B, 31B), the original instruct model, the
**QAT model before rounding** (`-qat-q4_0-unquantized`) and the 4-bit release (`-qat-q4_0-gguf`). That makes it
possible to measure the two effects separately.

## Versions compared, per size

| Code | Version | Source |
|---|---|---|
| O | Original instruct model, BF16 | `google/gemma-4-<size>-it`, converted to GGUF BF16 |
| U | QAT model before rounding, BF16 | `google/gemma-4-<size>-it-qat-q4_0-unquantized`, converted to GGUF BF16 |
| Q | Google's 4-bit QAT release | `google/gemma-4-<size>-it-qat-q4_0-gguf` (the file as published) |
| O4 | The original rounded to Q4_0 without QAT (control) | `llama-quantize O16 Q4_0` (O16: the F16 conversion, see "Pipeline changes") |
| U4 | The QAT model rounded to Q4_0 here | `llama-quantize U16 Q4_0` |
| O8, U8 | Q8_0 copies, for the accuracy runs | `llama-quantize O/U Q8_0` |

All repositories are pinned to a revision, every file's SHA-256 is recorded, and conversion and quantization use
llama.cpp at commit `0a1ef309c`, the same build as every other result in this project. No importance matrix is
used for O4 or U4.

**Same file format as Google's.** Google's Q4_0 files store every large weight matrix in Q4_0, the token embeddings
(which double as the output layer) in Q6_K, and norms and the expert router at full precision. O4 and U4 must differ
from Q only in how the weights were trained, not in such format choices. So, before any KL comparison, the
per-tensor formats of O4 and U4 are checked against Q's, tensor by tensor; where they differ, O4 and U4 are re-made
with `llama-quantize --tensor-type` / `--token-embedding-type` overrides until they match exactly.

Checked before writing this protocol:
- All ten Hugging Face repositories are public (not gated) and Apache-2.0.
- All ten share one tokenizer (identical `tokenizer.json`).
- The E2B and E4B QAT checkpoints lack the key/value projections of their last 20 and 18 layers. Those layers
  reuse an earlier layer's key/value cache (`num_kv_shared_layers` in the config; `src/models/gemma4.cpp` in
  llama.cpp), so the weights are unused in the original too. Every other tensor matches in name, shape and type.
- E2B and E4B use a chat template that differs from the larger models' in one respect: with thinking off, the
  larger models' template adds an empty thought block. Each model is therefore scored on text rendered with its
  own template.

## Measurements

### 1. Fidelity: KL-divergence of next-token predictions

Method as in Dutta et al. (NeurIPS 2024) and lab notebook findings 30–32: `llama-perplexity --kl-divergence-base`
saves a reference model's predictions on a fixed text, then `--kl-divergence` compares a candidate against them.

**Text:** the Gemma 4 26B-A4B Q8_0 model's own answers (thinking off, temperature 0) to the 210 MMLU-Pro and 50
GSM8K questions already in `results/tasks.jsonl` (tags `gemma4-mmlupro-v1`, `gemma4-mmlupro-v1-rerun4k`,
`gemma4-mmlupro-v2`, `gemma4-gsm8k-nothink-v1`). Only answers that finished are used, each with its question
rendered by the target model's chat template, in one fixed pseudo-random order (seed 0), so that the scored part
covers every subject and both tasks rather than the first few subjects. Scored with context 512 over the **first
100 chunks**, with chat markers read as special tokens (`patches/llama-perplexity-parse-special.patch`). The same
answers are used for every size.

**Comparisons per size** (reference → candidate):

| Comparison | What it measures |
|---|---|
| O → U | Retraining drift |
| U → Q | Rounding loss of the QAT release |
| O → Q | Total distance of the release from the original (what a user gets) |
| O → O4 | Rounding loss without QAT (control, same 4-bit format) |
| U → U4, Q → U4, and a byte comparison of U4's and Q's tensor data | Whether Google's file is a plain Q4_0 rounding of U (exploratory) |
| O → O8 | The error of the Q8_0 stand-in used for accuracy |
| O → O16 | What the F16 conversion step alone changes (added in the pilot; see "Pipeline changes") |
| O → O | The comparison's own noise floor (added in the pilot; see "Pipeline changes") |

**Reported for each:** mean KL-divergence with its standard error, the 99th and 99.9th percentile and maximum, and
the share of tokens where both versions pick the same top token.

**Validity check:** the KL pass's "Mean PPL(base)" must be within 2% of the reference run's own perplexity
(`tools/quality.py` checks this). Otherwise the stored reference is clipped by llama.cpp's 16-nat window (finding
22), and the comparison is reported as distorted, not dropped.

### 2. Task accuracy: MMLU-Pro

- **Sample:** 100 questions per subject, 1,400 in total (`groupby(category).sample(100, random_state=0)`). It
  contains the 210 questions already used (checked), so the 26B results extend the existing sample.
- **Settings:** as in finding 48: temperature 0, thinking off, at most 4,096 answer tokens, graded by the final
  "Answer: X". An answer cut off at the limit counts as wrong.
- **Versions:** O8 (reference), U8, Q and O4.
- **Power:** with the ~10% disagreement seen between versions so far, 1,400 paired questions detect a difference of
  about 2.4 points (80% power, two-sided 5%: (1.96 + 0.84) × √(0.10 / n)), against about 6 points for 210.

## Hypotheses (stated in advance; results are reported whatever their direction)

- **H1, rounding:** QAT makes the model robust to 4-bit rounding: KL(U → Q) < KL(O → O4) at every size.
- **H2, where the distance comes from:** most of the release's distance from the original is retraining drift,
  not rounding: KL(O → U) > KL(U → Q).
- **H3, accuracy:** Q differs from O8 on MMLU-Pro (two-sided). Also tested: Q against O4, and U8 against O8.
- **Exploratory:** whether Google's Q4_0 file equals a plain Q4_0 rounding of U.

## Analysis

- **Accuracy:** exact two-sided McNemar (sign) test on the questions where two versions disagree, with the paired
  difference and its 95% interval, as in `tools/task_eval.py --report`. The **primary** tests are Q against O8, one
  per size; across sizes they form one family, corrected with Holm–Bonferroni. Q against O4 and U8 against O8 are
  **secondary**: reported with uncorrected p-values and labelled as such. Cut-off answers are reported, and a
  further analysis counts only questions both versions finished.
- **KL-divergence:** a difference counts as clear only when the two 95% intervals don't overlap.
- **Stopping rule:** the sample size is fixed at 1,400 per version. No accuracy differences are looked at before a
  version's run is complete. A run interrupted by a crash or a loss of mains power is resumed for the remaining
  questions only, and the interruption is recorded.

## Scope and order

1. **Pilot, E2B:** the full pipeline end to end: download, conversion, quantization, the KL comparisons, and a
   timing check of the accuracy run. The pilot's KL results count as data. If the pipeline has to change,
   the change is recorded and E2B is rerun with the final pipeline.
2. **E2B accuracy, then E4B and 12B:** accuracy for E2B and E4B; all KL comparisons for E4B and 12B.
3. **26B-A4B and 31B:** all KL comparisons, once enough disk space is free (each needs roughly 80–100 GB at the
   conversion step). 26B accuracy: the existing 210 answers extended to 1,400 for O8 and Q, plus O4 and U8.
   Those 210 were analysed before this protocol existed (finding 48), so the 26B result is reported twice: on all
   1,400 questions and on the 1,190 new ones alone. For 26B, O8 is the unsloth Q8_0 file those answers came from
   (`unsloth/gemma-4-26B-A4B-it-GGUF` @ `c099eb48`), and its distance from O is measured like any other O → O8.
4. **Accuracy for 12B and 31B** only if a timing check, run before any of their accuracy data, projects at most
   24 hours per version on this laptop. Dense models read every weight for each token, so this may rule them out.

Every run goes through the project's preflight check. KL runs load models memory-mapped, and one whose weights
exceed free RAM may stream from the SSD (`--allow-ssd`): predictions are unaffected, only speed suffers, and speed
isn't measured in this study.

## Pipeline changes found in the E2B pilot (recorded before any KL result)

1. **Converter environment.** llama.cpp's requirements at `0a1ef309c` pin transformers 4.57.6, which can't read
   Gemma 4's tokenizer configuration. llama.cpp #21617 had set transformers 5.5.1 because it "is required to be
   able to convert Gemma4 models"; a later commit lowered the pin to fix CI in an unrelated pull request. Conversion
   therefore runs in its own environment with transformers 5.5.1 (`tools/requirements-convert-gemma4.txt`). Checked:
   the converted files' tokenizer metadata (13 fields) and architecture settings (20 fields) are identical to
   Google's own Q4_0 GGUF.
2. **An F16 conversion as the input for O4 and U4.** Google's Q4_0 files store `per_layer_model_proj` in F16, and
   llama-quantize never converts that tensor (`tensor_allows_quantization`), so a BF16 input left it in BF16 and the
   format check failed. Each checkpoint is therefore also converted with `--outtype f16` (O16, U16), and O4 and U4 are
   made from those. Storing the weights as F16 changes **none** of U's 4.63 billion values and 501,026 of O's (0.011%,
   tiny values below F16's normal range, mostly in the per-layer input gates). To measure that step on its own, O → O16
   is added to the KL comparisons.
3. **A wider storage window for the saved reference.** The first pilot pass (tag `qat-study-v1`) failed the validity
   check: with O as the reference, PPL(base) recomputed from the saved predictions was 5.59 against 5.82 from the run
   itself (ratio 0.962; the limit is 2%). E2B is less sure of the 26B model's text than the 26B is (perplexity 5.8
   against 2.96), so more true tokens fall below llama.cpp's 16-nat window and are stored at its floor. The threshold
   was **not** loosened. Instead the window was made a setting (`patches/llama-perplexity-kld-window.patch`,
   `LLAMA_PERPLEXITY_KLD_WINDOW`; the default stays 16) and the study uses **48 nats**, applied to both passes. The
   16-bit storage step grows from 0.00024 to 0.00073 nats. Checked: with the default window the patched tool
   reproduces the first pass's result exactly (mean KLD 0.111564 ± 0.001561, same top token 91.635%). Added: O → O, to
   measure the comparison's own noise floor under the new window. E2B is rerun in full with the final pipeline (tag
   `qat-study-v2`); the first pass stays in the raw log and is reported as such.

## Limitations known in advance

- One laptop, one test text (written by the 26B model), and one benchmark for accuracy.
- The accuracy reference is Q8_0, not BF16: dense BF16 models are too slow to generate 1,400 answers here. The O → O8
  comparison bounds the difference.
- KL-divergence measures how *different* two versions are, not which one is better. Accuracy decides that.
