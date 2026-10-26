# Glossary

Plain-language explanations of the terms used in this project, from basic to
advanced. The numbers come from the measurements in LAB_NOTEBOOK.md.

## 1. The model itself

**LLM (Large Language Model):** a program that writes text by repeatedly predicting the
next small piece of text. ChatGPT, Gemma and gpt-oss are all LLMs.

**Token:** the piece of text an LLM reads and writes, usually part of a word.
"unbelievable" might be "un" + "believ" + "able". Roughly, 1 token ≈ ¾ of an English word.

**Parameters / weights:** the numbers inside the model that hold everything it learned.
"26B" means 26 billion of them. They are the model's memory and knowledge. More
parameters usually means more capable, but more storage.

**Layers:** the model processes each token through a stack of steps (Gemma 4 has 30).

**MoE (Mixture of Experts):** a model split into many specialist sub-networks
("experts"). For each token a "router" picks only a few. Gemma 4 has 128 experts per
layer and uses 8 per token.
*Analogy:* a hospital with 128 specialists, where each patient sees only the 8
relevant doctors.

**Active parameters:** the parameters actually used per token (Gemma 4: ~4B of 26B).
This is why big MoE models can run on a laptop.

## 2. A laptop's memory, and why it sets the limits

**VRAM:** the graphics card's own memory (6 GB here). Very fast, small.

**RAM:** main system memory (16 GB here). Bigger, about 3× slower than VRAM.

**SSD:** storage (the C: drive). Huge, much slower again.
*Analogy:* VRAM is your desk, RAM a bookshelf in the room, the SSD a library down the
street. You work fastest with the books on your desk.

**Bandwidth:** how many gigabytes per second can be read from a memory. Measured:
VRAM 162 GB/s, RAM 54 GB/s, the link between them (**PCIe**) 13 GB/s.

**Why bandwidth sets the speed:** to write one token the model must read all its
*active* weights once, so speed ≈ bandwidth ÷ bytes read per token.

**Roofline:** that theoretical maximum speed. "62% of roofline" means the setup reaches
62% of what the hardware physically allows.

**Offloading:** keeping part of the model in RAM (or on the SSD) because it doesn't fit
in VRAM. `--n-cpu-moe` and **auto-fit** decide which parts go where.

**mmap (memory mapping):** instead of loading the whole file into RAM, Windows reads
pieces from the SSD as needed and keeps recent ones cached.

**Paging:** re-reading pieces from the SSD because RAM is full. Slow. "Partially
SSD-streamed" means some paging happens.

**Cold vs warm:** at first few pieces are cached (cold: QAT ~15 tok/s). With use, the
frequently used experts stay in RAM (warm: ~28 tok/s).

## 3. Speed

**Generation / decode, tokens per second (tok/s):** how fast the answer is written.
~28 tok/s ≈ 20 words per second, faster than you read.

**Prompt processing (prefill):** how fast the model *reads* your input before answering.

**Time to first token:** the wait before the first word appears; set by prompt length ÷
prompt-processing speed.

**Thermal throttling:** the chip slowing itself down when hot. Tests use cool-down
pauses so heat doesn't distort comparisons.

**Power mode:** the Windows setting (best power efficiency ↔ best performance).
Switching to best performance gave +16% generation speed.

## 4. Making models smaller

**Precision / bits:** how many bits store each weight. Originals usually use **BF16**
(16 bits). Fewer bits: smaller and faster, but less precise.

**Quantization:** rounding weights to fewer bits *after* training.
*Analogy:* saving a photo as a smaller JPEG. Light compression is invisible; heavy
compression shows artifacts.

**Q8_0, Q4_0, Q3_K_M, IQ3_XXS, IQ2_M:** quantization formats. The number is roughly
the bits per weight: Q8 ≈ 8 bits (near-original), IQ2 ≈ 2–2.5 bits (heavily compressed).

**Lossy vs lossless:** lossy changes the model's behaviour somewhat; lossless keeps it
exactly the same.

**QAT (Quantization-Aware Training):** the model is *trained while already in 4-bit*,
so it learns to work well that way. Google's Gemma 4 QAT and gpt-oss's **MXFP4** format
are examples, which is why they can be small and still smart. "Still smart" isn't
automatic, though: on this laptop, Gemma 4 QAT scored 4.8 points below the near-original
Q8_0 on MMLU-Pro (lab-notebook finding 48). gpt-oss was *released* in MXFP4, so it has no
higher-precision original to lose against.

## 5. Measuring intelligence: did anything get lost?

**Benchmark:** a standard test set with known answers.
- **GSM8K:** grade-school math word problems. Too easy here: every version scored 98–100%.
- **MMLU-Pro:** harder multiple-choice questions (10 options) across 14 subjects.

**Ceiling effect:** a test so easy that everyone gets full marks, so it can't rank anyone.

**Flips:** questions where a compressed version's answer *differs* from the original's,
even when the overall score looks the same.

**Perplexity:** how "surprised" a model is by a text; lower is better. Perplexity 3 means
it is typically choosing among about 3 likely next tokens.

**KL divergence (KLD):** how different two models' predictions are, token by token.
0 = identical behaviour. Catches small changes a test score can miss.

**Same top token:** how often both models' first choice for the next token matches.

**In-distribution:** testing on the kind of text a model is meant for (chats), rather
than something unusual (raw Wikipedia text confused this chat model).

**Statistical significance (p-value):** how often luck alone would produce a gap at
least this big if the two models were really equally good. p = 0.34 means such a gap
would appear by chance 34% of the time, so it doesn't count as real. Scientists usually
want below 5% (p < 0.05).

**Sample size / statistical power:** more questions detect smaller differences. 70
questions only reliably detect gaps of about 10 points; 210 detect about 6. The QAT
gap of 4–5 points looked like noise at 70 questions (p = 0.45) and became significant
at 210 (p = 0.041).

## 6. Making it faster without losing quality

**Temperature 0 (greedy):** always pick the single most likely next token, so outputs
are repeatable and tests are fair.

**Speculative decoding:** a small, fast "draft" model guesses the next few tokens, and
the big model checks them all at once. Correct guesses are kept. In exact arithmetic,
at temperature 0, the final text would be identical to the big model alone. In
practice llama.cpp's results differ in the last digits when it processes several tokens
at once ("not bit-for-bit identical for different batch sizes", its server docs). So
where two words are almost equally likely, the choice can flip, e.g. "industrial model"
vs "industrial wave". Same model, same quality, slightly different wording. Whether it
is faster depends on the model (measured, see LAB_NOTEBOOK.md).
*Analogy:* an assistant drafts a few sentences; the expert approves or corrects them in
one read instead of writing every word.

**Draft model / EAGLE-3 / MTP head:** kinds of guessers. MTP (multi-token prediction) is
a small add-on trained together with the model.

**Acceptance rate:** how often the guesses are right.

**The MoE catch:** checking several guessed tokens can need many different experts,
which can cancel the speed-up (shown by Saxena et al., see REPORT.md, References). That's
why it is measured, not assumed.

## 7. Context

**Context window:** how much text (in tokens) the model can consider at once: the
conversation plus documents. Gemma 4: up to 262,144 tokens.

**KV cache:** the model's working memory of the conversation so far. It grows with
context; kept at full precision it costs memory but not quality.

**Sliding-window attention:** some layers only look back a fixed distance (e.g. 1,024
tokens), which keeps the KV cache small.

## 8. Tools

**llama.cpp:** the open-source program that runs these models on ordinary computers.
Its programs: **llama-server** (chat / API), **llama-bench** (speed),
**llama-perplexity** (quality).

**GGUF:** the model file format llama.cpp uses.

**Preflight (this project's tool):** checks *before* running that a model fits in the
memory free right now, like weighing your luggage before going to the airport.
