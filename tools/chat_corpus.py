"""Build an in-distribution evaluation text from a reference model's own chat answers.

Each item = the chat-template generation prompt the model actually saw (rendered from
the GGUF's own template, thinking off; byte-identical to llama-server /apply-template)
+ the reference's answer + the end-of-turn marker. Scored with llama-perplexity using
LLAMA_PERPLEXITY_PARSE_SPECIAL=1 (patches/llama-perplexity-parse-special.patch) so the
markers are real special tokens. This avoids the wikitext problem for instruct models
(lab-notebook finding 22): the reference is confident on its own outputs, so the 16-nat
storage window of --kl-divergence-base is respected.

  python tools/chat_corpus.py --tag gemma4-gsm8k-nothink-v1 --label Q8_0 --gguf REF.gguf --out FILE.txt
  # QAT study (docs/QAT_STUDY_PROTOCOL.md): several tags, fixed shuffled order, a template file
  python tools/chat_corpus.py --tag gemma4-mmlupro-v1,gemma4-mmlupro-v1-rerun4k,gemma4-mmlupro-v2,gemma4-gsm8k-nothink-v1 \\
      --label Q8_0 --shuffle-seed 0 --template chat_template.jinja --out FILE.txt
"""
import argparse
import json
import os
import random
import sys

import jinja2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preflight as pf  # noqa: E402
import task_eval  # noqa: E402

END_OF_TURN = "<turn|>\n"  # Gemma 4 template's end-of-turn marker


def render_prompt(template, user_content):
    env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(Exception(m))
    env.globals["strftime_now"] = lambda f: "01 Jan 2026"
    return env.from_string(template).render(messages=[{"role": "user", "content": user_content}], bos_token="",
                                            add_generation_prompt=True, enable_thinking=False)


def load_answers(tags, label):
    """The label's answers under these tags (comma list); a later tag replaces an earlier answer to the same
    question, as in task_eval --report. Keyed by (bench, question id)."""
    with open(task_eval.OUT, encoding="utf-8") as f:
        rows = list(map(json.loads, f))
    by = {}
    for tag in tags.split(","):
        for r in rows:
            if r["tag"] == tag and r["label"] == label:
                by[(r.get("bench", "gsm8k"), r["idx"])] = r
    return by


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True, help="comma list; later tags replace earlier answers to the same question")
    ap.add_argument("--label", required=True, help="reference model label in results/tasks.jsonl")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gguf", help="GGUF whose chat template to use")
    src.add_argument("--template", help="chat template file (.jinja) to use instead of a GGUF's")
    ap.add_argument("--per-category", type=int, default=15, help="MMLU-Pro sample the answers come from")
    ap.add_argument("--shuffle-seed", type=int, help="fixed pseudo-random order (default: bench, then question id)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    if a.gguf:
        template = pf.read_gguf(a.gguf)[0]["tokenizer.chat_template"]
    else:
        with open(a.template, encoding="utf-8") as f:
            template = f.read()
    answers = load_answers(a.tag, a.label)
    prompts = {}
    for bench in sorted({b for b, _ in answers}):
        prompts.update({(bench, pid): prompt
                        for pid, prompt, _ in task_eval.load_problems(bench, 10**6, a.per_category, 0)})
    keys = sorted(k for k, r in answers.items() if r["finish"] == "stop" and k in prompts)  # only complete answers
    if a.shuffle_seed is not None:
        random.Random(a.shuffle_seed).shuffle(keys)
    parts = [render_prompt(template, prompts[k]) + answers[k]["content"] + END_OF_TURN for k in keys]
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(parts))
    print(f"{len(parts)} conversations, {sum(map(len, parts))} chars -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
