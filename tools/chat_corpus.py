"""Build an in-distribution evaluation text from a reference model's own chat answers.

Each item = the chat-template generation prompt the model actually saw (rendered from
the GGUF's own template, thinking off; byte-identical to llama-server /apply-template)
+ the reference's answer + the end-of-turn marker. Scored with llama-perplexity using
LLAMA_PERPLEXITY_PARSE_SPECIAL=1 (patches/llama-perplexity-parse-special.patch) so the
markers are real special tokens. This avoids the wikitext problem for instruct models
(lab-notebook finding 22): the reference is confident on its own outputs, so the 16-nat
storage window of --kl-divergence-base is respected.

  python tools/chat_corpus.py --tag gemma4-gsm8k-nothink-v1 --label Q8_0 --gguf REF.gguf --out FILE.txt
"""
import argparse
import json
import os
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--label", required=True, help="reference model label in results/tasks.jsonl")
    ap.add_argument("--gguf", required=True, help="GGUF whose chat template to use")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    template = pf.read_gguf(a.gguf)[0]["tokenizer.chat_template"]
    with open(task_eval.OUT, encoding="utf-8") as f:
        rows = [r for r in map(json.loads, f) if r["tag"] == a.tag and r["label"] == a.label]
    bench = rows[0].get("bench", "gsm8k") if rows else "gsm8k"
    prompts = {pid: prompt for pid, prompt, _ in task_eval.load_problems(bench, 10**6, 5, 0)}
    parts = []
    for r in sorted(rows, key=lambda r: r["idx"]):
        if r["finish"] != "stop" or r["idx"] not in prompts:
            continue  # only complete answers
        parts.append(render_prompt(template, prompts[r["idx"]]) + r["content"] + END_OF_TURN)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(parts))
    print(f"{len(parts)} conversations, {sum(map(len, parts))} chars -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
