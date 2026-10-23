"""Task accuracy and answer flips on GSM8K (in-distribution quality check).

Each model answers the same fixed GSM8K test problems (OpenAI grade-school-math,
commit 3101c7d, test.jsonl sha256 3730d312...) through llama-server at temperature 0.
Reports accuracy and, against a reference model, answer "flips" (Dutta et al.,
"Accuracy is Not All You Need", NeurIPS 2024): problems whose final answer differs.

  python tools/task_eval.py MODEL.gguf --label NAME --tag TAG [--lm auto] [-n 30]
  python tools/task_eval.py --report --tag TAG --ref REFLABEL
"""
import argparse
import datetime
import json
import math
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import preflight as pf  # noqa: E402
from spec_bench import http, wait_ready  # noqa: E402

DATA = os.path.join(bench.LLM_HOME, "data", "gsm8k", "test.jsonl")
MMLU_PRO = os.path.join(bench.LLM_HOME, "data", "TIGER-Lab__MMLU-Pro", "data", "test-00000-of-00001.parquet")  # rev b189ec7
OUT = os.path.join(pf.ROOT, "results", "tasks.jsonl")
INSTRUCTION = ("Solve the following math problem. End your reply with a final line of the form "
               "'#### <number>' containing only the numeric answer.\n\n")
LETTERS = "ABCDEFGHIJ"


def load_problems(bench, n, per_category, seed):
    """Fixed, reproducible problem lists: [(id, prompt, ground_truth)]."""
    if bench == "gsm8k":
        with open(DATA, encoding="utf-8") as f:
            probs = [json.loads(line) for line in f][:n]
        return [(i, INSTRUCTION + p["question"], number(p["answer"].split("####")[-1])) for i, p in enumerate(probs)]
    import pandas as pd  # MMLU-Pro (TIGER-Lab, MIT): stratified sample, per_category per subject
    df = pd.read_parquet(MMLU_PRO)
    df = df.groupby("category").sample(n=per_category, random_state=seed)  # every category has >= 381
    out = []
    for _, r in df.sort_values("question_id").iterrows():
        opts = "\n".join(f"({LETTERS[k]}) {o}" for k, o in enumerate(r["options"]))
        prompt = (f"The following is a multiple-choice question about {r['category']}. Think step by step, "
                  f"then end your reply with a final line of the form 'Answer: X' where X is the letter "
                  f"of the correct option.\n\nQuestion: {r['question']}\n\nOptions:\n{opts}")
        out.append((int(r["question_id"]), prompt, r["answer"]))
    return out


def extract_letter(content):
    m = re.findall(r"[Aa]nswer\s*(?:is)?\s*[:：]?\s*\**\(?([A-J])\)?", content)
    return m[-1] if m else None


def number(s):
    s = s.replace(",", "").replace("$", "").strip()
    try:
        v = float(s)
        return int(v) if v == int(v) else v
    except ValueError:
        return None


def extract(content):
    m = re.findall(r"####\s*\$?\s*(-?[\d,]*\.?\d+)", content)
    if m:
        return number(m[-1])
    nums = re.findall(r"-?\d[\d,]*\.?\d*", content)
    return number(nums[-1]) if nums else None


def run(a):
    if not bench.on_ac():
        raise SystemExit("STOPPED: on battery")
    probs = load_problems(a.bench, a.n, a.per_category, a.seed)
    if a.ids:  # rerun only these problem ids (e.g. answers truncated at the token limit)
        keep = {int(x) for x in a.ids.split(",")}
        probs = [p for p in probs if p[0] in keep]
    if a.skip_from:  # extend an earlier sample: skip ids this label already answered under those tags
        done = {r["idx"] for r in load_rows() if r["tag"] in a.skip_from.split(",") and r["label"] == a.label}
        probs = [p for p in probs if p[0] not in done]
        print(f"skipping {len(done)} ids already answered under {a.skip_from}; {len(probs)} to run")
    cmd = [os.path.join(a.bin, "llama-server.exe"), "-m", a.model, "-c", "4096", "-t", "8", "-fa", "on",
           "-lm", a.lm, "-fitt", "256", "--jinja", "-np", "1", "--host", "127.0.0.1", "--port", str(a.port),
           "--no-webui"] + a.extra.split()
    print(subprocess.list2cmdline(cmd))
    log = open(os.path.join(bench.LLM_HOME, f"server-{a.tag}-{a.label}.log"), "w", encoding="utf-8",
               errors="replace")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=bench.llama_env())
    correct = 0
    try:
        if not wait_ready(a.port, proc):
            raise SystemExit("server failed to start")
        with open(OUT, "a", encoding="utf-8") as out:
            for i, (pid, prompt, gt) in enumerate(probs):
                body = {"messages": [{"role": "user", "content": prompt}],
                        "max_tokens": a.max_tokens, "temperature": 0, "seed": 1, "stream": False}
                if a.no_think:  # Gemma 4 template: enable_thinking (llama-server turns it on otherwise)
                    body["chat_template_kwargs"] = {"enable_thinking": False}
                r = http(a.port, "/v1/chat/completions", body, timeout=3600)
                ch = r["choices"][0]
                content = ch["message"].get("content") or ""
                if a.bench == "gsm8k":
                    pred = extract(content)
                    ok = pred is not None and gt is not None and abs(pred - gt) < 1e-6
                else:
                    pred = extract_letter(content)
                    ok = pred == gt
                correct += ok
                rec = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "tag": a.tag,
                       "bench": a.bench, "label": a.label, "model": a.model, "idx": pid, "gt": gt, "pred": pred,
                       "correct": ok,
                       "finish": ch.get("finish_reason"), "gen_n": r.get("timings", {}).get("predicted_n"),
                       "content": content, "reasoning": ch["message"].get("reasoning_content")}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                print(f"  [{i:2d}] gt {gt} pred {pred} {'OK' if ok else 'WRONG'} "
                      f"({rec['gen_n']} tok, {rec['finish']})  running {correct}/{i + 1}")
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


def sign_test(lost, gained):
    """Exact two-sided sign test on discordant pairs (McNemar exact): P(split this uneven | no difference)."""
    n = lost + gained
    if n == 0:
        return 1.0
    k = min(lost, gained)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load_rows():
    if not os.path.exists(OUT):
        return []
    with open(OUT, encoding="utf-8") as f:
        return list(map(json.loads, f))


def report(a):
    all_rows = load_rows()
    by = {}
    for tag in a.tag.split(","):  # union of tags in order, later ones replace the same (label, id)
        for r in all_rows:
            if r["tag"] == tag:
                by.setdefault(r["label"], {})[r["idx"]] = r
    for r in all_rows:
        if a.override_tag and r["tag"] == a.override_tag and r["label"] in by:
            by[r["label"]][r["idx"]] = r
    ref = by.get(a.ref, {})
    print(f"{'model':<14} {'accuracy':>16} {'truncated':>9} {'flips vs ' + a.ref:>16} {'ref OK→wrong':>12} {'ref wrong→OK':>12} {'p (sign)':>9}")
    for label, d in by.items():
        n = len(d)
        acc = sum(r["correct"] for r in d.values()) / n
        se = math.sqrt(acc * (1 - acc) / n)
        trunc = sum(r["finish"] == "length" for r in d.values())
        common = [i for i in d if i in ref]
        flips = sum(d[i]["pred"] != ref[i]["pred"] for i in common)
        lost = sum(ref[i]["correct"] and not d[i]["correct"] for i in common)
        gained = sum(not ref[i]["correct"] and d[i]["correct"] for i in common)
        print(f"{label:<14} {100 * acc:>7.1f}% ± {100 * se:4.1f}% {trunc:>9} {f'{flips}/{len(common)}':>16} "
              f"{lost:>12} {gained:>12} {sign_test(lost, gained):>9.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("model", nargs="?")
    ap.add_argument("--label")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bin", default=bench.DEFAULT_BIN)
    ap.add_argument("--lm", default="none")
    ap.add_argument("--bench", default="gsm8k", choices=["gsm8k", "mmlupro"])
    ap.add_argument("-n", type=int, default=30, help="gsm8k: first n problems")
    ap.add_argument("--per-category", type=int, default=5, help="mmlupro: questions per subject (14 subjects)")
    ap.add_argument("--seed", type=int, default=0, help="mmlupro: sampling seed (fixed for comparability)")
    ap.add_argument("--ids", default="", help="comma list of problem ids to run (subset of the fixed sample)")
    ap.add_argument("--override-tag", default="", help="report: rows of this tag replace same-id rows")
    ap.add_argument("--skip-from", default="", help="run: skip ids this label already answered under these tags (comma list)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--extra", default="")
    ap.add_argument("--no-think", action="store_true", help="disable the thinking channel (chat_template_kwargs)")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--ref", default="Q8_0")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
    if a.report:
        report(a)
    else:
        run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
