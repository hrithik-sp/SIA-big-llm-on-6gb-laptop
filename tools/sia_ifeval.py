"""IFEval through Sia: does the assistant follow explicit instructions ("no commas", "at least 300 words",
"answer in JSON", "end with this phrase") the way the base model does?

IFEval (Zhou et al., Google, arXiv 2311.07911): 541 prompts, each with instructions a program can check.
This sends a fixed random sample of them to Sia exactly as the owner would type them (Chat mode by default,
with Sia's real prompt, persona and tools) and writes the replies in IFEval's input_response_data format.
Score them with Google's own checker (google-research/instruction_following_eval, evaluation_main.py).

Any tool Sia asks to use is declined (no side effects) and counted: a writing task should need none, as in the
"irrelevance" category of the Berkeley Function Calling Leaderboard.

  python tools/sia_ifeval.py INPUT_DATA.jsonl OUT.jsonl [--n 100] [--seed 0] [--mode chat]
Resumable: prompts already in OUT.jsonl are skipped. Starts Sia without a window on temporary settings.
"""
import argparse
import http.client
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H = {"Origin": "http://127.0.0.1:8765", "Content-Type": "application/json"}


def call(method, path, body=None, timeout=60):
    c = http.client.HTTPConnection("127.0.0.1", 8765, timeout=timeout)
    c.request(method, path, json.dumps(body) if body is not None else None, H)
    r = c.getresponse()
    data = r.read()
    return r.status, json.loads(data) if data else None


def ask(mode, prompt):
    c = http.client.HTTPConnection("127.0.0.1", 8765, timeout=1800)
    c.request("POST", "/api/chat", json.dumps({"mode": mode, "messages": [{"role": "user", "content": prompt}]}), H)
    r = c.getresponse()
    t0, answer, tools = time.time(), "", []
    for line in r:
        if not line.startswith(b"data: "):
            continue
        ev = json.loads(line[6:])
        if ev.get("type") == "tool_request":
            tools.append(ev["label"])
            call("POST", "/api/tool-decision", {"id": ev["id"], "decision": "deny"})
        elif ev.get("type") == "tool" and ev.get("label") not in tools and ev.get("status") == "running":
            tools.append(ev["label"])  # tools that need no card (calculator, sandbox)
        for ch in ev.get("choices", []):
            answer += (ch.get("delta") or {}).get("content") or ""
    return answer.strip(), tools, time.time() - t0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="chat")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.input, encoding="utf-8")]
    random.Random(a.seed).shuffle(rows)
    rows = rows[:a.n]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["prompt"] for l in open(a.out, encoding="utf-8")}
    todo = [r for r in rows if r["prompt"] not in done]
    print(f"{len(rows)} prompts, {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", 8765)) == 0:
            sys.exit("Sia is already running on port 8765. Quit it first (power button), then rerun.")
    tmp = Path(tempfile.mkdtemp())
    (tmp / "settings.json").write_text(json.dumps({"user_name": "Hrithik S P", "laptop_access": True}))
    code = ("import sys, pathlib; sys.argv=['server.py','--no-browser','--model','gemma']; "
            f"sys.path.insert(0, r'{ROOT / 'app'}'); import server; "
            f"server.SETTINGS_FILE = pathlib.Path(r'{tmp / 'settings.json'}'); server.DATA = pathlib.Path(r'{tmp}'); "
            "server.ACTIONS_LOG = server.DATA / 'actions.log'; server.main(sys.argv[1:])")
    srv = subprocess.Popen([sys.executable, "-c", code], cwd=str(ROOT), env=dict(os.environ))
    for _ in range(120):
        try:
            if call("GET", "/api/status", timeout=3)[0] == 200:
                break
        except OSError:
            time.sleep(0.5)
    try:
        ask(a.mode, "ok")  # load the model
        with open(a.out, "a", encoding="utf-8") as f:
            for i, r in enumerate(todo, 1):
                answer, tools, secs = ask(a.mode, r["prompt"])
                f.write(json.dumps({"prompt": r["prompt"], "response": answer, "key": r["key"], "tools": tools,
                                    "seconds": round(secs, 1), "mode": a.mode}, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{i}/{len(todo)} key {r['key']}: {len(answer.split())} words, {secs:.0f} s, tools {tools}", flush=True)
    finally:
        call("POST", "/api/quit", {})
        srv.wait(60)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
