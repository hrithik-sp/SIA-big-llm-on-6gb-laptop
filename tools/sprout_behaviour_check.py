"""Behaviour check for Sprout's Chat mode with the real model: speed, correctness and persona.

Scripted conversations based on a real chat, each scored automatically:
- greetings: short, varied, the owner's name used rarely, no filler laughter;
- relationship: the owner is the boss, not a friend;
- time and place: answered from the laptop, not guessed;
- a file search: found in one request, without asking "should I?";
- Wordle and Hangman: feedback consistent with the secret word, and no leaked word;
- a scheduling question with one right answer;
- reply speed: time to the first word on turns that use no tool.

  python tools/sprout_behaviour_check.py [--show]     (--show prints every reply)
Starts Sprout without a window on temporary settings (the user's own are untouched), then quits it.
"""
import datetime as dt
import http.client
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H = {"Origin": "http://127.0.0.1:8765", "Content-Type": "application/json"}
OWNER = "Hrithik"
NONE_RX = re.compile(r"\bnone of (those|these|the letters)\b|\bnothing in\b|not in the word|no letters", re.I)
ASK_RX = re.compile(r"\b(should i|do you want me to|want me to|shall i)\b", re.I)


def call(method, path, body=None, timeout=60):
    c = http.client.HTTPConnection("127.0.0.1", 8765, timeout=timeout)
    c.request(method, path, json.dumps(body) if body is not None else None, H)
    r = c.getresponse()
    data = r.read()
    return r.status, json.loads(data) if data else None


def send(history, text):
    """One user turn (tools auto-approved); appends both messages; returns a result dict."""
    history.append({"role": "user", "content": text})
    c = http.client.HTTPConnection("127.0.0.1", 8765, timeout=1800)
    c.request("POST", "/api/chat", json.dumps({"mode": "chat", "messages": history}), H)
    r = c.getresponse()
    t0, first, answer, tools_used, failed = time.time(), None, "", [], 0
    for line in r:
        if not line.startswith(b"data: "):
            continue
        ev = json.loads(line[6:])
        if ev.get("type") == "tool_request":
            tools_used.append(ev["label"])
            call("POST", "/api/tool-decision", {"id": ev["id"], "decision": "once"})
        elif ev.get("type") == "tool":
            if ev.get("status") not in ("done", "running"):
                failed += 1
            elif ev["label"] not in tools_used:  # tools that need no permission card (the game referee)
                tools_used.append(ev["label"])
        for ch in ev.get("choices", []):
            piece = (ch.get("delta") or {}).get("content") or ""
            if piece and first is None:
                first = time.time() - t0
            answer += piece
    answer = answer.strip()
    history.append({"role": "assistant", "content": answer})
    return {"user": text, "answer": answer, "first": first or 0.0, "total": time.time() - t0,
            "tools": tools_used, "failed": failed}


def scenarios():
    """(name, user turns, scorer(results) -> (passed, detail))."""
    def greet(rs):
        named = sum(OWNER.lower() in r["answer"].lower() for r in rs)
        laughs = sum(len(re.findall(r"\bha(ha)+\b|\blol\b", r["answer"], re.I)) for r in rs)
        quips = sum(bool(re.search(r"loop|hitting me with", r["answer"], re.I)) for r in rs)
        long_ = sum(len(r["answer"]) > 160 for r in rs)
        ok = named <= 1 and laughs == 0 and quips == 0 and long_ == 0
        return ok, f"name in {named}/4, laughs {laughs}, quips {quips}, long replies {long_}"

    def boss(rs):
        a = rs[0]["answer"].lower()
        casual = re.findall(r"friend|hanging out|buddy|\bbro\b|privilege", a)
        return ("boss" in a or "work for" in a) and not casual, f"boss: {'boss' in a}, off-tone words {casual}"

    def clock(rs):
        now = dt.datetime.now()
        times = {(now + dt.timedelta(minutes=d)).strftime(f) for d in (-2, -1, 0, 1, 2)
                 for f in ("%I:%M", "%H:%M")}
        times |= {t.lstrip("0") for t in times}
        hit = any(t in rs[0]["answer"] for t in times)
        return hit and not rs[0]["tools"], f"{'correct' if hit else 'WRONG'} time, tools {rs[0]['tools']}"

    def place(rs):
        a = rs[0]["answer"]
        return "Bengaluru" in a or "Bangalore" in a, a[:60]

    def find_id(rs):
        r = rs[0]
        found = bool(re.search(r"aadha?a?r", r["answer"], re.I)) and not re.search(
            r"(couldn't|can't|didn't|could not|did not) (find|see)|no (files?|results?)", r["answer"], re.I)
        asks = bool(ASK_RX.search(r["answer"]))
        return found and not asks and r["failed"] == 0, (f"found {found}, asks permission-style {asks}, "
                                                          f"tool calls {len(r['tools'])}, failed {r['failed']}")

    def wordle(rs):
        m = re.search(r"\bword (?:was|is)\W+([a-z]{5})\b", rs[-1]["answer"], re.I)
        if not m:
            return False, "no word revealed"
        word = m.group(1).upper()
        bad = []
        for r in rs[1:3]:  # every letter the reply calls absent must really be absent from the word
            absent = set()
            for grp in re.findall(r"((?:\b[A-Z]\b[,\s]*(?:and\s+)?)+)(?:is|are) not in the word", r["answer"]):
                absent |= set(re.findall(r"\b[A-Z]\b", grp))
            if NONE_RX.search(r["answer"]) and not absent:
                absent = set(r["user"].upper())
            if absent & set(word):
                bad.append(f"{r['user']}: {sorted(absent & set(word))}")
        used = all(any("Word game" in t for t in r["tools"]) for r in rs[1:])
        return not bad and used, f"word {word}; contradicted {bad or 'none'}; referee used every move {used}"

    def hangman(rs):
        leak = [r["user"] for r in rs[:-1] if re.search(r"\bthe word (is|was)\b", r["answer"], re.I)]
        m = re.search(r"\bword (?:was|is)\W+([a-z]+)\b", rs[-1]["answer"], re.I)
        boards = [b for r in rs[:-1] for b in re.findall(r"(?<!\S)(?:[A-Z_] ){2,}[A-Z_](?!\S)", r["answer"])]
        if leak or not m or not boards:
            return False, f"leak {leak}, revealed {bool(m)}, boards {len(boards)}"
        word, last = m.group(1).lower(), boards[-1].replace(" ", "").lower()
        fits = len(word) == len(last) and all(b == "_" or b == w for b, w in zip(last, word))
        absent = [r["user"].lower() for r in rs[1:-1]
                  if re.match(r"^\s*no\b", r["answer"], re.I) and r["user"].lower() in word]
        used = all(any("Word game" in t for t in r["tools"]) for r in rs[1:])
        return fits and not absent and used, (f"word {word!r} vs board {last!r}: fits {fits}; wrongly 'absent' "
                                               f"{absent}; referee used every move {used}")

    def schedule(rs):
        # every time stated as the finish must be 4:35 (worked steps like 2:45 may appear elsewhere)
        claims = [t for s in re.split(r"[.\n]", rs[0]["answer"]) if re.search(r"finish|done|wrap|free|end", s, re.I)
                  for t in re.findall(r"\b\d{1,2}:\d{2}\b", s) if not s.lower().count("start")]
        return bool(claims) and set(claims) <= {"4:35", "16:35"}, f"finish times stated {claims} (right: 4:35)"

    return [
        ("greetings", ["yo", "yo", "yo", "hi"], greet),
        ("boss, not friend", ["who am I to you?"], boss),
        ("time", ["what time rn"], clock),
        ("location", ["where am i right now?"], place),
        ("find my ID", ["find my aadhaar card on this laptop"], find_id),
        ("wordle", ["let's play wordle, you pick the word", "guest", "doing", "I give up, what was the word?"], wordle),
        ("hangman", ["let's play hangman, you pick a word", "o", "s", "r", "I give up"], hangman),
        ("schedule", ["I have 3 meetings of 45 minutes starting at 2 pm with 10-minute breaks between them. "
                      "When do I finish?"], schedule),
    ]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    show = "--show" in sys.argv
    tmp = Path(tempfile.mkdtemp())
    (tmp / "settings.json").write_text(json.dumps({"user_name": "Hrithik S P", "laptop_access": True}))
    code = ("import sys, pathlib; sys.argv=['server.py','--no-browser','--model','gemma']; "
            f"sys.path.insert(0, r'{ROOT / 'app'}'); import server; "
            f"server.SETTINGS_FILE = pathlib.Path(r'{tmp / 'settings.json'}'); server.DATA = pathlib.Path(r'{tmp}'); server.ACTIONS_LOG = server.DATA / 'actions.log'; server.main(sys.argv[1:])")
    import socket  # never test against the owner's running Sprout: it would get the questions and the actions
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", 8765)) == 0:
            sys.exit("Sprout is already running on port 8765. Quit it first (power button), then rerun.")
    srv = subprocess.Popen([sys.executable, "-c", code], cwd=str(ROOT), env=dict(os.environ))
    for _ in range(120):
        try:
            if call("GET", "/api/status", timeout=3)[0] == 200:
                break
        except OSError:
            time.sleep(0.5)
    passed, firsts = 0, []
    try:
        send([], "ok")  # load the model first, so the first scenario isn't timed with the cold start
        for name, turns, score in scenarios():
            history, results = [], []
            for t in turns:
                results.append(send(history, t))
            firsts += [r["first"] for r in results if not r["tools"]]
            ok, detail = score(results)
            passed += ok
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
            if show:
                for r in results:
                    print(f"    > {r['user']}\n      {r['answer']!r}  ({r['first']:.1f}s first word, tools {r['tools']})")
        print(f"\n{passed}/{len(scenarios())} scenarios passed; first word on non-tool turns: "
              f"median {statistics.median(firsts):.1f}s, max {max(firsts):.1f}s")
    finally:
        call("POST", "/api/quit", {})
        srv.wait(60)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
