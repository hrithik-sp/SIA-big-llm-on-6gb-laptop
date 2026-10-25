"""Reply latency of Sia's Chat mode over a multi-turn conversation, with the real model.

Measures, per turn: time to the first visible word, total time, and how many prompt tokens
llama-server had to process (prompt_n) vs reuse from its cache (cache_n). A conversation should
only process its newest message; re-processing the whole history means the prompt prefix
changed (for example a clock at the top of the system prompt). The run crosses a minute
boundary on purpose.

  python tools/sia_latency_check.py
Starts Sia without a window on temporary settings (the user's own are untouched), then quits it.
"""
import http.client
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H = {"Origin": "http://127.0.0.1:8765", "Content-Type": "application/json"}
TURNS = ["hi", "what time is it?", "how's my battery doing?", "ok thanks", "what's the date today?", "cool"]


def call(method, path, body=None, timeout=60):
    c = http.client.HTTPConnection("127.0.0.1", 8765, timeout=timeout)
    c.request(method, path, json.dumps(body) if body is not None else None, H)
    r = c.getresponse()
    data = r.read()
    return r.status, json.loads(data) if data else None


def turn(history):
    c = http.client.HTTPConnection("127.0.0.1", 8765, timeout=1800)
    c.request("POST", "/api/chat", json.dumps({"mode": "chat", "messages": history}), H)
    r = c.getresponse()
    t0, first, answer, timings, tools_used = time.time(), None, "", {}, []
    for line in r:
        if not line.startswith(b"data: "):
            continue
        ev = json.loads(line[6:])
        if ev.get("type") == "tool_request":
            tools_used.append(ev["label"])
            call("POST", "/api/tool-decision", {"id": ev["id"], "decision": "once"})
        for ch in ev.get("choices", []):
            piece = (ch.get("delta") or {}).get("content") or ""
            if piece and first is None:
                first = time.time() - t0
            answer += piece
        timings = ev.get("timings") or timings
    return answer.strip(), first or 0.0, time.time() - t0, timings, tools_used


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    tmp = Path(tempfile.mkdtemp())
    (tmp / "settings.json").write_text(json.dumps({"user_name": "Hrithik S P", "laptop_access": True}))
    code = ("import sys, pathlib; sys.argv=['server.py','--no-browser','--model','gemma']; "
            f"sys.path.insert(0, r'{ROOT / 'app'}'); import server; "
            f"server.SETTINGS_FILE = pathlib.Path(r'{tmp / 'settings.json'}'); server.DATA = pathlib.Path(r'{tmp}'); server.ACTIONS_LOG = server.DATA / 'actions.log'; server.main(sys.argv[1:])")
    import socket  # never test against the owner's running Sia: it would get the questions and the actions
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", 8765)) == 0:
            sys.exit("Sia is already running on port 8765. Quit it first (power button), then rerun.")
    srv = subprocess.Popen([sys.executable, "-c", code], cwd=str(ROOT), env=dict(os.environ))
    for _ in range(120):
        try:
            if call("GET", "/api/status", timeout=3)[0] == 200:
                break
        except OSError:
            time.sleep(0.5)
    history = []
    t_start = time.time()  # like opening the window: the model loads (and pre-reads the instructions)
    while time.time() - t_start < 180:  # while the user starts typing
        try:
            if (call("GET", "/api/status", timeout=3)[1] or {}).get("models", {}).get("gemma") == "loaded":
                break
        except OSError:
            pass
        time.sleep(0.5)
    print(f"model loaded {time.time() - t_start:.1f} s after start; typing for 8 s")
    time.sleep(8)
    try:
        print(f"{'turn':<28}{'first word':>11}{'total':>8}{'processed':>11}{'reused':>8}  tools / answer")
        for i, text in enumerate(TURNS):
            if i == 3:  # cross a minute boundary: the clock line changes
                time.sleep(61 - time.localtime().tm_sec)
            history.append({"role": "user", "content": text})
            answer, first, total, tm, used = turn(history)
            history.append({"role": "assistant", "content": answer})
            print(f"{text:<28}{first:>10.1f}s{total:>7.1f}s{tm.get('prompt_n', '?'):>11}{tm.get('cache_n', '?'):>8}  "
                  f"{used} {answer[:90]!r}", flush=True)
    finally:
        call("POST", "/api/quit", {})
        srv.wait(60)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
