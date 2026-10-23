"""Live check of Sprout's laptop tools with the real models: does the model call the right
tool, does the permission flow work (approve and deny), and does the answer use the result?

  python tools/sprout_tools_check.py
  python tools/sprout_tools_check.py "What time is it?" "Where am I?"   (your own questions, all approved)
Starts Sprout without a window on a temporary settings file (laptop access on, nothing
always-allowed), then quits it. The user's own settings are untouched.
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


def call(method, path, body=None, timeout=60):
    c = http.client.HTTPConnection("127.0.0.1", 8765, timeout=timeout)
    c.request(method, path, json.dumps(body) if body is not None else None, H)
    r = c.getresponse()
    data = r.read()
    return r.status, json.loads(data) if data else None


def chat(mode, text, model=None, decide="once"):
    body = {"mode": mode, "messages": [{"role": "user", "content": text}]}
    if model:
        body["model"] = model
    c = http.client.HTTPConnection("127.0.0.1", 8765, timeout=1800)
    c.request("POST", "/api/chat", json.dumps(body), H)
    r = c.getresponse()
    answer, events, t0 = "", [], time.time()
    for line in r:
        if not line.startswith(b"data: "):
            continue
        ev = json.loads(line[6:])
        if ev.get("type") == "tool_request":
            events.append(f"asked: {ev['label']}")
            call("POST", "/api/tool-decision", {"id": ev["id"], "decision": decide})
        elif ev.get("type") == "tool":
            events.append(f"{ev['status']}: {ev['label']}")
        elif ev.get("type") == "error":
            events.append("ERROR " + ev["error"][:200])
        for ch in ev.get("choices", []):
            answer += (ch.get("delta") or {}).get("content") or ""
    return answer.strip(), events, time.time() - t0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    tmp = Path(tempfile.mkdtemp())
    (tmp / "settings.json").write_text(json.dumps({"user_name": "Hrithik S P", "laptop_access": True}))
    env = dict(os.environ)
    code = ("import sys, runpy; sys.argv=['server.py','--no-browser','--model','gemma']; "
            f"import pathlib; sys.path.insert(0, r'{ROOT / 'app'}'); import server; "
            f"server.SETTINGS_FILE = pathlib.Path(r'{tmp / 'settings.json'}'); server.main(sys.argv[1:])")
    srv = subprocess.Popen([sys.executable, "-c", code], cwd=str(ROOT), env=env)
    for _ in range(120):
        try:
            if call("GET", "/api/status", timeout=3)[0] == 200:
                break
        except OSError:
            time.sleep(0.5)
    tests = [
        ("chat", "who's your boss?", None, "once"),
        ("chat", "How many browser tabs do I have open right now, and how loaded is my CPU?", None, "once"),
        ("chat", "What did I download today?", None, "once"),
        ("chat", f"Read the file {ROOT / 'app' / 'modes.json'} and tell me how many modes Sprout has.", None, "once"),
        ("chat", "What's my GPU temperature?", None, "deny"),
        ("think", "How much free space is on my C drive?", "gptoss", "once"),
    ]
    if len(sys.argv) > 1:
        tests = [("chat", q, None, "once") for q in sys.argv[1:]]
    try:
        for mode, q, model, decide in tests:
            a, ev, s = chat(mode, q, model, decide)
            print(f"\n> {q}  [{model or 'gemma'}, decision={decide}]\n  tools: {ev}\n  answer ({s:.0f}s): {a[:300]}", flush=True)
    finally:
        call("POST", "/api/quit", {})
        srv.wait(60)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
