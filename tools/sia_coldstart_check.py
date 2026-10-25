"""The first reply after Sia starts: re-reading the instructions vs restoring the saved warm-up.

Starts Sia twice on the same temporary data folder. Each time a message is sent as soon as the model is
up, the way a user who opens Sia and types straight away would, and the time to its first word is
measured. Run 1 has no saved warm-up (Sia pre-reads ~5,700 tokens of instructions and tools, then saves
that state); run 2 restores it. The log lines from app-data/logs/sia.log show the save and the restore.

  python tools/sia_coldstart_check.py
"""
import http.client
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
H = {"Origin": "http://127.0.0.1:8765", "Content-Type": "application/json"}


def call(method, path, body=None, timeout=60):
    c = http.client.HTTPConnection("127.0.0.1", 8765, timeout=timeout)
    c.request(method, path, json.dumps(body) if body is not None else None, H)
    r = c.getresponse()
    data = r.read()
    return r.status, json.loads(data) if data else None


def first_word(text):
    c = http.client.HTTPConnection("127.0.0.1", 8765, timeout=900)
    c.request("POST", "/api/chat", json.dumps({"mode": "chat", "messages": [{"role": "user", "content": text}]}), H)
    r = c.getresponse()
    t0, first, timings = time.time(), None, {}
    for line in r:
        if not line.startswith(b"data: "):
            continue
        ev = json.loads(line[6:])
        timings = ev.get("timings") or timings
        for ch in ev.get("choices", []):
            if (ch.get("delta") or {}).get("content") and first is None:
                first = time.time() - t0
    return first, timings


def run(tmp, label):
    code = ("import sys, pathlib; sys.argv=['server.py','--no-browser','--model','gemma']; "
            f"sys.path.insert(0, r'{ROOT / 'app'}'); import server; "
            f"server.SETTINGS_FILE = pathlib.Path(r'{tmp / 'settings.json'}'); server.DATA = pathlib.Path(r'{tmp}'); "
            "server.ACTIONS_LOG = server.DATA / 'actions.log'; server.main(sys.argv[1:])")
    free = psutil.virtual_memory().available / 2 ** 30
    t_start = time.time()
    srv = subprocess.Popen([sys.executable, "-c", code], cwd=str(ROOT))
    try:
        for _ in range(240):
            try:
                if (call("GET", "/api/status", timeout=3)[1] or {}).get("models", {}).get("gemma") == "loaded":
                    break
            except OSError:
                pass
            time.sleep(0.5)
        t_loaded = time.time() - t_start
        fw, tm = first_word("hi, what can you do for me?")
        print(f"{label}: free RAM {free:.1f} GB | model loaded after {t_loaded:.1f} s | first word "
              f"{fw:.1f} s after sending | llama: processed {tm.get('prompt_n')} reused {tm.get('cache_n')}", flush=True)
        log = tmp / "logs" / "sia.log"
        for _ in range(300):  # let the warm-up save or restore finish before quitting (it's logged when done)
            text = log.read_text(encoding="utf-8") if log.exists() else ""
            if "saved warm-" in text or "restored warm-" in text or "save failed" in text:
                break
            time.sleep(1)
    finally:
        call("POST", "/api/quit", {})
        srv.wait(60)
    log = tmp / "logs" / "sia.log"
    for line in (log.read_text(encoding="utf-8").splitlines() if log.exists() else []):
        if "warm-" in line or "restore" in line:
            print("   log:", line)
    if log.exists():
        log.unlink()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", 8765)) == 0:
            sys.exit("Sia is already running on port 8765. Quit it first (power button), then rerun.")
    tmp = Path(tempfile.mkdtemp())
    (tmp / "settings.json").write_text(json.dumps({"user_name": "Hrithik S P", "pronouns": "he/him"}))
    try:
        run(tmp, "run 1 (re-read the prompt)")
        files = list((tmp / "kv").glob("*.bin"))
        print("   saved:", [(f.name, f"{f.stat().st_size / 2 ** 20:.0f} MB") for f in files])
        run(tmp, "run 2 (restore the saved warm-up)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
