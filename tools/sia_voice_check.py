"""Voice mode end to end with the real models, without a microphone or speakers.

Each question is spoken by Sia's own voice, resampled to 16 kHz and streamed into /api/voice/audio
in 100 ms chunks at real-time pace, as the browser does. The script then times the voice turn:
  end of speech -> transcript (VAD + speech recognition)
  transcript -> first words of the reply (the chat model, in voice mode)
  first sentence -> its audio ready (text to speech)
and checks the reply is speakable (no markdown) and the memory used with everything loaded.

  python tools/sia_voice_check.py ["question" ...]
Starts Sia without a window on temporary settings and data, then quits it.
"""
import http.client
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import voice  # noqa: E402

H = {"Origin": "http://127.0.0.1:8765", "Content-Type": "application/json"}
QUESTIONS = ["Hey Sia, what time is it right now?", "Who created you?",
             "How many days are left until Christmas?", "Give me one quick tip to focus better while studying."]


def call(method, path, body=None, timeout=120, raw=False):
    c = http.client.HTTPConnection("127.0.0.1", 8765, timeout=timeout)
    data = body if isinstance(body, (bytes, bytearray)) else (json.dumps(body) if body is not None else None)
    c.request(method, path, data, H)
    r = c.getresponse()
    out = r.read()
    return r.status, (out if raw else (json.loads(out) if out else None))


def speech_pcm(text):
    """Sia's voice saying the question, as 16 kHz 16-bit PCM with silence around it."""
    with wave.open(io.BytesIO(voice.speak(text))) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        sr = w.getframerate()
    x16 = np.interp(np.linspace(0, len(x) - 1, int(len(x) * 16000 / sr)), np.arange(len(x)), x)
    audio = np.concatenate([np.zeros(4800), x16, np.zeros(16000)])
    return (audio * 32767).astype("<i2").tobytes(), len(x16) / 16000


def events_reader(store):
    c = http.client.HTTPConnection("127.0.0.1", 8765, timeout=600)
    c.request("POST", "/api/voice/events", "{}", H)
    r = c.getresponse()
    for line in r:
        if line.startswith(b"data: "):
            ev = json.loads(line[6:])
            ev["t"] = time.time()
            store.append(ev)
            if ev["type"] == "closed":
                return


def chat(history, spoken=True):
    c = http.client.HTTPConnection("127.0.0.1", 8765, timeout=600)
    c.request("POST", "/api/chat", json.dumps({"mode": "chat", "messages": history, "voice": spoken}), H)
    r = c.getresponse()
    t0, first, answer = time.time(), None, ""
    chat.timings = []
    for line in r:
        if not line.startswith(b"data: "):
            continue
        ev = json.loads(line[6:])
        if ev.get("type") == "tool_request":
            call("POST", "/api/tool-decision", {"id": ev["id"], "decision": "once"})
        if ev.get("timings"):
            tm = ev["timings"]
            chat.timings.append(f"processed {tm.get('prompt_n')} reused {tm.get('cache_n')} "
                                f"({tm.get('prompt_ms', 0) / 1000:.1f}s), {tm.get('predicted_per_second', 0):.0f} tok/s")
        for ch in ev.get("choices", []):
            piece = (ch.get("delta") or {}).get("content") or ""
            if piece and first is None:
                first = time.time()
            answer += piece
    return answer.strip(), (first or time.time()) - t0


def rss_mb():
    total = 0
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
            if "llama-server" in (p.info["name"] or "") or "server.main" in cmd:
                total += p.memory_info().rss
        except (psutil.Error, TypeError):
            continue
    return total / 2 ** 20


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    with socket.socket() as s:  # never test against the owner's running Sia
        if s.connect_ex(("127.0.0.1", 8765)) == 0:
            sys.exit("Sia is already running on port 8765. Quit it first (power button), then rerun.")
    questions = [q for q in sys.argv[1:] if not q.startswith("--")] or QUESTIONS
    voice.load()  # this process speaks the questions, then frees its copy so only Sia's models use RAM
    clips = {q: speech_pcm(q) for q in questions}
    voice.unload()
    import gc
    gc.collect()
    tmp = Path(tempfile.mkdtemp())
    (tmp / "settings.json").write_text(json.dumps({"user_name": "Hrithik S P", "laptop_access": True}))
    code = ("import sys, pathlib; sys.argv=['server.py','--no-browser','--model','gemma']; "
            f"sys.path.insert(0, r'{ROOT / 'app'}'); import server; "
            f"server.SETTINGS_FILE = pathlib.Path(r'{tmp / 'settings.json'}'); server.DATA = pathlib.Path(r'{tmp}'); "
            "server.ACTIONS_LOG = server.DATA / 'actions.log'; server.main(sys.argv[1:])")
    srv = subprocess.Popen([sys.executable, "-c", code], cwd=str(ROOT), env=dict(os.environ))
    try:
        for _ in range(120):
            try:
                if call("GET", "/api/status", timeout=3)[0] == 200:
                    break
            except OSError:
                time.sleep(0.5)
        t0 = time.time()
        while time.time() - t0 < 240 and (call("GET", "/api/status")[1] or {}).get("models", {}).get("gemma") != "loaded":
            time.sleep(1)
        time.sleep(8)  # the warm-up reads the instructions
        answer, t_first = chat([{"role": "user", "content": "hi, quick check before we talk"}], spoken=False)
        print(f"text-only baseline: first words {t_first:.2f}s {chat.timings} free "
              f"{psutil.virtual_memory().available / 2 ** 30:.1f} GB", flush=True)
        st, r = call("POST", "/api/voice/start", {})
        print(f"voice start: {st} {r}  | RAM (Sia + llama-server) {rss_mb():.0f} MB, "
              f"free {psutil.virtual_memory().available / 2 ** 30:.1f} GB", flush=True)
        events = []
        threading.Thread(target=events_reader, args=(events,), daemon=True).start()
        history, rows = [], []
        for q in questions:
            pcm, dur = clips[q]
            n0 = len(events)
            start = time.time()
            for i in range(0, len(pcm), 3200):  # real-time pace, like a microphone
                call("POST", "/api/voice/audio", pcm[i:i + 3200])
                time.sleep(max(0.0, start + (i + 3200) / 32000 - time.time()))
            speech_end = start + 0.3 + dur
            tr = None
            for _ in range(200):
                tr = next((e for e in events[n0:] if e["type"] == "transcript"), None)
                if tr:
                    break
                time.sleep(0.05)
            if not tr:
                print(f"> {q}\n  NO TRANSCRIPT; events {[e['type'] for e in events[n0:]]}")
                continue
            history.append({"role": "user", "content": tr["text"]})
            t_tr = tr["t"] - speech_end
            answer, t_first = chat(history)
            history.append({"role": "assistant", "content": answer})
            first_sentence = re.split(r"(?<=[.!?])\s", answer, maxsplit=1)[0]
            t1 = time.time()
            st, wav = call("POST", "/api/voice/speak", {"text": first_sentence}, raw=True)
            t_tts = time.time() - t1
            total = t_tr + t_first + t_tts
            md = bool(re.search(r"\*\*|^#|^\s*[-*] |```", answer, re.M))
            rows.append(total)
            print(f"> {q}\n  heard: {tr['text']!r}\n  reply: {answer[:200]!r}\n  speech end->transcript {t_tr:.2f}s | "
                  f"->first words {t_first:.2f}s | first sentence->audio {t_tts:.2f}s ({len(wav)} B) | "
                  f"voice-to-voice {total:.2f}s | markdown {md}\n  llama: {chat.timings}", flush=True)
        print(f"\nRAM with voice loaded (Sia + llama-server): {rss_mb():.0f} MB; free "
              f"{psutil.virtual_memory().available / 2 ** 30:.1f} GB")
        if rows:
            print(f"voice-to-voice: median {sorted(rows)[len(rows) // 2]:.2f}s, max {max(rows):.2f}s over {len(rows)} turns")
        call("POST", "/api/voice/stop", {})
    finally:
        try:
            call("POST", "/api/quit", {})
        except OSError:
            pass
        srv.wait(60)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
