"""Voice mode through the real page: headless Edge, a fake microphone, the real mic button.

A question spoken by Sprout's own voice is played into Edge as its microphone
(--use-file-for-fake-audio-capture); the script clicks the mic button through the DevTools protocol (a
small Node.js client: Node 22 has WebSocket built in) and checks that the question is heard, answered
and spoken, and that the page threw no errors. Nothing appears on screen.

  python tools/sprout_ui_voice_check.py
Starts Sprout without a window on temporary settings and data, then quits it.
"""
import http.client
import io
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import voice  # noqa: E402

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
QUESTIONS = [q for q in sys.argv[1:] if not q.startswith("--")] or ["Who created you?"]
QUESTION = QUESTIONS[0]
GAP_S = 45  # between questions in the fake microphone's recording: room for each answer to be spoken
H = {"Origin": "http://127.0.0.1:8765", "Content-Type": "application/json"}

DRIVER = r"""
const port = process.argv[2]; const NQ = Number(process.argv[3] || 1);
const targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
const page = targets.find((t) => t.type === "page" && t.url.startsWith("http://127.0.0.1:8765"));
console.error("targets:", JSON.stringify(targets.map((t) => [t.type, t.url])));
const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0; const waiting = new Map(); const errors = [];
ws.onmessage = (m) => {
  const msg = JSON.parse(m.data);
  if (msg.id && waiting.has(msg.id)) { waiting.get(msg.id)(msg); waiting.delete(msg.id); }
  if (msg.method === "Runtime.exceptionThrown") errors.push(msg.params.exceptionDetails.exception?.description || msg.params.exceptionDetails.text);
  if (msg.method === "Runtime.consoleAPICalled" && msg.params.type === "error") errors.push(msg.params.args.map((a) => a.value || a.description).join(" "));
};
await new Promise((r) => (ws.onopen = r));
const send = (method, params = {}) => new Promise((r) => { const i = ++id; waiting.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const evaluate = async (expr) => (await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true })).result;
await send("Runtime.enable");
let ready = null;
for (let i = 0; i < 60; i++) { ready = await evaluate("typeof S !== 'undefined' && !!S.config"); if (ready.result?.value) break; await new Promise((r) => setTimeout(r, 500)); }
console.error("ready:", JSON.stringify(ready));
const click = await evaluate("document.querySelector('#voice').click(), 'clicked'");
const t0 = Date.now(); let state = null, toasts = new Set();
while (Date.now() - t0 < 90000 + NQ * 60000) {
  state = (await evaluate(`JSON.stringify({ on: V.on, toast: document.querySelector('#toast').hidden ? '' : document.querySelector('#toast').textContent,
    messages: S.chat ? S.chat.messages.map((m) => ({ role: m.role, text: m.content || m.answer || '', error: m.error || '', stats: m.stats || '' })) : [],
    spoken: V.recent.map((r) => r.text), streaming: S.streaming, placeholder: document.querySelector('#input').placeholder })`)).result?.value;
  const s = JSON.parse(state || "{}");
  if (s.toast) toasts.add(s.toast);
  if (s.messages && s.messages.filter((m) => m.role === "assistant" && m.text).length >= NQ && !s.streaming && s.spoken.length) break;
  await new Promise((r) => setTimeout(r, 1000));
}
console.log(JSON.stringify({ click: click.result?.value, state: JSON.parse(state || "{}"), toasts: [...toasts], errors, seconds: (Date.now() - t0) / 1000 }));
ws.close();
"""


def call(method, path, body=None, timeout=60):
    c = http.client.HTTPConnection("127.0.0.1", 8765, timeout=timeout)
    c.request(method, path, json.dumps(body) if body is not None else None, H)
    r = c.getresponse()
    data = r.read()
    return r.status, json.loads(data) if data else None


def question_wav(path):
    """The questions in Sprout's voice at 48 kHz, GAP_S apart, then silence (Edge loops the file)."""
    voice.load_tts()
    parts = [np.zeros(48000 * 6)]  # speak after the mic is on
    for q in QUESTIONS:
        with wave.open(io.BytesIO(voice.speak(q))) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
            sr = w.getframerate()
        x48 = np.interp(np.linspace(0, len(x) - 1, int(len(x) * 48000 / sr)), np.arange(len(x)), x)
        parts += [x48, np.zeros(48000 * GAP_S - len(x48))]
    voice.unload()
    audio = np.concatenate(parts + [np.zeros(48000 * 60)])
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes((audio * 32767).astype("<i2").tobytes())


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", 8765)) == 0:
            sys.exit("Sprout is already running on port 8765. Quit it first (power button), then rerun.")
    tmp = Path(tempfile.mkdtemp())
    question_wav(tmp / "question.wav")
    (tmp / "settings.json").write_text(json.dumps({"user_name": "Hrithik S P", "pronouns": "he/him"}))
    (tmp / "driver.mjs").write_text(DRIVER, encoding="utf-8")
    code = ("import sys, pathlib; sys.argv=['server.py','--no-browser','--model','gemma']; "
            f"sys.path.insert(0, r'{ROOT / 'app'}'); import server; "
            f"server.SETTINGS_FILE = pathlib.Path(r'{tmp / 'settings.json'}'); server.DATA = pathlib.Path(r'{tmp}'); "
            "server.ACTIONS_LOG = server.DATA / 'actions.log'; server.main(sys.argv[1:])")
    srv = subprocess.Popen([sys.executable, "-c", code], cwd=str(ROOT))
    edge = None
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
        edge = subprocess.Popen([EDGE, "--headless=new", "--disable-gpu", f"--user-data-dir={tmp / 'edge'}",
                                 "--remote-debugging-port=9333", "--use-fake-ui-for-media-stream",
                                 "--use-fake-device-for-media-stream", f"--use-file-for-fake-audio-capture={tmp / 'question.wav'}",
                                 "--autoplay-policy=no-user-gesture-required", "http://127.0.0.1:8765/"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(4)
        out = subprocess.run(["node", str(tmp / "driver.mjs"), "9333", str(len(QUESTIONS))], capture_output=True, text=True,
                             timeout=200 + 60 * len(QUESTIONS))
        print(out.stderr.strip()[-1500:])
        print(out.stdout.strip())
        r = json.loads(out.stdout.strip().splitlines()[-1]) if out.stdout.strip() else {}
        msgs = r.get("state", {}).get("messages", [])
        default = QUESTION == "Who created you?"
        ok = (not r.get("errors") and any(m["role"] == "user" and m["text"] for m in msgs)
              and any(m["role"] == "assistant" and m["text"] and (not default or "Hrithik" in m["text"]) for m in msgs)
              and r["state"].get("spoken"))
        print("\nRESULT:", "PASS" if ok else "FAIL", f"({r.get('seconds', 0):.0f} s)")
    finally:
        if edge:
            edge.kill()
        try:
            call("POST", "/api/quit", {})
        except OSError:
            pass
        srv.wait(60)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
