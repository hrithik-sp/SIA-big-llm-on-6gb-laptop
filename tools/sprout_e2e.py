"""End-to-end check of the Sprout app (app/server.py) with the real models.

Starts Sprout without a window, then drives its API like the UI does: Chat identity,
See (image text + shapes), Read (PDF page citation), Library (retrieval + citation),
Think deep (gpt-oss, model switch), Stop, and Quit (no llama-server left behind).
Each scenario checks for a specific expected answer.

  python tools/sprout_e2e.py TESTDIR      # TESTDIR holds vision_test.png, lecture.pdf, notes_*.txt
"""
import base64
import http.client
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
PORT = 8765
ORIGIN = {"Origin": f"http://127.0.0.1:{PORT}"}
TEST_LIB = "e2e-test"  # never the user's own Library


def req(method, path, body=None, headers=None, timeout=900):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    h = {**ORIGIN, **(headers or {})}
    if isinstance(body, (dict, list)):
        body, h["Content-Type"] = json.dumps(body), "application/json"
    c.request(method, path, body, h)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, json.loads(data) if data else None


def chat(mode, messages, model=None, think=None, stop_after_s=None):
    """Stream one answer; returns (answer, reasoning, meta, timings, seconds)."""
    body = {"mode": mode, "messages": messages, "library": TEST_LIB}
    if model:
        body["model"] = model
    if think is not None:
        body["think"] = think
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=1800)
    c.request("POST", "/api/chat", json.dumps(body), {**ORIGIN, "Content-Type": "application/json"})
    r = c.getresponse()
    t0, answer, reasoning, meta, timings = time.time(), "", "", {}, {}
    while True:
        if stop_after_s and time.time() - t0 > stop_after_s:
            c.close()  # what the Stop button does: drop the connection
            break
        line = r.readline()
        if not line:
            break
        if not line.startswith(b"data: "):
            continue
        ev = json.loads(line[6:])
        if ev.get("type") == "meta":
            meta = ev
        elif ev.get("type") == "error":
            raise RuntimeError(ev["error"])
        elif "choices" in ev:
            d = ev["choices"][0].get("delta", {})
            answer += d.get("content") or ""
            reasoning += d.get("reasoning_content") or ""
            timings = ev.get("timings") or timings
    c.close()
    return answer, reasoning, meta, timings, time.time() - t0


def user(text, images=(), docs=()):
    t = "\n\n".join(f'<document name="{n}">\n{d}\n</document>' for n, d in docs)
    t = (t + "\n\n" + text) if t else text
    if not images:
        return {"role": "user", "content": t}
    return {"role": "user", "content": [{"type": "text", "text": t}] +
            [{"type": "image_url", "image_url": {"url": u}} for u in images]}


RESULTS = []


def check(name, ok, detail):
    RESULTS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def llama_procs():
    return [p for p in psutil.process_iter(["name"]) if (p.info["name"] or "").lower() == "llama-server.exe"]


def main():
    tdir = Path(sys.argv[1])
    if llama_procs():
        sys.exit("a llama-server is already running; stop it first")
    ram0 = psutil.virtual_memory().available
    proc = subprocess.Popen([sys.executable, str(ROOT / "app" / "server.py"), "--no-browser", "--no-autoquit"],
                            cwd=str(ROOT))
    for _ in range(120):
        if proc.poll() is not None:
            sys.exit(f"Sprout exited during startup (code {proc.returncode})")
        try:
            if req("GET", "/api/status", timeout=3)[0] == 200:
                break
        except OSError:
            time.sleep(0.5)
    st = req("GET", "/api/status")[1]
    print("status:", st, flush=True)

    # Chat: identity + speed (first request also waits for the model to load)
    a, _, meta, tm, s = chat("chat", [user("What's your name? Answer in one short sentence.")])
    check("chat identity", "sprout" in a.lower(), f"{a.strip()[:100]!r} ({s:.1f} s incl. load, {tm.get('predicted_per_second', 0):.1f} tok/s)")

    # See: text + shapes in a generated image
    img = "data:image/png;base64," + base64.b64encode((tdir / "vision_test.png").read_bytes()).decode()
    a, _, meta, tm, s = chat("see", [user("Extract all the text in this image, then list the shapes and their colours.", [img])])
    al = a.lower()
    check("see: reads text", "sprout test 42" in al, f"{s:.1f} s")
    check("see: shapes/colours", "circle" in al and "red" in al and ("rectangle" in al or "square" in al), a.strip().replace("\n", " ")[:220])

    # Read: PDF through the extract endpoint, answer must cite page 2
    code, ex = req("POST", "/api/extract", (tdir / "lecture.pdf").read_bytes(), {"X-Filename": "lecture.pdf"})
    check("read: pdf extraction", code == 200 and "14 November" in ex["text"], f"{ex.get('pages')} pages, {ex.get('chars')} chars")
    a, _, _, tm, s = chat("read", [user("When is the project deadline, and on which page does it say so?",
                                        docs=[("lecture.pdf", ex["text"])])])
    check("read: answer + page", "14 november" in a.lower() and "2" in a, a.strip().replace("\n", " ")[:200])

    # Library: index two notes files, ask a question only one answers, expect its citation
    for f in ("notes_networks.txt", "notes_databases.txt"):
        code, r = req("POST", "/api/kb/add", (tdir / f).read_bytes(), {"X-Filename": f, "X-Library": TEST_LIB})
        check(f"library: add {f}", code == 200, r)
    a, _, meta, _, s = chat("library", [user("What does ACID stand for according to my notes?")])
    srcs = [x["file"] for x in meta.get("sources", [])]
    check("library: retrieval ranks the right file first", srcs[:1] == ["notes_databases.txt"], srcs)
    check("library: answer + citation", "atomicity" in a.lower() and "[" in a, a.strip().replace("\n", " ")[:200])

    req("POST", "/api/kb/delete", {"name": TEST_LIB})

    # Think deep: gpt-oss (Sprout swaps the chat server), a problem with a known answer
    a, reasoning, meta, tm, s = chat("think", [user("How many ways are there to tile a 2x10 board with 1x2 dominoes? Give the number.")])
    check("think: model + answer", meta.get("model") == "gptoss" and "89" in a,
          f"{a.strip()[-120:]!r}; reasoning {len(reasoning)} chars; {s:.1f} s incl. model switch; {tm.get('predicted_per_second', 0):.1f} tok/s")

    # Stop: drop the connection mid-answer; the next request must still work promptly
    chat("chat", [user("Write a 2000-word story about a lighthouse.")], stop_after_s=6)
    a, _, _, _, s = chat("chat", [user("Reply with just the word OK.")], model="gptoss")
    check("stop: next request works", "ok" in a.lower() and s < 60, f"{a.strip()!r} in {s:.1f} s")

    # Quit: everything stops and memory comes back
    req("POST", "/api/quit")
    proc.wait(60)
    time.sleep(5)
    left = llama_procs()
    check("quit: no llama-server left", not left, f"{len(left)} left; RAM available now "
          f"{psutil.virtual_memory().available / 2**30:.1f} GiB (before start {ram0 / 2**30:.1f})")
    n_ok = sum(ok for _, ok in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
