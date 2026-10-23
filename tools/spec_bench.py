"""Speculative decoding benchmark with llama-server (llama-bench has no speculative mode).

For each configuration: start llama-server (auto-fit with the draft model included,
so a draft's VRAM cost moves experts to the CPU, as in real use), send the fixed
prompts of tools/spec_prompts.json at temperature 0, and record per prompt:
generation tok/s, prompt tok/s, drafted / accepted tokens, and the output text.

Losslessness: greedy speculative decoding must produce the baseline's text. Runs
are compared against the baseline (no draft) of the same tag by tools/spec_report.py.

  python tools/spec_bench.py MODEL.gguf --tag NAME                       # baseline
  python tools/spec_bench.py MODEL.gguf --tag NAME --spec "--spec-type draft-eagle3 -md DRAFT.gguf"
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import preflight as pf  # noqa: E402

PROMPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec_prompts.json")
OUT = os.path.join(pf.ROOT, "results", "spec.jsonl")


def http(port, path, body=None, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_ready(port, proc, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            if http(port, "/health", timeout=5).get("status") == "ok":
                return True
        except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
            pass
        time.sleep(2)
    return False


SAMPLING = {
    "greedy": {"temperature": 0},
    # Gemma 4 generation_config.json (Google): what the Sprout app uses
    "gemma": {"temperature": 1.0, "top_k": 64, "top_p": 0.95},
}


def chat(port, prompt, max_tokens, sampling="greedy"):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "seed": 1, "stream": False, **SAMPLING[sampling]}
    r = http(port, "/v1/chat/completions", body)
    msg = r["choices"][0]["message"]
    text = (msg.get("reasoning_content") or "") + "\n<<content>>\n" + (msg.get("content") or "")
    return text, r.get("timings", {})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("model")
    ap.add_argument("--bin", default=bench.DEFAULT_BIN)
    ap.add_argument("--spec", default="", help="speculative args for llama-server ('' = baseline)")
    ap.add_argument("--label", default="", help="short name for this configuration")
    ap.add_argument("-c", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--fitt", type=int, default=256)
    ap.add_argument("--lm", default="none", help="load mode; 'auto' (mmap) for a reference that exceeds RAM")
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--cool-to", type=float, default=62)
    ap.add_argument("--extra", default="", help="extra llama-server args")
    ap.add_argument("--sampling", default="greedy", choices=list(SAMPLING),
                    help="greedy = temperature 0 (losslessness check); gemma = Google's recommended sampling")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)

    if not bench.on_ac():
        raise SystemExit("STOPPED: laptop is on battery; timed runs require AC power")
    with open(PROMPTS, encoding="utf-8") as f:
        prompts = json.load(f)

    cmd = [os.path.join(a.bin, "llama-server.exe"), "-m", a.model, "-c", str(a.c), "-t", "8", "-fa", "on",
           "-lm", a.lm, "-fitt", str(a.fitt), "--jinja", "-np", "1", "--host", "127.0.0.1",
           "--port", str(a.port), "--no-webui"] + bench.split_args(a.spec) + bench.split_args(a.extra)
    print(subprocess.list2cmdline(cmd))
    temp, waited = bench.cool_down(a.cool_to)
    log_path = os.path.join(bench.LLM_HOME, f"server-{a.tag}-{a.label or 'base'}.log")
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=bench.llama_env())
    mon = bench.Monitor(proc.pid)
    rows = []
    try:
        if not wait_ready(a.port, proc):
            raise SystemExit(f"server failed to start (see {log_path})")
        ram = pf.ram_available()
        # with mmap, low free RAM is expected (clean, file-backed page cache the OS can drop);
        # with private allocations (-lm none) it would mean pagefile thrashing
        if a.lm == "none" and ram < 0.5 * pf.GiB:
            raise SystemExit(f"STOPPED: only {ram / pf.GiB:.2f} GiB RAM left after loading; unsafe")
        chat(a.port, "Say hello.", 16)  # warm-up, not recorded
        for p in prompts:
            text, tm = chat(a.port, p["prompt"], a.max_tokens, a.sampling)
            drafted, accepted = tm.get("draft_n"), tm.get("draft_n_accepted")
            row = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "tag": a.tag,
                   "label": a.label or ("baseline" if not a.spec else a.spec), "spec": a.spec,
                   "model": a.model, "prompt_id": p["id"], "gen_tps": tm.get("predicted_per_second"),
                   "prompt_tps": tm.get("prompt_per_second"), "gen_n": tm.get("predicted_n"),
                   "draft_n": drafted, "draft_accepted": accepted,
                   "accept_rate": (accepted / drafted) if drafted else None,
                   "text_sha": hashlib.sha256(text.encode()).hexdigest()[:16], "text": text,
                   "sampling": a.sampling, "start_temp_c": temp, "power_mode": bench.power_mode.get_mode() if bench.power_mode else None,
                   "cmd": cmd}
            rows.append(row)
            acc = f", accepted {accepted}/{drafted} ({100 * accepted / drafted:.0f}%)" if drafted else ""
            print(f"  {p['id']:<10} gen {row['gen_tps']:.2f} tok/s ({row['gen_n']} tok), "
                  f"prompt {row['prompt_tps']:.1f} tok/s{acc}")
    finally:
        hw = mon.stop()
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    on_ac = bench.on_ac()
    with open(OUT, "a", encoding="utf-8") as f:
        for r in rows:
            r["hw"], r["valid"] = hw, on_ac
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if rows:
        n = sum(r["gen_n"] for r in rows)
        t = sum(r["gen_n"] / r["gen_tps"] for r in rows)
        print(f"overall: {n / t:.2f} tok/s over {n} generated tokens; VRAM peak {hw['vram_used_peak_mib']} MiB, "
              f"RAM free min {hw['ram_available_min_gib']} GiB" + ("" if on_ac else "  [INVALID: power unplugged]"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
