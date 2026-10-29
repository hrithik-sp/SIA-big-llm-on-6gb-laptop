"""Quality of a quantized model as KL-divergence against a reference (llama-perplexity).

Method: Dutta et al., "Accuracy is Not All You Need" (NeurIPS 2024) — compressed
models can match accuracy while answers flip, so measure the distance to the
reference: mean/percentile KLD and "same top p" (100% − flip rate). Implementation:
llama.cpp tools/perplexity (--kl-divergence-base / --kl-divergence).

  # 1) reference logits (once per reference model, text, context and chunk count)
  python tools/quality.py base REF.gguf --chunks 100 [--allow-ssd]
  # 2) each candidate against that reference
  python tools/quality.py kld QUANT.gguf --base-model REF.gguf --chunks 100

Results are appended to results/quality.jsonl. Before each run the same preflight
gate as tools/bench.py checks VRAM/RAM; a reference that does not fit in RAM may
only run with --allow-ssd (weights stream from SSD: slow, values unaffected).
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import preflight as pf  # noqa: E402

DATA = os.path.join(bench.LLM_HOME, "data", "wikitext-2-raw", "wiki.test.raw")
KLD_DIR = os.path.join(bench.LLM_HOME, "kld")
OUT = os.path.join(pf.ROOT, "results", "quality.jsonl")

# "Mean    KLD:   0.012345 ±   0.000123", "99.0%   KLD:   0.1", "Same top p:  95.123 ± 0.1 %"
_STAT = re.compile(r"^\s*([A-Za-z0-9.%()/ ΔÎ\-]+?)\s*:\s*(-?[\d.]+)\s*(?:±|Â±)?\s*(-?[\d.]+)?")
_FINAL = re.compile(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)")


def kld_path(ref, ctx, chunks, data=DATA, window=None):
    name = os.path.splitext(os.path.basename(ref))[0]
    # wikitext keeps the original names (existing files); other texts get their own
    tag = "" if os.path.abspath(data) == os.path.abspath(DATA) else "-" + os.path.splitext(os.path.basename(data))[0]
    # a reference saved with a non-default window must never be compared under another one
    win = f"-w{window:g}" if window and window != 16 else ""
    return os.path.join(KLD_DIR, f"{name}{tag}-c{ctx}-ch{chunks}{win}.kld")


def parse(text):
    stats, in_stats = {}, False
    for line in text.splitlines():
        if line.startswith("====== Perplexity statistics") or line.startswith("Mean PPL(Q)"):
            in_stats = True
        if not in_stats:
            continue
        m = _STAT.match(line)
        if m:
            key = re.sub(r"\s+", " ", m.group(1).replace("Î", "Δ")).strip()
            stats[key] = float(m.group(2))
            if m.group(3):
                stats[key + " ±"] = float(m.group(3))
    m = _FINAL.search(text)
    if m:
        stats["Final PPL"], stats["Final PPL ±"] = float(m.group(1)), float(m.group(2))
    return stats


def gate(model, ctx, b, ub, lm, allow_ssd, ram_margin, fitt=256):
    """Same check as tools/bench.py: auto-fit placement + llama.cpp-measured buffers."""
    m = pf.load_model(model)
    _, ngl, ot = bench.fit_params(bench.DEFAULT_BIN, model, ctx, b, ub, "f16", "f16", "on", fitt)
    meas = bench.measured_compute(bench.DEFAULT_BIN, model, ctx, b, ub, "f16", "on", ngl, ot, 0)
    gpu_dev = next((k for k in meas if k != "Host"), None)
    cfg = pf.Config(ngl=ngl, n_ctx=ctx, ubatch=ub, overrides=pf.parse_overrides(ot),
                    mmap=lm in ("auto", "mmap", "mmap+mlock"),
                    compute_mib=meas[gpu_dev][2] if gpu_dev else 0, host_compute_mib=meas["Host"][2],
                    kv_gpu_mib=meas[gpu_dev][1] if gpu_dev else None)
    p = pf.plan(m, cfg)
    gpu, ram = pf.gpu_memory(), pf.ram_available()
    gpu_ok = gpu is not None and p.gpu_need <= gpu[0]
    ram_ok = p.ram_need + ram_margin * pf.GiB <= ram
    print(f"preflight: VRAM need {p.gpu_need / pf.GiB:.2f} / free {gpu[0] / pf.GiB:.2f} GiB; "
          f"RAM need {p.ram_need / pf.GiB:.2f} (+{ram_margin} margin) / available {ram / pf.GiB:.2f} GiB")
    if not gpu_ok:
        raise SystemExit("DO NOT RUN: does not fit in VRAM")
    if not ram_ok and not allow_ssd:
        raise SystemExit("DO NOT RUN: CPU-side weights exceed free RAM (use --allow-ssd only for a "
                         "reference run, where speed does not matter)")
    return ngl, ot, meas, not ram_ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("mode", choices=["base", "kld"])
    ap.add_argument("model")
    ap.add_argument("--base-model", help="reference model whose logits were saved (kld mode)")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("-c", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=100)
    ap.add_argument("-b", type=int, default=2048)
    ap.add_argument("-ub", type=int, default=512)
    ap.add_argument("-t", type=int, default=8)
    ap.add_argument("--lm", default=None, help="load mode (default: none for kld, auto=mmap for base)")
    ap.add_argument("--allow-ssd", action="store_true")
    ap.add_argument("--bin", default=bench.DEFAULT_BIN, help="bin dir with llama-perplexity.exe")
    ap.add_argument("--extra", default="", help="extra llama-perplexity args (use --extra=\"-x ...\")")
    ap.add_argument("--fitt", type=int, default=256, help="auto-fit VRAM margin in MiB (placement)")
    ap.add_argument("--parse-special", action="store_true",
                    help="tokenize special tokens in --data (chat corpora; needs the perplexity patch)")
    ap.add_argument("--engine", default="llama", choices=["llama", "ik"],
                    help="ik: ik_llama.cpp flags (--no-mmap instead of -lm); placement from mainline fit")
    ap.add_argument("--ram-margin", type=float, default=1.0)
    ap.add_argument("--tag", default="", help="study tag stored with the record (e.g. qat-study-v1)")
    ap.add_argument("--kld-window", type=float, default=None,
                    help="nats of log-probability below the top one that the reference keeps (llama.cpp: 16; needs "
                         "patches/llama-perplexity-kld-window.patch). Use the same value for 'base' and 'kld'")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

    ref = a.model if a.mode == "base" else a.base_model
    if not ref:
        ap.error("kld mode needs --base-model")
    kld_file = kld_path(ref, a.c, a.chunks, a.data, a.kld_window)
    if a.mode == "kld" and not os.path.exists(kld_file):
        ap.error(f"reference logits not found: {kld_file} (run 'base' first)")
    # The reference does not fit in RAM: mmap lets it stream from SSD instead of pagefile thrash.
    lm = a.lm or ("auto" if a.mode == "base" else "none")

    ngl, ot, meas, ssd = gate(a.model, a.c, a.b, a.ub, lm, a.allow_ssd, a.ram_margin, a.fitt)
    print(f"placement: -ngl {ngl}, {ot.count('=CPU')} CPU overrides (fitt {a.fitt})")
    os.makedirs(KLD_DIR, exist_ok=True)
    cmd = [os.path.join(a.bin, "llama-perplexity.exe"), "-m", a.model, "-f", a.data,
           "-c", str(a.c), "--chunks", str(a.chunks), "-b", str(a.b), "-ub", str(a.ub), "-t", str(a.t),
           "-fa", "on", "-ngl", str(ngl)]
    if a.engine == "ik":
        cmd += ["--no-mmap"] if lm == "none" else []
    else:
        cmd += ["-lm", lm]
    if ot:
        cmd += ["-ot", ot]
    cmd += a.extra.split()
    cmd += ["--kl-divergence-base", kld_file] + (["--kl-divergence"] if a.mode == "kld" else [])
    print(subprocess.list2cmdline(cmd))

    t0 = time.time()
    env_extra = {"LLAMA_PERPLEXITY_PARSE_SPECIAL": "1"} if a.parse_special else {}
    if a.kld_window:
        env_extra["LLAMA_PERPLEXITY_KLD_WINDOW"] = str(a.kld_window)
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=bench.llama_env(env_extra))
    dt = time.time() - t0
    text = r.stdout + "\n" + r.stderr
    log = os.path.join(KLD_DIR, os.path.basename(kld_file) + f".{a.mode}.{a.engine}.{os.path.basename(a.model)}.log")
    with open(log, "w", encoding="utf-8") as f:
        f.write(text)
    if r.returncode != 0:
        print(text[-3000:])
        raise SystemExit(f"llama-perplexity failed ({r.returncode}); log: {log}")

    stats = parse(text)
    rec = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "tag": a.tag, "mode": a.mode, "engine": a.engine,
           "extra": a.extra, "parse_special": a.parse_special, "kld_window": a.kld_window or 16,
           "model": a.model, "reference": ref, "data": a.data, "ctx": a.c, "chunks": a.chunks,
           "seconds": round(dt), "streamed_from_ssd": ssd, "placement": {"ngl": ngl, "ot": ot},
           "llama_cpp_memory_mib": meas, "kld_file": kld_file, "log": log, "cmd": cmd, "stats": stats}
    if a.mode == "kld" and "Mean PPL(base)" in stats:
        # the stored reference keeps a 16-nat window per token (perplexity.cpp log_softmax);
        # if PPL(base) recomputed from it disagrees with the base run, KLD is distorted
        base_runs = []
        if os.path.exists(OUT):
            with open(OUT, encoding="utf-8") as f:
                base_runs = [r for r in map(json.loads, f) if r["mode"] == "base" and r["kld_file"] == kld_file]
        if base_runs and "Final PPL" in base_runs[-1]["stats"]:
            ref_ppl = base_runs[-1]["stats"]["Final PPL"]
            ratio = stats["Mean PPL(base)"] / ref_ppl
            rec["base_ppl_check"] = {"base_run": ref_ppl, "kld_pass": stats["Mean PPL(base)"], "ratio": ratio}
            if abs(ratio - 1) > 0.02:
                severity = "KLD is distorted" if abs(ratio - 1) > 0.25 else \
                    "a few true tokens are clipped; KLD is only slightly affected, rankings hold"
                print(f"  WARNING: PPL(base) from stored logits {stats['Mean PPL(base)']:.2f} != base run "
                      f"{ref_ppl:.2f} (ratio {ratio:.3f}): reference partly outside the 16-nat storage "
                      f"window; {severity}")
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    keys = ["Final PPL", "Mean PPL(Q)", "Mean PPL(base)", "Mean KLD", "99.0% KLD", "Same top p", "RMS Δp"]
    for k in keys:
        if k in stats:
            print(f"  {k:<16} {stats[k]:.6f}" + (f" ± {stats[k + ' ±']:.6f}" if k + " ±" in stats else ""))
    print(f"done in {dt / 60:.1f} min; appended to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
