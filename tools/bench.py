"""Preflight-gated llama-bench sweeps with hardware monitoring and roofline comparison.

For every configuration in the sweep:
  1. placement: either what llama.cpp's auto-fit would choose (asked from
     llama-fit-params, then pinned explicitly so the run is deterministic), or an
     explicit --n-cpu-moe with all layers on the GPU;
  2. preflight: refuse to run if it does not fit the VRAM/RAM free right now;
  3. llama-bench (-o jsonl) while sampling GPU memory/temperature/clock/power and
     the benchmark process's RAM;
  4. one record per test appended to results/runs.jsonl, including the decode
     roofline and the fraction of it achieved.

Example:
  python tools/bench.py MODEL.gguf --place auto,0,4 -d 0,4096 --tag baseline
"""
import argparse
import datetime
import itertools
import json
import os
import random
import re
import shlex
import statistics
import subprocess
import sys
import threading
import time

import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preflight as pf  # noqa: E402

try:
    import power_mode  # Windows power slider overlay, recorded with every run
except OSError:
    power_mode = None

ROOT = pf.ROOT
# Large files (llama.cpp, models, data) live outside the repo: LLM_HOME, default ~/llm
LLM_HOME = os.environ.get("LLM_HOME") or os.path.join(os.path.expanduser("~"), "llm")
DEFAULT_BIN = os.path.join(LLM_HOME, "llama.cpp", "build-cuda", "bin")


def llama_env(extra=None):
    """Environment for llama.cpp processes, with the *persisted* PATH.

    A shell started before the CUDA Toolkit was installed lacks CUDA\\vX\\bin\\x64
    (cudart64_13.dll, cublas64_13.dll) and the binaries fail with 0xC0000135.
    `extra` adds variables such as GGML_OP_OFFLOAD_MIN_BATCH for one run.
    """
    env = dict(os.environ)
    env.update(extra or {})
    if sys.platform == "win32":
        import winreg
        parts = []
        for root, key in ((winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
                          (winreg.HKEY_CURRENT_USER, "Environment")):
            try:
                with winreg.OpenKey(root, key) as k:
                    parts.append(os.path.expandvars(winreg.QueryValueEx(k, "Path")[0]))
            except OSError:
                pass
        env["PATH"] = ";".join(parts + [env.get("PATH", "")])
    return env


class Monitor:
    """Samples nvidia-smi every 500 ms and the benchmark process's RAM while it runs."""

    FIELDS = ["memory.used", "temperature.gpu", "clocks.sm", "power.draw", "utilization.gpu"]

    def __init__(self, pid):
        self.proc = psutil.Process(pid)
        self.gpu, self.rss, self.avail = [], [], []
        self._stop = threading.Event()
        self.smi = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={','.join(self.FIELDS)}", "--format=csv,noheader,nounits", "-lms", "500"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        # system-wide disk reads: mmap'd weights evicted from RAM come back as page-ins from SSD
        self.disk0 = psutil.disk_io_counters().read_bytes
        self.threads = [threading.Thread(target=f, daemon=True) for f in (self._read_gpu, self._read_ram)]
        for t in self.threads:
            t.start()

    def _read_gpu(self):
        for line in self.smi.stdout:
            row = []
            for x in line.split(","):
                try:
                    row.append(float(x))
                except ValueError:  # "[N/A]"
                    row.append(None)
            if len(row) == len(self.FIELDS):
                self.gpu.append(row)

    def _read_ram(self):
        while not self._stop.wait(0.5):
            try:
                self.rss.append(self.proc.memory_info().rss)
            except psutil.Error:
                break
            self.avail.append(psutil.virtual_memory().available)

    def stop(self):
        self._stop.set()
        self.smi.terminate()
        for t in self.threads:
            t.join(3)
        col = {f: [r[i] for r in self.gpu if r[i] is not None] for i, f in enumerate(self.FIELDS)}
        busy = [r[2] for r in self.gpu if r[4] is not None and r[4] >= 30 and r[2] is not None]
        return {
            "vram_used_peak_mib": max(col["memory.used"], default=None),
            "gpu_temp_max_c": max(col["temperature.gpu"], default=None),
            "sm_clock_busy_median_mhz": statistics.median(busy) if busy else None,
            "power_max_w": max(col["power.draw"], default=None),
            "rss_peak_gib": round(max(self.rss) / pf.GiB, 2) if self.rss else None,
            "ram_available_min_gib": round(min(self.avail) / pf.GiB, 2) if self.avail else None,
            "disk_read_gib": round((psutil.disk_io_counters().read_bytes - self.disk0) / pf.GiB, 2),
            "samples": len(self.gpu),
        }


MAX_DISK_SHARE = 0.02  # a test is invalid if SSD reads could explain >2% of the bytes it needed


def disk_share(rec, cpu_weights, cpu_read_per_tok, test_disk):
    """Upper bound on the share of a test's CPU-side weight traffic that came from disk.

    Disk counters are system-wide (Defender, indexer, ...), so raw bytes are not a
    verdict. Compare them with what the timed runs had to read from RAM: decode reads
    the active CPU weights once per token; each prompt ubatch touches all CPU weights.
    """
    if test_disk is None:
        return None
    reps = len(rec.get("samples_ns") or []) or 1
    if rec["n_gen"] > 0:
        needed = reps * rec["n_gen"] * cpu_read_per_tok
    else:
        needed = reps * -(-rec["n_prompt"] // rec["n_ubatch"]) * cpu_weights
    return test_disk / needed if needed else None


# src/llama-moe-cache.cpp: "llama_moe_cache: ubatch <= 8: hits = H, misses = M, hit rate = R%, uploaded = U MiB"
MOE_CACHE_STATS = re.compile(r"llama_moe_cache: ubatch <= \d+: hits = (\d+), misses = (\d+), "
                             r"hit rate = ([\d.]+)%, uploaded = ([\d.]+) MiB")


def disk_read_bytes():
    return psutil.disk_io_counters().read_bytes


def run_with_phases(cmd, env_extra=None):
    """Run llama-bench --progress; return (stdout, stderr_tail, exit code, phases).

    phases[k] = disk bytes read during the timed part of benchmark k (from its first
    warmup line to the next benchmark's start), so model loading, which reads the
    whole file once, is not mistaken for weights being paged from SSD mid-test.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env=llama_env(env_extra))
    mon = Monitor(proc.pid)
    events, tail, cache = [], [], []

    def read_err():
        for line in proc.stderr:
            tail.append(line)
            del tail[:-200]
            m = MOE_CACHE_STATS.search(line)
            if m:  # printed when each test's context is freed, i.e. once per test, in order
                cache.append({"hits": int(m.group(1)), "misses": int(m.group(2)),
                              "hit_rate": float(m.group(3)) / 100, "uploaded_mib": float(m.group(4))})
            if line.startswith("llama-bench: benchmark "):
                k = int(line.split()[2].split("/")[0])
                events.append((k, "starting" if line.rstrip().endswith("starting") else "run", disk_read_bytes()))

    t = threading.Thread(target=read_err, daemon=True)
    t.start()
    out = proc.stdout.read()
    proc.wait()
    t.join(5)
    end = disk_read_bytes()
    hw = mon.stop()
    first_run, next_start = {}, {}
    for k, kind, d in events:
        if kind == "run":
            first_run.setdefault(k, d)
        elif k - 1 in first_run:
            next_start.setdefault(k - 1, d)
    phases = {k: next_start.get(k, end) - d for k, d in first_run.items()}
    if first_run:
        hw["disk_read_load_gib"] = round((min(first_run.values()) - mon.disk0) / pf.GiB, 2)
    hw["moe_cache"] = cache
    return out, "".join(tail), proc.returncode, hw, phases


def measured_compute(bin_dir, model, ctx, b, ub, ctk, fa, ngl, ot, ncmoe):
    """llama.cpp's own dry-run buffer sizes for this exact placement, in MiB.

    Returns {"CUDA0": (model, context, compute), "Host": (...)} from
    llama-fit-params --fit-print on (no weights are loaded). -b must match the
    benchmark: the effective micro-batch is min(-b, -ub).
    """
    cmd = [os.path.join(bin_dir, "llama-fit-params.exe"), "-m", model, "-c", str(ctx), "-b", str(b),
           "-ub", str(ub), "-ctk", ctk, "-ctv", ctk, "-fa", fa, "-ngl", str(ngl), "-fitp", "on"]
    if ncmoe:
        cmd += ["-ncmoe", str(ncmoe)]
    if ot:
        cmd += ["-ot", ot]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=llama_env())
    out = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 4 and all(x.isdigit() for x in parts[1:]):
            out[parts[0]] = tuple(int(x) for x in parts[1:])
    if r.returncode != 0 or "Host" not in out:
        raise RuntimeError(f"llama-fit-params -fitp failed ({r.returncode}):\n{r.stderr[-3000:]}")
    return out


def gpu_temp():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return float(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def on_ac():
    """True if the laptop runs on mains power. On battery Windows switches to the DC
    power-mode overlay and caps the GPU (measured: 30 W instead of 56 W, pp512 -26%)."""
    b = psutil.sensors_battery()
    return True if b is None else bool(b.power_plugged)


def cool_down(target_c, max_wait_s=600):
    """Laptops hit a thermal wall under sustained load (ATSInfer, arXiv:2607.10183, Fig. 4/12),
    so every configuration starts from the same GPU temperature."""
    t0 = time.time()
    temp = gpu_temp()
    while temp is not None and temp > target_c and time.time() - t0 < max_wait_s:
        time.sleep(5)
        temp = gpu_temp()
    return temp, round(time.time() - t0)


def split_args(s):
    r"""Split a command-line fragment Windows-style: backslashes in paths are kept (POSIX shlex
    drops them: C:\a\b -> C:ab), quotes group words and are removed."""
    return [t[1:-1] if len(t) > 1 and t[0] == t[-1] and t[0] in "'\"" else t
            for t in shlex.split(s, posix=False)]


def fit_params(bin_dir, model, ctx, b, ub, ctk, ctv, fa, fitt):
    """Ask llama-fit-params what auto-fit would choose: returns (n_ctx, ngl, -ot spec)."""
    cmd = [os.path.join(bin_dir, "llama-fit-params.exe"), "-m", model, "-c", str(ctx), "-b", str(b),
           "-ub", str(ub), "-ctk", ctk, "-ctv", ctv, "-fa", fa, "-fitt", str(fitt)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=llama_env())
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"llama-fit-params failed ({r.returncode}):\n{r.stderr[-3000:]}")
    toks = shlex.split(r.stdout.strip().splitlines()[-1])
    args = dict(zip(toks[::2], toks[1::2]))
    return int(args["-c"]), int(args["-ngl"]), args.get("-ot", "")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("model")
    ap.add_argument("--bin", default=DEFAULT_BIN,
                    help="comma list of build bin dirs (llama-bench.exe, llama-fit-params.exe) to compare")
    ap.add_argument("--vary", default="", help="'|'-separated alternative extra llama-bench args, e.g. "
                                                 "\"|-C 0x5555 --cpu-strict 1|--prio 2\" ('' = none)")
    ap.add_argument("--engine", default="llama", choices=["llama", "ik"],
                    help="ik = ik_llama.cpp llama-bench: placement and memory checks still come from "
                         "mainline llama-fit-params (identical tensor placement); -mmp instead of -lm, "
                         "JSON output, no -d/--delay/--progress")
    ap.add_argument("--fit-bin", default=DEFAULT_BIN, help="mainline bin dir for llama-fit-params (ik engine)")
    ap.add_argument("--place", default="auto", help="comma list: 'auto' (llama.cpp fit) and/or --n-cpu-moe values")
    ap.add_argument("--fitt", default="1024", help="comma list of fit margins in MiB (for 'auto')")
    ap.add_argument("-b", type=int, default=2048, help="logical batch size (llama-bench default 2048)")
    ap.add_argument("-ub", default="512", help="comma list of ubatch sizes (each must be <= -b)")
    ap.add_argument("-t", default="8", help="comma list of thread counts")
    ap.add_argument("-ctk", default="f16", help="comma list of KV cache types (K and V use the same)")
    ap.add_argument("-fa", default="1", choices=["0", "1"])
    ap.add_argument("--lm", default="auto", help="comma list of load modes: auto,none,mmap,mlock,mmap+mlock,dio")
    ap.add_argument("--mcm", default="0", help="comma list of --moe-cache-mib sizes (needs the patched llama-bench)")
    ap.add_argument("--env", default="", help="sweep one environment variable: NAME=v1,v2 ('-' = unset)")
    ap.add_argument("-p", type=int, default=512)
    ap.add_argument("-n", type=int, default=128)
    ap.add_argument("-d", default="0", help="comma list of depths (filled context) for each test")
    ap.add_argument("-r", type=int, default=5, help="repetitions")
    ap.add_argument("--delay", type=int, default=5, help="seconds of cool-down between tests")
    ap.add_argument("--allow-ssd", action="store_true", help="run even if CPU weights exceed free RAM")
    ap.add_argument("--ram-margin", type=float, default=1.0, help="GiB of RAM headroom required above the estimate")
    ap.add_argument("--cool-to", type=float, default=55, help="wait until the GPU is at or below this temp (C)")
    ap.add_argument("--rounds", type=int, default=1, help="repeat the whole sweep (thermal/noise control)")
    ap.add_argument("--shuffle", action="store_true", help="randomise configuration order in each round")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra", default="", help="extra llama-bench arguments")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run", action="store_true", help="preflight and print commands only")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "runs.jsonl"))
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)  # show progress live when piped

    m = pf.load_model(a.model)
    with open(os.path.join(ROOT, "results", "bandwidth.json")) as f:
        bw = json.load(f)
    depths = [int(x) for x in a.d.split(",")]
    ctx = a.p + a.n + max(depths)
    fa_on = a.fa == "1"
    env_name, _, env_vals = a.env.partition("=")
    env_vals = env_vals.split(",") if env_name else ["-"]
    bins = a.bin.split(",")
    varies = a.vary.split("|") if a.vary else [""]
    sweep = list(itertools.product(a.place.split(","), a.fitt.split(","), a.ub.split(","),
                                   a.t.split(","), a.ctk.split(","), a.lm.split(","), a.mcm.split(","),
                                   env_vals, bins, varies))
    # fit margins only matter for 'auto'; drop duplicate explicit placements
    sweep = list(dict.fromkeys((pl, ft if pl == "auto" else None, ub, t, ck, lm, mc, ev, bn, vr)
                               for pl, ft, ub, t, ck, lm, mc, ev, bn, vr in sweep))
    print(f"{len(sweep)} configuration(s) x {a.rounds} round(s), ctx {ctx}, depths {depths}\n")
    # llama.cpp caps n_batch at n_ctx and n_ubatch at n_batch (src/llama-context.cpp), so with a
    # short context different -b/-ub values silently run the same test
    capped = sorted({u for u in a.ub.split(",") if min(a.b, int(u)) > ctx}, key=int)
    if capped:
        print(f"WARNING: ctx {ctx} < ubatch {','.join(capped)}: llama.cpp caps the micro-batch at "
              f"{ctx}, so these ubatch values are not what runs. Use -d or a longer -p.\n")
    rng = random.Random(a.seed)
    order = []
    for rnd in range(1, a.rounds + 1):
        batch = list(sweep)
        if a.shuffle:
            rng.shuffle(batch)
        order += [(rnd, *c) for c in batch]

    rows = []
    for rnd, place, fitt, ub, t, ctk, lm, mcm, envv, bin_dir, vary in order:
        env_extra = {env_name: envv} if env_name and envv != "-" else {}
        ub_i = int(ub)
        if ub_i > a.b:
            print(f"[ub={ub}] SKIPPED: -ub {ub} > -b {a.b} would silently run with ub={a.b}")
            continue
        fit_dir = a.fit_bin if a.engine == "ik" else bin_dir
        if place == "auto":
            _, ngl, ot = fit_params(fit_dir, a.model, ctx, a.b, ub_i, ctk, ctk, "on" if fa_on else "off", fitt)
            label, ncmoe = f"auto(fitt={fitt}): -ngl {ngl}", 0
        else:
            ngl, ot, ncmoe = 99, "", int(place)
            label = f"-ncmoe {ncmoe}"
        label += f" lm={lm}" + (f" mcm={mcm}" if mcm != "0" else "") + \
            "".join(f" {k}={v}" for k, v in env_extra.items()) + \
            (f" [{os.path.basename(os.path.dirname(bin_dir.rstrip(os.sep)))}]" if len(bins) > 1 else "") + \
            (" [ik_llama.cpp]" if a.engine == "ik" else "") + \
            (f" {{{vary}}}" if vary else "")
        meas = measured_compute(fit_dir, a.model, ctx, a.b, ub_i, ctk, "on" if fa_on else "off", ngl, ot, ncmoe)
        gpu_dev = next((k for k in meas if k != "Host"), None)
        cfg = pf.Config(ngl=ngl, n_cpu_moe=ncmoe, n_ctx=ctx, ctk=ctk, ctv=ctk, ubatch=ub_i,
                        flash_attn=fa_on, overrides=pf.parse_overrides(ot),
                        mmap=lm in ("auto", "mmap", "mmap+mlock"),
                        compute_mib=meas[gpu_dev][2] if gpu_dev else 0,
                        host_compute_mib=meas["Host"][2], moe_cache_mib=float(mcm),
                        kv_gpu_mib=meas[gpu_dev][1] if gpu_dev else None)
        p = pf.plan(m, cfg)
        # cross-check preflight's weight placement against llama.cpp's own numbers
        if gpu_dev:
            ours, theirs = p.gpu_weights / pf.MiB, meas[gpu_dev][0]
            if abs(ours - theirs) > max(64, 0.02 * theirs):
                print(f"  NOTE: preflight GPU weights {ours:.0f} MiB vs llama.cpp {theirs} MiB")
        gpu, ram = pf.gpu_memory(), pf.ram_available()
        gpu_ok = gpu is not None and p.gpu_need <= gpu[0]
        ram_ok = p.ram_need + a.ram_margin * pf.GiB <= ram or a.allow_ssd
        head = f"[r{rnd} {label} ub={ub} t={t} ctk={ctk}]"
        if not (gpu_ok and ram_ok):
            print(f"{head} SKIPPED by preflight: VRAM need {p.gpu_need / pf.GiB:.2f}/"
                  f"{(gpu[0] if gpu else 0) / pf.GiB:.2f} GiB, RAM need {p.ram_need / pf.GiB:.2f}/{ram / pf.GiB:.2f} GiB")
            continue

        common = ["-r", str(a.r), "-p", str(a.p), "-n", str(a.n), "-ngl", str(ngl), "-b", str(a.b), "-ub", ub,
                  "-t", t, "-ctk", ctk, "-ctv", ctk, "-fa", a.fa]
        if a.engine == "ik":
            if depths != [0] or lm not in ("auto", "none", "mmap"):
                print(f"{head} SKIPPED: ik llama-bench has no -d, and only -mmp 0|1 for load mode")
                continue
            cmd = [os.path.join(bin_dir, "llama-bench.exe"), "-m", a.model, "-o", "json", *common,
                   "-mmp", "0" if lm == "none" else "1"]
            if ncmoe:
                cmd += ["--n-cpu-moe", str(ncmoe)]
            if ot:
                cmd += ["-ot", ot]  # ik parses ',' between overrides (common parse_buft_overrides)
        else:
            cmd = [os.path.join(bin_dir, "llama-bench.exe"), "-m", a.model, "-o", "jsonl", "--progress",
                   "--delay", str(a.delay), "-d", a.d, *common, "-lm", lm]
            if ncmoe:
                cmd += ["-ncmoe", str(ncmoe)]
            if ot:
                cmd += ["-ot", ot.replace(",", ";")]  # llama-bench: ',' separates sweep groups, ';' overrides
        if mcm != "0":
            cmd += ["--moe-cache-mib", mcm, "-v"]  # -v: the cache logs its hit rate when each test ends
        cmd += split_args(a.extra) + split_args(vary)
        print(f"{head} preflight OK (VRAM {p.gpu_need / pf.GiB:.2f} GiB est.)\n  {subprocess.list2cmdline(cmd)}")
        if a.dry_run:
            continue

        if not on_ac():
            print("  STOPPED: laptop is on battery; timed runs require AC power")
            break
        start_temp, waited = cool_down(a.cool_to)
        print(f"  GPU at {start_temp} C after {waited} s cool-down")
        out, err, code, hw, phases = run_with_phases(cmd, env_extra)
        hw["start_temp_c"], hw["cooldown_s"] = start_temp, waited
        hw["power_mode"] = power_mode.get_mode() if power_mode else None
        hw["on_ac_after"] = on_ac()
        if not hw["on_ac_after"]:
            print("  WARNING: power was unplugged during this run; its results are marked invalid")
        if code != 0:
            print(f"  FAILED (exit {code}):\n{err[-3000:]}")
            continue

        if a.engine == "ik":  # one JSON array; no per-phase disk data (no --progress)
            start = out.find("[")
            recs = json.loads(out[start:]) if start >= 0 else []
            for rec in recs:
                rec.setdefault("n_depth", 0)
        else:
            recs = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
        paged = False
        for k, rec in enumerate(recs, start=1):
            test_disk = phases.get(k)
            share = disk_share(rec, p.cpu_weights, p.cpu_read_per_tok, test_disk)
            valid = (share is None or share <= MAX_DISK_SHARE) and hw["on_ac_after"]
            paged |= not valid
            roof = pct = None
            if rec["n_gen"] > 0:
                pd = pf.plan(m, pf.Config(**{**pf.asdict(cfg), "depth": rec["n_depth"]}))
                roof = pf.roofline_tps(pd, bw["gpu_read_GBps"], bw["cpu_read_GBps"])
                pct = rec["avg_ts"] / roof
            test = f"pp{rec['n_prompt']}" if rec["n_prompt"] else f"tg{rec['n_gen']}"
            test += f"@d{rec['n_depth']}" if rec["n_depth"] else ""
            row = {
                "time": datetime.datetime.now().isoformat(timespec="seconds"), "tag": a.tag, "round": rnd,
                "label": label, "test": test, "tps": rec["avg_ts"], "tps_sd": rec["stddev_ts"],
                "roofline_tps": roof, "pct_roofline": pct, "cmd": cmd,
                "placement": {"ngl": ngl, "n_cpu_moe": ncmoe, "ot": ot, "fitt": fitt},
                "env": env_extra,
                "llama_cpp_memory_mib": meas,
                "preflight": pf.asdict(p), "hw": hw, "llama_bench": rec,
                "disk_read_during_test_gib": round(test_disk / pf.GiB, 3) if test_disk is not None else None,
                "disk_share": share,
                "valid": valid,
                # one stats line per test is expected; if counts differ, keep all and flag them
                "moe_cache": (None if mcm == "0" else hw["moe_cache"][k - 1]
                              if len(hw["moe_cache"]) == len(recs) else {"unaligned": hw["moe_cache"]}),
            }
            rows.append(row)
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
        print(f"  done: VRAM peak {hw['vram_used_peak_mib']} MiB, GPU max {hw['gpu_temp_max_c']} C, "
              f"RSS peak {hw['rss_peak_gib']} GiB, RAM free min {hw['ram_available_min_gib']} GiB, "
              f"disk read {hw['disk_read_gib']} GiB (load {hw.get('disk_read_load_gib')} GiB), "
              f"during timed runs: {', '.join(f'{v / pf.GiB:.2f}' for v in phases.values())} GiB")
        if paged:
            print(f"  WARNING: disk reads could explain >{MAX_DISK_SHARE:.0%} of a test's weight traffic "
                  "(weights paged from SSD; RAM too small) -> affected tests are marked valid=false")

    if rows:
        print(f"\n{'config':<34} {'test':<12} {'tok/s':>14} {'roofline':>9} {'%':>5}")
        for r in rows:
            roof = f"{r['roofline_tps']:.1f}" if r["roofline_tps"] else "-"
            pct = f"{100 * r['pct_roofline']:.0f}" if r["pct_roofline"] else "-"
            name = f"{r['label']} ub={r['llama_bench']['n_ubatch']} t={r['llama_bench']['n_threads']}"
            flag = "" if r["valid"] else "  INVALID (SSD paging)"
            print(f"{name:<34} {r['test']:<12} {r['tps']:>8.2f} ±{r['tps_sd']:<5.2f} {roof:>9} {pct:>5}{flag}")
        print(f"\nappended {len(rows)} record(s) to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
