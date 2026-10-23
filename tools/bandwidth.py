"""Measure sustained memory read bandwidth of system RAM (CPU) and VRAM (GPU).

LLM token generation is memory-bandwidth bound: every generated token must stream
all *active* weights once. These measured numbers are the roofline that later
benchmarks are compared against (achieved tok/s vs. theoretical max tok/s).

Footprint: ~512 MiB of system RAM and ~1 GiB of VRAM, freed on exit.
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def cpu_read_bw(mib: int, threads: int, reps: int) -> float:
    """Multi-threaded read bandwidth in GB/s.

    Prefers torch (AVX2/AVX-512 vectorised, OpenMP-threaded reduction). numpy's
    float sum is compute-bound (~3.5 GB/s/thread here), so it under-reports.
    """
    n = mib * 1024 * 1024 // 4
    try:
        import torch
        torch.set_num_threads(threads)
        x = torch.ones(n, dtype=torch.float32)
        x.sum()
        best = 0.0
        for _ in range(reps):
            t0 = time.perf_counter()
            x.sum()
            best = max(best, x.numel() * 4 / (time.perf_counter() - t0) / 1e9)
        return best
    except ImportError:
        pass
    buf = np.ones(n, dtype=np.float32)
    chunks = np.array_split(buf, threads)
    best = 0.0
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(np.sum, chunks))  # warm-up: fault in pages, spin up threads
        for _ in range(reps):
            t0 = time.perf_counter()
            list(ex.map(np.sum, chunks))
            dt = time.perf_counter() - t0
            best = max(best, buf.nbytes / dt / 1e9)
    return best


def gpu_read_bw(mib: int, reps: int):
    try:
        import torch
    except ImportError:
        return None, "torch not installed"
    if not torch.cuda.is_available():
        return None, f"torch {torch.__version__} has no CUDA"
    n = mib * 1024 * 1024 // 4
    x = torch.ones(n, dtype=torch.float32, device="cuda")
    x.sum()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = 0.0
    for _ in range(reps):
        start.record()
        x.sum()
        end.record()
        torch.cuda.synchronize()
        dt = start.elapsed_time(end) / 1e3
        best = max(best, x.numel() * 4 / dt / 1e9)
    name = torch.cuda.get_device_name(0)
    del x
    torch.cuda.empty_cache()
    return best, name


def h2d_bw(mib: int, reps: int):
    """Host->GPU copy bandwidth (GB/s) over PCIe, from pinned and pageable host memory.

    Bounds MoE-cache misses and weight uploads during prompt processing.
    """
    try:
        import torch
    except ImportError:
        return {}
    if not torch.cuda.is_available():
        return {}
    res = {}
    for pinned in (True, False):
        x = torch.empty(mib * 1024 * 1024, dtype=torch.uint8, pin_memory=pinned)
        y = torch.empty_like(x, device="cuda")
        y.copy_(x)
        torch.cuda.synchronize()
        best = 0.0
        for _ in range(reps):
            t0 = time.perf_counter()
            y.copy_(x, non_blocking=pinned)
            torch.cuda.synchronize()
            best = max(best, x.numel() / (time.perf_counter() - t0) / 1e9)
        res["h2d_pinned_GBps" if pinned else "h2d_pageable_GBps"] = round(best, 1)
        del x, y
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cpu-mib", type=int, default=512)
    ap.add_argument("--gpu-mib", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=os.cpu_count() // 2)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--json", help="write results to this file")
    a = ap.parse_args()

    res = {"cpu_threads": a.threads}
    for t in sorted({1, a.threads, os.cpu_count()}):
        res[f"cpu_read_GBps_{t}t"] = round(cpu_read_bw(a.cpu_mib, t, a.reps), 1)
    res["cpu_read_GBps"] = max(v for k, v in res.items() if k.startswith("cpu_read_GBps_"))
    g, info = gpu_read_bw(a.gpu_mib, a.reps)
    res["gpu"] = info
    res["gpu_read_GBps"] = round(g, 1) if g else None
    res.update(h2d_bw(256, a.reps))

    print(json.dumps(res, indent=2))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
