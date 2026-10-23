"""Summarise results/runs.jsonl: one row per test, re-scored with the current validity rule.

  python tools/report.py [--tag TAG] [--md]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import MAX_DISK_SHARE, disk_share  # noqa: E402
from preflight import GiB, ROOT  # noqa: E402


def load(path, tag=None):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if tag is None or r["tag"] == tag]


def validity(share, hw):
    """Same rule as bench.py: SSD share within limit AND still on AC power after the run
    (records from before the AC check have no on_ac_after field)."""
    if hw.get("on_ac_after") is False:
        return "NO (battery)"
    if share is not None and share > MAX_DISK_SHARE:
        return "NO (SSD)"
    return "yes"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", default=os.path.join(ROOT, "results", "runs.jsonl"))
    ap.add_argument("--tag")
    ap.add_argument("--md", action="store_true", help="markdown table")
    a = ap.parse_args(argv)

    rows = load(a.file, a.tag)
    head = ["tag", "rnd", "config", "test", "tok/s", "± sd", "% roof", "disk share", "RAM min GiB",
            "GPU start/max C", "SM MHz", "valid"]
    out = []
    for r in rows:
        lb, pre = r["llama_bench"], r["preflight"]
        dgib = r.get("disk_read_during_test_gib")
        share = disk_share(lb, pre["cpu_weights"], pre["cpu_read_per_tok"],
                           dgib * GiB if dgib is not None else None)
        hw = r["hw"]
        out.append([
            r["tag"], str(r.get("round", 1)), r["label"], r["test"], f"{r['tps']:.2f}", f"{r['tps_sd']:.2f}",
            f"{100 * r['pct_roofline']:.0f}" if r.get("pct_roofline") else "-",
            f"{100 * share:.2f}%" if share is not None else "-",
            f"{hw.get('ram_available_min_gib')}",
            f"{hw.get('start_temp_c', '-')}/{hw.get('gpu_temp_max_c')}",
            f"{hw.get('sm_clock_busy_median_mhz')}",
            validity(share, hw),
        ])
    if a.md:
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for row in out:
            print("| " + " | ".join(row) + " |")
    else:
        w = [max(len(h), *(len(row[i]) for row in out)) if out else len(h) for i, h in enumerate(head)]
        print("  ".join(h.ljust(w[i]) for i, h in enumerate(head)))
        for row in out:
            print("  ".join(c.ljust(w[i]) for i, c in enumerate(row)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
