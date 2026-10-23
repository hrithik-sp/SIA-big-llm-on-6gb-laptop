"""Summarise the daily-use RAM-pressure test (tools/ram_pressure.ps1).

One row per (target, hold level): RAM actually held by the helper, decode tok/s of the
first rep (cold) and the mean of the last 3 reps (warm), SSD share, minimum available RAM,
validity. Records of an interrupted session stay in the file; --latest keeps only the most
recent run of each test (--since ISO-TIME selects by time instead).

  python tools/pressure_report.py --latest [--md]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preflight import ROOT  # noqa: E402
from report import validity  # noqa: E402

RUNS = os.path.join(ROOT, "results", "runs.jsonl")
HELD = os.path.join(ROOT, "results", "rampressure.jsonl")


def load(path, since):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if r.get("tag", "").startswith("rampressure-") and r.get("time", "") >= since]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", default="", help="ISO time; only records at or after it")
    ap.add_argument("--latest", action="store_true", help="only the most recent run of each test")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args(argv)
    held = {r["tag"]: r["held_gib"] for r in load(HELD, a.since)}  # last record per tag wins
    head = ["target", "held GiB", "cold tok/s", "warm tok/s", "disk share", "RAM min GiB", "valid"]
    out = []
    runs = load(RUNS, a.since)
    if a.latest:  # the last record per tag wins, like the helper's records above
        runs = list({r["tag"]: r for r in runs}.values())
    for r in runs:
        ts = r["llama_bench"].get("samples_ts") or []
        warm = ts[-3:] if len(ts) >= 3 else ts
        target = r["tag"].removeprefix("rampressure-").rsplit("-hold", 1)[0]
        out.append([target, f"{held.get(r['tag'], '?')}", f"{ts[0]:.1f}" if ts else "-",
                    f"{sum(warm) / len(warm):.1f}" if warm else "-",
                    f"{100 * r['disk_share']:.1f}%" if r.get("disk_share") is not None else "-",
                    f"{r['hw'].get('ram_available_min_gib')}", validity(r.get("disk_share"), r["hw"])])
    out.sort(key=lambda row: (row[0], float(row[1]) if row[1] != "?" else -1))
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
