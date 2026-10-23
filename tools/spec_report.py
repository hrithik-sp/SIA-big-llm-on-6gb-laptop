"""Compare speculative-decoding runs with the baseline of the same tag (results/spec.jsonl).

Per configuration: overall generation tok/s (token-weighted), speedup vs baseline,
draft acceptance rate, and losslessness = share of prompts whose temperature-0 output
is identical to the baseline's.

  python tools/spec_report.py --tag NAME
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preflight import ROOT  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--file", default=os.path.join(ROOT, "results", "spec.jsonl"))
    a = ap.parse_args(argv)
    with open(a.file, encoding="utf-8") as f:
        rows = [r for r in map(json.loads, f) if r["tag"] == a.tag and r.get("valid", True)]
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    # the reference is the first no-draft label ("baseline", or "base" in older tags)
    ref = next((lbl for lbl in by if lbl in ("baseline", "base")), next(iter(by), None))
    base = {r["prompt_id"]: r for r in by.get(ref, [])}

    def overall(rs):
        n = sum(r["gen_n"] for r in rs)
        return n / sum(r["gen_n"] / r["gen_tps"] for r in rs) if rs else float("nan")

    b_tps = overall(by.get(ref, []))
    print(f"{'config':<44} {'gen tok/s':>9} {'speedup':>8} {'accept':>7} {'identical':>10}")
    for label, rs in by.items():
        d = sum(r["draft_n"] or 0 for r in rs)
        acc = sum(r["draft_accepted"] or 0 for r in rs)
        same = [r["text_sha"] == base[r["prompt_id"]]["text_sha"] for r in rs if r["prompt_id"] in base]
        tps = overall(rs)
        print(f"{label[:44]:<44} {tps:>9.2f} {tps / b_tps:>7.2f}x "
              f"{(f'{100 * acc / d:.0f}%' if d else '-'):>7} {f'{sum(same)}/{len(same)}':>10}")
        for r in rs:
            b = base.get(r["prompt_id"])
            sp = r["gen_tps"] / b["gen_tps"] if b else float("nan")
            ar = f"{100 * r['accept_rate']:.0f}%" if r.get("accept_rate") is not None else "-"
            # how far the greedy output stays identical to the baseline's (characters)
            same = os.path.commonprefix([r["text"], b["text"]]) if b else ""
            agree = f"{len(same)}/{len(b['text'])} chars identical" if b else ""
            print(f"    {r['prompt_id']:<12} {r['gen_tps']:>7.2f} tok/s  x{sp:.2f}  accept {ar:>4}  {agree}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
