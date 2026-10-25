"""Smart Turn (app/turn.py) on the English rows of its official test set: how often does it cut a speaker off
(says "finished" when the turn was not) and how often does it make a finished turn wait?

Data: one parquet shard of pipecat-ai/smart-turn-data-v3.2-test (CC BY 4.0), e.g. data/train-00000-of-00010.parquet.
Needs soundfile (FLAC) and pyarrow:  python tools/sia_turn_eval.py SHARD.parquet
"""
import collections
import io
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import turn  # noqa: E402


def main():
    rows = pq.read_table(sys.argv[1]).to_pylist()
    res = collections.defaultdict(lambda: [0, 0, 0, 0])  # finished+said finished, cut off, held, needless wait
    for r in rows:
        if r["language"] != "eng":
            continue
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != turn.RATE:
            x = np.interp(np.linspace(0, len(x) - 1, int(len(x) * turn.RATE / sr)), np.arange(len(x)), x)
        said, real = turn.complete_probability(x) > 0.5, bool(r["endpoint_bool"])
        for key in ("synthetic" if r["synthetic"] else "human", r["dataset"]):
            res[key][0 if said and real else 1 if said else 2 if not real else 3] += 1
    for key, (tp, fp, tn, fn) in sorted(res.items()):
        n = tp + fp + tn + fn
        print(f"{key:24} n={n:4}  accuracy {100 * (tp + tn) / n:5.1f}%  cut off {fp}/{fp + tn} unfinished, "
              f"held {fn}/{fn + tp} finished")


if __name__ == "__main__":
    main()
