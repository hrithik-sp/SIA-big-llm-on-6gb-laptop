"""QAT fidelity study (docs/QAT_STUDY_PROTOCOL.md): prepare the versions of one Gemma 4 size and compare them.

For a size, the versions are O (original instruct model), U (Google's QAT model before rounding), Q (Google's
Q4_0 QAT release), O4 and U4 (O and U rounded to Q4_0 here, in exactly Q's per-tensor formats) and O8 and U8
(Q8_0 copies for the accuracy runs).

  python tools/qat_study.py convert E2B       # O and U: Hugging Face checkpoints -> GGUF BF16
  python tools/qat_study.py quantize E2B      # O4, U4 (formats copied from Q, then checked), O8, U8
  python tools/qat_study.py hashes E2B        # SHA-256 of every file -> results/qat_study_files.jsonl
  python tools/qat_study.py kld E2B           # the protocol's KL comparisons (tools/quality.py)
  python tools/qat_study.py compare-q E2B     # is Google's Q a plain Q4_0 rounding of U? (tensor bytes)
  python tools/qat_study.py report E2B

Downloads use tools/fetch_model.ps1 at the pinned revisions in SIZES. Every model run goes through the same
preflight gate as tools/bench.py (inside tools/quality.py).
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import preflight as pf  # noqa: E402

LLAMA = os.path.join(bench.LLM_HOME, "llama.cpp")
MODELS = os.path.join(bench.LLM_HOME, "models")
STUDY = os.path.join(MODELS, "qat-study")
CORPUS = os.path.join(bench.LLM_HOME, "data", "qat-study")
FILES_OUT = os.path.join(pf.ROOT, "results", "qat_study_files.jsonl")
TAG = "qat-study-v1"

# Pinned revisions (Hugging Face commit SHAs). template: which chat-template family the size uses (the protocol's
# two renderings of the test text).
SIZES = {
    "E2B": {"O": ("google/gemma-4-E2B-it", "3e22461f65e89153144f8adb70e3b8c2cc9845a7"),
            "U": ("google/gemma-4-E2B-it-qat-q4_0-unquantized", "6befbaca7398925921802abd1f277b495b78b738"),
            "Q": ("google/gemma-4-E2B-it-qat-q4_0-gguf", "675cff42a74c774d6cb76f76d8eacb49b48c9b93",
                  "gemma-4-E2B_q4_0-it.gguf"),
            "template": "small"},
}


def gguf_path(size, code):
    """Study files carry the size in their name: tools/quality.py names reference logits after the file."""
    if code == "Q":
        repo, _, name = SIZES[size]["Q"]
        return os.path.join(MODELS, repo.replace("/", "__"), name)
    kind = {"O": "bf16", "U": "bf16", "O4": "q4_0", "U4": "q4_0", "O8": "q8_0", "U8": "q8_0"}[code]
    return os.path.join(STUDY, size, f"gemma-4-{size}-{code}-{kind}.gguf")


def corpus_path(size):
    return os.path.join(CORPUS, f"corpus-{SIZES[size]['template']}-template.txt")


def run(cmd, log):
    """Run a command, teeing its output to a log file; raise on failure."""
    print(subprocess.list2cmdline(cmd), flush=True)
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=bench.llama_env())
    if r.returncode != 0:
        raise SystemExit(f"failed ({r.returncode}); log: {log}")


def tensor_types(path):
    """{tensor name: ggml type name} from a GGUF header (gguf-py from the llama.cpp tree)."""
    sys.path.insert(0, os.path.join(LLAMA, "gguf-py"))
    from gguf import GGUFReader
    return {t.name: t.tensor_type.name for t in GGUFReader(path).tensors}


def override_lines(types):
    """llama-quantize --tensor-type-file lines that force every tensor to the given type. Patterns are regexes
    searched in the tensor name (src/llama-quant.cpp), so each name is escaped and anchored to match only itself."""
    return [f"^{re.escape(name)}$={t.lower()}" for name, t in sorted(types.items())]


LOSSLESS_FLOAT = {"F32", "BF16"}  # a BF16 value stored as F32 is the same number


def type_mismatches(ours, ref):
    """Tensors present in both files whose storage differs in a way that changes values."""
    out = []
    for name in sorted(set(ours) & set(ref)):
        a, b = ours[name], ref[name]
        if a != b and not (a in LOSSLESS_FLOAT and b in LOSSLESS_FLOAT):
            out.append((name, a, b))
    return out


def sha256(path, chunk=16 * 2 ** 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def cmd_convert(a):
    for code in ("O", "U"):
        repo, rev = SIZES[a.size][code]
        src, dst = os.path.join(MODELS, repo.replace("/", "__")), gguf_path(a.size, code)
        if os.path.exists(dst):
            print(f"exists: {dst}")
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        run([sys.executable, os.path.join(LLAMA, "convert_hf_to_gguf.py"), src, "--outtype", "bf16",
             "--outfile", dst + ".part"], dst + ".convert.log")
        os.replace(dst + ".part", dst)  # a conversion cut short never looks finished


def cmd_quantize(a):
    q_types = tensor_types(gguf_path(a.size, "Q"))
    tt = os.path.join(STUDY, a.size, "q-tensor-types.txt")
    with open(tt, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(override_lines(q_types)) + "\n")
    exe = os.path.join(LLAMA, "build-cuda", "bin", "llama-quantize.exe")
    for src, dst, ftype, extra in (("O", "O4", "Q4_0", ["--tensor-type-file", tt]),
                                   ("U", "U4", "Q4_0", ["--tensor-type-file", tt]),
                                   ("O", "O8", "Q8_0", []), ("U", "U8", "Q8_0", [])):
        out = gguf_path(a.size, dst)
        if not os.path.exists(out):
            run([exe, *extra, gguf_path(a.size, src), out + ".part", ftype], out + ".quantize.log")
            os.replace(out + ".part", out)
        if ftype == "Q4_0":
            ours = tensor_types(out)
            bad = type_mismatches(ours, q_types)
            only_ours, only_q = sorted(set(ours) - set(q_types)), sorted(set(q_types) - set(ours))
            print(f"{dst}: {len(set(ours) & set(q_types))} tensors shared with Q, format mismatches: {len(bad)}, "
                  f"only in {dst}: {len(only_ours)}, only in Q: {len(only_q)}")
            for m in bad[:20]:
                print("   MISMATCH", *m)
            for n in only_ours[:6]:
                print("   only in", dst, n, ours[n])
            if bad:
                raise SystemExit(f"{dst} does not match Q's formats; see above (protocol: formats must match)")


def cmd_hashes(a):
    with open(FILES_OUT, "a", encoding="utf-8") as f:
        for code in ("O", "U", "Q", "O4", "U4", "O8", "U8"):
            p = gguf_path(a.size, code)
            if not os.path.exists(p):
                print(f"MISSING {p}")
                continue
            rec = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "tag": TAG, "size": a.size,
                   "code": code, "file": os.path.basename(p), "bytes": os.path.getsize(p), "sha256": sha256(p),
                   "source": SIZES[a.size].get(code[0] if code != "Q" else "Q")[:2]}
            f.write(json.dumps(rec) + "\n")
            print(f"{code:<3} {rec['sha256'][:16]}  {rec['bytes'] / 2 ** 30:6.2f} GiB  {rec['file']}")


# (reference, [candidates]) in protocol order. Each reference's saved predictions are deleted once its candidates
# are done (they are ~134 MB per chunk); the logs and results stay.
KLD_PLAN = [("O", ["U", "Q", "O4", "O8"]), ("U", ["Q", "U4"]), ("Q", ["U4"])]


def cmd_kld(a):
    import quality
    data = corpus_path(a.size)
    # mmap for every run: predictions don't depend on the load mode, and speed isn't measured in this study
    common = ["--data", data, "--parse-special", "-c", "512", "--chunks", "100", "--tag", TAG, "--lm", "auto"]
    common += ["--allow-ssd"] if a.allow_ssd else []
    for ref, cands in KLD_PLAN:
        if a.only and ref not in a.only.split(","):
            continue
        ref_p = gguf_path(a.size, ref)
        kld_file = quality.kld_path(ref_p, 512, 100, data)
        if not os.path.exists(kld_file):
            quality.main(["base", ref_p, *common])
        for c in cands:
            quality.main(["kld", gguf_path(a.size, c), "--base-model", ref_p, *common])
        if not a.keep_logits:
            os.remove(kld_file)
            print(f"deleted reference logits {kld_file} (results and logs kept)")


def cmd_compare_q(a):
    """Byte comparison of U4's and Q's tensor data: identical bytes mean Google's file is a plain Q4_0 of U."""
    sys.path.insert(0, os.path.join(LLAMA, "gguf-py"))
    import numpy as np
    from gguf import GGUFReader
    u4, q = GGUFReader(gguf_path(a.size, "U4")), GGUFReader(gguf_path(a.size, "Q"))
    qt = {t.name: t for t in q.tensors}
    same = differ = 0
    for t in u4.tensors:
        o = qt.get(t.name)
        if o is None:
            continue
        if t.tensor_type == o.tensor_type and np.array_equal(np.asarray(t.data).view(np.uint8),
                                                             np.asarray(o.data).view(np.uint8)):
            same += 1
        else:
            differ += 1
            if differ <= 10:
                print(f"   differs: {t.name} ({t.tensor_type.name} vs {o.tensor_type.name})")
    print(f"{a.size}: {same} tensors byte-identical, {differ} differ, of {same + differ} shared")


def cmd_report(a):
    rows = []
    with open(os.path.join(pf.ROOT, "results", "quality.jsonl"), encoding="utf-8") as f:
        for r in map(json.loads, f):
            if r.get("tag") == TAG and r["mode"] == "kld" and f"gemma-4-{a.size}" in os.path.basename(r["reference"]):
                rows.append(r)
    name = lambda p: re.sub(r"\.gguf$", "", os.path.basename(p))
    print(f"{'reference -> candidate':<52} {'mean KLD':>16} {'99.9% KLD':>10} {'max':>8} {'same top':>9} {'PPL check':>9}")
    for r in rows:
        s, chk = r["stats"], (r.get("base_ppl_check") or {}).get("ratio")
        print(f"{name(r['reference']) + ' -> ' + name(r['model']):<52} "
              f"{s.get('Mean KLD', float('nan')):>8.5f} ± {s.get('Mean KLD ±', float('nan')):<7.5f}"
              f"{s.get('99.9% KLD', float('nan')):>10.4f} {s.get('Maximum KLD', float('nan')):>8.3f} "
              f"{s.get('Same top p', float('nan')):>8.2f}% {f'{chk:.3f}' if chk else '-':>9}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("command", choices=["convert", "quantize", "hashes", "kld", "compare-q", "report"])
    ap.add_argument("size", choices=list(SIZES))
    ap.add_argument("--only", default="", help="kld: only these references (comma list of O, U, Q)")
    ap.add_argument("--keep-logits", action="store_true", help="kld: keep each reference's saved predictions")
    ap.add_argument("--allow-ssd", action="store_true",
                    help="kld: let a model that exceeds free RAM stream from the SSD (values unaffected; the "
                         "protocol allows it because speed isn't measured)")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    {"convert": cmd_convert, "quantize": cmd_quantize, "hashes": cmd_hashes, "kld": cmd_kld,
     "compare-q": cmd_compare_q, "report": cmd_report}[a.command](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
