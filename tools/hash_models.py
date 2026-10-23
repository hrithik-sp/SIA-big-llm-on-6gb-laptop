"""SHA-256 of every model file the results used, for provenance (docs/LAB_NOTEBOOK.md, "Files used").

Compares each local file with the LFS SHA-256 that Hugging Face publishes for the pinned
revision, so a reader can check they have byte-identical files.

  python tools/hash_models.py            # writes results/model_files.json
"""
import hashlib
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402

# (repo, revision, file, role) for every model file used in the results or by Sprout
FILES = [
    ("ggml-org/gpt-oss-20b-GGUF", "15dcfc7e7c608c6161507211ed43c5b590801b4b", "gpt-oss-20b-MXFP4.gguf", "gpt-oss-20b, released MXFP4"),
    ("ggml-org/gpt-oss-20b-GGUF", "15dcfc7e7c608c6161507211ed43c5b590801b4b", "eagle3-gpt-oss-20b-Q8_0.gguf", "EAGLE-3 draft for gpt-oss (finding 35)"),
    ("google/gemma-4-26B-A4B-it-qat-q4_0-gguf", "d1c082be9cf3c8a514acf63b8761f4b41935842e", "gemma-4-26B_q4_0-it.gguf", "Gemma 4 26B-A4B QAT Q4_0 (Google)"),
    ("google/gemma-4-26B-A4B-it-qat-q4_0-gguf", "d1c082be9cf3c8a514acf63b8761f4b41935842e", "gemma-4-26B-it-mmproj.gguf", "Gemma 4 vision projector (Sprout)"),
    ("unsloth/gemma-4-26B-A4B-it-qat-GGUF", "main", "mtp-gemma-4-26B-A4B-it.gguf", "QAT MTP head (speculative decoding)"),
    ("unsloth/gemma-4-26B-A4B-it-GGUF", "c099eb48e663fd284577b04978a94ffccb261841", "gemma-4-26B-A4B-it-Q8_0.gguf", "Gemma 4 Q8_0, quality reference"),
    ("unsloth/gemma-4-26B-A4B-it-GGUF", "c099eb48e663fd284577b04978a94ffccb261841", "mtp-gemma-4-26B-A4B-it.gguf", "MTP head for Q8_0"),
    ("unsloth/gemma-4-26B-A4B-it-GGUF", "c099eb48e663fd284577b04978a94ffccb261841", "gemma-4-26B-A4B-it-UD-IQ2_M.gguf", "lossy comparison"),
    ("unsloth/gemma-4-26B-A4B-it-GGUF", "c099eb48e663fd284577b04978a94ffccb261841", "gemma-4-26B-A4B-it-UD-IQ3_XXS.gguf", "lossy comparison"),
    ("unsloth/gemma-4-26B-A4B-it-GGUF", "c099eb48e663fd284577b04978a94ffccb261841", "gemma-4-26B-A4B-it-UD-Q3_K_M.gguf", "lossy comparison"),
    ("ggml-org/embeddinggemma-2-GGUF", "58faf90940886df2d62be555c8765d00dac362c5", "embeddinggemma-2-Q8_0.gguf", "Sprout Library search"),
]


def sha256(path, chunk=16 * 2 ** 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def published_sha(repo, rev, name):
    url = f"https://huggingface.co/api/models/{repo}/tree/{rev}"
    with urllib.request.urlopen(url, timeout=30) as r:
        for e in json.load(r):
            if e.get("path") == name:
                return (e.get("lfs") or {}).get("oid"), e.get("size")
    return None, None


def main():
    out = []
    for repo, rev, name, role in FILES:
        path = os.path.join(bench.LLM_HOME, "models", repo.replace("/", "__"), name)
        if not os.path.exists(path):
            print(f"MISSING {path}")
            continue
        local = sha256(path)
        pub, size = published_sha(repo, rev, name)
        if rev == "main":  # pin the revision actually checked
            with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=30) as r:
                rev = json.load(r)["sha"]
        ok = local == pub
        out.append({"repo": repo, "revision": rev, "file": name, "role": role, "bytes": os.path.getsize(path),
                    "sha256": local, "matches_published": ok})
        print(f"{'OK      ' if ok else 'MISMATCH'} {name}  {local[:16]}…  ({repo}@{rev[:8]})", flush=True)
    dst = os.path.join(bench.ROOT, "results", "model_files.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {dst}; {sum(o['matches_published'] for o in out)}/{len(out)} match Hugging Face")


if __name__ == "__main__":
    main()
