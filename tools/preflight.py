"""Preflight: will this llama.cpp configuration fit on this machine *right now*?

Reads only the GGUF header (no weights are loaded), models where llama.cpp places
every tensor for a given -ngl / --n-cpu-moe / context / KV-cache type, and compares
that with the VRAM and RAM free at this moment. Also prints a bandwidth roofline
(Williams, Waterman and Patterson, CACM 2009): the maximum decode tok/s physically
possible for that placement, using the bandwidths measured by tools/bandwidth.py.

Weight sizes are exact (taken from tensor offsets, so any quant type works).
KV cache is exact for standard attention and for sliding-window models whose GGUF
stores the per-layer pattern (src/llama-kv-cache-iswa.cpp); without the pattern it
is an upper bound. The compute buffer is an estimate; once a run succeeds, pass the real
"CUDA0 compute buffer size" from llama.cpp's log via --compute-mib.

Exit code: 0 = fits, 1 = does not fit, 2 = bad input.
"""
import argparse
import json
import os
import re
import struct
import subprocess
import sys
from dataclasses import asdict, dataclass

MiB = 1024**2
GiB = 1024**3
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- GGUF header

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
           7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9


class _Skipped:
    """Stand-in for a large or string array (e.g. the vocab): keeps only its length."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


class _Reader:
    def __init__(self, f):
        self.f = f

    def scalar(self, t):
        fmt = _SCALAR[t]
        return struct.unpack(fmt, self.f.read(struct.calcsize(fmt)))[0]

    def u32(self):
        return self.scalar(4)

    def u64(self):
        return self.scalar(10)

    def string(self):
        return self.f.read(self.u64()).decode("utf-8", "replace")

    def value(self, t):
        if t == _STRING:
            return self.string()
        if t != _ARRAY:
            return self.scalar(t)
        et, n = self.u32(), self.u64()
        if et in _SCALAR:
            size = struct.calcsize(_SCALAR[et])
            if n <= 65536:
                return list(struct.unpack(f"<{n}{_SCALAR[et][1]}", self.f.read(n * size)))
            self.f.seek(n * size, 1)
        elif et == _STRING:
            for _ in range(n):
                self.f.seek(self.u64(), 1)
        else:
            for _ in range(n):
                self.value(et)
        return _Skipped(n)


@dataclass
class Tensor:
    name: str
    shape: tuple
    nbytes: int
    layer: int | None
    expert: bool  # routed-expert tensor (ffn_*_exps); shared experts (_shexp) are dense


_BLK = re.compile(r"^blk\.(\d+)\.")
_SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def shard_paths(path):
    m = _SHARD.search(path)
    if not m:
        return [path]
    n = int(m.group(2))
    paths = [f"{path[:m.start()]}-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"missing shards: {missing}")
    return paths


def read_gguf(path):
    """Return (metadata, tensors). Tensor sizes come from offsets, so no quant-type table is needed."""
    meta, tensors = {}, []
    for p in shard_paths(path):
        with open(p, "rb", buffering=1 << 20) as f:
            if f.read(4) != b"GGUF":
                raise ValueError(f"{p}: not a GGUF file")
            r = _Reader(f)
            version = r.u32()
            if version < 2:
                raise ValueError(f"{p}: GGUF v{version} is not supported")
            n_tensors, n_kv = r.u64(), r.u64()
            kv = {}
            for _ in range(n_kv):
                key = r.string()
                kv[key] = r.value(r.u32())
            infos = []
            for _ in range(n_tensors):
                name = r.string()
                shape = tuple(r.u64() for _ in range(r.u32()))
                r.u32()  # ggml type
                infos.append((r.u64(), name, shape))
            align = kv.get("general.alignment", 32)
            data_start = -(-f.tell() // align) * align
        data_len = os.path.getsize(p) - data_start
        infos.sort()
        if infos and data_len <= infos[-1][0]:
            raise ValueError(f"{p}: file is truncated (incomplete download?)")
        ends = [off for off, _, _ in infos[1:]] + [data_len]
        for (off, name, shape), end in zip(infos, ends):
            m = _BLK.match(name)
            tensors.append(Tensor(name, shape, end - off, int(m.group(1)) if m else None,
                                  "_exps" in name))
        meta = meta or kv
    return meta, tensors


# ---------------------------------------------------------------- model

# Default SWA period passed to load_swa_pattern() when the GGUF has no pattern
# (src/models/*.cpp). Verified: gpt-oss (openai-moe.cpp: load_swa_pattern(ml, 2)).
SWA_DEFAULT_PERIOD = {"gpt-oss": 2}

@dataclass
class Model:
    path: str
    arch: str
    n_layer: int
    n_vocab: int
    n_embd: int
    n_head: list
    n_head_kv: list
    k_dim: int
    v_dim: int
    n_ff_max: int
    n_expert: int
    n_expert_used: int
    ctx_train: int | None
    swa: int | None
    swa_layers: list            # per-layer True = sliding-window layer ([] if not in metadata)
    k_dim_swa: int
    v_dim_swa: int
    mla_rank: int | None
    rope_dim: int
    tensors: list

    @property
    def weight_bytes(self):
        return sum(t.nbytes for t in self.tensors)


def load_model(path):
    meta, tensors = read_gguf(path)
    arch = meta["general.architecture"]

    def g(key, default=None):
        return meta.get(f"{arch}.{key}", default)

    n_layer = int(g("block_count"))

    def per_layer(v):
        return [int(x) for x in v] if isinstance(v, list) else [int(v)] * n_layer

    n_embd = int(g("embedding_length"))
    n_head = per_layer(g("attention.head_count", 0))
    hkv = g("attention.head_count_kv")
    n_head_kv = per_layer(hkv) if hkv is not None else list(n_head)
    head_dim = n_embd // max(n_head) if max(n_head) else 0
    k_dim = int(g("attention.key_length", head_dim))
    v_dim = int(g("attention.value_length", k_dim))
    n_expert, n_expert_used = int(g("expert_count", 0)), int(g("expert_used_count", 0))
    ff = g("feed_forward_length", 0)
    ff = max(ff) if isinstance(ff, list) else int(ff)
    ff_exp = int(g("expert_feed_forward_length", 0))
    tokens = meta.get("tokenizer.ggml.tokens")
    emb = next((t for t in tensors if t.name == "token_embd.weight"), None)
    n_vocab = (len(tokens) if tokens is not None else
               int(g("vocab_size", 0)) or (emb.shape[-1] if emb else 0))
    # llama_model_base::load_swa_pattern: a per-layer array if present; otherwise a period n
    # (metadata scalar, else the architecture's default) with layer il SWA iff il % n < n - 1
    pattern = g("attention.sliding_window_pattern", SWA_DEFAULT_PERIOD.get(arch))
    if isinstance(pattern, list) and len(pattern) == n_layer:
        swa_layers = [bool(x) for x in pattern]
    elif isinstance(pattern, int) and g("attention.sliding_window"):
        swa_layers = [pattern == 0 or (il % pattern < pattern - 1) for il in range(n_layer)]
    else:
        swa_layers = []
    k_dim_swa = int(g("attention.key_length_swa", k_dim))
    return Model(
        path=path, arch=arch, n_layer=n_layer, n_vocab=n_vocab, n_embd=n_embd,
        n_head=n_head, n_head_kv=n_head_kv, k_dim=k_dim, v_dim=v_dim,
        n_ff_max=max(ff, ff_exp * max(n_expert_used, 1)),
        n_expert=n_expert, n_expert_used=n_expert_used,
        ctx_train=g("context_length"), swa=g("attention.sliding_window"),
        swa_layers=swa_layers, k_dim_swa=k_dim_swa,
        v_dim_swa=int(g("attention.value_length_swa", v_dim)),
        mla_rank=g("attention.kv_lora_rank"), rope_dim=int(g("rope.dimension_count", 0)),
        tensors=tensors,
    )


# ---------------------------------------------------------------- placement

KV_BYTES = {"f32": 4, "f16": 2, "bf16": 2, "q8_0": 34 / 32, "q5_1": 24 / 32,
            "q5_0": 22 / 32, "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32}


@dataclass
class Config:
    ngl: int = 99
    n_cpu_moe: int = 0
    n_ctx: int = 4096
    ctk: str = "f16"
    ctv: str = "f16"
    ubatch: int = 512
    flash_attn: bool = True
    kv_offload: bool = True
    mmap: bool = True
    depth: int = 0              # context already filled when measuring decode speed
    overrides: tuple = ()       # (regex, "cpu"|"gpu") pairs, as from -ot; first match wins
    moe_cache_mib: float = 0    # --moe-cache-mib: extra VRAM for cached host experts
    kv_gpu_mib: float | None = None  # measured GPU KV buffer (llama-fit-params -fitp "context")
    compute_mib: float | None = None       # GPU compute buffer (measured, from llama-fit-params -fitp)
    host_compute_mib: float | None = None  # host compute buffer (measured); default: same as GPU
    cuda_overhead_mib: float = 350   # CUDA context + cuBLAS workspace on Windows
    host_overhead_mib: float = 512
    margin_mib: float = 256


@dataclass
class Plan:
    gpu_weights: int
    cpu_weights: int
    gpu_kv: int
    cpu_kv: int
    compute: int
    gpu_need: int
    ram_need: int
    gpu_read_per_tok: float
    cpu_read_per_tok: float
    compute_host: int = 0


def kv_bytes_per_token(m: Model, layer: int, cfg: Config) -> float:
    bk, bv = KV_BYTES[cfg.ctk], KV_BYTES[cfg.ctv]
    if m.mla_rank:  # DeepSeek-style MLA: compressed latent + rope part (upper bound)
        return (m.mla_rank + m.rope_dim) * bk + m.mla_rank * bv
    if is_swa_layer(m, layer):
        return m.n_head_kv[layer] * (m.k_dim_swa * bk + m.v_dim_swa * bv)
    return m.n_head_kv[layer] * (m.k_dim * bk + m.v_dim * bv)


def is_swa_layer(m: Model, layer: int) -> bool:
    return bool(m.swa and m.swa_layers and m.swa_layers[layer])


def kv_cells(m: Model, layer: int, cfg: Config) -> int:
    """KV cells per layer. src/llama-kv-cache-iswa.cpp: the SWA cache holds
    GGML_PAD(min(n_ctx, n_swa + n_ubatch), 256) cells; other layers hold n_ctx.
    Without a per-layer pattern in the GGUF, assume full context (upper bound)."""
    if not is_swa_layer(m, layer):
        return cfg.n_ctx
    return -(-min(cfg.n_ctx, m.swa + cfg.ubatch) // 256) * 256  # padded after the min, as upstream


def compute_estimate(m: Model, cfg: Config) -> int:
    """Rough peak of llama.cpp's reused compute buffer (the allocator reuses memory between ops)."""
    if cfg.compute_mib is not None:
        return int(cfg.compute_mib * MiB)
    ub = min(cfg.ubatch, cfg.n_ctx)
    logits = m.n_vocab * ub * 4
    ffn = 3 * ub * m.n_ff_max * 4
    attn = 0 if cfg.flash_attn else max(m.n_head) * cfg.n_ctx * ub * 4
    return int(max(logits, ffn, attn) * 1.25)


# llama.cpp common/common.h: LLM_FFN_EXPS_REGEX and llm_ffn_block_regex()
FFN_EXPS_REGEX = r"\.ffn_(up|down|gate|gate_up)_(ch|)exps"


def parse_overrides(spec: str):
    """Parse llama.cpp -ot syntax 'regex=BUFT,regex=BUFT' (llama-bench uses ';').

    CPU and CUDA_Host buffers live in system RAM; any other CUDA buffer is VRAM.
    """
    out = []
    for item in filter(None, re.split(r"[,;]", spec)):
        pat, _, buft = item.rpartition("=")
        if not pat:
            raise ValueError(f"bad override '{item}'")
        b = buft.strip().upper()
        out.append((pat, "cpu" if b in ("CPU", "CUDA_HOST") or b.startswith("CPU") else "gpu"))
    return tuple(out)


def plan(m: Model, cfg: Config) -> Plan:
    L = m.n_layer
    # llama.cpp src/llama-model.cpp: the output layer is layer index L and layers are
    # offloaded from the back, i_gpu_start = L + 1 - ngl; so -ngl N puts the output layer
    # and the last N-1 blocks on the GPU.
    gpu_start = max(0, L + 1 - cfg.ngl)
    gpu_layers = set(range(gpu_start, L)) if cfg.ngl > 0 else set()
    out_dev = "gpu" if cfg.ngl > 0 else "cpu"
    frac = m.n_expert_used / m.n_expert if m.n_expert else 1.0
    overrides = [(re.compile(f"blk\\.{i}{FFN_EXPS_REGEX}"), "cpu") for i in range(cfg.n_cpu_moe)]
    overrides += [(re.compile(p), d) for p, d in cfg.overrides]
    w = {"gpu": 0, "cpu": 0}
    rd = {"gpu": 0.0, "cpu": 0.0}
    emb = None
    for t in m.tensors:
        if t.layer is None:
            if "token_embd" in t.name:  # input embeddings always stay on the CPU; 1 row read per token
                dev, per_tok = "cpu", t.nbytes / max(t.shape[-1], 1)
                if t.name == "token_embd.weight":
                    emb = t
                w[dev] += t.nbytes
                rd[dev] += per_tok
                continue
            dev, per_tok = out_dev, t.nbytes
        else:
            dev = "gpu" if t.layer in gpu_layers else "cpu"
            per_tok = t.nbytes * frac if t.expert else t.nbytes
        dev = next((d for rx, d in overrides if rx.search(t.name)), dev)
        w[dev] += t.nbytes
        rd[dev] += per_tok
    tied = emb is not None and not any(t.name == "output.weight" for t in m.tensors)
    if tied:
        # tied embeddings: llama.cpp loads a duplicate of token_embd on the output device
        if out_dev == "gpu":
            w["gpu"] += emb.nbytes
        rd[out_dev] += emb.nbytes

    kv = {"gpu": 0.0, "cpu": 0.0}
    for i in range(L):
        dev = "gpu" if (i in gpu_layers and cfg.kv_offload) else "cpu"
        per_tok = kv_bytes_per_token(m, i, cfg)
        kv[dev] += per_tok * kv_cells(m, i, cfg)
        # decode at depth d reads the cached keys/values: a SWA layer only its window
        rd[dev] += per_tok * min(cfg.depth, m.swa if is_swa_layer(m, i) else cfg.depth)
    if cfg.kv_gpu_mib is not None:  # llama.cpp's measured KV ("context") buffer is authoritative
        kv["gpu"] = cfg.kv_gpu_mib * MiB

    comp = compute_estimate(m, cfg)
    comp_host = int(cfg.host_compute_mib * MiB) if cfg.host_compute_mib is not None else comp
    uses_gpu = w["gpu"] > 0 or kv["gpu"] > 0
    gpu_need = (w["gpu"] + kv["gpu"] + comp + (cfg.cuda_overhead_mib + cfg.margin_mib + cfg.moe_cache_mib) * MiB
                if uses_gpu else 0)
    # With mmap only the embedding rows of tokens actually seen are paged in, unless
    # the CPU also computes the (tied) output layer from token_embd, which reads all of it.
    cold = emb.nbytes if (emb is not None and cfg.mmap and not (tied and out_dev == "cpu")) else 0
    ram_need = w["cpu"] - cold + kv["cpu"] + comp_host + cfg.host_overhead_mib * MiB
    if cfg.mmap:
        # Measured on Windows (gpt-oss-20b): pages of tensors uploaded to the
        # GPU stay mapped in the process working set, competing with the CPU-side experts;
        # with 10.4 GiB free, RAM ran down to 0.36 GiB and experts were re-read from SSD.
        ram_need += w["gpu"]
    return Plan(int(w["gpu"]), int(w["cpu"]), int(kv["gpu"]), int(kv["cpu"]), comp,
                int(gpu_need), int(ram_need), rd["gpu"], rd["cpu"], comp_host)


def min_cpu_moe(m: Model, cfg: Config, gpu_free: int):
    """Smallest --n-cpu-moe that fits in VRAM, or None if even all-experts-on-CPU does not fit."""
    for n in range(m.n_layer + 1):
        c = Config(**{**asdict(cfg), "n_cpu_moe": n})
        if plan(m, c).gpu_need <= gpu_free:
            return n
    return None


def roofline_tps(p: Plan, gpu_bw: float, cpu_bw: float) -> float:
    """Upper bound on decode tok/s: every active byte read once per token, layers run serially."""
    t = p.gpu_read_per_tok / (gpu_bw * 1e9) + p.cpu_read_per_tok / (cpu_bw * 1e9)
    return 1 / t if t else float("inf")


# ---------------------------------------------------------------- live resources

def gpu_memory():
    """(free, total) bytes of GPU 0, or None if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout
        free, total = (int(x) for x in out.strip().splitlines()[0].split(","))
        return free * MiB, total * MiB
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def ram_available():
    import psutil
    return psutil.virtual_memory().available


# ---------------------------------------------------------------- CLI

def _g(b):
    return f"{b / GiB:6.2f} GiB"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="path to .gguf (first shard if split)")
    ap.add_argument("-ngl", "--n-gpu-layers", type=int, default=99)
    ap.add_argument("--n-cpu-moe", type=int, default=0)
    ap.add_argument("-ot", "--override-tensor", default="", help="llama.cpp -ot spec, e.g. as printed by llama-fit-params")
    ap.add_argument("--auto-moe", action="store_true", help="find the smallest --n-cpu-moe that fits")
    ap.add_argument("-c", "--ctx-size", type=int, default=4096)
    ap.add_argument("-ctk", "--cache-type-k", default="f16", choices=KV_BYTES)
    ap.add_argument("-ctv", "--cache-type-v", default="f16", choices=KV_BYTES)
    ap.add_argument("-ub", "--ubatch-size", type=int, default=512)
    ap.add_argument("-fa", "--flash-attn", default="on", choices=["on", "off"])
    ap.add_argument("-nkvo", "--no-kv-offload", action="store_true")
    ap.add_argument("--no-mmap", action="store_true")
    ap.add_argument("--depth", type=int, default=0, help="filled context for the roofline")
    ap.add_argument("--compute-mib", type=float, help="measured GPU compute buffer (llama-fit-params -fitp on)")
    ap.add_argument("--host-compute-mib", type=float, help="measured host compute buffer (llama-fit-params -fitp on)")
    ap.add_argument("--bandwidth", default=os.path.join(ROOT, "results", "bandwidth.json"))
    ap.add_argument("--json", help="also write the result to this file")
    ap.add_argument("--top", type=int, default=0, help="list the N largest non-expert tensors")
    a = ap.parse_args(argv)

    try:
        m = load_model(a.model)
        overrides = parse_overrides(a.override_tensor)
    except (OSError, ValueError, KeyError, re.error) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    cfg = Config(ngl=a.n_gpu_layers, n_cpu_moe=a.n_cpu_moe, n_ctx=a.ctx_size,
                 ctk=a.cache_type_k, ctv=a.cache_type_v, ubatch=a.ubatch_size,
                 flash_attn=a.flash_attn == "on", kv_offload=not a.no_kv_offload,
                 mmap=not a.no_mmap, depth=a.depth, compute_mib=a.compute_mib,
                 host_compute_mib=a.host_compute_mib, overrides=overrides)
    gpu = gpu_memory()
    ram = ram_available()

    if a.auto_moe and m.n_expert:
        if gpu is None:
            print("error: --auto-moe needs nvidia-smi", file=sys.stderr)
            return 2
        n = min_cpu_moe(m, cfg, gpu[0])
        if n is None:
            print("Even with every expert on the CPU the rest does not fit in VRAM; lower -ngl.")
            return 1
        cfg.n_cpu_moe = n

    p = plan(m, cfg)
    if a.top:
        dense = sorted((t for t in m.tensors if not t.expert), key=lambda t: -t.nbytes)
        n_exp = sum(t.nbytes for t in m.tensors if t.expert)
        print(f"expert tensors {_g(n_exp)}, non-expert {_g(m.weight_bytes - n_exp)}; largest non-expert:")
        for t in dense[:a.top]:
            elems = 1
            for d in t.shape:
                elems *= d
            print(f"  {t.name:34} {str(t.shape):24} {t.nbytes / MiB:9.1f} MiB  {8 * t.nbytes / elems:5.2f} bits/weight")
        print()
    print(f"model     {os.path.basename(m.path)}  [{m.arch}]  {_g(m.weight_bytes)} weights")
    print(f"          {m.n_layer} layers, vocab {m.n_vocab}"
          + (f", {m.n_expert} experts / {m.n_expert_used} active" if m.n_expert else ""))
    print(f"config    -ngl {cfg.ngl} --n-cpu-moe {cfg.n_cpu_moe} -c {cfg.n_ctx} "
          f"-ctk {cfg.ctk} -ctv {cfg.ctv} -ub {cfg.ubatch} -fa {'on' if cfg.flash_attn else 'off'}"
          + (" --no-mmap" if not cfg.mmap else "")
          + (f" -ot {a.override_tensor}" if cfg.overrides else ""))
    print()
    print(f"VRAM      weights {_g(p.gpu_weights)}  kv {_g(p.gpu_kv)}  compute~{_g(p.compute)}  "
          f"+overhead/margin {(cfg.cuda_overhead_mib + cfg.margin_mib) / 1024:.2f} GiB")
    print(f"RAM       weights {_g(p.cpu_weights)}  kv {_g(p.cpu_kv)}  compute~{_g(p.compute_host)}  "
          f"+process {cfg.host_overhead_mib / 1024:.2f} GiB")
    print()

    ok = True
    if gpu is None:
        print("GPU       nvidia-smi unavailable: cannot check VRAM")
        ok = p.gpu_need == 0
    else:
        fit = p.gpu_need <= gpu[0]
        ok &= fit
        print(f"GPU       need {_g(p.gpu_need)} / free {_g(gpu[0])}  -> "
              + ("OK" if fit else "DOES NOT FIT (OOM, or silent spill to shared memory)"))
    ram_fit = p.ram_need <= ram
    if ram_fit:
        print(f"RAM       need {_g(p.ram_need)} / available {_g(ram)}  -> OK")
    elif cfg.mmap:
        print(f"RAM       need {_g(p.ram_need)} / available {_g(ram)}  -> WARNING: CPU-side weights "
              "won't stay resident; mmap will re-read them from SSD (SSD-bound speed)")
    else:
        ok = False
        print(f"RAM       need {_g(p.ram_need)} / available {_g(ram)}  -> DOES NOT FIT (--no-mmap "
              "would push into the pagefile)")
    if m.swa and not m.swa_layers:
        print(f"note      sliding-window attention ({m.swa}) without a per-layer pattern in the GGUF: "
              "KV shown as full-context upper bound")

    tps = None
    if os.path.exists(a.bandwidth):
        with open(a.bandwidth) as f:
            bw = json.load(f)
        if bw.get("gpu_read_GBps") and bw.get("cpu_read_GBps"):
            tps = roofline_tps(p, bw["gpu_read_GBps"], bw["cpu_read_GBps"])
            print(f"roofline  {p.gpu_read_per_tok / GiB:.2f} GiB/token from VRAM + "
                  f"{p.cpu_read_per_tok / GiB:.2f} GiB/token from RAM -> max {tps:.1f} tok/s decode")
    print()
    print("VERDICT   " + ("SAFE TO RUN" if ok else "DO NOT RUN"))

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"model": m.path, "config": asdict(cfg), "plan": asdict(p),
                       "gpu_free": gpu[0] if gpu else None, "ram_available": ram,
                       "roofline_tps": tps, "ok": ok}, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
