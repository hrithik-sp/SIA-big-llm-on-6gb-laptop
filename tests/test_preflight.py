"""Tests for tools/preflight.py against synthetic GGUF files with known contents."""
import os
import struct
import sys
import tempfile
import unittest
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import preflight as pf  # noqa: E402

ALIGN = 32
N_LAYER, N_EMBD, N_VOCAB = 4, 64, 100
EXP = 8192  # bytes of each routed-expert tensor


def _str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _kv(key, v):
    out = _str(key)
    if isinstance(v, str):
        return out + struct.pack("<I", 8) + _str(v)
    if isinstance(v, bool):
        return out + struct.pack("<I?", 7, v)
    if isinstance(v, int):
        return out + struct.pack("<II", 4, v)
    if isinstance(v, list) and all(isinstance(x, str) for x in v):
        return out + struct.pack("<IIQ", 9, 8, len(v)) + b"".join(_str(x) for x in v)
    if isinstance(v, list) and all(isinstance(x, bool) for x in v):
        return out + struct.pack("<IIQ", 9, 7, len(v)) + struct.pack(f"<{len(v)}?", *v)
    if isinstance(v, list) and all(isinstance(x, int) for x in v):
        return out + struct.pack("<IIQ", 9, 4, len(v)) + struct.pack(f"<{len(v)}I", *v)
    raise TypeError(v)


def write_gguf(path, meta, tensors, truncate=0):
    """tensors: list of (name, shape, nbytes). Data is zero-filled and 32-byte aligned."""
    hdr = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta))
    hdr += b"".join(_kv(k, v) for k, v in meta.items())
    off, data = 0, b""
    for name, shape, nb in tensors:
        hdr += _str(name) + struct.pack("<I", len(shape)) + struct.pack(f"<{len(shape)}Q", *shape)
        hdr += struct.pack("<IQ", 0, off)
        pad = -nb % ALIGN
        data += b"\0" * (nb + pad)
        off += nb + pad
    hdr += b"\0" * (-len(hdr) % ALIGN)
    blob = hdr + data
    with open(path, "wb") as f:
        f.write(blob[:len(blob) - truncate])


def model_tensors(tied=True):
    t = [("token_embd.weight", (N_EMBD, N_VOCAB), 6400), ("output_norm.weight", (N_EMBD,), 256)]
    if not tied:
        t.append(("output.weight", (N_EMBD, N_VOCAB), 6400))
    for i in range(N_LAYER):
        t += [(f"blk.{i}.attn_q.weight", (N_EMBD, 16), 1024),
              (f"blk.{i}.ffn_gate_inp.weight", (N_EMBD, 8), 512),
              (f"blk.{i}.ffn_gate_exps.weight", (N_EMBD, 16, 8), EXP),
              (f"blk.{i}.ffn_up_exps.weight", (N_EMBD, 16, 8), EXP),
              (f"blk.{i}.ffn_down_exps.weight", (16, N_EMBD, 8), EXP)]
    return t


META = {
    "general.architecture": "llama",
    "llama.block_count": N_LAYER,
    "llama.embedding_length": N_EMBD,
    "llama.attention.head_count": 4,
    "llama.attention.head_count_kv": 2,
    "llama.expert_count": 8,
    "llama.expert_used_count": 2,
    "llama.feed_forward_length": 128,
    "tokenizer.ggml.tokens": [f"t{i}" for i in range(N_VOCAB)],
}

# exact arithmetic: no estimated compute buffer or overheads
EXACT = pf.Config(compute_mib=0, cuda_overhead_mib=0, host_overhead_mib=0, margin_mib=0, n_ctx=10)


class PreflightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.tied = os.path.join(cls.dir.name, "tied.gguf")
        cls.untied = os.path.join(cls.dir.name, "untied.gguf")
        write_gguf(cls.tied, META, model_tensors(tied=True))
        write_gguf(cls.untied, META, model_tensors(tied=False))
        cls.m = pf.load_model(cls.tied)
        cls.total = sum(nb for _, _, nb in model_tensors(tied=True))

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_header_parsed_exactly(self):
        m = self.m
        self.assertEqual((m.arch, m.n_layer, m.n_vocab, m.n_expert, m.n_expert_used),
                         ("llama", N_LAYER, N_VOCAB, 8, 2))
        self.assertEqual({t.name: t.nbytes for t in m.tensors},
                         {n: nb for n, _, nb in model_tensors(tied=True)})
        self.assertEqual(m.weight_bytes, self.total)
        self.assertEqual(sum(t.expert for t in m.tensors), 3 * N_LAYER)
        self.assertEqual((m.k_dim, m.v_dim), (16, 16))

    def test_full_offload_duplicates_tied_embeddings(self):
        p = pf.plan(self.m, EXACT)
        self.assertEqual(p.cpu_weights, 6400)                 # token_embd stays on CPU
        self.assertEqual(p.gpu_weights, self.total - 6400 + 6400)  # ...and a copy feeds the output layer

    def test_untied_model_has_no_duplicate(self):
        p = pf.plan(pf.load_model(self.untied), EXACT)
        self.assertEqual(p.cpu_weights, 6400)
        self.assertEqual(p.gpu_weights, self.total + 6400 - 6400)

    def test_n_cpu_moe_moves_first_layers_experts(self):
        p = pf.plan(self.m, replace(EXACT, n_cpu_moe=2))
        self.assertEqual(p.cpu_weights, 6400 + 2 * 3 * EXP)
        self.assertEqual(p.gpu_weights, self.total - 2 * 3 * EXP)

    def test_partial_ngl_follows_llama_cpp_layer_numbering(self):
        # i_gpu_start = n_layer + 1 - ngl: -ngl 2 -> output layer + last block on GPU
        per_layer = 1024 + 512 + 3 * EXP
        output = 256 + 6400  # output_norm + tied token_embd duplicate
        p = pf.plan(self.m, replace(EXACT, ngl=2))
        self.assertEqual(p.gpu_weights, per_layer + output)
        p = pf.plan(self.m, replace(EXACT, ngl=N_LAYER + 1))  # everything
        self.assertEqual(p.gpu_weights, N_LAYER * per_layer + output)
        p = pf.plan(self.m, replace(EXACT, ngl=0))
        self.assertEqual((p.gpu_weights, p.gpu_need), (0, 0))

    def test_override_matches_n_cpu_moe(self):
        ot = pf.parse_overrides(r"blk\.0\.ffn_(up|down|gate|gate_up)_(ch|)exps=CPU;"
                                r"blk\.1\.ffn_(up|down|gate|gate_up)_(ch|)exps=CPU")
        self.assertEqual(pf.plan(self.m, replace(EXACT, overrides=ot)),
                         pf.plan(self.m, replace(EXACT, n_cpu_moe=2)))

    def test_fit_style_override_moves_all_experts(self):
        # pattern format printed by llama-fit-params (common/fit.cpp pattern_moe_all)
        ot = pf.parse_overrides(r"blk\.\d+\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU")
        p = pf.plan(self.m, replace(EXACT, overrides=ot))
        self.assertEqual(p.cpu_weights, 6400 + N_LAYER * 3 * EXP)

    def test_override_first_match_wins_and_can_target_gpu(self):
        ot = pf.parse_overrides(r"blk\.3\.ffn_up_exps=CUDA0,blk\.\d+\.ffn_.*_exps=CPU")
        p = pf.plan(self.m, replace(EXACT, overrides=ot))
        self.assertEqual(p.cpu_weights, 6400 + N_LAYER * 3 * EXP - EXP)
        # an override can also pull a tensor of a CPU layer onto the GPU
        p = pf.plan(self.m, replace(EXACT, ngl=0, overrides=pf.parse_overrides(r"blk\.0\.attn_q=CUDA0")))
        self.assertEqual(p.gpu_weights, 1024)

    def test_override_parse(self):
        self.assertEqual(pf.parse_overrides("a=CPU,b=CUDA0;c=CUDA_Host"),
                         (("a", "cpu"), ("b", "gpu"), ("c", "cpu")))
        with self.assertRaises(ValueError):
            pf.parse_overrides("noequals")

    def test_moe_cache_adds_vram(self):
        base = pf.plan(self.m, EXACT).gpu_need
        self.assertEqual(pf.plan(self.m, replace(EXACT, moe_cache_mib=512)).gpu_need, base + 512 * pf.MiB)

    def test_sliding_window_kv_matches_llama_cpp_formula(self):
        # Gemma-4-like: layers 0-2 SWA (8 kv heads x 256), layer 3 global (2 kv heads x 512)
        meta = {**META, "llama.attention.head_count_kv": [8, 8, 8, 2],
                "llama.attention.key_length": 512, "llama.attention.value_length": 512,
                "llama.attention.key_length_swa": 256, "llama.attention.value_length_swa": 256,
                "llama.attention.sliding_window": 1024,
                "llama.attention.sliding_window_pattern": [True, True, True, False]}
        path = os.path.join(self.dir.name, "swa.gguf")
        write_gguf(path, meta, model_tensors())
        m = pf.load_model(path)
        swa_tok, glob_tok = 8 * (256 + 256) * 2, 2 * (512 + 512) * 2
        # n_ctx 4096, ub 512: SWA layers hold PAD(min(4096, 1024+512), 256) = 1536 cells
        p = pf.plan(m, replace(EXACT, n_ctx=4096, ubatch=512))
        self.assertEqual(p.gpu_kv, 3 * swa_tok * 1536 + glob_tok * 4096)
        # small context: PAD(min(640, 1536), 256) = 768 cells (padding after the min, as upstream)
        p = pf.plan(m, replace(EXACT, n_ctx=640, ubatch=512))
        self.assertEqual(p.gpu_kv, 3 * swa_tok * 768 + glob_tok * 640)
        # decode at depth 4096 reads only the window on SWA layers
        p0 = pf.plan(m, replace(EXACT, n_ctx=8192, depth=0))
        p1 = pf.plan(m, replace(EXACT, n_ctx=8192, depth=4096))
        self.assertAlmostEqual(p1.gpu_read_per_tok - p0.gpu_read_per_tok, 3 * swa_tok * 1024 + glob_tok * 4096)
        # a measured KV size overrides the estimate
        self.assertEqual(pf.plan(m, replace(EXACT, kv_gpu_mib=380)).gpu_kv, 380 * pf.MiB)

    def test_swa_period_fallback_matches_llama_cpp(self):
        # no per-layer array: period from metadata, il % n < n - 1 -> SWA (set_swa_pattern)
        meta = {**META, "llama.attention.sliding_window": 128, "llama.attention.sliding_window_pattern": 2}
        path = os.path.join(self.dir.name, "swa_period.gguf")
        write_gguf(path, meta, model_tensors())
        self.assertEqual(pf.load_model(path).swa_layers, [True, False, True, False])

    def test_kv_cache_size(self):
        # 2 kv heads * (16 k + 16 v) * 2 bytes = 128 B/token/layer
        p = pf.plan(self.m, EXACT)
        self.assertEqual((p.gpu_kv, p.cpu_kv), (128 * 10 * N_LAYER, 0))
        p = pf.plan(self.m, replace(EXACT, ctk="q8_0", ctv="q8_0"))
        self.assertEqual(p.gpu_kv, int(2 * 32 * 34 / 32 * 10 * N_LAYER))
        p = pf.plan(self.m, replace(EXACT, kv_offload=False))
        self.assertEqual((p.gpu_kv, p.cpu_kv), (0, 128 * 10 * N_LAYER))

    def test_per_token_reads_only_active_experts(self):
        p = pf.plan(self.m, EXACT)
        dense = 256 + N_LAYER * (1024 + 512)
        self.assertAlmostEqual(p.gpu_read_per_tok, dense + N_LAYER * 3 * EXP * 2 / 8 + 6400)
        self.assertAlmostEqual(p.cpu_read_per_tok, 6400 / N_VOCAB)  # one embedding row

    def test_ram_need_excludes_cold_embeddings_only_with_mmap(self):
        # mmap: token_embd stays cold, but GPU-uploaded tensors stay mapped in RAM too
        p = pf.plan(self.m, replace(EXACT, n_cpu_moe=1))
        self.assertEqual(p.ram_need, 3 * EXP + p.gpu_weights)
        p = pf.plan(self.m, replace(EXACT, n_cpu_moe=1, mmap=False))
        self.assertEqual(p.ram_need, 6400 + 3 * EXP)              # no mmap: CPU tensors only
        p = pf.plan(self.m, replace(EXACT, ngl=0))                # tied output computed on CPU
        self.assertEqual(p.ram_need, self.total + 128 * 10 * N_LAYER)  # all of token_embd + KV in RAM

    def test_min_cpu_moe_finds_smallest_fitting_value(self):
        need2 = pf.plan(self.m, replace(EXACT, n_cpu_moe=2)).gpu_need
        self.assertEqual(pf.min_cpu_moe(self.m, EXACT, need2), 2)
        self.assertEqual(pf.min_cpu_moe(self.m, EXACT, need2 - 1), 3)
        self.assertIsNone(pf.min_cpu_moe(self.m, EXACT, 1000))

    def test_roofline(self):
        p = pf.Plan(0, 0, 0, 0, 0, 0, 0, gpu_read_per_tok=1e9, cpu_read_per_tok=1e9)
        self.assertAlmostEqual(pf.roofline_tps(p, 100, 50), 1 / (0.01 + 0.02))

    def test_truncated_file_rejected(self):
        bad = os.path.join(self.dir.name, "bad.gguf")
        write_gguf(bad, META, model_tensors(), truncate=3 * EXP)
        with self.assertRaisesRegex(ValueError, "truncated"):
            pf.load_model(bad)

    def test_missing_shard_rejected(self):
        with self.assertRaises(FileNotFoundError):
            pf.shard_paths(os.path.join(self.dir.name, "m-00001-of-00002.gguf"))

    def test_cli_exit_codes(self):
        orig = pf.gpu_memory
        try:  # a fixed GPU, so the result doesn't depend on the machine running the tests
            pf.gpu_memory = lambda: (6 * 1024 ** 3, 6 * 1024 ** 3)
            self.assertEqual(pf.main([self.tied, "-c", "64", "--bandwidth", "none"]), 0)
            pf.gpu_memory = lambda: None  # no nvidia-smi (e.g. CI): GPU layers can't be checked
            self.assertEqual(pf.main([self.tied, "-c", "64", "--bandwidth", "none"]), 1)
        finally:
            pf.gpu_memory = orig
        self.assertEqual(pf.main([os.path.join(self.dir.name, "nope.gguf")]), 2)


if __name__ == "__main__":
    unittest.main()
