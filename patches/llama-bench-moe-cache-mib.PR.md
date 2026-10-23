# llama-bench : add --moe-cache-mib

The MoE expert cache (#29887, SLRU since #30637) is configured through
`llama_context_params.moe_cache_size`, but `llama-bench` has no way to set it, so its
effect can't be measured with the standard benchmark tool.

This adds `--moe-cache-mib <n>` to `llama-bench`, following the existing context-param
plumbing (same pattern as `-fitc`):

- accepts a comma-separated list, so cache sizes can be swept like other params;
- negative values are rejected as invalid parameters;
- the value is written as `moe_cache_mib` in csv/json/jsonl/sql output (INT field);
  the markdown column `mcm` is shown only when it differs from the default (0 = disabled).

Usage:

```
llama-bench -m model.gguf -ngl 99 -ncmoe 18 -p 0 -n 128 --moe-cache-mib 0,512,2048 -v
```

With `-v`, the cache's existing log lines report size, slots, and per-run
hits/misses/hit rate/uploaded MiB.

## Testing

Windows 11, RTX 3050 6 GB Laptop (PCIe 4.0 x8), MSVC 19.29 + CUDA 13.4,
base `0a1ef309c`:

- builds cleanly; `--help` lists the option; `--moe-cache-mib -5` → `error: invalid
  parameter for argument: --moe-cache-mib`;
- gpt-oss-20b MXFP4, `-ngl 99 -ncmoe 18 -lm none -p 0 -n 32 -r 1 --moe-cache-mib 0,512 -v`:
  the cache is created (`CUDA0 MoE cache size = 504.27 MiB for 7261.52 MiB of host
  experts`, 39 slots), the markdown/JSON output carries the value, and the stats line
  is printed per test (`hits = 0, misses = 2376, hit rate = 0.00%` at this deliberately
  undersized cache: ~2 slots per layer vs 4 active experts, the thrashing case noted
  in #30637).
