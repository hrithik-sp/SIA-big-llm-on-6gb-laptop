"""Detect whether the NVIDIA driver silently spills VRAM into system RAM.

Allocates 256 MiB blocks until CUDA reports out-of-memory, capped at total VRAM
+ 512 MiB. With "CUDA - Sysmem Fallback Policy = Prefer No Sysmem Fallback" the
allocations stop below total VRAM; with fallback enabled they go past it (the
excess lives in shared system memory and is ~10x slower). The overshoot is
bounded to 512 MiB of system RAM. Exit code: 0 = no fallback, 1 = fallback.
"""
import sys

import torch

BLOCK = 256 * 1024**2


def main():
    total = torch.cuda.get_device_properties(0).total_memory
    cap = total + 2 * BLOCK
    blocks = []
    try:
        while len(blocks) * BLOCK < cap:
            blocks.append(torch.empty(BLOCK, dtype=torch.uint8, device="cuda"))
            blocks[-1].fill_(1)  # touch it, so the allocation is really backed
    except torch.cuda.OutOfMemoryError:
        pass
    got = len(blocks) * BLOCK
    del blocks
    torch.cuda.empty_cache()
    spill = got > total
    print(f"VRAM total {total / 1024**3:.2f} GiB, allocated {got / 1024**3:.2f} GiB before OOM -> "
          + ("SYSMEM FALLBACK IS ON (silent spill to system RAM)" if spill else "no fallback (OOM is loud)"))
    return 1 if spill else 0


if __name__ == "__main__":
    sys.exit(main())
