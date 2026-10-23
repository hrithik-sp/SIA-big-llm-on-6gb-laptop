"""Hold N GiB of RAM the way open apps do, for the daily-use RAM-pressure test.

The memory is written (committed and resident), and every page is touched again every
--touch-s seconds, so Windows keeps it in the working set like an active browser tab
instead of trimming an idle allocation to the pagefile (which would flatter the test).
Safety: allocation stops early if available RAM would fall below --keep-free GiB, and
the amount actually held is reported.

  python tools/ram_hold.py 4            # hold 4 GiB until killed / Ctrl+C
  python tools/ram_hold.py 4 --seconds 600
Prints "READY held=X.XX GiB available=Y.YY GiB" once allocated.
"""
import argparse
import sys
import time

import psutil

GiB = 1024 ** 3
CHUNK = 256 * 1024 ** 2
PAGE = 4096


def touch(buf):
    """Write one byte per page: keeps the page resident without a full memset."""
    v = (buf[0] + 1) & 0xFF
    buf[::PAGE] = bytes([v]) * len(range(0, len(buf), PAGE))


def hold(gib, keep_free_gib):
    chunks, held = [], 0
    target = int(gib * GiB)
    while held < target:
        size = min(CHUNK, target - held)
        if psutil.virtual_memory().available - size < keep_free_gib * GiB:
            print(f"stopping early: available RAM would drop below {keep_free_gib} GiB", flush=True)
            break
        b = bytearray(size)  # zero-filled: the OS commits it
        touch(b)
        chunks.append(b)
        held += size
    return chunks, held


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("gib", type=float)
    ap.add_argument("--keep-free", type=float, default=2.0, help="never leave less than this available (GiB)")
    ap.add_argument("--touch-s", type=float, default=5.0)
    ap.add_argument("--seconds", type=float, default=0, help="exit after this long (0 = until killed)")
    a = ap.parse_args(argv)
    chunks, held = hold(a.gib, a.keep_free)
    print(f"READY held={held / GiB:.2f} GiB available={psutil.virtual_memory().available / GiB:.2f} GiB", flush=True)
    t0 = time.time()
    try:
        while not a.seconds or time.time() - t0 < a.seconds:
            time.sleep(a.touch_s)
            for c in chunks:
                touch(c)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
