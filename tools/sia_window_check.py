"""Check Sia's standalone window (pywebview): hidden start, instant show, close-to-hide,
second launch reuses the window, memory cost, and a clean quit.

  python tools/sia_window_check.py      (shows the Sia window briefly on screen)
"""
import ctypes
import http.client
import os
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
PYW = str(Path(sys.executable).with_name("pythonw.exe"))
H = {"Origin": "http://127.0.0.1:8765", "Content-Type": "application/json"}
u32 = ctypes.windll.user32
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def post(path):
    c = http.client.HTTPConnection("127.0.0.1", 8765, timeout=10)
    c.request("POST", path, "{}", H)
    return c.getresponse().status


def sia_windows():
    """Top-level windows titled 'Sia' owned by the Sia backend: [(hwnd, visible)]."""
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        n = u32.GetWindowTextLengthW(hwnd)
        if n:
            buf = ctypes.create_unicode_buffer(n + 1)
            u32.GetWindowTextW(hwnd, buf, n + 1)
            if buf.value == "Sia":
                found.append((hwnd, bool(u32.IsWindowVisible(hwnd))))
        return True
    u32.EnumWindows(cb, 0)
    return found


def capture(hwnd):
    """The window's own pixels via PrintWindow(PW_RENDERFULLCONTENT), whatever is in front."""
    from PIL import Image
    rect = wintypes.RECT()
    u32.GetWindowRect(hwnd, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    g32 = ctypes.windll.gdi32
    hdc = u32.GetWindowDC(hwnd)
    mdc = g32.CreateCompatibleDC(hdc)
    bmp = g32.CreateCompatibleBitmap(hdc, w, h)
    g32.SelectObject(mdc, bmp)
    u32.PrintWindow(hwnd, mdc, 2)
    buf = ctypes.create_string_buffer(w * h * 4)

    class BMI(ctypes.Structure):
        _fields_ = [("biSize", wintypes.DWORD), ("biWidth", ctypes.c_long), ("biHeight", ctypes.c_long),
                    ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                    ("biSizeImage", wintypes.DWORD), ("a", ctypes.c_long), ("b", ctypes.c_long),
                    ("c", wintypes.DWORD), ("d", wintypes.DWORD)]
    bmi = BMI(ctypes.sizeof(BMI), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
    g32.GetDIBits(mdc, bmp, 0, h, buf, ctypes.byref(bmi), 0)
    g32.DeleteObject(bmp); g32.DeleteDC(mdc); u32.ReleaseDC(hwnd, hdc)
    return Image.frombuffer("RGBA", (w, h), buf, "raw", "BGRA", 0, 1).convert("RGB")


def wait_for(cond, timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return time.time() - t0
        time.sleep(0.05)
    return None


def tree_mb(pid):
    p = psutil.Process(pid)
    procs = [p] + p.children(recursive=True)
    return sum(x.memory_info().rss for x in procs if x.is_running()) / 2 ** 20, [x.name() for x in procs]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    proc = subprocess.Popen([PYW, str(ROOT / "app" / "server.py"), "--background"], cwd=str(ROOT))
    up = wait_for(lambda: any(True for _ in sia_windows()), 30)
    check("background start creates the window", up is not None, f"{up and round(up, 1)} s after launch")
    check("…and keeps it hidden", sia_windows() and not any(v for _, v in sia_windows()), str(sia_windows()))
    time.sleep(15)  # the hidden page has loaded by now: it must not have woken the model
    mb, names = tree_mb(proc.pid)
    no_model = not any(n.lower() == "llama-server.exe" for n in names)
    check("no model loaded in the background (after 15 s)", no_model, f"Sia process tree {mb:.0f} MB: {sorted(set(names))}")

    post("/api/show")
    shown = wait_for(lambda: any(v for _, v in sia_windows()), 10)
    check("summon shows the window", shown is not None, f"{shown and round(shown * 1000)} ms")
    time.sleep(3)  # let the page draw
    hwnd = next(h for h, v in sia_windows() if v)
    shot = capture(hwnd)
    out = Path(os.environ.get("TEMP", ".")) / "sia-window.png"
    shot.save(out)
    dark = sum(1 for p in shot.resize((64, 64)).getdata() if max(p[:3]) < 40) / 4096
    check("window renders Sia (captured from the window itself)", dark > 0.6,
          f"{shot.size}, {dark:.0%} dark pixels (Sia's background is near-black), saved to {out}")

    u32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE, as the title-bar X does
    hidden = wait_for(lambda: not any(v for _, v in sia_windows()), 10)
    check("closing hides (Sia stays running)", hidden is not None and proc.poll() is None,
          f"hidden after {hidden and round(hidden * 1000)} ms; backend alive: {proc.poll() is None}")

    t2 = time.time()
    second = subprocess.run([sys.executable, str(ROOT / "app" / "server.py")], cwd=str(ROOT), timeout=60)
    shown2 = wait_for(lambda: any(v for _, v in sia_windows()), 10)
    check("second launch reuses the window", shown2 is not None and len(sia_windows()) == 1,
          f"second process exited {second.returncode} in {time.time() - t2:.1f} s; windows: {len(sia_windows())}")

    p = psutil.Process(proc.pid)
    procs = [p] + [c for c in p.children(recursive=True) if c.name().lower() != "llama-server.exe"]
    mb = sum(x.memory_full_info().uss for x in procs) / 2 ** 20  # unique memory: RSS counts shared pages repeatedly
    check("unique memory of Sia itself (window + backend, model excluded)", mb < 300,
          f"{mb:.0f} MB in {len(procs)} processes: {sorted(set(x.name() for x in procs))}")
    post("/api/quit")
    gone = wait_for(lambda: proc.poll() is not None, 30)
    time.sleep(2)
    left = [p.info["name"] for p in psutil.process_iter(["name", "ppid"]) if p.info["name"] == "llama-server.exe"]
    check("quit stops everything", gone is not None and not sia_windows() and not left,
          f"exit after {gone and round(gone, 1)} s; windows left {len(sia_windows())}; llama-server left {len(left)}")
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")


if __name__ == "__main__":
    main()
