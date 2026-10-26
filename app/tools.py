"""Laptop tools Sia's model may call, with the user's permission.

Security model:
- The reading tools defined here are read-only: they never change a setting or write, delete,
  move, run or send anything, and Sia itself never reaches the network. The action tools
  registered in TOOLS (open, control, window, power, ...) live in actions.py, with their own
  rules, their own Settings switch and an approval card for every use. Two notes: the battery report
  is written by Windows to a temp file that is deleted at once, and Windows' own location
  service (when location is on) may use its network lookup to find the position.
- No shell is exposed to the model; each tool is a fixed Python function, and unexpected
  arguments are dropped.
- Paths go through safe_path(): local drives only (no network, device or stream paths),
  symlinks resolved, and places that hold secrets are refused even if the user approved
  the call (keys, browser profiles, credential stores, registry hives, password vaults,
  wallets, .env). Files over 50 MB are refused; reads are truncated.
- Results are size-limited, and the backend asks the user before each call (the card
  shows the exact path) unless they chose "Always allow" for that tool (app/server.py).
- A file's contents could try to steer the model ("now read X"); the worst outcome is a
  further read or action request the user can deny, since nothing runs without its card.
"""
import ctypes
import datetime as dt
import fnmatch
import json
import math
import os
import platform
import re
import subprocess
import time
import uuid
import zipfile
from ctypes import wintypes
from pathlib import Path

import psutil

import actions
import calc
import games
import sandbox

APP = Path(__file__).resolve().parent
GiB = 1024 ** 3

# --------------------------------------------------------------------------- path safety
# General file access is read-only and still refuses places that hold secrets, even when the
# user approved the call: keys, browser profiles (passwords, cookies), credential stores,
# registry hives, password-manager vaults, wallets, .env files. Only local drive paths.
BLOCKED_DIRS = [re.compile(p, re.I) for p in (
    r"\\\.ssh(\\|$)", r"\\\.gnupg(\\|$)", r"\\\.aws(\\|$)", r"\\\.azure(\\|$)", r"\\\.kube(\\|$)", r"\\\.docker(\\|$)",
    r"\\AppData\\Local\\(Google\\Chrome|Microsoft\\Edge|BraveSoftware\\Brave-Browser|Vivaldi|Opera Software)[^\\]*\\User Data",
    r"\\AppData\\Roaming\\(Mozilla\\Firefox|Opera Software)",
    r"\\AppData\\(Local|Roaming)\\Microsoft\\(Credentials|Protect|Vault|Crypto|SystemCertificates)",
    r"\\Windows\\System32\\config(\\|$)", r"\\Windows\\(System32|SysWOW64)\\(LogFiles\\)?Sru",
    r"\\(KeePass|KeePassXC|1Password|Bitwarden|LastPass|Dashlane|Proton Pass)(\\|$)",
    r"\\(Electrum|Exodus|Ethereum|Bitcoin|MetaMask)(\\|$)",
    r"\\AppData\\Roaming\\(Telegram Desktop|Signal|WhatsApp)(\\|$)",
)]
BLOCKED_NAMES = [re.compile(p, re.I) for p in (
    r"^id_(rsa|dsa|ecdsa|ed25519)", r"\.(pem|key|pfx|p12|ppk|kdbx|kdb|keystore|jks|asc|gpg)$", r"^\.env(\..*)?$",
    r"^(credentials|secrets?)(\..*)?$", r"^(\.git-credentials|\.netrc|_netrc|\.npmrc|\.pypirc|\.pgpass)$",
    r"^(login data|cookies|web data|history|key4\.db|logins\.json|cert9\.db)$", r"^wallet\.dat$",
    r"^(ntuser\.dat|ntuser\.dat\.log.*|sam|security|system|software|usrclass\.dat)$",
)]
MAX_READ_BYTES = 50 * 2 ** 20


def safe_path(path, want="any"):
    """Resolve a user/model-supplied path; refuse non-local, device, stream and secret locations."""
    raw = str(path or "").strip().strip('"')
    if not raw:
        raise ValueError("no path given")
    raw = os.path.expandvars(os.path.expanduser(raw))
    if raw.startswith(("\\\\", "//")):
        raise PermissionError("network and device paths are not allowed")
    p = Path(os.path.realpath(raw))
    s = str(p)
    if not re.match(r"^[A-Za-z]:\\", s) or ":" in s[2:]:
        raise PermissionError("only local drive paths are allowed")
    if any(rx.search(s) for rx in BLOCKED_DIRS) or any(rx.search(p.name) for rx in BLOCKED_NAMES):
        raise PermissionError("this location can hold passwords, keys or other secrets, so Sia won't read it")
    if want == "dir" and not p.is_dir():
        raise FileNotFoundError(f"no such folder: {s}")
    if want == "file" and not p.is_file():
        raise FileNotFoundError(f"no such file: {s}")
    return p


def _known_folder(guid):
    """Real location of a Windows known folder (handles OneDrive / redirected folders)."""
    class GUID(ctypes.Structure):
        _fields_ = [("d1", wintypes.DWORD), ("d2", wintypes.WORD), ("d3", wintypes.WORD), ("d4", ctypes.c_ubyte * 8)]
    u = uuid.UUID(guid)
    g = GUID(u.fields[0], u.fields[1], u.fields[2], (ctypes.c_ubyte * 8)(*u.bytes[8:]))
    p = ctypes.c_wchar_p()
    if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(g), 0, None, ctypes.byref(p)) != 0:
        return None
    path = p.value
    ctypes.windll.ole32.CoTaskMemFree(p)
    return Path(path)


FOLDERS = {  # the only folders a tool may list, chosen by name
    "downloads": "374DE290-123F-4565-9164-39C4925E467B",
    "desktop": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641",
    "documents": "FDD39AD0-238F-46AF-ADB4-6C85480369C7",
    "recent": "AE50C081-EBD2-438A-8655-8A092E34987A",  # Windows' "recently opened" shortcuts
}


def _folder(name):
    p = _known_folder(FOLDERS[name]) if os.name == "nt" else None
    return p if p and p.is_dir() else Path.home() / name.capitalize()


def _gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip().split(", ")
        return {"name": out[0], "temp_c": int(out[1]), "load_pct": int(out[2]),
                "vram_used_gb": round(int(out[3]) / 1024, 1), "vram_total_gb": round(int(out[4]) / 1024, 1)}
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


# --------------------------------------------------------------------------- tools
SIA = "Sia (this assistant and its AI model)"


def system_vitals():
    vm, batt = psutil.virtual_memory(), psutil.sensors_battery()
    disk = psutil.disk_usage("C:\\")
    try:  # Sia itself (this backend, its window and its model servers): grouped, not called a hog
        me = psutil.Process(os.getpid())
        own = {me.pid} | {c.pid for c in me.children(recursive=True)}
    except psutil.Error:
        own = set()
    procs = []
    for p in psutil.process_iter(["name", "memory_info"]):
        try:
            name = SIA if p.pid in own else p.info["name"]
            procs.append((p.info["memory_info"].rss, name))
        except (psutil.Error, AttributeError):
            continue
    by_app = {}
    for rss, name in procs:
        by_app[name] = by_app.get(name, 0) + rss
    top = sorted(by_app.items(), key=lambda x: -x[1])[:6]
    return {
        "cpu_load_pct": psutil.cpu_percent(interval=0.5),
        "cpu_cores": psutil.cpu_count(),
        "ram": {"used_gb": round((vm.total - vm.available) / GiB, 1), "free_gb": round(vm.available / GiB, 1),
                "total_gb": round(vm.total / GiB, 1)},
        "gpu": _gpu(),
        "battery": {"percent": round(batt.percent), "plugged_in": batt.power_plugged} if batt else None,
        "disk_c": {"free_gb": round(disk.free / GiB, 1), "total_gb": round(disk.total / GiB, 1)},
        "uptime_hours": round((time.time() - psutil.boot_time()) / 3600, 1),
        "top_apps_by_memory": [{"app": n, "gb": round(b / GiB, 2)} for n, b in top],
        "note": f"'{SIA}' is this assistant and its model: expected, so don't suggest closing it; "
                "mapped model files count partly as cache that Windows frees when other apps need it.",
    }


def browser_tabs(include_titles=False):
    args = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(APP / "tabs.ps1")]
    if include_titles:
        args.append("-Titles")
    out = subprocess.run(args, capture_output=True, text=True, timeout=30,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip() or "{}"
    data = json.loads(out)
    for b in data.values():
        b["titles"] = [t.split(" - Memory usage")[0] for t in b.get("titles", [])][:60] if include_titles else []
        if not include_titles:
            b.pop("titles", None)
    total = sum(b["tabs"] for b in data.values())
    out = {"total_tabs": total, "by_browser": data, "note": "Counts tabs in open, visible browser windows."}
    names = {"chrome.exe": "Chrome", "msedge.exe": "Edge", "firefox.exe": "Firefox", "brave.exe": "Brave"}
    running = sorted({names[p.info["name"].lower()] for p in psutil.process_iter(["name"])
                      if (p.info["name"] or "").lower() in names})
    hidden = [b for b in running if b not in data]
    if hidden:  # e.g. Chrome keeps running after its last window closes
        out["running_without_windows"] = hidden
    return out


def recent_files(folder="downloads", days=1, limit=25):
    folder = str(folder).lower()
    if folder not in FOLDERS:
        return {"error": f"folder must be one of {sorted(FOLDERS)}"}
    days = max(0.0, min(float(days), 365.0))
    limit = max(1, min(int(limit), 50))
    root = _folder(folder)
    start = dt.datetime.combine(dt.date.today(), dt.time()) - dt.timedelta(days=max(days - 1, 0)) if days >= 1 \
        else dt.datetime.now() - dt.timedelta(days=days)
    items = []
    with os.scandir(root) as it:
        for e in it:
            try:
                st = e.stat()
            except OSError:
                continue
            t = dt.datetime.fromtimestamp(max(st.st_mtime, st.st_ctime))  # ctime = when it arrived on this PC
            if t >= start:
                items.append((t, e.name, e.is_dir(), st.st_size))
    items.sort(reverse=True)
    if folder == "recent":  # shortcuts to what was opened; the shortcut's time is when it was opened
        items = [(t, n.removesuffix(".lnk"), d, 0) for t, n, d, s in items
                 if n.lower() not in ("desktop.ini", "automaticdestinations", "customdestinations")]
    return {
        "folder": folder, "since": start.strftime("%Y-%m-%d %H:%M"), "count": len(items),
        "total_size_mb": round(sum(s for *_, s in items) / 2 ** 20, 1),
        "items": [{"name": n, "type": "folder" if d else "file", "size_mb": round(s / 2 ** 20, 2),
                   "time": t.strftime("%Y-%m-%d %H:%M")} for t, n, d, s in items[:limit]],
    }


def running_apps(include_titles=False):
    """Apps with a visible window right now; with include_titles, each window's title (which shows
    e.g. the file open in an editor or the page in a browser)."""
    names, windows = set(), []
    if os.name == "nt":
        u32 = ctypes.windll.user32

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def cb(hwnd, _):
            n = u32.GetWindowTextLengthW(hwnd)
            if u32.IsWindowVisible(hwnd) and n:
                pid = wintypes.DWORD()
                u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                try:
                    name = psutil.Process(pid.value).name().removesuffix(".exe")
                except psutil.Error:
                    return True
                buf = ctypes.create_unicode_buffer(n + 1)
                u32.GetWindowTextW(hwnd, buf, n + 1)
                if name.lower() == "explorer" and buf.value != "Program Manager":
                    name = "File Explorer"  # a folder window (the shell itself is "Program Manager")
                elif name.lower() == "applicationframehost":
                    name = buf.value  # Store apps (Calculator, Settings) live in this host: use their title
                names.add(name)
                windows.append({"app": name, "title": buf.value[:200]})
            return True
        u32.EnumWindows(cb, 0)
    skip = {"explorer", "TextInputHost", "ApplicationFrameHost", "SystemSettings", "ShellExperienceHost",
            "Program Manager"}
    apps = sorted(t for t in names if t not in skip)
    out = {"count": len(apps), "apps": apps}
    if include_titles:
        out["windows"] = [w for w in windows if w["app"] not in skip and w["title"] not in ("Program Manager",)]
    return out


def list_folder(path, limit=100):
    p = safe_path(path, "dir")
    limit = max(1, min(int(limit), 300))
    items, n_files, n_dirs = [], 0, 0
    with os.scandir(p) as it:
        for e in it:
            try:
                st = e.stat()
                is_dir = e.is_dir()
            except OSError:
                continue
            n_dirs += is_dir
            n_files += not is_dir
            items.append((st.st_mtime, e.name, is_dir, st.st_size))
    items.sort(reverse=True)
    return {"folder": str(p), "files": n_files, "folders": n_dirs,
            "items": [{"name": n, "type": "folder" if d else "file", "size_mb": round(s / 2 ** 20, 2),
                       "modified": dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")}
                      for t, n, d, s in items[:limit]],
            "shown": min(limit, len(items)), "sorted": "newest first"}


def read_file(path, max_chars=16000):
    """Text, code, Markdown, CSV, JSON, PDF or Word (.docx); binary files are refused."""
    p = safe_path(path, "file")
    max_chars = max(500, min(int(max_chars), 20000))
    size = p.stat().st_size
    if size > MAX_READ_BYTES:
        return {"error": f"file is {size / 2 ** 20:.0f} MB; Sia reads files up to 50 MB"}
    ext = p.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        pages = PdfReader(str(p)).pages
        text = "\n\n".join(f"[page {i + 1}]\n" + (pg.extract_text() or "") for i, pg in enumerate(pages))
    elif ext == ".docx":
        with zipfile.ZipFile(p) as z:
            xml = z.read("word/document.xml").decode("utf-8", "replace")
        text = re.sub(r"<[^>]+>", "", xml.replace("</w:p>", "\n"))
    else:
        raw = p.read_bytes()[: max_chars * 4 + 4096]
        if b"\x00" in raw[:8192]:
            return {"error": f"{p.name} is a binary file ({ext or 'no extension'}); Sia only reads text, PDF and Word files"}
        text = raw.decode("utf-8", errors="replace")
    return {"file": str(p), "size_kb": round(size / 1024, 1), "truncated": len(text) > max_chars,
            "text": text[:max_chars]}


SEARCH_SKIP = {"appdata", "node_modules", "site-packages", "__pycache__", ".git", ".venv", "venv", "cache",
               "temp", "tmp"}  # program and library data: skipped unless the search starts inside them


def user_folders():
    """The user's real folders (Windows known folders; OneDrive-redirected ones included)."""
    out = {"home": str(Path.home())}
    for name, guid in (("desktop", FOLDERS["desktop"]), ("documents", FOLDERS["documents"]),
                       ("downloads", FOLDERS["downloads"]), ("pictures", "33E28130-4E1E-4676-835A-98395C3BC3BB")):
        p = _known_folder(guid) if os.name == "nt" else None
        out[name] = str(p if p and p.is_dir() else Path.home() / name.capitalize())
    return out


def find_files(name, folder="~", limit=40):
    """Find files or folders by name under a folder; max ~8 s. Several spellings can be joined with |
    (aadhaar|aadhar); wildcards like *.pdf work. From the home folder, the personal folders are searched
    first and program data (AppData, node_modules, caches) is skipped."""
    patterns = [p.strip().lower() for p in str(name or "").split("|") if p.strip()]
    if not patterns:
        return {"error": "give a name or pattern to look for"}
    patterns = [p if any(c in p for c in "*?") else f"*{p}*" for p in patterns]
    root = safe_path(folder, "dir")
    limit = max(1, min(int(limit), 100))
    roots = [root]
    if root == Path.home().resolve():  # personal folders first, then the rest of home
        roots = [Path(p) for k, p in user_folders().items() if k != "home"] + [root]
    found, seen, scanned, t0 = [], set(), 0, time.time()
    for start in roots:
        for dirpath, dirnames, filenames in os.walk(start):
            dirnames[:] = [d for d in dirnames if not d.startswith(("$", ".")) and d.lower() not in SEARCH_SKIP
                           and not any(rx.search(os.path.join(dirpath, d)) for rx in BLOCKED_DIRS)]
            for n in dirnames + filenames:
                scanned += 1
                low = n.lower()
                if any(fnmatch.fnmatch(low, p) for p in patterns) and not any(rx.search(n) for rx in BLOCKED_NAMES):
                    full = os.path.join(dirpath, n)
                    if full.lower() in seen:
                        continue
                    seen.add(full.lower())
                    try:
                        st = os.stat(full)
                    except OSError:
                        continue
                    found.append({"path": full, "type": "folder" if n in dirnames else "file",
                                  "size_mb": round(st.st_size / 2 ** 20, 2),
                                  "modified": dt.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")})
                    if len(found) >= limit:
                        break
            if len(found) >= limit or time.time() - t0 > 8:
                break
        if len(found) >= limit or time.time() - t0 > 8:
            break
    return {"searched": [str(r) for r in roots], "patterns": patterns, "matches": found, "entries_scanned": scanned,
            "complete": time.time() - t0 <= 8 and len(found) < limit}


def _wifi():
    """Connected Wi-Fi network and signal (netsh), or None."""
    try:
        out = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True, text=True, timeout=5,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"^\s*SSID\s*:\s*(.+)$", out, re.M)
    sig = re.search(r"^\s*Signal\s*:\s*(.+)$", out, re.M)
    return {"network": m.group(1).strip(), "signal": sig.group(1).strip() if sig else None} if m else None


def system_info():
    uname = platform.uname()
    drives = []
    for part in psutil.disk_partitions(all=False):
        try:
            u = psutil.disk_usage(part.mountpoint)
            drives.append({"drive": part.device, "type": part.fstype, "free_gb": round(u.free / GiB, 1),
                           "total_gb": round(u.total / GiB, 1)})
        except OSError:
            continue
    stats, addrs = psutil.net_if_stats(), psutil.net_if_addrs()
    nets = [{"adapter": n, "up": s.isup, "speed_mbps": s.speed,
             "ipv4": [a.address for a in addrs.get(n, []) if a.family == 2]} for n, s in stats.items() if s.isup]
    wifi = _wifi()
    return {"os": f"{uname.system} {uname.release} (build {uname.version})", "computer": uname.node,
            "cpu": platform.processor(), "cpu_cores": psutil.cpu_count(logical=False), "cpu_threads": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / GiB, 1), "gpu": _gpu(), "drives": drives,
            "network_adapters_up": nets, "wifi": wifi,
            "booted": dt.datetime.fromtimestamp(psutil.boot_time()).strftime("%Y-%m-%d %H:%M")}


def running_processes(sort="memory", limit=15):
    limit = max(1, min(int(limit), 50))
    procs = list(psutil.process_iter(["name", "pid", "memory_info"]))
    for p in procs:
        try:
            p.cpu_percent(None)
        except psutil.Error:
            pass
    time.sleep(0.5)
    rows = []
    for p in procs:
        try:
            rows.append({"name": p.info["name"], "pid": p.info["pid"], "cpu_pct": round(p.cpu_percent(None), 1),
                         "ram_mb": round(p.info["memory_info"].rss / 2 ** 20)})
        except (psutil.Error, AttributeError):
            continue
    key = "cpu_pct" if str(sort).lower().startswith("cpu") else "ram_mb"
    rows.sort(key=lambda r: -r[key])
    return {"total_processes": len(rows), "sorted_by": key, "top": rows[:limit]}


def installed_apps(filter=""):
    import winreg
    seen = {}
    roots = [(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
             (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
             (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall")]
    for hive, key in roots:
        try:
            with winreg.OpenKey(hive, key) as k:
                for i in range(winreg.QueryInfoKey(k)[0]):
                    try:
                        with winreg.OpenKey(k, winreg.EnumKey(k, i)) as sub:
                            name = winreg.QueryValueEx(sub, "DisplayName")[0]
                            try:
                                ver = winreg.QueryValueEx(sub, "DisplayVersion")[0]
                            except OSError:
                                ver = ""
                            seen[name] = ver
                    except OSError:
                        continue
        except OSError:
            continue
    f = str(filter or "").lower()
    apps = sorted((n, v) for n, v in seen.items() if f in n.lower())
    return {"count": len(apps), "apps": [{"name": n, "version": v} for n, v in apps[:200]]}


def clipboard_text():
    """Text currently on the clipboard (first 10,000 characters)."""
    u32, k32 = ctypes.windll.user32, ctypes.windll.kernel32
    u32.GetClipboardData.restype = ctypes.c_void_p
    k32.GlobalLock.restype = ctypes.c_void_p
    k32.GlobalLock.argtypes = [ctypes.c_void_p]
    k32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    if not u32.OpenClipboard(None):
        return {"error": "clipboard is busy"}
    try:
        h = u32.GetClipboardData(13)  # CF_UNICODETEXT
        if not h:
            return {"text": "", "note": "no text on the clipboard"}
        ptr = k32.GlobalLock(h)
        try:
            text = ctypes.wstring_at(ptr)
        finally:
            k32.GlobalUnlock(h)
    finally:
        u32.CloseClipboard()
    return {"chars": len(text), "text": text[:10000]}


def _powershell_json(category):
    out = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(APP / "devices.ps1"),
                          "-Category", category], capture_output=True, text=True, timeout=40,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
    return json.loads(out or "{}")


def _displays():
    """Resolution and refresh rate of each active display (EnumDisplaySettings)."""
    class DEVMODE(ctypes.Structure):
        _fields_ = [("name", ctypes.c_wchar * 32), ("spec", wintypes.WORD), ("drv", wintypes.WORD),
                    ("size", wintypes.WORD), ("extra", wintypes.WORD), ("fields", wintypes.DWORD),
                    ("pos", ctypes.c_long * 2), ("orient", wintypes.DWORD), ("fixed", wintypes.DWORD),
                    ("color", ctypes.c_short), ("duplex", ctypes.c_short), ("yres", ctypes.c_short),
                    ("tt", ctypes.c_short), ("collate", ctypes.c_short), ("form", ctypes.c_wchar * 32),
                    ("logpix", wintypes.WORD), ("bits", wintypes.DWORD), ("width", wintypes.DWORD),
                    ("height", wintypes.DWORD), ("flags", wintypes.DWORD), ("hz", wintypes.DWORD),
                    ("rest", wintypes.DWORD * 8)]

    class DISPLAY_DEVICE(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("name", ctypes.c_wchar * 32), ("string", ctypes.c_wchar * 128),
                    ("flags", wintypes.DWORD), ("id", ctypes.c_wchar * 128), ("key", ctypes.c_wchar * 128)]
    u32, out, i = ctypes.windll.user32, [], 0
    while True:
        dd = DISPLAY_DEVICE(cb=ctypes.sizeof(DISPLAY_DEVICE))
        if not u32.EnumDisplayDevicesW(None, i, ctypes.byref(dd), 0):
            break
        i += 1
        if not dd.flags & 1:  # DISPLAY_DEVICE_ATTACHED_TO_DESKTOP
            continue
        dm = DEVMODE(size=ctypes.sizeof(DEVMODE))
        if u32.EnumDisplaySettingsW(dd.name, -1, ctypes.byref(dm)):  # ENUM_CURRENT_SETTINGS
            out.append({"adapter": dd.string, "resolution": f"{dm.width} x {dm.height}", "refresh_hz": dm.hz,
                        "primary": bool(dd.flags & 4)})
    return out


DEVICE_CATEGORIES = ("battery", "display", "audio", "bluetooth", "usb", "printers", "startup", "internet", "model",
                     "games")
GAME_PUBLISHERS = re.compile(r"riot games|electronic arts|ubisoft|rockstar|activision|blizzard|hoyoverse|mihoyo|"
                             r"bandai namco|square enix|capcom|sega|bethesda|2k|cd projekt|krafton|epic games inc",
                             re.I)


def installed_games():
    """Games from Steam, Epic and Xbox libraries, plus programs by known game publishers."""
    out = {"steam": [], "epic": [], "xbox": [], "other": []}
    steam = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Steam"
    libs = {steam}
    try:
        vdf = (steam / "config" / "libraryfolders.vdf").read_text(encoding="utf-8", errors="replace")
        libs |= {Path(p.replace("\\\\", "\\")) for p in re.findall(r'"path"\s+"([^"]+)"', vdf)}
    except OSError:
        pass
    for lib in libs:
        for acf in (lib / "steamapps").glob("appmanifest_*.acf"):
            m = re.search(r'"name"\s+"([^"]+)"', acf.read_text(encoding="utf-8", errors="replace"))
            if m and not re.search(r"redistributable|proton|steam linux runtime|steamvr", m.group(1), re.I):
                out["steam"].append(m.group(1))
    for item in Path(os.environ.get("ProgramData", r"C:\ProgramData"), "Epic", "EpicGamesLauncher", "Data",
                     "Manifests").glob("*.item"):
        try:
            out["epic"].append(json.loads(item.read_text(encoding="utf-8")).get("DisplayName", item.stem))
        except (OSError, ValueError):
            continue
    for part in psutil.disk_partitions(all=False):
        xb = Path(part.mountpoint) / "XboxGames"
        if xb.is_dir():
            out["xbox"] += sorted(d.name for d in xb.iterdir() if d.is_dir() and d.name.lower() != "gamesave")
    if os.name == "nt":
        import winreg
        for hive, key in ((winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
                          (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
                          (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall")):
            try:
                with winreg.OpenKey(hive, key) as k:
                    for i in range(winreg.QueryInfoKey(k)[0]):
                        try:
                            with winreg.OpenKey(k, winreg.EnumKey(k, i)) as sub:
                                name = winreg.QueryValueEx(sub, "DisplayName")[0]
                                pub = winreg.QueryValueEx(sub, "Publisher")[0]
                        except OSError:
                            continue
                        if GAME_PUBLISHERS.search(pub) and name not in out["other"]:
                            out["other"].append(f"{name} ({pub})")
            except OSError:
                continue
    out["count"] = sum(len(v) for v in out.values() if isinstance(v, list))
    return out


def device_info(category):
    """Read-only device details: one category per call."""
    category = str(category or "").lower()
    if category not in DEVICE_CATEGORIES:
        return {"error": f"category must be one of {list(DEVICE_CATEGORIES)}"}
    if category == "games":
        return installed_games()
    data = _powershell_json(category)
    if category == "battery":
        b = psutil.sensors_battery()
        if b:
            data.update(percent=round(b.percent), plugged_in=b.power_plugged)
            if not b.power_plugged and b.secsleft not in (psutil.POWER_TIME_UNLIMITED, psutil.POWER_TIME_UNKNOWN):
                data["time_left_min"] = round(b.secsleft / 60)
        data["note"] = "health = full-charge capacity / design capacity (Windows battery report)"
    elif category == "internet":
        data["wifi"] = _wifi()
        data["wifi_radio"] = actions._ps("radios").get("wifi", "unknown")
    if category == "bluetooth":
        # an installed adapter and paired devices don't mean Bluetooth is on: read the radio itself
        data["bluetooth_radio"] = actions._ps("radios").get("bluetooth", "unknown")
        data["note"] = "bluetooth_radio is whether Bluetooth is switched on; the adapter is just the hardware"
    elif category == "display":
        data["monitors"] = [m.strip("\x00 ") for m in data.get("monitors", []) if m.strip("\x00 ")]
        data["displays"] = _displays()
    return data


_PLACES = []


def _places():
    if not _PLACES:
        with open(APP / "data" / "places.tsv", encoding="utf-8") as f:
            for line in f:
                if not line.startswith("#"):
                    n, region, country, lat, lon, pop = line.rstrip("\n").split("\t")
                    _PLACES.append((n, region, country, float(lat), float(lon), int(pop)))
    return _PLACES


def _km(lat1, lon1, lat2, lon2):
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * math.asin(math.sqrt(a))


def _direction(lat1, lon1, lat2, lon2):
    """Compass direction from point 1 to point 2."""
    y = math.sin(math.radians(lon2 - lon1)) * math.cos(math.radians(lat2))
    x = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) -
         math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(lon2 - lon1)))
    return ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"][
        round(math.degrees(math.atan2(y, x)) % 360 / 45) % 8]


def nearest_place(lat, lon):
    """Name a position offline (GeoNames): inside a city of 1M+ (within 25 km of its centre) the city
    with direction and distance from its centre, otherwise the nearest town."""
    rows = [(_km(lat, lon, r[3], r[4]), r) for r in _places() if abs(r[3] - lat) < 3 and abs(r[4] - lon) < 3]
    if not rows:
        rows = [(_km(lat, lon, r[3], r[4]), r) for r in _places()]
    d, r = min(rows)
    big = [(dd, rr) for dd, rr in rows if rr[5] >= 1_000_000 and dd <= 25]
    if big:
        dd, rr = min(big)
        where = ("near the city centre" if dd < 1.5 else
                 f"about {round(dd)} km {_direction(rr[3], rr[4], lat, lon)} of the city centre")
        return {"place": f"{rr[0]}, {rr[1]}, {rr[2]}", "where_in_city": where}
    return {"place": f"{r[0]}, {r[1]}, {r[2]}",
            "where": f"about {round(d)} km {_direction(r[3], r[4], lat, lon)} of {r[0]}"}


def region_and_timezone():
    """Windows' home region and the time zone (both offline settings)."""
    region = None
    if os.name == "nt":
        k32 = ctypes.windll.kernel32
        buf = ctypes.create_unicode_buffer(128)
        if k32.GetGeoInfoW(k32.GetUserGeoID(16), 8, buf, 128, 0):  # GEOCLASS_NATION, GEO_FRIENDLYNAME
            region = buf.value
    off = time.strftime("%z")
    return {"windows_region": region, "time_zone": time.tzname[time.localtime().tm_isdst],
            "utc_offset": f"{off[:3]}:{off[3:]}" if len(off) == 5 else off}


def location():
    """Where the laptop is: Windows location (if on) named offline, plus region and time zone."""
    out = region_and_timezone()
    pos = _powershell_json("location")
    if "lat" in pos:
        out.update(latitude=round(pos["lat"], 4), longitude=round(pos["lon"], 4), accuracy_m=pos.get("accuracy_m"))
        out.update(nearest_place(pos["lat"], pos["lon"]))
        out["source"] = "Windows location service; place names from GeoNames (offline)"
    else:
        out["position"] = (f"unavailable (Windows location status: {pos.get('status')}, permission: "
                           f"{pos.get('permission')}). Turn on Settings > Privacy & security > Location.")
    return out


def laptop_summary():
    """One line describing this laptop (maker, model, CPU, GPU, RAM), from the registry and nvidia-smi."""
    parts = []
    if os.name == "nt":
        import winreg

        def reg(path, name):
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as k:
                    return str(winreg.QueryValueEx(k, name)[0]).strip()
            except OSError:
                return ""
        maker = reg(r"HARDWARE\DESCRIPTION\System\BIOS", "SystemManufacturer")
        model = reg(r"HARDWARE\DESCRIPTION\System\BIOS", "SystemProductName")
        cpu = reg(r"HARDWARE\DESCRIPTION\System\CentralProcessor\0", "ProcessorNameString")
        parts += [p for p in (" ".join(x for x in (maker, model) if x), cpu) if p]
    gpu = _gpu()
    if gpu:
        parts.append(gpu["name"] if "GB" in gpu["name"] else f"{gpu['name']} {gpu['vram_total_gb']:.0f} GB")
    nominal = 2 ** round(math.log2(psutil.virtual_memory().total / GiB))  # 15.3 GiB usable = a 16 GB laptop
    parts.append(f"{nominal} GB RAM")
    return ", ".join(parts)


def now_context(days=False):
    """One line for every request: the laptop's current date, time, time zone and region. With days=True, also the
    next seven dates with their weekdays: asked for a study plan "starting today" on a Monday, Sia once headed it
    "Tuesday (Today)"."""
    rt, now = region_and_timezone(), dt.datetime.now()
    clock = now.strftime("%I:%M %p").lstrip("0")
    region = f" Windows region: {rt['windows_region']}." if rt["windows_region"] else ""
    ahead = ""
    if days:
        d = [now + dt.timedelta(days=i) for i in range(8)]
        ahead = (f" Dates: today {d[0]:%A} {d[0].day} {d[0]:%b}, tomorrow {d[1]:%A} {d[1].day} {d[1]:%b}, then "
                 + ", ".join(f"{x:%A} {x.day} {x:%b}" for x in d[2:]) + ".")
    return (f"Right now on this laptop: {now:%A, %d %B %Y}, {now:%H:%M} ({clock}), "
            f"{rt['time_zone']} (UTC{rt['utc_offset']}).{region}{ahead}")


# --------------------------------------------------------------------------- registry
TOOLS = {
    "system_vitals": {
        "fn": system_vitals, "label": "Check system vitals",
        "reads": "CPU, memory, GPU, battery, disk space and which apps use the most memory",
        "schema": {"type": "object", "properties": {}},
        "description": "Live laptop vitals: CPU load, RAM, GPU temperature/load/VRAM, battery, free disk "
                       "space, uptime, and the apps using the most memory.",
    },
    "browser_tabs": {
        "fn": browser_tabs, "label": "Count open browser tabs",
        "reads": "how many tabs are open in Chrome/Edge/Firefox (and their titles, if asked)",
        "schema": {"type": "object", "properties": {
            "include_titles": {"type": "boolean", "description": "also return tab titles (only if the user asked for them)"}}},
        "description": "Count the tabs open in the user's browsers (Chrome, Edge, Brave, Firefox), per browser.",
    },
    "recent_files": {
        "fn": recent_files, "label": "List recent files",
        "reads": "names, sizes and times of recent files in Downloads, Desktop or Documents, or recently opened files",
        "schema": {"type": "object", "properties": {
            "folder": {"type": "string", "enum": sorted(FOLDERS),
                       "description": "which folder; 'recent' = files the user recently opened anywhere"},
            "days": {"type": "number", "description": "how far back: 1 = today, 7 = this week"},
            "limit": {"type": "integer", "description": "max items to list (1-50)"}},
            "required": ["folder"]},
        "description": "List files added or changed recently in the user's Downloads, Desktop or Documents "
                       "folder (e.g. 'what did I download today'), or with folder='recent' the files the user "
                       "recently opened anywhere (e.g. 'what did I open recently').",
    },
    "running_apps": {
        "fn": running_apps, "label": "List open apps",
        "reads": "the apps that have a window open (and their window titles, if asked)",
        "schema": {"type": "object", "properties": {
            "include_titles": {"type": "boolean", "description": "also return each window's title, which shows "
                                                                 "e.g. the file open in an editor or the page in a browser"}}},
        "description": "List the apps that currently have a window open; with include_titles, what each window shows "
                       "(e.g. 'what's open in VS Code').",
    },
    "list_folder": {
        "fn": list_folder, "label": "List a folder",
        "reads": "the names, sizes and dates of the items in one folder",
        "schema": {"type": "object", "properties": {
            "path": {"type": "string", "description": "folder path, e.g. C:\\Users\\me\\Documents or ~\\Pictures"},
            "limit": {"type": "integer", "description": "max items (1-300)"}}, "required": ["path"]},
        "description": "List what is inside a folder on the laptop (newest first).",
    },
    "read_file": {
        "fn": read_file, "label": "Read a file",
        "reads": "the text of one file (text, code, PDF or Word)",
        "schema": {"type": "object", "properties": {
            "path": {"type": "string", "description": "full file path"},
            "max_chars": {"type": "integer", "description": "how much text to read (500-20000)"}}, "required": ["path"]},
        "description": "Read a text, code, Markdown, CSV, JSON, PDF or Word (.docx) file on the laptop.",
    },
    "find_files": {
        "fn": find_files, "label": "Search for files",
        "reads": "file and folder names (not contents) under one folder",
        "schema": {"type": "object", "properties": {
            "name": {"type": "string", "description": "part of the name; join spelling variants with | "
                                                      "(e.g. aadhaar|aadhar|adhar), or a pattern like *.pdf"},
            "folder": {"type": "string", "description": "where to search; default ~ (the user's home: personal "
                                                        "folders first, program data skipped), or e.g. D:\\"},
            "limit": {"type": "integer", "description": "max matches (1-100)"}}, "required": ["name"]},
        "description": "Find files or folders by name on the laptop. One call with all likely spellings joined "
                       "by | usually finds it.",
    },
    "system_info": {
        "fn": system_info, "label": "Read system information",
        "reads": "OS version, CPU, GPU, RAM, drives, network adapters and the Wi-Fi network name",
        "schema": {"type": "object", "properties": {}},
        "description": "Laptop hardware and system details: OS, CPU, GPU, RAM, drives and free space, network, Wi-Fi.",
    },
    "running_processes": {
        "fn": running_processes, "label": "List running processes",
        "reads": "running programs with their CPU and memory use",
        "schema": {"type": "object", "properties": {
            "sort": {"type": "string", "enum": ["memory", "cpu"]},
            "limit": {"type": "integer", "description": "how many (1-50)"}}},
        "description": "List the processes using the most memory or CPU right now.",
    },
    "installed_apps": {
        "fn": installed_apps, "label": "List installed apps",
        "reads": "the names and versions of installed programs",
        "schema": {"type": "object", "properties": {
            "filter": {"type": "string", "description": "optional: only names containing this"}}},
        "description": "List the programs installed on the laptop (optionally filtered by name).",
    },
    "device_info": {
        "fn": device_info, "label": "Read device details",
        "reads": "one kind of device detail: battery health, display, audio, Bluetooth, USB, printers, startup apps, "
                 "internet connection or laptop model",
        "schema": {"type": "object", "properties": {
            "category": {"type": "string", "enum": list(DEVICE_CATEGORIES),
                         "description": "battery (health, capacity, time left), display (resolution, refresh rate, "
                                        "brightness), audio (volume, mute, devices), bluetooth (paired devices, "
                                        "connected or not), usb (connected devices), printers, startup (apps that "
                                        "start with Windows), internet (online or not, Wi-Fi name and signal, IP, gateway), games (installed "
                                        "games: Steam, Epic, Xbox, others), model (laptop "
                                        "maker, model, BIOS)"}},
            "required": ["category"]},
        "description": "Read details about the laptop's devices and settings, one category per call.",
    },
    "location": {
        "fn": location, "label": "Check location",
        "reads": "the laptop's position from Windows location (named offline), its region and time zone",
        "schema": {"type": "object", "properties": {}},
        "description": "Where the laptop is right now: nearest town and city, coordinates and accuracy (from "
                       "Windows location), plus the Windows region and time zone.",
    },
    "word_game": {
        "fn": games.word_game, "label": "Word game", "auto": True,  # reads nothing from the laptop: no permission card
        "reads": "nothing from the laptop (Sia's own game engine)",
        "schema": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["start", "guess", "hint", "reveal"],
                       "description": "start: begin a game (a running one continues unless new=true); guess: score "
                                      "the user's guess; hint: when the user asks for a hint; reveal: only when the "
                                      "user gives up or asks for the word"},
            "game": {"type": "string", "enum": ["wordle", "hangman"], "description": "for start"},
            "guess": {"type": "string", "description": "for guess: the user's word (Wordle) or letter/word (Hangman)"},
            "new": {"type": "boolean", "description": "for start: true only if the user asked for a fresh game"}},
            "required": ["action"]},
        "description": "Referee for Wordle and Hangman: it picks and keeps the secret word and scores every guess. "
                       "Pass every move of the user to it, then tell the user its 'summary' in full (every letter's "
                       "result, the board, what's left); never invent feedback or a secret word.",
    },
    "calculate": {
        "fn": calc.calculate, "label": "Calculate", "auto": True,  # reads nothing from the laptop: no card
        "reads": "nothing from the laptop (Sia's exact calculator)",
        "schema": {"type": "object", "properties": {
            "expression": {"type": "string", "description":
                "e.g. 876*88894, (1+0.05/12)**120, 15% * 2400, sqrt(2), 25!. Several steps separated by ; with "
                "named values: 'dose = 15*72; dose/250'. No units or thousands commas. Operators + - * / // % ** "
                "(^ works too). Functions: sqrt cbrt root(x,n) exp ln log(x) (natural) log(x,base) log10 log2 "
                "sin cos tan asin acos atan atan2 (radians) sind cosd tand asind acosd atand (degrees) sinh cosh "
                "tanh hypot factorial comb perm gcd lcm abs floor ceil round(x,digits) (half up) min max sum mean "
                "median stdev pstdev variance; constants pi e tau."},
            "format": {"type": "string", "enum": ["indian", "international", "words"],
                       "description": "also write the result out: 'indian' = lakh/crore grouping (12,34,567) and "
                                      "words; 'international' = 1,234,567 and million/billion words; 'words' = both"}},
            "required": ["expression"]},
        "description": "Exact calculator. Use it for EVERY calculation, even 11*2: whole numbers of any length are "
                       "exact, fractions and money have no rounding error. Report its 'result' exactly as given.",
    },
    "run_python": {
        "fn": sandbox.run_python, "label": "Run Python", "auto": True,  # sandboxed: no files, network or programs
        "reads": "nothing from the laptop (a locked-down Python sandbox)",
        "schema": {"type": "object", "properties": {
            "code": {"type": "string", "description":
                "A short Python program. The value of its last line is returned, and print() output too. "
                "Available: math, datetime (local time), calendar, statistics, string, re, json, itertools, "
                "collections, random; "
                "exact(\"expression\") gives calculate's exact result as text, words(number, \"indian\" or "
                "\"international\") spells a number. No files, internet, input() or installing packages."}},
            "required": ["code"]},
        "description": "Run a short Python program in a sandbox for anything that needs logic or data handling: dates "
                       "and durations, unit conversions, sorting or counting lists, text processing, tables, "
                       "multi-step calculations. Use calculate for a single formula.",
    },
    # ---- actions: they change something, so each one is approved by the user first (see actions.py)
    "open": {
        "fn": actions.open_, "label": "Open", "action": True,
        "does": "opens an app from the Start menu, a document or folder, a website, or a Settings page",
        "schema": {"type": "object", "properties": {
            "what": {"type": "string", "enum": ["app", "file", "folder", "website", "settings"]},
            "target": {"type": "string", "description": "app name (e.g. Spotify), full path, https:// address, or a "
                                                        "Settings page: " + ", ".join(sorted(actions.SETTINGS_PAGES))}},
            "required": ["what", "target"]},
        "description": "Open an app, file, folder, website or Windows Settings page. Programs and scripts can't be "
                       "opened as files; apps open by name.",
    },
    "control": {
        "fn": actions.control, "label": "Change", "action": True,
        "does": "changes volume, mute, brightness, Bluetooth, Wi-Fi, dark/light theme, or media playback",
        "schema": {"type": "object", "properties": {
            "setting": {"type": "string", "enum": ["volume", "mute", "brightness", "bluetooth", "wifi", "theme", "media"]},
            "value": {"type": "string", "description": "volume/brightness: 0-100; mute/bluetooth/wifi: on or off; "
                                                       "theme: dark or light; media: play, pause, next, previous or stop"}},
            "required": ["setting", "value"]},
        "description": "Change a laptop setting: volume, mute, brightness, Bluetooth, Wi-Fi, theme, or control the "
                       "music/video playing. Call it directly (no need to read the current value first): it reports the value "
                       "before and after.",
    },
    "window": {
        "fn": actions.window, "label": "Window", "action": True,
        "does": "brings an app's window to the front, minimises or maximises it, or closes it like clicking X",
        "schema": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["focus", "minimize", "maximize", "close"]},
            "app": {"type": "string", "description": "the app's name, e.g. Spotify or Chrome"}},
            "required": ["action", "app"]},
        "description": "Bring an app to the front, minimise, maximise, or close it (closing is like clicking X; the "
                       "app may ask to save; nothing is force-killed).",
    },
    "power": {
        "fn": actions.power_, "label": "Power", "action": True,
        "does": "locks the screen, puts the laptop to sleep, or restarts or shuts it down after 60 s (cancellable)",
        "schema": {"type": "object", "properties": {
            "mode": {"type": "string", "enum": ["lock", "sleep", "restart", "shutdown", "cancel"]}},
            "required": ["mode"]},
        "description": "Lock the screen, sleep, restart or shut down (restart/shutdown wait 60 s; 'cancel' stops them).",
    },
    "screenshot": {
        "fn": actions.take_screenshot, "label": "Take a screenshot", "action": True,
        "does": "saves a screenshot of the screen to Pictures\\Screenshots",
        "schema": {"type": "object", "properties": {}},
        "description": "Take a screenshot and save it to Pictures\\Screenshots.",
    },
    "copy": {
        "fn": actions.copy_to_clipboard, "label": "Copy to the clipboard", "action": True,
        "does": "puts text on the clipboard (replacing what's there)",
        "schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        "description": "Copy text to the clipboard so the user can paste it.",
    },
    "save_note": {
        "fn": actions.save_note, "label": "Save a note", "action": True,
        "does": "saves text as a new file in Documents\\Sia (never overwrites a file)",
        "schema": {"type": "object", "properties": {
            "name": {"type": "string", "description": "file name, e.g. shopping list.txt (.txt, .md, .csv or .json)"},
            "text": {"type": "string"}}, "required": ["name", "text"]},
        "description": "Save text (a note, a draft, a list) as a new file in Documents\\Sia.",
    },
    "reminder": {
        "fn": actions.reminder, "label": "Reminder", "action": True,
        "does": "sets, lists or cancels a reminder or timer, shown as a Windows notification with a sound",
        "schema": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["set", "list", "cancel"]},
            "text": {"type": "string", "description": "for set: what to remind about"},
            "seconds": {"type": "number", "description": "for set: seconds from now (timers, e.g. 10); give "
                                                         "seconds OR minutes, not both"},
            "minutes": {"type": "number", "description": "for set: minutes from now (instead of seconds)"},
            "at": {"type": "string", "description": "for set, instead of minutes: a time like 17:30 or 5:30 pm"},
            "id": {"type": "string", "description": "for cancel: the reminder's id from list (or give text: "
                                                    "words of the reminder, e.g. oven; 'all' for every pending one; or neither for the latest)"}},
            "required": ["action"]},
        "description": "Set a reminder or timer (in N seconds, N minutes, or at a time), list pending ones, or "
                       "cancel one. It pops up as a Windows notification with a sound.",
    },
    "undo": {
        "fn": actions.undo, "label": "Undo", "action": True,
        "does": "puts back something Sia did earlier in this session (a setting's old value, closes what it "
                "opened, cancels a timer or shutdown, restores the clipboard)",
        "schema": {"type": "object", "properties": {
            "what": {"type": "string", "description": "'last' for the most recent action, or words naming it "
                                                      "(e.g. volume, bluetooth, theme, spotify, timer, shutdown)"}}},
        "description": "Undo or cancel something Sia did earlier: put a setting back, close an app or file it "
                       "opened, cancel a timer/reminder or a restart/shutdown, restore the clipboard.",
    },
    "clipboard_text": {
        "fn": clipboard_text, "label": "Read the clipboard",
        "reads": "the text currently copied to the clipboard",
        "schema": {"type": "object", "properties": {}},
        "description": "Read the text the user has currently copied to the clipboard.",
    },
}


# ---- MCP plug-ins (mcp_host.py, mcp.json): their tools join TOOLS under <server>_<tool>
MCP = {"host": None}


def register_mcp(host):
    """Add the plug-ins' allowed tools. Each keeps its policy: auto (no card), read, action, or ask (a card every
    time, never always-allowed). 'requires' says which Settings switch it needs: access, actions or none."""
    MCP["host"] = host
    for name, t in host.tools.items():
        spec = host.servers[t["server"]].spec
        TOOLS[name] = {
            "fn": (lambda n: lambda **kw: host.call(n, kw))(name), "label": t["tool"], "mcp": t["server"],
            "auto": t["policy"] == "auto", "action": t["policy"] in ("action", "ask"),
            "ask": t["policy"] == "ask", "requires": t["rule"].get("requires", spec.get("requires")),
            "reads": "", "does": t["rule"].get("does", ""), "schema": t["schema"], "description": t["description"]}
    if "memory" in host.servers:
        register_forget()


FORGET_SKIP = {"my", "the", "a", "an", "about", "that", "this", "name", "names", "i", "me", "forget", "all",
               "everything", "you", "know", "remember", "of", "please", "s", "is", "what"}


def _memory_rows():
    host = MCP["host"]
    srv = host.servers.get("memory") if host else None
    if not srv:
        return None, []
    path = Path(srv._expand(srv.spec.get("env", {}).get("MEMORY_FILE_PATH", "")))
    try:
        return path, [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    except (OSError, ValueError):
        return path, []


def _forget_plan(what):
    """Which remembered facts match the words (all of them must appear): whole entities whose name matches or
    whose every fact matches, else the single matching facts."""
    words = [w for w in re.findall(r"[a-z0-9]+", str(what).lower()) if w not in FORGET_SKIP]
    _, rows = _memory_rows()
    entities, facts = [], []
    for r in rows:
        if r.get("type") != "entity" or not words:
            continue
        hits = [o for o in r.get("observations", [])
                if all(w in f"{r['name']} {r.get('entityType', '')} {o}".lower() for w in words)]
        if all(w in r["name"].lower() for w in words) or (hits and len(hits) == len(r.get("observations", []))):
            entities.append(r["name"])
        elif hits:
            facts.append({"entityName": r["name"], "observations": hits})
    return entities, facts


def forget(what):
    entities, facts = _forget_plan(what)
    if not entities and not facts:
        _, rows = _memory_rows()
        return {"error": "nothing remembered matches that; nothing was forgotten",
                "remembered": [r.get("name") for r in rows if r.get("type") == "entity"]}
    host = MCP["host"]
    if entities:
        host.servers["memory"].call("delete_entities", {"entityNames": entities})
    if facts:
        host.servers["memory"].call("delete_observations", {"deletions": facts})
    left = _forget_plan(what)
    return {"forgot": entities + [f"{f['entityName']}: {o}" for f in facts for o in f["observations"]],
            "verified": not left[0] and not left[1]}


def forget_label(what):
    entities, facts = _forget_plan(what)
    parts = [f"everything about {e}" for e in entities] + [f"{f['entityName']}: {o}" for f in facts
                                                           for o in f["observations"]]
    return "Forget: " + "; ".join(parts) if parts else "Forget (nothing matches)"


def register_forget():
    """Sia's own 'forget' on top of the Memory plug-in: the user names what to forget in plain words and the
    card lists exactly which stored facts match (the model once guessed a fact's wording, nothing matched, and
    it still said it had forgotten)."""
    TOOLS["forget"] = {
        "fn": forget, "label": "Forget", "action": True, "requires": None, "plugin": "memory",
        "does": "deletes the matching facts from Sia's memory",
        "schema": {"type": "object", "properties": {"what": {"type": "string", "description":
                   "the person, thing or fact to forget, in the user's words (e.g. 'my sister', 'HP Victus')"}},
                   "required": ["what"]},
        "description": "Forget remembered facts about a person, thing or topic (the user approves the exact list)."}


def always_ask(name):
    """Tools that need a fresh approval every time (never 'Always allow')."""
    return bool(TOOLS.get(name, {}).get("ask"))


def openai_tools(names=None):
    return [{"type": "function", "function": {"name": n, "description": t["description"], "parameters": t["schema"]}}
            for n, t in TOOLS.items() if names is None or n in names]


def run(name, args_json):
    """Run a registered tool with validated arguments; returns a JSON-able dict."""
    if name not in TOOLS:
        return {"error": f"unknown tool {name}"}
    try:
        args = json.loads(args_json or "{}")
        if not isinstance(args, dict):
            raise ValueError("arguments must be an object")
        allowed = set(TOOLS[name]["schema"].get("properties", {}))
        args = {k: v for k, v in args.items() if k in allowed}  # drop anything unexpected
        if name == "reminder":
            args = actions.fix_reminder_args(args)  # the duration the user stated wins
        result = TOOLS[name]["fn"](**args)
        if TOOLS[name].get("action") and name != "undo" and isinstance(result, dict) and "error" not in result:
            actions.record(name, args, result)  # so it can be undone later
        return result
    except Exception as e:  # report the failure to the model instead of crashing the chat
        return {"error": f"{type(e).__name__}: {e}"}


def nothing_to_do(name, args_json):
    """An error result when a cancel or undo has nothing to act on, so no card is shown for it."""
    try:
        args = json.loads(args_json or "{}")
    except ValueError:
        return None
    if name == "forget" and isinstance(args, dict) and not any(_forget_plan(args.get("what", ""))):
        return forget(args.get("what", ""))  # nothing matches: say so without a card
    if TOOLS.get(name, {}).get("mcp") and isinstance(args, dict):  # e.g. a blocked shortcut: refused, no card
        refused = MCP["host"].check(name, args)
        if refused:
            return {"error": refused + "; nothing was done"}
    if name == "reminder" and str(args.get("action")) == "cancel":
        picks, pending = actions._find_reminders(args.get("id", ""), args.get("text", ""))
        if not picks:
            return {"error": "no matching pending reminder; nothing was cancelled",
                    "pending": [r["text"] for r in pending]}
    if name == "undo" and not actions._undo_candidate(args.get("what", "last")):
        return {"error": "nothing Sia did in this session matches that; nothing was undone"}
    return None


def describe(name, args_json):
    """One line for the permission card."""
    t = TOOLS.get(name)
    if not t:
        return f"Unknown tool {name}"
    try:
        args = json.loads(args_json or "{}")
    except ValueError:
        args = {}
    if name == "forget":
        return forget_label(args.get("what", "") if isinstance(args, dict) else "")
    if t.get("mcp"):
        return MCP["host"].describe(name, args if isinstance(args, dict) else {})
    if name == "reminder" and isinstance(args, dict):
        args = actions.fix_reminder_args(args)  # the card shows what will really be set
    extra = ""
    if name == "recent_files":
        extra = f" ({args.get('folder', 'downloads')}, last {args.get('days', 1)} day(s))"
    if name == "browser_tabs" and args.get("include_titles"):
        extra = " (including tab titles)"
    if name in ("list_folder", "read_file"):
        extra = f": {args.get('path', '?')}"
    if name == "find_files":
        extra = f" named '{args.get('name', '?')}' in {args.get('folder', '~')}"
    if name == "device_info":
        extra = f": {args.get('category', '?')}"
    if name == "word_game":
        act = args.get("action", "?")
        extra = ": " + " ".join(str(x) for x in (act, args.get("game") if act == "start" else None,
                                                  str(args.get("guess") or "").upper() or None) if x)
    # actions: the card says exactly what will happen
    if name == "open":
        return f"Open {args.get('what', '?')}: {args.get('target', '?')}"
    if name == "control":
        s, v = str(args.get("setting", "?")), str(args.get("value", "?"))
        return {"volume": f"Set volume to {v}%", "brightness": f"Set brightness to {v}%", "mute": f"Turn mute {v}",
                "bluetooth": f"Turn Bluetooth {v}", "wifi": f"Turn Wi-Fi {v}", "theme": f"Switch to {v} theme",
                "media": f"Media: {v}"}.get(s, f"Change {s} to {v}")
    if name == "window":
        a, app = str(args.get("action", "?")), args.get("app", "?")
        return {"close": f"Close {app} (like clicking X)", "focus": f"Bring {app} to the front",
                "minimize": f"Minimise {app}", "maximize": f"Maximise {app}"}.get(a, f"{a} {app}")
    if name == "power":
        m = str(args.get("mode", "?"))
        return {"lock": "Lock the screen", "sleep": "Put the laptop to sleep",
                "restart": "Restart the laptop in 60 seconds", "shutdown": "Shut down the laptop in 60 seconds",
                "cancel": "Cancel the restart/shutdown"}.get(m, f"Power: {m}")
    if name == "copy":
        return f"Copy {len(str(args.get('text', '')))} characters to the clipboard"
    if name == "save_note":
        return f"Save a note: Documents\\Sia\\{args.get('name', 'note')}"
    if name == "calculate":
        expr = str(args.get("expression", ""))
        return "Calculate: " + (expr if len(expr) <= 80 else expr[:77] + "...")
    if name == "undo":
        return actions.undo_description(args.get("what", "last"))
    if name == "reminder" and str(args.get("action")) == "cancel":
        picks, _ = actions._find_reminders(args.get("id", ""), args.get("text", ""))  # the card names what goes
        if len(picks) > 1:
            return f"Cancel all {len(picks)} reminders: " + ", ".join(r["text"] for r in picks)
        return f"Cancel the reminder: {picks[0]['text']}" if picks else "Cancel a reminder (none matches)"
    if name == "reminder":
        if str(args.get("action")) == "set":
            if args.get("seconds") is not None or args.get("minutes") is not None:
                total = (float(args["seconds"]) if args.get("seconds") is not None
                         else 60 * float(args["minutes"]))
                when = f"in {total:.0f} s" if total < 120 else f"in {total / 60:.0f} min"
            else:
                when = f"at {args.get('at', '?')}"
            return f"Remind you {when}: {args.get('text', '')}"
        return f"Reminders: {args.get('action', '?')}"
    return t["label"] + extra
