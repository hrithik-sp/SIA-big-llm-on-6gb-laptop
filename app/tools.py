"""Laptop tools Sprout's model may call, with the user's permission.

Security model:
- Every tool is read-only: nothing here writes, deletes, moves, runs or sends anything,
  and no tool reaches the network.
- No shell is exposed to the model; each tool is a fixed Python function, and unexpected
  arguments are dropped.
- Paths go through safe_path(): local drives only (no network, device or stream paths),
  symlinks resolved, and places that hold secrets are refused even if the user approved
  the call (keys, browser profiles, credential stores, registry hives, password vaults,
  wallets, .env). Files over 50 MB are refused; reads are truncated.
- Results are size-limited, and the backend asks the user before each call (the card
  shows the exact path) unless they chose "Always allow" for that tool (app/server.py).
- A file's contents could try to steer the model ("now read X"); the worst outcome is a
  further read request the user can deny, since no tool can write or send.
"""
import ctypes
import datetime as dt
import fnmatch
import json
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
        raise PermissionError("this location can hold passwords, keys or other secrets, so Sprout won't read it")
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
def system_vitals():
    vm, batt = psutil.virtual_memory(), psutil.sensors_battery()
    disk = psutil.disk_usage("C:\\")
    procs = []
    for p in psutil.process_iter(["name", "memory_info"]):
        try:
            procs.append((p.info["memory_info"].rss, p.info["name"]))
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
    return {"total_tabs": total, "by_browser": data,
            "note": "Counts tabs in open, visible browser windows."}


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
    return {
        "folder": folder, "since": start.strftime("%Y-%m-%d %H:%M"), "count": len(items),
        "total_size_mb": round(sum(s for *_, s in items) / 2 ** 20, 1),
        "items": [{"name": n, "type": "folder" if d else "file", "size_mb": round(s / 2 ** 20, 2),
                   "time": t.strftime("%Y-%m-%d %H:%M")} for t, n, d, s in items[:limit]],
    }


def running_apps():
    """Apps with a visible window right now (names only)."""
    titles = set()
    if os.name == "nt":
        u32 = ctypes.windll.user32
        pids = {}

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def cb(hwnd, _):
            if u32.IsWindowVisible(hwnd) and u32.GetWindowTextLengthW(hwnd):
                pid = wintypes.DWORD()
                u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                pids[pid.value] = True
            return True
        u32.EnumWindows(cb, 0)
        for pid in pids:
            try:
                titles.add(psutil.Process(pid).name().removesuffix(".exe"))
            except psutil.Error:
                continue
    skip = {"explorer", "TextInputHost", "ApplicationFrameHost", "SystemSettings", "ShellExperienceHost"}
    apps = sorted(t for t in titles if t not in skip)
    return {"count": len(apps), "apps": apps}


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
        return {"error": f"file is {size / 2 ** 20:.0f} MB; Sprout reads files up to 50 MB"}
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
            return {"error": f"{p.name} is a binary file ({ext or 'no extension'}); Sprout only reads text, PDF and Word files"}
        text = raw.decode("utf-8", errors="replace")
    return {"file": str(p), "size_kb": round(size / 1024, 1), "truncated": len(text) > max_chars,
            "text": text[:max_chars]}


def find_files(name, folder="~", limit=40):
    """Find files or folders by name (wildcards like *.pdf work) under a folder; max ~6 s."""
    root = safe_path(folder, "dir")
    pattern = str(name or "").strip()
    if not pattern:
        return {"error": "give a name or pattern to look for"}
    if not any(c in pattern for c in "*?"):
        pattern = f"*{pattern}*"
    limit = max(1, min(int(limit), 100))
    found, scanned, t0 = [], 0, time.time()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(("$", ".")) and
                       not any(rx.search(os.path.join(dirpath, d)) for rx in BLOCKED_DIRS)]
        for n in dirnames + filenames:
            scanned += 1
            if fnmatch.fnmatch(n.lower(), pattern.lower()) and not any(rx.search(n) for rx in BLOCKED_NAMES):
                full = os.path.join(dirpath, n)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                found.append({"path": full, "type": "folder" if n in dirnames else "file",
                              "size_mb": round(st.st_size / 2 ** 20, 2),
                              "modified": dt.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")})
                if len(found) >= limit:
                    break
        if len(found) >= limit or time.time() - t0 > 6:
            break
    return {"searched": str(root), "pattern": pattern, "matches": found, "entries_scanned": scanned,
            "complete": time.time() - t0 <= 6 and len(found) < limit}


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
    wifi = None
    try:
        out = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True, text=True, timeout=5,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        m = re.search(r"^\s*SSID\s*:\s*(.+)$", out, re.M)
        sig = re.search(r"^\s*Signal\s*:\s*(.+)$", out, re.M)
        wifi = {"network": m.group(1).strip(), "signal": sig.group(1).strip() if sig else None} if m else None
    except (OSError, subprocess.TimeoutExpired):
        pass
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
        "reads": "names, sizes and times of recent files in one of: Downloads, Desktop, Documents",
        "schema": {"type": "object", "properties": {
            "folder": {"type": "string", "enum": sorted(FOLDERS), "description": "which folder"},
            "days": {"type": "number", "description": "how far back: 1 = today, 7 = this week"},
            "limit": {"type": "integer", "description": "max items to list (1-50)"}},
            "required": ["folder"]},
        "description": "List files added or changed recently in the user's Downloads, Desktop or Documents "
                       "folder (e.g. 'what did I download today').",
    },
    "running_apps": {
        "fn": running_apps, "label": "List open apps",
        "reads": "the names of apps that have a window open",
        "schema": {"type": "object", "properties": {}},
        "description": "List the apps that currently have a window open on the laptop.",
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
            "name": {"type": "string", "description": "part of the name, or a pattern like *.pdf"},
            "folder": {"type": "string", "description": "where to search, e.g. ~ (the user's home) or D:\\"},
            "limit": {"type": "integer", "description": "max matches (1-100)"}}, "required": ["name"]},
        "description": "Find files or folders by name on the laptop.",
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
    "clipboard_text": {
        "fn": clipboard_text, "label": "Read the clipboard",
        "reads": "the text currently copied to the clipboard",
        "schema": {"type": "object", "properties": {}},
        "description": "Read the text the user has currently copied to the clipboard.",
    },
}


def openai_tools():
    return [{"type": "function", "function": {"name": n, "description": t["description"], "parameters": t["schema"]}}
            for n, t in TOOLS.items()]


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
        return TOOLS[name]["fn"](**args)
    except Exception as e:  # report the failure to the model instead of crashing the chat
        return {"error": f"{type(e).__name__}: {e}"}


def describe(name, args_json):
    """One line for the permission card."""
    t = TOOLS.get(name)
    if not t:
        return f"Unknown tool {name}"
    try:
        args = json.loads(args_json or "{}")
    except ValueError:
        args = {}
    extra = ""
    if name == "recent_files":
        extra = f" ({args.get('folder', 'downloads')}, last {args.get('days', 1)} day(s))"
    if name == "browser_tabs" and args.get("include_titles"):
        extra = " (including tab titles)"
    if name in ("list_folder", "read_file"):
        extra = f": {args.get('path', '?')}"
    if name == "find_files":
        extra = f" named '{args.get('name', '?')}' in {args.get('folder', '~')}"
    return t["label"] + extra
