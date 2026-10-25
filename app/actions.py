"""Actions Sprout's model may take on the laptop, each one approved by the user first.

Security model (in addition to the per-action approval card in app/server.py):
- A fixed set of actions with validated arguments; no shell, no free-form commands. Arguments reach
  PowerShell as plain parameters, never as code.
- Never: delete, move or overwrite files; install or uninstall; run scripts, installers or programs
  from files; force-kill processes; touch password or key locations.
- "Open" launches only apps registered in the Start menu, opens documents and folders with their
  default app (executable and script types are refused), and websites over http(s) only.
- New files are only written inside Documents\\Sprout, never over an existing file.
- Power actions (sleep, restart, shut down) are marked dangerous: they always need a fresh
  approval; restart and shut down wait 60 s and can be cancelled.
- Every action is appended to app-data/logs/actions.log.
"""
import ctypes
import datetime as dt
import json
import os
import re
import subprocess
import threading
import time
import uuid
from ctypes import wintypes
from pathlib import Path

import psutil

APP = Path(__file__).resolve().parent
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
# opening these runs code, so "open" refuses them (apps open through the Start menu instead)
RUNNABLE = {".exe", ".com", ".bat", ".cmd", ".ps1", ".psm1", ".vbs", ".vbe", ".js", ".jse", ".wsf", ".wsh", ".hta",
            ".msi", ".msp", ".msix", ".appx", ".appxbundle", ".scr", ".pif", ".cpl", ".reg", ".lnk", ".url", ".jar",
            ".py", ".pyw", ".application", ".appref-ms", ".gadget", ".inf", ".sys", ".dll", ".sh", ".settingcontent-ms"}
SETTINGS_PAGES = {  # Windows Settings pages that "open_settings" may show (ms-settings: URIs)
    "bluetooth": "bluetooth", "wifi": "network-wifi", "network": "network-status", "display": "display",
    "sound": "sound", "battery": "batterysaver", "power": "powersleep", "storage": "storagesense",
    "apps": "appsfeatures", "startup apps": "startupapps", "notifications": "notifications",
    "focus": "focus", "updates": "windowsupdate", "privacy": "privacy", "location": "privacy-location",
    "mouse": "mousetouchpad", "keyboard": "typing", "personalization": "personalization",
    "colors": "colors", "night light": "nightlight", "date and time": "dateandtime", "about": "about",
    "default apps": "defaultapps", "accounts": "yourinfo", "vpn": "network-vpn", "printers": "printers",
}


def _ps(action, value="", flag=0, timeout=40):
    out = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(APP / "actions.ps1"),
                          "-Action", action, "-Value", str(value), "-Flag", str(int(flag))],
                         capture_output=True, text=True, timeout=timeout, creationflags=NO_WINDOW).stdout.strip()
    return json.loads(out or "{}")


# --------------------------------------------------------------------------- open
_START_APPS = {"t": 0.0, "apps": []}


def _start_apps():
    if time.time() - _START_APPS["t"] > 300:  # Get-StartApps takes ~0.5 s; refresh every 5 min
        _START_APPS.update(t=time.time(), apps=_ps("start_apps").get("apps", []))
    return _START_APPS["apps"]


def _match_app(name):
    name = re.sub(r"\s+", " ", str(name or "").strip().lower())
    if not name:
        return None, []
    apps = _start_apps()
    exact = [a for a in apps if a["name"].lower() == name]
    starts = [a for a in apps if a["name"].lower().startswith(name)]
    words = [a for a in apps if all(w in a["name"].lower() for w in name.split())]
    for group in (exact, starts, words):
        if group:
            # skip uninstallers, help and readme entries that share the app's name
            good = [a for a in group if not re.search(r"uninstall|readme|help|manual|website", a["name"], re.I)]
            if good:
                return min(good, key=lambda a: len(a["name"])), [a["name"] for a in good[:5]]
    return None, []


def open_app(name):
    """Launch an app that is in the Start menu (by its name, e.g. Spotify, Notepad, VS Code)."""
    aliases = {"vs code": "visual studio code", "vscode": "visual studio code", "code": "visual studio code",
               "chrome": "google chrome", "edge": "microsoft edge", "word": "word", "excel": "excel",
               "settings": "settings", "file explorer": "file explorer", "explorer": "file explorer",
               "task manager": "task manager", "calc": "calculator", "cmd": None, "powershell": None, "terminal": None}
    key = str(name or "").strip().lower()
    if key in aliases and aliases[key] is None:
        return {"error": "Sprout doesn't open command shells (they could run anything)."}
    app, candidates = _match_app(aliases.get(key, key))
    if not app:
        return {"error": f"No app named '{name}' in the Start menu."}
    if re.search(r"powershell|command prompt|terminal|cmd\b", app["name"], re.I):
        return {"error": "Sprout doesn't open command shells (they could run anything)."}
    subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{app['id']}"], creationflags=NO_WINDOW)
    # report what really happened: wait briefly for the app's window (Store apps can start slowly)
    words = [w for w in re.findall(r"[a-z0-9]+", app["name"].lower()) if len(w) > 2][:2]
    seen = False
    for _ in range(24):
        time.sleep(0.25)
        if any(all(w in (pn + " " + t).lower() for w in words) for _, pn, t in _windows_of_any()):
            seen = True
            break
    return {"launched": app["name"], "window_seen": seen,
            "note": "" if seen else "launch requested; its window hasn't appeared yet (it may still be starting)",
            "other_matches": [c for c in candidates if c != app["name"]][:4]}


def _windows_of_any():
    """All visible titled top-level windows: [(hwnd, process name, title)]."""
    u32, out = ctypes.windll.user32, []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        n = u32.GetWindowTextLengthW(hwnd)
        if u32.IsWindowVisible(hwnd) and n:
            pid = wintypes.DWORD()
            u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            try:
                pname = psutil.Process(pid.value).name().lower()
            except psutil.Error:
                return True
            buf = ctypes.create_unicode_buffer(n + 1)
            u32.GetWindowTextW(hwnd, buf, n + 1)
            out.append((hwnd, pname, buf.value))
        return True
    u32.EnumWindows(cb, 0)
    return out


def open_file(path):
    """Open a document or folder with its default app (not programs or scripts)."""
    import tools  # safe_path: local paths only, secret locations refused
    p = tools.safe_path(path, "any")
    if p.is_file() and p.suffix.lower() in RUNNABLE:
        return {"error": f"{p.name} is a program or script; Sprout doesn't run files. Ask to open an app by name instead."}
    os.startfile(str(p))
    return {"opened": str(p), "kind": "folder" if p.is_dir() else "file"}


def open_website(url):
    url = str(url or "").strip()
    if not re.match(r"^https?://[^\s]+$", url, re.I):
        return {"error": "only http(s) web addresses can be opened"}
    os.startfile(url)
    return {"opened": url, "note": "opened in the default browser"}


def open_settings(page):
    key = str(page or "").strip().lower()
    if key not in SETTINGS_PAGES:
        return {"error": f"page must be one of {sorted(SETTINGS_PAGES)}"}
    os.startfile(f"ms-settings:{SETTINGS_PAGES[key]}")
    return {"opened": f"Settings > {key}"}


# --------------------------------------------------------------------------- media, sound, display, radios
def media(control):
    keys = {"play_pause": 0xB3, "next": 0xB0, "previous": 0xB1, "stop": 0xB2}
    vk = keys.get(str(control).lower())
    if vk is None:
        return {"error": f"control must be one of {sorted(keys)}"}
    u32 = ctypes.windll.user32
    u32.keybd_event(vk, 0, 0, 0)
    u32.keybd_event(vk, 0, 2, 0)  # KEYEVENTF_KEYUP
    return {"sent": control}


def set_volume(level=None, mute=None):
    if level is None and mute is None:
        return {"error": "give a level (0-100) or mute true/false"}
    value = "" if level is None else max(0, min(100, int(float(level))))
    flag = 0 if mute is None else (1 if mute else 2)
    return _ps("volume", value, flag)


def set_brightness(level):
    return _ps("brightness", max(0, min(100, int(float(level)))))


def set_radio(radio, on):
    radio = str(radio).lower().replace("-", "")
    if radio not in ("bluetooth", "wifi"):
        return {"error": "radio must be bluetooth or wifi"}
    return _ps("radio", radio, 1 if on else 0)


def set_theme(mode):
    import winreg
    mode = str(mode).lower()
    if mode not in ("dark", "light"):
        return {"error": "mode must be dark or light"}
    key = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key, 0, winreg.KEY_READ | winreg.KEY_SET_VALUE) as k:
        before = "light" if winreg.QueryValueEx(k, "AppsUseLightTheme")[0] else "dark"
        for name in ("AppsUseLightTheme", "SystemUsesLightTheme"):
            winreg.SetValueEx(k, name, 0, winreg.REG_DWORD, 1 if mode == "light" else 0)
    # tell Windows to redraw: both messages are needed, or the taskbar can keep the old colour
    # (it once stayed white after switching back to dark)
    res = wintypes.DWORD()
    for param in ("ImmersiveColorSet", "WindowsThemeElement"):
        ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x001A, 0, ctypes.c_wchar_p(param), 0x0002, 3000,
                                                 ctypes.byref(res))
    return {"before": before, "after": mode}


# --------------------------------------------------------------------------- windows and power
def close_app(name):
    """Ask an app's windows to close, like clicking X (the app may ask to save). Never force-kills."""
    target = str(name or "").strip().lower().removesuffix(".exe")
    if not target:
        return {"error": "say which app to close"}
    if target in ("sprout", "pythonw", "python", "llama-server", "explorer", "dwm", "winlogon", "csrss"):
        return {"error": f"Sprout won't close {name}."}
    u32 = ctypes.windll.user32
    closed = set()

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        if u32.IsWindowVisible(hwnd) and u32.GetWindowTextLengthW(hwnd):
            pid = wintypes.DWORD()
            u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            try:
                pname = psutil.Process(pid.value).name().lower().removesuffix(".exe")
            except psutil.Error:
                return True
            n = u32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(n + 1)
            u32.GetWindowTextW(hwnd, buf, n + 1)
            segments = [x.strip() for x in buf.value.lower().split(" - ")]  # "Downloads - File Explorer"
            if target == pname or target in segments or target == buf.value.lower():
                u32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
                closed.add(pname)
        return True
    u32.EnumWindows(cb, 0)
    if not closed:
        return {"error": f"No open window of '{name}' was found."}
    return {"asked_to_close": sorted(closed), "note": "Closed like clicking X; the app may ask to save first."}


def lock_screen():
    ctypes.windll.user32.LockWorkStation()
    return {"locked": True}


def power(mode):
    mode = str(mode).lower()
    if mode == "sleep":
        ctypes.windll.powrprof.SetSuspendState(0, 0, 0)
        return {"sleeping": True}
    if mode in ("restart", "shutdown"):
        subprocess.run(["shutdown", "/r" if mode == "restart" else "/s", "/t", "60",
                        "/c", f"Sprout: {mode} in 60 seconds (asked for in Sprout). Cancel: ask Sprout, or run shutdown /a."],
                       creationflags=NO_WINDOW, timeout=10)
        return {"scheduled": mode, "in_seconds": 60, "cancel": "ask Sprout to cancel the shutdown"}
    if mode == "cancel":
        r = subprocess.run(["shutdown", "/a"], creationflags=NO_WINDOW, capture_output=True, text=True, timeout=10)
        return {"cancelled": r.returncode == 0}
    return {"error": "mode must be sleep, restart, shutdown or cancel"}


# --------------------------------------------------------------------------- screenshot, clipboard, notes
def take_screenshot():
    from PIL import ImageGrab
    import tools
    folder = Path(tools.user_folders()["pictures"]) / "Screenshots"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"Sprout {dt.datetime.now():%Y-%m-%d %H.%M.%S}.png"
    ImageGrab.grab(all_screens=True).save(path)
    return {"saved": str(path)}


def copy_to_clipboard(text):
    text = str(text or "")
    if not text:
        return {"error": "nothing to copy"}
    try:  # remembered so "undo" can put the user's previous clipboard text back
        import tools
        CLIPBOARD_BEFORE["text"] = tools.clipboard_text().get("text", "")
    except Exception:  # noqa: BLE001 (clipboard busy or not text)
        CLIPBOARD_BEFORE["text"] = None
    u32, k32 = ctypes.windll.user32, ctypes.windll.kernel32
    k32.GlobalAlloc.restype = ctypes.c_void_p
    k32.GlobalLock.restype = ctypes.c_void_p
    k32.GlobalLock.argtypes = [ctypes.c_void_p]
    k32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    u32.SetClipboardData.argtypes = [wintypes.UINT, ctypes.c_void_p]
    data = (text + "\0").encode("utf-16-le")
    h = k32.GlobalAlloc(0x0002, len(data))  # GMEM_MOVEABLE
    ctypes.memmove(k32.GlobalLock(h), data, len(data))
    k32.GlobalUnlock(h)
    if not u32.OpenClipboard(None):
        return {"error": "the clipboard is busy; try again"}
    try:
        u32.EmptyClipboard()
        u32.SetClipboardData(13, h)  # CF_UNICODETEXT; the clipboard now owns the memory
    finally:
        u32.CloseClipboard()
    return {"copied_chars": len(text)}


def save_note(name, text):
    """Save text as a new file in Documents\\Sprout (never overwrites; text formats only)."""
    import tools
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", str(name or "").strip())[:80] or "note"
    stem, ext = os.path.splitext(name)
    if ext.lower() not in (".txt", ".md", ".csv", ".json"):
        stem, ext = name, ".txt"
    folder = Path(tools.user_folders()["documents"]) / "Sprout"
    folder.mkdir(parents=True, exist_ok=True)
    path, i = folder / f"{stem}{ext}", 1
    while path.exists():  # never overwrite
        i += 1
        path = folder / f"{stem} ({i}){ext}"
    path.write_text(str(text or ""), encoding="utf-8")
    return {"saved": str(path), "chars": len(str(text or ""))}


# --------------------------------------------------------------------------- reminders
REMINDERS = {"file": None, "items": [], "lock": threading.Lock()}


def _save_reminders():
    if REMINDERS["file"]:
        REMINDERS["file"].write_text(json.dumps(REMINDERS["items"], indent=1), encoding="utf-8")


def init_reminders(path):
    """Load saved reminders and start the clock that shows them as Windows notifications."""
    REMINDERS["file"] = Path(path)
    try:
        REMINDERS["items"] = json.loads(REMINDERS["file"].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        REMINDERS["items"] = []

    def loop():
        while True:
            now = time.time()
            due = []
            with REMINDERS["lock"]:
                for r in REMINDERS["items"]:
                    if not r.get("done") and r["at"] <= now:
                        r["done"] = True
                        due.append(r)
                if due:
                    _save_reminders()
            for r in due:
                try:
                    _ps("toast", r["text"])
                except (OSError, subprocess.SubprocessError, ValueError):
                    pass
            time.sleep(1)  # timers can be seconds long
    threading.Thread(target=loop, daemon=True).start()


def set_reminder(text, minutes=None, at=None, seconds=None):
    text = str(text or "").strip()[:200] or "Timer"
    now = dt.datetime.now()
    if seconds is not None or minutes is not None:  # a timer: "in 10 seconds", "in 25 minutes"
        # seconds and minutes are alternatives: the model once sent minutes=1 with seconds=30 for "30 seconds"
        total = float(seconds) if seconds is not None else 60 * float(minutes)
        when = now + dt.timedelta(seconds=max(5.0, min(total, 7 * 24 * 3600)))
    elif at:
        m = re.match(r"^\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*$", str(at), re.I)
        if not m:
            return {"error": "give the time like 17:30 or 5:30 pm"}
        h, mnt, ap = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").lower()
        h = h % 12 + (12 if ap == "pm" else 0) if ap else h
        if not (0 <= h < 24 and 0 <= mnt < 60):
            return {"error": "that isn't a valid time"}
        when = now.replace(hour=h, minute=mnt, second=0, microsecond=0)
        if when <= now:
            when += dt.timedelta(days=1)
    else:
        return {"error": "give minutes from now or a time"}
    item = {"id": uuid.uuid4().hex[:8], "text": text, "at": when.timestamp(), "done": False}
    with REMINDERS["lock"]:
        REMINDERS["items"].append(item)
        _save_reminders()
    secs = (when - now).total_seconds()
    fmt = "%A %d %B, %I:%M:%S %p" if secs < 120 else "%A %d %B, %I:%M %p"  # seconds only for short timers
    return {"reminder": text, "at": when.strftime(fmt).replace(" 0", " "),
            "in": f"{secs:.0f} seconds" if secs < 120 else f"{secs / 60:.0f} minutes",
            "note": "shown as a Windows notification with a sound while Sprout is running"}


def list_reminders():
    with REMINDERS["lock"]:
        items = [r for r in REMINDERS["items"] if not r.get("done")]
    return {"reminders": [{"id": r["id"], "text": r["text"],
                           "at": dt.datetime.fromtimestamp(r["at"]).strftime("%a %d %b, %I:%M %p")} for r in items]}


def _find_reminders(id="", text=""):
    """The pending reminders meant: all of them ("clear all my reminders"), one by id, by words of its text
    ("the oven one"; the model has passed the name as the id), or else the most recent one."""
    pending = [r for r in REMINDERS["items"] if not r.get("done")]
    if {str(text).lower().strip(), str(id).lower().strip()} & {"all", "every", "everything", "all of them"}:
        return pending, pending
    pick = [r for r in pending if id and r["id"] == str(id)]
    for words_from in (text, id):
        if not pick and words_from:
            words = [w for w in re.findall(r"[a-z0-9]+", str(words_from).lower())
                     if w not in ("the", "my", "a", "reminder", "timer", "alarm", "to", "for")]
            pick = [r for r in pending if words and all(w in r["text"].lower() for w in words)]
    if not pick and not id and not text:
        pick = pending[-1:]
    return pick[-1:], pending


def cancel_reminder(id="", text=""):
    with REMINDERS["lock"]:
        picks, pending = _find_reminders(id, text)
        if not picks:
            return {"error": "no matching pending reminder", "pending": [x["text"] for x in pending]}
        for r in picks:
            r["done"] = True
        _save_reminders()
        return {"cancelled": picks[0]["text"] if len(picks) == 1 else [r["text"] for r in picks]}


# --------------------------------------------------------------------------- grouped entry points (the model's tools)
# Few tools with clear choices keep the prompt small (every tool's description is read each time)
# and make it easy for the model to pick the right one.
def _windows_of(target):
    """Visible top-level windows whose process name or title matches target: [(hwnd, process name, title)]."""
    u32, found = ctypes.windll.user32, []
    target = target.lower().removesuffix(".exe")

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        n = u32.GetWindowTextLengthW(hwnd)
        if u32.IsWindowVisible(hwnd) and n:
            pid = wintypes.DWORD()
            u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            try:
                pname = psutil.Process(pid.value).name().lower().removesuffix(".exe")
            except psutil.Error:
                return True
            buf = ctypes.create_unicode_buffer(n + 1)
            u32.GetWindowTextW(hwnd, buf, n + 1)
            title = buf.value
            segments = [x.strip() for x in title.lower().split(" - ")]
            if target == pname or target in segments or (len(target) >= 4 and target in pname):
                found.append((hwnd, pname, title))
        return True
    u32.EnumWindows(cb, 0)
    return found


def window(action, app):
    action, app = str(action).lower(), str(app or "").strip()
    if action == "close":
        return close_app(app)
    wins = _windows_of(app)
    if not wins:
        return {"error": f"No open window of '{app}' was found."}
    u32 = ctypes.windll.user32
    hwnd, pname, title = wins[0]
    if action == "focus":
        u32.ShowWindow(hwnd, 9)  # SW_RESTORE
        u32.SetForegroundWindow(hwnd)
    elif action == "minimize":
        u32.ShowWindow(hwnd, 6)
    elif action == "maximize":
        u32.ShowWindow(hwnd, 3)
    else:
        return {"error": "action must be focus, minimize, maximize or close"}
    return {"done": action, "window": title[:120]}


def open_(what, target):
    what = str(what).lower()
    fn = {"app": open_app, "file": open_file, "folder": open_file, "website": open_website,
          "settings": open_settings}.get(what)
    if not fn:
        return {"error": "what must be app, file, folder, website or settings"}
    return fn(target)


def control(setting, value=""):
    setting, value = str(setting).lower(), str(value or "").strip().lower()
    on = value in ("on", "true", "1", "yes", "enable", "enabled")
    if setting == "volume":
        if not re.match(r"^\d{1,3}$", value):
            return {"error": "volume needs a number 0-100"}
        return set_volume(level=int(value))
    if setting == "mute":
        return set_volume(mute=on)
    if setting == "brightness":
        if not re.match(r"^\d{1,3}$", value):
            return {"error": "brightness needs a number 0-100"}
        return set_brightness(int(value))
    if setting in ("bluetooth", "wifi"):
        return set_radio(setting, on)
    if setting == "theme":
        return set_theme(value)
    if setting == "media":
        keys = {"play": "play_pause", "pause": "play_pause", "play pause": "play_pause", "play_pause": "play_pause",
                "resume": "play_pause", "next": "next", "skip": "next", "previous": "previous", "prev": "previous",
                "back": "previous", "stop": "stop"}
        return media(keys.get(value, value))
    return {"error": "setting must be volume, mute, brightness, bluetooth, wifi, theme or media"}


def power_(mode):
    return lock_screen() if str(mode).lower() == "lock" else power(mode)


USER_TEXT = {"text": None}  # the user's newest message, set by the server
UNIT_SECONDS = {"s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1, "m": 60, "min": 60, "mins": 60,
                "minute": 60, "minutes": 60, "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600}


def duration_in(text):
    """Seconds stated in the user's words ("10 seconds", "1 minute 30 seconds", "half an hour"), or None."""
    t = str(text or "").lower()
    total, found = 0.0, False
    words = r"seconds?|secs?|minutes?|mins?|hours?|hrs?"
    pairs = re.findall(rf"\b(\d+(?:\.\d+)?)\s*({words})\b", t) + re.findall(r"\b(\d+(?:\.\d+)?)([smh])\b", t)
    pairs += re.findall(rf"\b(half an?|an?)\s+({words})\b", t)  # "an hour", "half an hour"; never "6 am"
    for num, unit in pairs:
        n = 0.5 if num.startswith("half") else 1.0 if num in ("a", "an") else float(num)
        total += n * UNIT_SECONDS[unit]
        found = True
    return total if found else None


def fix_reminder_args(args):
    """The model has turned "30 seconds" into 30 minutes, and "10 seconds" into 10 minutes: when the user's
    message states a duration, that duration is used (and shown on the approval card)."""
    if str(args.get("action", "")).lower() != "set" or args.get("at"):
        return args
    secs = duration_in(USER_TEXT["text"])
    if secs:
        args = {k: v for k, v in args.items() if k not in ("minutes", "seconds")}
        args["seconds"] = secs
    return args


def reminder(action, text="", minutes=None, at=None, id="", seconds=None):
    action = str(action).lower()
    if action == "set":
        return set_reminder(text, minutes, at, seconds)
    if action == "list":
        return list_reminders()
    if action == "cancel":
        return cancel_reminder(id, text)
    return {"error": "action must be set, list or cancel"}


def is_dangerous(name, args):
    """Actions that always need a fresh approval (never 'Always allow')."""
    return name == "power" and str(args.get("mode", "")).lower() in ("sleep", "restart", "shutdown")



# --------------------------------------------------------------------------- undo
CLIPBOARD_BEFORE = {"text": None}
HISTORY = []  # this session's actions, newest last: what was done and how to put it back


def record(name, args, result):
    """Remember an action and how to undo it (called after every successful action)."""
    undo, desc = None, None
    before = result.get("before") if isinstance(result, dict) else None
    if name == "control":
        s = str(args.get("setting", "")).lower()
        if s in ("volume", "mute") and isinstance(before, dict):
            undo = ("set_volume", {"level": before["volume_pct"], "mute": before["muted"]})
            desc = f"volume back to {before['volume_pct']}%" + (" (muted)" if before["muted"] else "")
        elif s == "brightness" and before is not None:
            undo, desc = ("set_brightness", {"level": before}), f"brightness back to {before}%"
        elif s in ("bluetooth", "wifi") and before:
            undo = ("set_radio", {"radio": s, "on": str(before).lower() == "on"})
            desc = f"{'Bluetooth' if s == 'bluetooth' else 'Wi-Fi'} back {str(before).lower()}"
        elif s == "theme" and before:
            undo, desc = ("set_theme", {"mode": before}), f"{before} theme back"
        elif s == "media" and result.get("sent") == "play_pause":
            undo, desc = ("media", {"control": "play_pause"}), "play/pause toggled back"
    elif name == "open":
        what = str(args.get("what", "")).lower()
        if what == "app" and result.get("launched"):
            undo, desc = ("close_app", {"name": result["launched"]}), f"close {result['launched']}"
        elif what in ("file", "folder") and result.get("opened"):
            label = Path(result["opened"]).name
            undo, desc = ("close_app", {"name": label}), f"close {label}"
        elif what == "website":
            desc = "can't close a browser tab; close it in the browser"
    elif name == "window":
        a, app = str(args.get("action", "")).lower(), str(args.get("app", ""))
        if a in ("minimize", "maximize"):
            undo, desc = ("restore_window", {"app": app}), f"restore {app}'s window"
    elif name == "power" and str(args.get("mode", "")).lower() in ("restart", "shutdown"):
        undo, desc = ("power", {"mode": "cancel"}), f"cancel the {args['mode']}"
    elif name == "reminder" and str(args.get("action", "")).lower() == "set" and REMINDERS["items"]:
        rid = REMINDERS["items"][-1]["id"]
        undo, desc = ("cancel_reminder", {"id": rid}), f"cancel the reminder '{REMINDERS['items'][-1]['text']}'"
    elif name == "copy" and CLIPBOARD_BEFORE["text"] is not None:
        undo, desc = ("copy_to_clipboard", {"text": CLIPBOARD_BEFORE["text"] or " "}), "put the previous clipboard text back"
    elif name in ("screenshot", "save_note"):
        desc = f"Sprout doesn't delete files; it's at {result.get('saved', '?')}"
    HISTORY.append({"n": len(HISTORY) + 1, "tool": name, "args": args, "undo": undo, "desc": desc,
                    "done": False, "label": f"{name} {json.dumps(args)[:120]}"})
    del HISTORY[:-50]


def restore_window(app):
    wins = _windows_of(app)
    if not wins:
        return {"error": f"No open window of '{app}' was found."}
    ctypes.windll.user32.ShowWindow(wins[0][0], 9)  # SW_RESTORE
    return {"restored": wins[0][2][:120]}


def _undo_candidate(what="last"):
    what = str(what or "last").lower().strip()
    pool = [h for h in reversed(HISTORY) if not h["done"]]
    if what not in ("", "last", "that", "it", "everything"):
        same = {"timer": "reminder", "alarm": "reminder", "sound": "volume", "music": "media", "song": "media",
                "screen": "brightness", "dark": "theme", "light": "theme", "mode": "theme"}
        filler = {"the", "my", "back", "undo", "that", "it", "please", "cancel", "put", "change", "set", "turn",
                  "revert", "last", "a", "to", "you", "did", "what", "just"}
        words = [same.get(w, w) for w in re.findall(r"[a-z0-9]+", what) if w not in filler]
        pool = [h for h in pool if all(w in (h["label"] + " " + (h["desc"] or "")).lower() for w in words)]
    return pool[0] if pool else None


def undo(what="last"):
    """Undo one earlier action of this session: the last one, or the latest that matches (e.g. "volume")."""
    h = _undo_candidate(what)
    if not h:
        return {"error": "nothing Sprout did in this session matches that"}
    if not h["undo"]:
        return {"error": f"that can't be undone: {h['desc'] or 'no way back for this action'}"}
    fn, kw = h["undo"]
    result = globals()[fn](**kw)
    if not (isinstance(result, dict) and "error" in result):
        h["done"] = True
    return {"undid": h["desc"], "result": result}


def undo_description(what="last"):
    h = _undo_candidate(what)
    return f"Undo: {h['desc']}" if h and h["undo"] else "Undo an earlier action"
