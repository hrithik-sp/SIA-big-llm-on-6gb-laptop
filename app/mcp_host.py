"""Sia as an MCP host: tools from Model Context Protocol servers (https://modelcontextprotocol.io).

MCP is the common plug-in format for AI tools, so existing servers (Windows-MCP for operating apps,
the reference Memory server, and many more) work in Sia without new code. Servers are declared in
app/mcp.json. Like llama.cpp's own MCP support (tools/server/README.md, "MCP servers"), only the
stdio transport is used: each server is a child process exchanging one JSON-RPC message per line.

Sia keeps control of every call. A tool reaches the model only if mcp.json lists it, and its rule
there sets:
  policy   "auto" (no card: reads nothing private), "read" (card; may be always-allowed),
           "action" (card; may be always-allowed), "ask" (card every time, never always-allowed);
  fixed    arguments Sia sets itself (hidden from the model);
  allowed  the only values an argument may take (e.g. App modes without launch_executable);
  blocked  values refused before any card (e.g. the Win+R shortcut);
  label    the approval card's text, a template over the arguments ({target} names a UI element).
Tools a server offers but mcp.json doesn't list are never shown or callable.
"""
import json
import os
import re
import subprocess
import threading
from pathlib import Path

APP = Path(__file__).resolve().parent
LLM = Path(os.environ.get("LLM_HOME") or Path.home() / "llm")
PROTOCOL = "2025-06-18"
TIMEOUT_S = 60
MAX_TEXT = 24000


class Server:
    """One MCP server process, started on first use and restarted if it dies."""

    def __init__(self, name, spec, data_dir, on_spawn=None):
        self.name, self.spec, self.data_dir, self.on_spawn = name, spec, Path(data_dir), on_spawn
        self.proc, self.next_id, self.waiting = None, 1, {}
        self.lock = threading.Lock()

    def _expand(self, s):
        return str(s).replace("{LLM_HOME}", str(LLM)).replace("{DATA}", str(self.data_dir)).replace("{APP}", str(APP))

    def _start(self):
        env = os.environ.copy()
        env.update({k: self._expand(v) for k, v in (self.spec.get("env") or {}).items()})
        cmd = [self._expand(self.spec["command"])] + [self._expand(a) for a in self.spec.get("args", [])]
        self.data_dir.joinpath("logs").mkdir(parents=True, exist_ok=True)
        log = open(self.data_dir / "logs" / f"mcp-{self.name}.log", "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=env,
                                     cwd=self._expand(self.spec.get("cwd", "{DATA}")),
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if self.on_spawn:
            self.on_spawn(self.proc)
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        self._request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                     "clientInfo": {"name": "sia", "version": "1.2"}}, started=True)
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _send(self, msg):
        self.proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def _reader(self, proc):
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if "id" in msg and ("result" in msg or "error" in msg):
                slot = self.waiting.get(msg["id"])
                if slot:
                    slot["msg"] = msg
                    slot["event"].set()
            elif "id" in msg and "method" in msg:  # a request from the server: ping, or features Sia lacks
                reply = {"jsonrpc": "2.0", "id": msg["id"]}
                if msg["method"] == "ping":
                    reply["result"] = {}
                else:
                    reply["error"] = {"code": -32601, "message": "not supported by Sia"}
                try:
                    self._send(reply)
                except OSError:
                    break
        for slot in list(self.waiting.values()):  # the process ended: fail whatever was waiting
            slot["msg"] = {"error": {"message": f"the {self.name} plug-in stopped"}}
            slot["event"].set()

    def _request(self, method, params, started=False, timeout=TIMEOUT_S):
        if not started and (self.proc is None or self.proc.poll() is not None):
            self._start()
        rid = self.next_id
        self.next_id += 1
        slot = {"event": threading.Event(), "msg": None}
        self.waiting[rid] = slot
        try:
            self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
            if not slot["event"].wait(timeout):
                self.kill()  # a hung call (e.g. a frozen app blocking the accessibility tree): start fresh next time
                raise TimeoutError(f"the {self.name} plug-in didn't answer within {timeout} s (an app may be frozen)")
        finally:
            self.waiting.pop(rid, None)
        if "error" in slot["msg"]:
            raise RuntimeError(slot["msg"]["error"].get("message", "error"))
        return slot["msg"]["result"]

    def list_tools(self):
        with self.lock:
            tools, cursor = [], None
            while True:
                r = self._request("tools/list", {"cursor": cursor} if cursor else {})
                tools += r.get("tools", [])
                cursor = r.get("nextCursor")
                if not cursor:
                    return tools

    def call(self, tool, args, timeout=TIMEOUT_S):
        with self.lock:
            r = self._request("tools/call", {"name": tool, "arguments": args or {}}, timeout=timeout)
        texts = [c.get("text", "") for c in r.get("content", []) if c.get("type") == "text"]
        skipped = [c.get("type") for c in r.get("content", []) if c.get("type") != "text"]
        out = {"text": "\n".join(texts)[:MAX_TEXT]}
        if r.get("structuredContent") is not None and not texts:
            out["data"] = r["structuredContent"]
        if skipped:
            out["note"] = f"{len(skipped)} non-text part(s) ({', '.join(sorted(set(skipped)))}) not shown"
        if r.get("isError"):
            return {"error": out["text"] or "the tool failed"}
        return out

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(3)
            except (OSError, subprocess.TimeoutExpired):
                self.kill()

    def kill(self):
        """Kill the server and its children (a venv's python.exe is a launcher that starts the real one)."""
        if self.proc and self.proc.poll() is None:
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], capture_output=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


class Host:
    """All configured servers and their allowed tools, named <server>_<tool> like llama.cpp does."""

    def __init__(self, config_path, data_dir, on_spawn=None):
        self.config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.data_dir = Path(data_dir)
        self.servers, self.tools, self.errors, self.last_snapshot = {}, {}, {}, ""
        for name, spec in self.config.get("servers", {}).items():
            if spec.get("enabled", True):
                self.servers[name] = Server(name, spec, data_dir, on_spawn)

    def _cache_file(self):
        return self.data_dir / "mcp_tools.json"

    def load(self, use_cache=True):
        """Read each server's tool list (what the model sees). The list is cached with the config it came
        from, so later starts don't launch the servers until a tool is actually used."""
        key = json.dumps(self.config, sort_keys=True)
        try:
            cached = json.loads(self._cache_file().read_text(encoding="utf-8")) if use_cache else None
        except (OSError, ValueError):
            cached = None
        offered = dict(cached["offered"]) if cached and cached.get("config") == key else {}
        for name, srv in self.servers.items():
            if name in offered:
                continue
            try:
                offered[name] = srv.list_tools()
            except Exception as e:  # a missing or broken plug-in must not stop Sia
                self.errors[name] = f"{type(e).__name__}: {e}"
            finally:
                if srv.spec.get("stop_after_listing", True):
                    srv.stop()  # started again on its first real call
        if not self.errors:
            try:
                self._cache_file().write_text(json.dumps({"config": key, "offered": offered}), encoding="utf-8")
            except OSError:
                pass
        for name, tools in offered.items():
            if name not in self.servers:
                continue
            allowed = self.servers[name].spec.get("tools", {})
            for t in tools:
                rule = allowed.get(t["name"])
                if not rule or rule.get("policy") == "off":
                    continue  # not listed in mcp.json: never shown to the model, never callable
                schema = json.loads(json.dumps(t.get("inputSchema") or {"type": "object", "properties": {}}))
                props = schema.setdefault("properties", {})
                for k in list(rule.get("fixed", {})) + rule.get("hide", []):  # set by Sia, not by the model
                    props.pop(k, None)
                for k, values in rule.get("allowed", {}).items():
                    if k in props:
                        props[k] = {"type": "string", "enum": values, "default": values[0]}
                if "required" in schema:
                    schema["required"] = [r for r in schema["required"] if r in props]
                for k in list(props):
                    props[k] = _lean(props[k])
                self.tools[f"{name}_{t['name']}"] = {
                    "server": name, "tool": t["name"], "policy": rule.get("policy", "ask"), "rule": rule,
                    "description": rule.get("description") or (t.get("description") or "")[:1200],
                    "schema": schema}
        return self.tools

    def check(self, full_name, args):
        """An error if Sia refuses these arguments (before any approval card is shown), else None."""
        t = self.tools.get(full_name)
        if not t:
            return f"unknown or disallowed tool {full_name}"
        rule = t["rule"]
        for k, values in rule.get("allowed", {}).items():
            if k in args and args[k] is not None and str(args[k]) not in values:
                return f"{k}={args[k]!r} isn't allowed in Sia (allowed: {', '.join(values)})"
        for k, values in rule.get("blocked", {}).items():
            v = str(args.get(k, "")).lower().replace(" ", "")
            if v and v in values:
                return f"{k}={args[k]!r} is blocked in Sia for safety"
        for k, pattern in rule.get("blocked_regex", {}).items():
            if args.get(k) is not None and re.search(pattern, str(args[k]), re.I):
                return f"{k}={args[k]!r} is blocked in Sia for safety"
        return None

    def element(self, label):
        """The line of the last screen snapshot that describes a UI element id, for readable cards."""
        for line in self.last_snapshot.splitlines():
            cells = [c.strip() for c in line.replace("|", "\t").split("\t") if c.strip()]
            if cells and cells[0] == str(label):
                return " ".join(cells[1:4])[:80]
        return None

    def describe(self, full_name, args):
        t = self.tools.get(full_name)
        if not t:
            return f"Unknown tool {full_name}"
        values = {k: _flat(v) for k, v in (args or {}).items()}
        if values.get("label") is not None:
            found = self.element(values["label"])
            values["target"] = f"'{found}'" if found else f"element {values['label']}"
        elif values.get("loc"):
            values["target"] = f"the point ({values['loc']})"
        else:
            values["target"] = "the current position"
        template = t["rule"].get("label", t["tool"])
        try:
            text = template.format_map(_Blank(values))
        except (ValueError, IndexError, KeyError):
            text = template
        return " ".join(text.split())[:200]

    def call(self, full_name, args):
        t = self.tools.get(full_name)
        if not t:
            return {"error": f"unknown or disallowed tool {full_name}"}
        refused = self.check(full_name, args or {})
        if refused:
            return {"error": refused}
        args = {k: v for k, v in (args or {}).items() if v is not None}
        args.update(t["rule"].get("fixed", {}))
        try:
            out = self.servers[t["server"]].call(t["tool"], args, timeout=t["rule"].get("timeout_s", TIMEOUT_S))
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        if t["rule"].get("is_snapshot") and "text" in out:
            text = out["text"]
            try:  # Windows-MCP returns its text parts as a JSON list
                text = "\n".join(json.loads(text))
            except (ValueError, TypeError):
                pass
            out["text"] = self.last_snapshot = text[:MAX_TEXT]
        return out

    def stop(self):
        for srv in self.servers.values():
            srv.stop()


def _lean(prop):
    """A shorter schema for the same argument (every tool definition is read on every request): "X or null"
    becomes X, the null default and the "title" go; the argument stays optional."""
    if not isinstance(prop, dict):
        return prop
    p = {k: v for k, v in prop.items() if k != "title" and not (k == "default" and v is None)}
    options = [o for o in p.get("anyOf", []) if o.get("type") != "null"]
    if "anyOf" in p and options:
        p.pop("anyOf")
        p.update(_lean(options[0]))  # the first non-null form (e.g. the [x, y] list for a location)
    if isinstance(p.get("items"), dict):
        p["items"] = _lean(p["items"])
    if isinstance(p.get("properties"), dict):
        p["properties"] = {k: _lean(v) for k, v in p["properties"].items()}
    return p


def _flat(v):
    """Arguments as card text: [{"entityName": "Hrithik", "contents": ["likes tea"]}] -> "Hrithik: likes tea"."""
    if isinstance(v, list):
        return "; ".join(_flat(x) for x in v)
    if isinstance(v, dict):
        head = v.get("name") or v.get("entityName") or v.get("from") or ""
        rest = [_flat(x) for k, x in v.items() if k not in ("name", "entityName", "from", "entityType") and x]
        return (f"{head}: " if head and rest else head) + ", ".join(rest)
    return str(v)


class _Blank(dict):
    def __missing__(self, key):
        return ""
