"""A minimal MCP server over stdio for tests: one JSON-RPC message per line, three tools."""
import json
import sys

TOOLS = [
    {"name": "Echo", "description": "Echo text back.", "inputSchema": {
        "type": "object", "properties": {"text": {"type": "string"}, "secret_flag": {"type": "boolean"}},
        "required": ["text"]}},
    {"name": "Press", "description": "Press keys.", "inputSchema": {
        "type": "object", "properties": {"shortcut": {"type": "string"}, "mode": {"type": "string"}}}},
    {"name": "Hidden", "description": "Not listed in the config.", "inputSchema": {"type": "object", "properties": {}}},
]

for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue
    method, params = msg.get("method"), msg.get("params") or {}
    if method == "initialize":
        result = {"protocolVersion": params.get("protocolVersion"), "capabilities": {"tools": {}},
                  "serverInfo": {"name": "fake", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        result = {"content": [{"type": "text", "text": json.dumps({"called": params["name"],
                                                                    "args": params.get("arguments", {})})}]}
    else:
        print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": "no"}}), flush=True)
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}), flush=True)
