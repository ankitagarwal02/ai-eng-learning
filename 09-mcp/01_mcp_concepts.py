"""
01_mcp_concepts.py — MCP protocol at a glance
==============================================

Illustrates the JSON-RPC message shapes that MCP hosts and servers exchange.
This is not a working server — see 02_mcp_server_basic.py for that. Use this
file to understand what's on the wire.
"""

import json


CAPABILITY_DISCOVERY_REQUEST = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "example-client", "version": "1.0.0"},
    },
}

CAPABILITY_DISCOVERY_RESPONSE = {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "protocolVersion": "2025-06-18",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"subscribe": True},
            "prompts": {},
        },
        "serverInfo": {"name": "company-utils", "version": "1.0.0"},
    },
}

LIST_TOOLS_REQUEST = {
    "jsonrpc": "2.0", "id": 2,
    "method": "tools/list",
}

LIST_TOOLS_RESPONSE = {
    "jsonrpc": "2.0", "id": 2,
    "result": {
        "tools": [
            {"name": "get_weather",
             "description": "Get current weather for a city",
             "inputSchema": {"type": "object",
                             "properties": {"city": {"type": "string"}},
                             "required": ["city"]}},
        ]
    },
}

CALL_TOOL_REQUEST = {
    "jsonrpc": "2.0", "id": 3,
    "method": "tools/call",
    "params": {"name": "get_weather", "arguments": {"city": "Paris"}},
}

CALL_TOOL_RESPONSE = {
    "jsonrpc": "2.0", "id": 3,
    "result": {"content": [{"type": "text", "text": "{\"city\":\"Paris\",\"temp_c\":22}"}]},
}


if __name__ == "__main__":
    print("MCP Protocol Reference — key JSON-RPC exchanges")
    print("=" * 78)

    for label, msg in [
        ("→ 1. Client initializes", CAPABILITY_DISCOVERY_REQUEST),
        ("← 2. Server capabilities", CAPABILITY_DISCOVERY_RESPONSE),
        ("→ 3. List tools",         LIST_TOOLS_REQUEST),
        ("← 4. Tools returned",     LIST_TOOLS_RESPONSE),
        ("→ 5. Call a tool",        CALL_TOOL_REQUEST),
        ("← 6. Tool result",        CALL_TOOL_RESPONSE),
    ]:
        print(f"\n{label}")
        print(json.dumps(msg, indent=2))

    print("""

✅ MCP concepts:

  • Wire format: JSON-RPC 2.0 over stdio or SSE
  • Server exposes: tools (functions), resources (data), prompts (templates)
  • Client discovers capabilities on connect, then calls tools/reads resources
  • Standardized envelope — the SAME server works with Claude Desktop,
    Cursor, VS Code Copilot, custom LangGraph agents, etc.

Next: 02_mcp_server_basic.py (working server with mcp SDK).
""")
