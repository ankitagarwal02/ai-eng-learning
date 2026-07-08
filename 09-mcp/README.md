# Phase 9 — Model Context Protocol (MCP)

> **Real-life analogy:** MCP is the USB-C of AI. Instead of building a custom integration for
> every AI app × every data source, MCP defines one universal plug. Any MCP-aware host
> (Claude Desktop, VS Code, Cursor, custom agents) can connect to any MCP server.

---

## The 3 primitives

```
┌──────────────────────────────────────────────────────────┐
│  RESOURCES   read-only data          e.g. company://policies/hr │
│  TOOLS       executable actions      e.g. get_weather(city)     │
│  PROMPTS     reusable prompt templates                          │
└──────────────────────────────────────────────────────────┘
```

## Architecture

```
   ┌──────────────┐   JSON-RPC   ┌──────────────┐
   │  MCP HOST    │◀────────────▶│  MCP SERVER  │
   │  (Claude,    │   (stdio     │  (yours)     │
   │   VS Code,   │   or SSE)    │              │
   │   agent)     │              │              │
   └──────────────┘              └──────────────┘
```

## Why MCP?

Before MCP, every AI app had to build custom connectors for every data source. MCP standardizes
the plug — write one MCP server for your internal API, and it works with Claude Desktop,
Cursor, VS Code Copilot, LangGraph, and every future MCP host.

## Folder structure

```
09-mcp/
├── README.md
├── 01_mcp_concepts.py         ← Message shapes, capability discovery
├── 02_mcp_server_basic.py     ← Working MCP server: tools + resources
├── 03_mcp_server_advanced.py  ← Prompts, auth, streaming, Pydantic schemas
├── 04_mcp_client.py           ← Programmatic MCP client
└── requirements.txt
```

## Quickstart

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather-server")

@mcp.tool()
def get_weather(city: str) -> dict:
    """Get current weather for a city."""
    return {"city": city, "temp_c": 22}

@mcp.resource("company://policies/hr")
def hr_policy() -> str:
    return "Full HR policy text..."

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

That's a complete MCP server. Test it with:
```
mcp dev 02_mcp_server_basic.py
```

Prereqs: Phase 8 (understand tools & agent loop).
Next: Phase 10 (Multi-agents — where MCP tools shine).
