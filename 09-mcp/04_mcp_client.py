"""
04_mcp_client.py — Programmatic MCP client
============================================

WHAT THIS FILE TEACHES
----------------------
  • Spawn a stdio MCP server as a subprocess
  • Discover its capabilities (tools, resources, prompts)
  • Call a tool with arguments, handle the response
  • Read a resource
  • Retrieve a prompt template
  • Integrate MCP tool calls with an OpenAI-style agent loop
  • Handle errors gracefully

HOW TO RUN
----------
    pip install mcp
    python 04_mcp_client.py

The client spawns 02_mcp_server_basic.py (from this folder) and drives it.

REAL-WORLD SCENARIO
-------------------
You're building a custom AI application (not Claude Desktop). Your app needs
to leverage tools defined in an MCP server. This file shows how — the same
pattern any host uses to consume MCP.
"""

import os
import asyncio
import json
from pathlib import Path

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    _has_mcp = True
except ImportError:
    _has_mcp = False


HERE = Path(__file__).parent
SERVER_SCRIPT = HERE / "02_mcp_server_basic.py"


async def demo_discovery():
    """Connect to server, list its capabilities."""
    server_params = StdioServerParameters(command="python", args=[str(SERVER_SCRIPT)])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("─── Server capabilities ──────────────────")

            tools = await session.list_tools()
            print(f"\nTools ({len(tools.tools)}):")
            for t in tools.tools:
                print(f"  • {t.name}  — {t.description}")

            resources = await session.list_resources()
            print(f"\nResources ({len(resources.resources)}):")
            for r in resources.resources:
                print(f"  • {r.uri}  — {r.name if r.name else '(unnamed)'}")

            try:
                prompts = await session.list_prompts()
                print(f"\nPrompts ({len(prompts.prompts)}):")
                for p in prompts.prompts:
                    print(f"  • {p.name}  — {p.description}")
            except Exception:
                print("\n(Server doesn't expose prompts)")


async def demo_call_tool():
    """Call a specific tool."""
    server_params = StdioServerParameters(command="python", args=[str(SERVER_SCRIPT)])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("\n─── Calling get_weather('Paris') ─────────")
            result = await session.call_tool("get_weather", {"city": "Paris"})
            for block in result.content:
                if hasattr(block, "text"):
                    print(f"  ← {block.text}")

            print("\n─── Calling search_docs ─────────────────")
            result = await session.call_tool("search_docs", {"query": "expense", "n": 2})
            for block in result.content:
                if hasattr(block, "text"):
                    print(f"  ← {block.text}")


async def demo_read_resource():
    server_params = StdioServerParameters(command="python", args=[str(SERVER_SCRIPT)])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("\n─── Reading resource company://policies/hr ──")
            result = await session.read_resource("company://policies/hr")
            for content in result.contents:
                if hasattr(content, "text"):
                    print(content.text[:200] + "..." if len(content.text) > 200 else content.text)


async def demo_agent_loop_with_mcp():
    """Show how an agent loop uses MCP tools.

    Pattern:
      1. Client discovers tools
      2. Convert MCP tool schemas → OpenAI function schemas
      3. LLM chooses which to call
      4. Client invokes the MCP tool
      5. Result fed back to LLM
    """
    server_params = StdioServerParameters(command="python", args=[str(SERVER_SCRIPT)])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools

            # Convert MCP tools → OpenAI function schema
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.inputSchema or {"type": "object", "properties": {}},
                    },
                }
                for t in mcp_tools
            ]

            print("\n─── Simulated agent loop (MOCK_MODE) ────")
            print(f"LLM sees {len(openai_tools)} tools:")
            for t in openai_tools:
                print(f"  • {t['function']['name']}({list((t['function']['parameters'].get('properties') or {}).keys())})")

            # In a real agent, you'd now call OpenAI with these `tools` and let the
            # model decide. When it responds with a tool_call, invoke it via
            # session.call_tool(name, args). The result goes back into messages.

            # Simulate that path:
            print("\n[LLM decides to call get_weather with city='Tokyo']")
            r = await session.call_tool("get_weather", {"city": "Tokyo"})
            for block in r.content:
                if hasattr(block, "text"):
                    print(f"[tool result → back to LLM] {block.text}")


async def main():
    if not _has_mcp:
        print("Install MCP SDK: pip install mcp")
        return
    if not SERVER_SCRIPT.exists():
        print(f"Server script not found: {SERVER_SCRIPT}")
        return

    print("=" * 78)
    print("MCP Client demo")
    print("=" * 78)
    await demo_discovery()
    await demo_call_tool()
    await demo_read_resource()
    await demo_agent_loop_with_mcp()


if __name__ == "__main__":
    asyncio.run(main() if _has_mcp else main())
    print("""
✅ Summary — MCP client pattern:

Every MCP-aware host (Claude Desktop, VS Code, custom agent) does exactly this:

  1. Spawn or connect to server (stdio or SSE)
  2. Initialize session
  3. Discover: list_tools / list_resources / list_prompts
  4. For agent loops:
       - Map MCP tool schemas → your LLM's function-calling schema
       - When LLM chooses a tool, call session.call_tool(name, args)
       - Feed the result back into messages
  5. Close session at end

This means: write ONE MCP server for your data, and it works with every
current + future MCP-aware AI app. That's the point of the protocol.

Try this: point Claude Desktop at 03_mcp_server_advanced.py. You'll suddenly
have file-system tools in Claude, using the SAME server this file drives.
""")
