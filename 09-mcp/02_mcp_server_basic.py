"""
02_mcp_server_basic.py — Working MCP server with tools and resources
=====================================================================

STATUS: Starter — full working example using the mcp Python SDK.

HOW TO RUN
----------
    pip install mcp
    python 02_mcp_server_basic.py     # runs on stdio
    # or:
    mcp dev 02_mcp_server_basic.py    # local inspector UI

Then in Claude Desktop or another MCP host, add this server to config.
"""

import os

try:
    from mcp.server.fastmcp import FastMCP
    _has_mcp = True
except ImportError:
    _has_mcp = False
    print("Install mcp: pip install mcp")


HR_POLICY_TEXT = """\
Company HR Policy (v2026.07)
============================

PTO:      15 days/year, accrues at 1.25 days/month.
Sick:     10 paid sick days/year, credited Jan 1.
Personal: 3 personal days/year, must use by Dec 31.
Parental: 16 weeks paid.
Bereavement: 5 days paid on immediate family loss.
"""

FAKE_DOCS = {
    "onboarding": "Welcome to the team! Setup guides at internal/onboarding.",
    "expenses":   "Submit expenses via /expenses. Reimbursed weekly.",
    "security":   "Report security issues to security@example.com.",
}


if _has_mcp:
    mcp = FastMCP("company-utils")

    # ─── TOOL: get_weather ────────────────────────────────
    @mcp.tool()
    def get_weather(city: str) -> dict:
        """Get current weather (mock)."""
        return {"city": city, "temp_c": 22, "condition": "sunny"}

    # ─── TOOL: search_docs ────────────────────────────────
    @mcp.tool()
    def search_docs(query: str, n: int = 3) -> list[str]:
        """Search internal docs by keyword."""
        query = query.lower()
        matched = [f"{key}: {text}" for key, text in FAKE_DOCS.items()
                   if query in key or query in text.lower()]
        return matched[:n] or [f"No docs matched '{query}'"]

    # ─── RESOURCE: HR policy ──────────────────────────────
    @mcp.resource("company://policies/hr")
    def hr_policy() -> str:
        """Full HR policy text."""
        return HR_POLICY_TEXT

    if __name__ == "__main__":
        mcp.run(transport="stdio")
else:
    print("""
The MCP SDK is not installed. Preview of what this file contains:

  Tool:      get_weather(city)          → {city, temp_c, condition}
  Tool:      search_docs(query, n)      → [str, ...]
  Resource:  company://policies/hr      → full HR policy text

Install and try:
    pip install mcp
    python 02_mcp_server_basic.py

Then add to Claude Desktop config:
    "servers": {
        "company-utils": {"command": "python", "args": ["path/to/02_mcp_server_basic.py"]}
    }
""")
