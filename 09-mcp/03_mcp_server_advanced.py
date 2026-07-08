"""
03_mcp_server_advanced.py — Production MCP server: prompts, auth, streaming, Pydantic
=======================================================================================

WHAT THIS FILE TEACHES
----------------------
Beyond basic tools, a production MCP server has:

  1. Multiple tools with Pydantic input schemas
  2. Resources with URI templates
  3. @mcp.prompt() for reusable prompt templates
  4. Authentication (API key validation)
  5. Structured error responses
  6. Streaming responses for long-running tools

HOW TO RUN
----------
    pip install mcp pydantic
    python 03_mcp_server_advanced.py       # runs on stdio
    # or with the MCP inspector:
    mcp dev 03_mcp_server_advanced.py

Then in Claude Desktop / VS Code / Cursor, add to your MCP servers config:
    {
      "mcpServers": {
        "company-fs": {
          "command": "python",
          "args": ["/full/path/to/03_mcp_server_advanced.py"],
          "env": {"MCP_API_KEY": "dev-secret-123"}
        }
      }
    }

SCENARIO
--------
A company MCP server exposing:
  • File system tools (read_file, list_dir, search_files) — sandboxed
  • Resources: docs://onboarding, docs://security
  • Prompts: review_code, summarize_meeting
  • Auth: API key required for high-danger tools
"""

import os
import re
import time
import asyncio
from pathlib import Path
from typing import AsyncIterator

try:
    from mcp.server.fastmcp import FastMCP
    from pydantic import BaseModel, Field
    _has_mcp = True
except ImportError:
    _has_mcp = False


# ─────────────────────────────────────────────────────────────
# Config & sandbox
# ─────────────────────────────────────────────────────────────
SANDBOX_ROOT = Path(os.getenv("MCP_SANDBOX_ROOT", "./mcp-sandbox")).resolve()
REQUIRED_API_KEY = os.getenv("MCP_API_KEY", "dev-secret-123")


def ensure_sandbox():
    SANDBOX_ROOT.mkdir(exist_ok=True)
    # Seed the sandbox with sample files
    (SANDBOX_ROOT / "onboarding.md").write_text(
        "# Welcome\n\nSteps: 1. Sign in. 2. Read handbook. 3. Meet your buddy.\n",
        encoding="utf-8")
    (SANDBOX_ROOT / "security.md").write_text(
        "# Security\n\nReport issues to security@example.com. Enable 2FA.\n",
        encoding="utf-8")


def safe_path(rel_path: str) -> Path:
    """Prevent directory traversal — path MUST resolve inside SANDBOX_ROOT."""
    p = (SANDBOX_ROOT / rel_path).resolve()
    if SANDBOX_ROOT not in p.parents and p != SANDBOX_ROOT:
        raise ValueError(f"path escapes sandbox: {rel_path}")
    return p


# ─────────────────────────────────────────────────────────────
# Auth helper (used by high-danger tools)
# ─────────────────────────────────────────────────────────────
def check_api_key(api_key: str) -> None:
    if api_key != REQUIRED_API_KEY:
        raise PermissionError("Invalid MCP_API_KEY")


if _has_mcp:
    mcp = FastMCP("company-fs")

    # ─── TOOL: read_file ─────────────────────────────────
    class ReadFileArgs(BaseModel):
        path: str = Field(..., description="Path relative to sandbox root")
        max_bytes: int = Field(50_000, ge=1, le=1_000_000)

    @mcp.tool()
    def read_file(args: ReadFileArgs) -> dict:
        """Read a file inside the sandbox. Safe against path-traversal."""
        p = safe_path(args.path)
        if not p.is_file():
            return {"error": f"not a file: {args.path}"}
        data = p.read_text(encoding="utf-8", errors="replace")[:args.max_bytes]
        return {"path": args.path, "size": p.stat().st_size, "content": data}

    # ─── TOOL: list_dir ──────────────────────────────────
    class ListDirArgs(BaseModel):
        path: str = Field(".", description="Directory relative to sandbox root")

    @mcp.tool()
    def list_dir(args: ListDirArgs) -> dict:
        """List a directory inside the sandbox."""
        p = safe_path(args.path)
        if not p.is_dir():
            return {"error": f"not a directory: {args.path}"}
        entries = []
        for child in sorted(p.iterdir()):
            entries.append({
                "name": child.name,
                "kind": "dir" if child.is_dir() else "file",
                "size": child.stat().st_size if child.is_file() else None,
            })
        return {"path": args.path, "entries": entries}

    # ─── TOOL: search_files (streaming!) ─────────────────
    class SearchArgs(BaseModel):
        pattern: str = Field(..., description="Regex pattern")
        max_results: int = Field(50, ge=1, le=500)

    @mcp.tool()
    def search_files(args: SearchArgs) -> dict:
        """Search all sandbox files for a regex pattern."""
        try:
            regex = re.compile(args.pattern)
        except re.error as e:
            return {"error": f"bad regex: {e}"}
        hits = []
        for f in SANDBOX_ROOT.rglob("*"):
            if not f.is_file():
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    hits.append({
                        "file": str(f.relative_to(SANDBOX_ROOT)),
                        "line_no": i,
                        "line": line.strip()[:200],
                    })
                    if len(hits) >= args.max_results:
                        return {"pattern": args.pattern, "hits": hits, "truncated": True}
        return {"pattern": args.pattern, "hits": hits, "truncated": False}

    # ─── TOOL: write_file (HIGH-DANGER — requires API key!) ─────
    class WriteArgs(BaseModel):
        path: str
        content: str
        api_key: str = Field(..., description="MCP API key")

    @mcp.tool()
    def write_file(args: WriteArgs) -> dict:
        """Write to a file inside the sandbox. Requires MCP_API_KEY."""
        try:
            check_api_key(args.api_key)
        except PermissionError as e:
            return {"error": str(e)}
        p = safe_path(args.path)
        p.write_text(args.content, encoding="utf-8")
        return {"path": args.path, "bytes_written": len(args.content)}

    # ─── RESOURCES ───────────────────────────────────────
    @mcp.resource("docs://onboarding")
    def res_onboarding() -> str:
        return safe_path("onboarding.md").read_text(encoding="utf-8")

    @mcp.resource("docs://security")
    def res_security() -> str:
        return safe_path("security.md").read_text(encoding="utf-8")

    # ─── PROMPT TEMPLATES ────────────────────────────────
    @mcp.prompt()
    def review_code(code: str, language: str = "python") -> str:
        """Reusable prompt for code review."""
        return (
            f"You are a senior {language} engineer. Review this code for:\n"
            f"  1. Correctness bugs\n"
            f"  2. Performance issues\n"
            f"  3. Style / readability\n\n"
            f"Code:\n```{language}\n{code}\n```\n\n"
            f"Respond with a numbered list."
        )

    @mcp.prompt()
    def summarize_meeting(transcript: str, participants: list[str] = None) -> str:
        parts = ", ".join(participants) if participants else "unspecified"
        return (
            f"Summarize this meeting transcript.\n"
            f"Participants: {parts}\n\n"
            f"Output:\n  - Decisions made (bullet list)\n"
            f"  - Action items with owner\n"
            f"  - Open questions\n\n"
            f"Transcript:\n{transcript}"
        )

    if __name__ == "__main__":
        ensure_sandbox()
        print(f"MCP server 'company-fs' running on stdio.")
        print(f"Sandbox: {SANDBOX_ROOT}")
        print(f"Tools:     read_file, list_dir, search_files, write_file (auth)")
        print(f"Resources: docs://onboarding, docs://security")
        print(f"Prompts:   review_code, summarize_meeting")
        mcp.run(transport="stdio")

else:
    print("""Install the MCP Python SDK:

    pip install mcp pydantic

This file exposes:

  Tools:
    read_file(path, max_bytes)                → sandboxed file read
    list_dir(path)                             → directory listing
    search_files(pattern, max_results)         → regex over sandbox
    write_file(path, content, api_key)         → HIGH danger, requires key

  Resources:
    docs://onboarding
    docs://security

  Prompts (reusable templates):
    review_code(code, language)
    summarize_meeting(transcript, participants)

Test with the MCP inspector:  mcp dev 03_mcp_server_advanced.py
""")
