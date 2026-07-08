"""
02_streaming_responses.py — Server-Sent Events (SSE) for token streaming
========================================================================

WHAT THIS FILE TEACHES
----------------------
• StreamingResponse + async generator = SSE-compatible stream.
• The SSE wire format: `data: <json>\\n\\n`, terminator `[DONE]`.
• Handling errors mid-stream (send error event, close).
• Client-side reading of a stream with httpx.

HOW TO RUN
----------
    uvicorn 02_streaming_responses:app --reload --port 8000
    # then in another terminal:
    curl -N http://localhost:8000/stream/hello

Or in-process:
    python 02_streaming_responses.py

REAL-WORLD SCENARIO
-------------------
Recreate the ChatGPT-style typing effect for your own model. Same protocol as
OpenAI Chat Completions with `stream=True`.
"""

import os
import json
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

MOCK_MODE = not os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Streaming Chat API", version="1.0.0")


class ChatRequest(BaseModel):
    message: str
    model: str = "gpt-4o-mini"


# ─────────────────────────────────────────────────────────────
# The streaming source — mock token generator
# ─────────────────────────────────────────────────────────────
async def mock_token_stream(text: str):
    """Emit words one at a time with tiny delays. Real code would await the
    LLM SDK's streaming API (e.g., openai.chat.completions.create(stream=True))."""
    words = text.split()
    for i, w in enumerate(words):
        await asyncio.sleep(0.05)
        yield w + (" " if i < len(words) - 1 else "")


async def sse_chat_stream(req: ChatRequest):
    """Convert a token stream to SSE frames.

    Each frame:
        data: {"choices":[{"delta":{"content":"..."}}]}\n\n

    The final frame is `data: [DONE]\n\n`.
    """
    reply = f"[MOCK {req.model}] echo: {req.message}"
    try:
        async for tok in mock_token_stream(reply):
            payload = {"choices": [{"delta": {"content": tok}}]}
            yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        # Errors mid-stream: send one final event so the client knows.
        err = {"error": {"type": type(e).__name__, "message": str(e)}}
        yield f"data: {json.dumps(err)}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/chat/stream", tags=["Chat"])
async def stream_chat(req: ChatRequest):
    """The endpoint returns a StreamingResponse that yields SSE frames."""
    return StreamingResponse(sse_chat_stream(req), media_type="text/event-stream")


# Simpler GET endpoint you can test with curl directly:
@app.get("/stream/{msg}", tags=["Chat"])
async def stream_get(msg: str):
    """Simpler endpoint — GET /stream/hello  streams back tokens."""
    return StreamingResponse(
        sse_chat_stream(ChatRequest(message=msg)),
        media_type="text/event-stream",
    )


# ─────────────────────────────────────────────────────────────
# In-process demo — read a stream as a client
# ─────────────────────────────────────────────────────────────
async def _demo():
    try:
        import httpx
    except ImportError:
        print("(Install httpx to run the in-process demo)")
        return

    from httpx import ASGITransport
    print("Streaming demo — reading SSE frames as they arrive:\n")

    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with client.stream("POST", "/chat/stream",
                                 json={"message": "Explain FastAPI streaming in one sentence."}) as r:
            async for raw_line in r.aiter_lines():
                if not raw_line.startswith("data:"):
                    continue
                payload = raw_line[5:].strip()
                if payload == "[DONE]":
                    print("\n  ← [DONE]")
                    break
                data = json.loads(payload)
                if "error" in data:
                    print(f"\n  ← error: {data['error']}")
                    break
                content = data["choices"][0]["delta"]["content"]
                print(content, end="", flush=True)


if __name__ == "__main__":
    print("=" * 70)
    print("SSE Streaming demo")
    print("=" * 70)
    asyncio.run(_demo())
    print("""

✅ Summary — SSE streaming:

  • StreamingResponse + async generator = trivial FastAPI streaming
  • SSE frame format: `data: <json>\\n\\n` per token, `[DONE]` to close
  • Errors mid-stream: emit an error frame, then close
  • Clients (browsers, httpx, OpenAI SDK) all speak this natively

You now match the wire protocol of OpenAI Chat Completions streaming.

Next: 03_ai_api_server.py — a complete OpenAI-compatible server.
""")
