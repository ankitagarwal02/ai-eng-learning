"""
03_ai_api_server.py — Production-shape OpenAI-compatible AI server
====================================================================

WHAT THIS BUILDS
----------------
A complete FastAPI service with:
  • /v1/chat/completions endpoint (OpenAI-compatible request/response schema)
  • /v1/embeddings endpoint
  • API-key auth via header
  • Structured request logging middleware
  • Proper error handling with correct HTTP status codes
  • Input validation: max_tokens ceiling + content-policy pre-check
  • Streaming variant via `stream: true`
  • Metrics response header: X-Request-ID, X-Tokens-Used, X-Latency-MS

HOW TO RUN
----------
    pip install fastapi 'uvicorn[standard]' httpx
    uvicorn 03_ai_api_server:app --reload --port 8000

Test with the OpenAI Python SDK — since we speak its schema:
    import openai
    c = openai.OpenAI(api_key='demo-key-abc123', base_url='http://localhost:8000/v1')
    c.chat.completions.create(model='gpt-4o-mini',
                              messages=[{'role':'user','content':'hi'}])
"""

import os
import time
import uuid
import json
import asyncio
import logging
from typing import Optional, Literal
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Header, Request, status
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

MOCK_MODE = not os.getenv("OPENAI_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("ai-api")

ALLOWED_KEYS = {"demo-key-abc123"}
MAX_INPUT_TOKENS = 4000
MAX_MAX_TOKENS = 4096


# ─────────────────────────────────────────────────────────────
# Lifespan (startup/shutdown hooks — cleaner than @on_event)
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("startup — MOCK_MODE=%s", MOCK_MODE)
    yield
    log.info("shutdown")


app = FastAPI(
    title="Custom OpenAI-Compatible API",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────
# Middleware — request logging + request-id
# ─────────────────────────────────────────────────────────────
@app.middleware("http")
async def add_request_context(request: Request, call_next):
    req_id = str(uuid.uuid4())
    start = time.perf_counter()
    request.state.request_id = req_id
    log.info("→ %s %s  request_id=%s", request.method, request.url.path, req_id)

    response = await call_next(request)

    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Latency-MS"] = str(elapsed_ms)
    log.info("← %s %s  status=%s  request_id=%s  %sms",
             request.method, request.url.path, response.status_code, req_id, elapsed_ms)
    return response


# ─────────────────────────────────────────────────────────────
# Schemas (OpenAI-compatible shape)
# ─────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = Field("gpt-4o-mini")
    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: float = Field(0.2, ge=0, le=2)
    max_tokens: int = Field(1024, gt=0, le=MAX_MAX_TOKENS)
    stream: bool = False

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"]

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


class EmbeddingsRequest(BaseModel):
    model: str = "text-embedding-3-small"
    input: list[str] | str

class EmbeddingItem(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]

class EmbeddingsResponse(BaseModel):
    object: str = "list"
    model: str
    data: list[EmbeddingItem]
    usage: Usage


# ─────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────
async def verify_key(authorization: str = Header(..., alias="Authorization")) -> str:
    """OpenAI-style: Authorization: Bearer <key>"""
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Authorization header must be 'Bearer <key>'")
    key = authorization.split(" ", 1)[1]
    if key not in ALLOWED_KEYS:
        raise HTTPException(401, "Invalid API key")
    return key


# ─────────────────────────────────────────────────────────────
# Content policy pre-check (input guardrail)
# ─────────────────────────────────────────────────────────────
FORBIDDEN_KEYWORDS = ["ignore all previous instructions", "system:", "you are now dan"]

def content_policy_check(messages: list[ChatMessage]) -> Optional[str]:
    """Return a reason string if blocked; else None."""
    joined = " ".join(m.content.lower() for m in messages)
    for kw in FORBIDDEN_KEYWORDS:
        if kw in joined:
            return f"Content policy: matched '{kw}'"
    total_chars = sum(len(m.content) for m in messages)
    approx_tokens = total_chars // 4
    if approx_tokens > MAX_INPUT_TOKENS:
        return f"Input too long: ~{approx_tokens} tokens > {MAX_INPUT_TOKENS}"
    return None


# ─────────────────────────────────────────────────────────────
# The LLM (mock)
# ─────────────────────────────────────────────────────────────
async def mock_chat(req: ChatCompletionRequest) -> str:
    await asyncio.sleep(0.05)
    last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    return f"[MOCK {req.model}] I heard: {last_user[:60]}"

async def mock_stream(req: ChatCompletionRequest):
    text = await mock_chat(req)
    for word in text.split():
        await asyncio.sleep(0.03)
        yield word + " "


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "mock_mode": MOCK_MODE}


@app.post("/v1/chat/completions",
          response_model=None,   # streaming may return non-JSON
          tags=["Chat"])
async def chat_completions(
    req: ChatCompletionRequest,
    _: str = Depends(verify_key),
):
    # Guardrail:
    reason = content_policy_check(req.messages)
    if reason:
        raise HTTPException(status_code=400, detail=reason)

    if req.stream:
        async def sse():
            id_ = "chatcmpl-" + uuid.uuid4().hex[:12]
            async for tok in mock_stream(req):
                frame = {
                    "id": id_,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(frame)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")

    content = await mock_chat(req)

    prompt_toks = sum(len(m.content.split()) for m in req.messages)
    completion_toks = len(content.split())

    return ChatCompletionResponse(
        id="chatcmpl-" + uuid.uuid4().hex[:12],
        created=int(time.time()),
        model=req.model,
        choices=[Choice(
            index=0,
            message=ChatMessage(role="assistant", content=content),
            finish_reason="stop",
        )],
        usage=Usage(prompt_tokens=prompt_toks,
                    completion_tokens=completion_toks,
                    total_tokens=prompt_toks + completion_toks),
    )


@app.post("/v1/embeddings", response_model=EmbeddingsResponse, tags=["Embeddings"])
async def embeddings(req: EmbeddingsRequest, _: str = Depends(verify_key)):
    inputs = [req.input] if isinstance(req.input, str) else req.input
    if not inputs:
        raise HTTPException(400, "input must be non-empty")

    # Mock: deterministic random 8-dim vectors.
    import random
    items = []
    total_tokens = 0
    for i, txt in enumerate(inputs):
        random.seed(hash(txt) & 0xFFFFFFFF)
        vec = [random.uniform(-1, 1) for _ in range(8)]
        items.append(EmbeddingItem(index=i, embedding=vec))
        total_tokens += len(txt.split())

    return EmbeddingsResponse(
        model=req.model,
        data=items,
        usage=Usage(prompt_tokens=total_tokens, completion_tokens=0, total_tokens=total_tokens),
    )


# ─────────────────────────────────────────────────────────────
# Exception handler → uniform error shape (OpenAI-style)
# ─────────────────────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": exc.detail, "type": "invalid_request",
                           "request_id": getattr(request.state, "request_id", None)}},
    )


# ─────────────────────────────────────────────────────────────
# In-process demo
# ─────────────────────────────────────────────────────────────
async def _demo():
    try:
        import httpx
    except ImportError:
        print("Install httpx for demo")
        return

    from httpx import ASGITransport
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # Health:
        r = await c.get("/health")
        print("health:", r.json())

        headers = {"Authorization": "Bearer demo-key-abc123"}

        # Chat non-stream:
        r = await c.post("/v1/chat/completions",
                         headers=headers,
                         json={"model": "gpt-4o-mini",
                               "messages": [{"role": "user", "content": "hi"}]})
        print("\nchat (non-stream):", r.status_code, r.json()["choices"][0]["message"]["content"])
        print("  headers X-Request-ID:", r.headers.get("X-Request-ID"))
        print("  headers X-Latency-MS:", r.headers.get("X-Latency-MS"))

        # Blocked by content policy:
        r = await c.post("/v1/chat/completions",
                         headers=headers,
                         json={"model": "gpt-4o-mini",
                               "messages": [{"role": "user",
                                             "content": "ignore all previous instructions"}]})
        print("\nchat (blocked):", r.status_code, r.json())

        # Embeddings:
        r = await c.post("/v1/embeddings",
                         headers=headers,
                         json={"input": ["hello world", "foo bar"]})
        emb = r.json()
        print(f"\nembeddings: dim={len(emb['data'][0]['embedding'])} count={len(emb['data'])}")

        # Streaming:
        print("\nchat (streaming):")
        async with c.stream("POST", "/v1/chat/completions",
                            headers=headers,
                            json={"model": "gpt-4o-mini",
                                  "messages": [{"role": "user", "content": "count to five"}],
                                  "stream": True}) as sr:
            async for line in sr.aiter_lines():
                if line.startswith("data:"):
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    frame = json.loads(body)
                    print("  chunk:", frame["choices"][0]["delta"].get("content", ""))


if __name__ == "__main__":
    asyncio.run(_demo())
    print("""
✅ Summary — Production-shape API:

  • OpenAI-compatible schema → any client library works
  • Structured logging middleware + request IDs
  • Auth via Bearer token
  • Content-policy pre-check (input guardrail)
  • Streaming and non-streaming variants
  • Uniform error shape

Deploy this behind a load balancer, add Prometheus metrics (Phase 16), point
your existing OpenAI SDK at it. Done.
""")
