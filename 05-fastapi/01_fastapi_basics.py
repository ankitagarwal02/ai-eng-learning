"""
01_fastapi_basics.py — FastAPI fundamentals for AI engineers
=============================================================

WHAT THIS FILE TEACHES
----------------------
• @app.get/post/put/delete decorators.
• Path params, query params, request body (Pydantic).
• Response models and status codes.
• Dependency injection with Depends().
• Background tasks.
• CORS middleware.
• Health check endpoint pattern.
• Rate limiting via dependency.

HOW TO RUN
----------
    pip install fastapi uvicorn[standard] httpx
    uvicorn 01_fastapi_basics:app --reload --port 8000

Then browse:
    http://localhost:8000/docs        (Swagger UI)
    http://localhost:8000/redoc       (ReDoc UI)
    http://localhost:8000/openapi.json (raw spec)

Or run in-process test at the bottom of this file:
    python 01_fastapi_basics.py

REAL-WORLD SCENARIO
-------------------
A simple AI chat endpoint that accepts
    {"model": "gpt-4o", "message": "hello"}
and returns
    {"response": "...", "tokens_used": N}
with API-key auth, rate-limiting dependency, and a background task that logs metrics.
"""

import os
import time
import asyncio
from collections import defaultdict, deque
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Header, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MOCK_MODE = not os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="Basic AI Chat API",
    description="Phase 5 tutorial — Pydantic + Depends + BackgroundTasks.",
    version="1.0.0",
)

# ─────────────────────────────────────────────────────────────
# SECTION 1: CORS (browsers)
# ─────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # in prod: list your explicit origins
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# SECTION 2: Request / response models
# ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    model: str = Field("gpt-4o-mini", description="Model to use")
    message: str = Field(..., min_length=1, max_length=4000)
    temperature: float = Field(0.2, ge=0, le=2)

class ChatResponse(BaseModel):
    response: str
    model: str
    tokens_used: int
    latency_ms: float


# ─────────────────────────────────────────────────────────────
# SECTION 3: Fake token store + auth dependency
# ─────────────────────────────────────────────────────────────
ALLOWED_KEYS = {"demo-key-abc123", "test-key-xyz789"}

async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    """Read `X-API-Key` header, block if not in allowlist."""
    if x_api_key not in ALLOWED_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# ─────────────────────────────────────────────────────────────
# SECTION 4: Rate limiter (dependency)
# ─────────────────────────────────────────────────────────────
_request_log: dict[str, deque] = defaultdict(deque)
LIMIT_PER_MINUTE = 30

async def rate_limit(api_key: str = Depends(verify_api_key)) -> str:
    now = time.time()
    q = _request_log[api_key]
    # Drop calls older than 60s
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail=f"Rate limit: {LIMIT_PER_MINUTE}/min")
    q.append(now)
    return api_key


# ─────────────────────────────────────────────────────────────
# SECTION 5: The "LLM call" (mock)
# ─────────────────────────────────────────────────────────────
async def call_llm(req: ChatRequest) -> ChatResponse:
    start = time.perf_counter()
    await asyncio.sleep(0.05)   # simulate network
    text = f"[MOCK {req.model}] echo: {req.message}"
    tokens = len(req.message.split()) + len(text.split())
    return ChatResponse(
        response=text,
        model=req.model,
        tokens_used=tokens,
        latency_ms=round((time.perf_counter() - start) * 1000, 1),
    )


# ─────────────────────────────────────────────────────────────
# SECTION 6: Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["Ops"])
def health() -> dict:
    """Kubernetes liveness/readiness probe. NEVER put slow work here."""
    return {"status": "ok", "mock_mode": MOCK_MODE}


@app.get("/models/{model_id}", tags=["Meta"])
def get_model_info(model_id: str, verbose: bool = False) -> dict:
    """Path param + query param. Path validates against the URL segment;
    query params come from `?verbose=true`."""
    info = {"model": model_id, "context": 128000}
    if verbose:
        info["price_per_1M_input"] = 0.15
        info["price_per_1M_output"] = 0.60
    return info


@app.post("/chat",
          response_model=ChatResponse,
          status_code=status.HTTP_200_OK,
          tags=["Chat"])
async def chat(
    req: ChatRequest,
    background_tasks: BackgroundTasks,
    _: str = Depends(rate_limit),        # runs verify_api_key + rate_limit
):
    """Main chat endpoint. All ingredients:
      • Pydantic request body validated automatically
      • Depends() enforces auth + rate limit before this fn runs
      • BackgroundTasks: metric logging AFTER response is sent
    """
    result = await call_llm(req)

    # Fire-and-forget — user response returns immediately.
    background_tasks.add_task(log_metrics, req.model, result.tokens_used, result.latency_ms)

    return result


def log_metrics(model: str, tokens: int, latency_ms: float) -> None:
    """In real prod: write to a metric store (StatsD, Prometheus, DataDog)."""
    print(f"  [metrics] model={model} tokens={tokens} latency={latency_ms}ms")


# ─────────────────────────────────────────────────────────────
# In-process demo — no browser needed
# ─────────────────────────────────────────────────────────────
async def _demo():
    """Runs a few requests through the app using httpx.AsyncClient
    (in-process, no real server needed)."""
    try:
        import httpx
    except ImportError:
        print("(Install httpx to run the in-process demo: pip install httpx)")
        return

    from httpx import ASGITransport

    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1) Health
        r = await client.get("/health")
        print(f"GET  /health              → {r.status_code}  {r.json()}")

        # 2) Model info with query param
        r = await client.get("/models/gpt-4o?verbose=true")
        print(f"GET  /models/gpt-4o?...   → {r.status_code}  {r.json()}")

        # 3) Chat with valid key
        r = await client.post("/chat",
                              json={"message": "hello world"},
                              headers={"X-API-Key": "demo-key-abc123"})
        print(f"POST /chat (valid key)    → {r.status_code}  {r.json()}")

        # 4) Chat with invalid key
        r = await client.post("/chat",
                              json={"message": "hi"},
                              headers={"X-API-Key": "wrong"})
        print(f"POST /chat (bad key)      → {r.status_code}  {r.json()}")

        # 5) Chat with invalid body
        r = await client.post("/chat",
                              json={"message": ""},
                              headers={"X-API-Key": "demo-key-abc123"})
        print(f"POST /chat (empty msg)    → {r.status_code}  (validation error)")


if __name__ == "__main__":
    print("=" * 70)
    print("FastAPI in-process demo")
    print("=" * 70)
    asyncio.run(_demo())
    print("""
✅ Summary — the FastAPI ingredients you'll reuse in every AI service:

  • Pydantic request/response models = zero-boilerplate validation
  • Depends() = clean auth + rate-limit + DI without middleware spaghetti
  • BackgroundTasks = metrics after response, not during
  • CORS = browsers won't block your app
  • /health = kubernetes/lb-ready probe

Run with real hot-reload server:
    uvicorn 01_fastapi_basics:app --reload

Next: 02_streaming_responses.py — SSE token streaming.
""")
