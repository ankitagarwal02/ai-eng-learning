# Phase 5 — FastAPI: Serving AI over HTTP

> **Real-life analogy:** FastAPI is the reception desk of your AI system. Mobile apps, web UIs,
> and other services talk to FastAPI, which validates requests, routes them to the right AI
> component, streams responses back, and hangs up.

---

## Why FastAPI (vs Flask, Django, raw ASGI)

```
                    Flask       FastAPI     Notes
Async support        no          native      Critical for parallel LLM calls
Auto-validation      manual      Pydantic    LLM apps live and die by types
OpenAPI docs         plugin      auto        /docs works out of the box
Type-first design    no          yes         Same models used everywhere
Streaming            awkward     SSE built-in ChatGPT-style token stream
Speed (req/s)        ~5k         ~15k        Starlette + uvloop under the hood
```

FastAPI = Starlette (routing/ASGI) + Pydantic (validation) + OpenAPI (docs).

---

## The Request Lifecycle

```
Client                    FastAPI                        AI backend
  │                          │                              │
  ├─ POST /v1/chat ─────────▶│                              │
  │                          ├─ Pydantic validates body    │
  │                          ├─ Depends(): auth check ─────│
  │                          ├─ Depends(): rate limit ─────│
  │                          ├─ Middleware: log request    │
  │                          ├─ Route handler runs ────────▶│  await LLM
  │                          │                              │
  │◀──── SSE token stream ───┤◀─── tokens ──────────────────│
  │                          │                              │
  │                          ├─ Middleware: log response
  │                          ├─ BackgroundTask: metrics
```

Every arrow is either sync (blocks the event loop) or async (yields to loop).
Everything I/O should be async.

---

## OpenAI-Compatible API Pattern

Almost every LLM tool (LangChain, LiteLLM, LlamaIndex, Continue.dev, curl,
OpenAI SDK, custom clients) speaks the OpenAI Chat Completions shape:

```json
POST /v1/chat/completions
{
  "model": "gpt-4o-mini",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user",   "content": "Hi"}
  ],
  "temperature": 0.2,
  "stream": false
}

Response:
{
  "id": "chatcmpl-...",
  "choices": [{"message": {"role": "assistant", "content": "Hello!"},
               "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
}
```

**Advantage:** ship your own OpenAI-compatible endpoint (routing to internal
models, custom RAG, etc.) — every client library works with it.

---

## Server-Sent Events (SSE) for streaming

```
Content-Type: text/event-stream

data: {"choices":[{"delta":{"content":"Hel"}}]}

data: {"choices":[{"delta":{"content":"lo"}}]}

data: [DONE]
```

The `data:` prefix + blank line delimiter is the SSE format. Browsers and every
LLM client library know how to parse it.

---

## Concept Table

| Term | Plain-English |
|------|---------------|
| **ASGI** | Async web server interface — the "new WSGI" |
| **Route decorator** | `@app.get("/x")` registers a URL handler |
| **Path/query/body param** | Data from URL path / `?key=val` / JSON body |
| **Pydantic body** | Request JSON auto-parsed into a typed model |
| **Depends()** | Dependency injection — auth, DB, config |
| **BackgroundTasks** | Fire-and-forget after response is sent |
| **CORS** | Browser same-origin policy escape hatch |
| **SSE** | Server-Sent Events — one-way streaming |
| **uvicorn** | The ASGI server you run FastAPI with |
| **OpenAPI** | Machine-readable API spec (auto-generated at /openapi.json) |

---

## Code Patterns

### Minimal AI endpoint

```python
from fastapi import FastAPI, Depends
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    model: str = "gpt-4o-mini"
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    return {"response": await llm_call(req.message)}
```

### Streaming endpoint

```python
from fastapi.responses import StreamingResponse

@app.post("/chat/stream")
async def stream(req: ChatRequest):
    async def gen():
        async for tok in llm_stream(req.message):
            yield f"data: {tok}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
```

### Auth via Depends

```python
from fastapi import Header, HTTPException

async def verify_key(x_api_key: str = Header(...)):
    if x_api_key not in ALLOWED_KEYS:
        raise HTTPException(401, "invalid key")

@app.post("/chat", dependencies=[Depends(verify_key)])
async def chat(...): ...
```

---

## Use Case Matrix

| Scenario | Right endpoint pattern |
|----------|------------------------|
| Simple Q&A UI | `POST /chat` non-streaming |
| ChatGPT-style typing | `POST /chat` with SSE streaming |
| Bulk classification | `POST /batch` with `BackgroundTasks` |
| Health check for k8s | `GET /health` returning `{"status": "ok"}` |
| OpenAI SDK compat | `POST /v1/chat/completions` |
| Multi-tenant | API key header → tenant lookup via `Depends` |

---

## Folder structure

```
05-fastapi/
├── README.md
├── 01_fastapi_basics.py       ← Routes, Pydantic, Depends, BackgroundTasks
├── 02_streaming_responses.py  ← SSE streaming with async generator
├── 03_ai_api_server.py        ← Full OpenAI-compatible server
└── requirements.txt
```

Run each server with:
```
uvicorn 03_ai_api_server:app --reload --port 8000
```
Then browse to http://localhost:8000/docs for auto-generated Swagger UI.
