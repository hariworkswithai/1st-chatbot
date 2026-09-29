"""FastAPI entrypoint: routes, error mapping, logging, CORS, rate limiting.

The browser talks only to these endpoints. The OpenRouter key never
leaves the backend process.
"""

from __future__ import annotations

import json
import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app.chat_service import ChatService
from app.config import get_settings
from app.llm.base import LLMError
from app.models import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    ResetRequest,
    ResetResponse,
    TokenUsage as APITokenUsage,
)
from app.ratelimit import SlidingWindowLimiter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# Never log at DEBUG in prod paths that could print headers/bodies.
logger = logging.getLogger("ai-study-assistant")

settings = get_settings()
logging.getLogger().setLevel(settings.log_level.upper())

app = FastAPI(title="AI Study Assistant", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

service = ChatService(settings)
limiter = SlidingWindowLimiter(
    max_requests=settings.app_rate_limit_per_minute, window_seconds=60.0
)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def _check_app_rate_limit(request: Request) -> JSONResponse | None:
    allowed, retry_after = limiter.allow(_client_ip(request))
    if allowed:
        return None
    return JSONResponse(
        status_code=429,
        content=ErrorResponse(
            error="Too many requests. Slow down and try again in a minute.",
            code="app_rate_limited",
        ).model_dump(),
        headers={"Retry-After": str(int(retry_after) + 1)},
    )


@app.exception_handler(LLMError)
async def _llm_error_handler(request: Request, exc: LLMError):
    # Safe message to the browser; details stay in server logs only.
    logger.warning("llm error path=%s code=%s err=%s", request.url.path, exc.code, exc)
    status = exc.status_code
    headers = (
        {"Retry-After": str(int(exc.retry_after) + 1)}
        if isinstance(exc.retry_after, (int, float)) and exc.retry_after
        else None
    )
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(error=exc.user_message, code=exc.code).model_dump(),
        headers=headers,
    )


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "configured": settings.has_api_key,
        "model": settings.openrouter_model,
    }


@app.get("/api/config")
async def public_config():
    """Non-secret info the UI may display (never the API key)."""
    return {
        "model": settings.openrouter_model,
        "has_fallback": bool(
            settings.openrouter_fallback_model
            and settings.openrouter_fallback_model != settings.openrouter_model
        ),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request):
    limited = await _check_app_rate_limit(request)
    if limited:
        return limited
    started = time.perf_counter()
    result, sid = await service.chat(body.message, body.session_id)
    latency_ms = int((time.perf_counter() - started) * 1000)
    usage = (
        APITokenUsage(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
        )
        if result.usage
        else None
    )
    return ChatResponse(
        reply=result.text,
        session_id=sid,
        model=result.model,
        usage=usage,
        latency_ms=latency_ms,
    )


@app.post("/api/chat/stream")
async def chat_stream(body: ChatRequest, request: Request):
    limited = await _check_app_rate_limit(request)
    if limited:
        return limited

    gen, sid = await service.stream_chat(body.message, body.session_id)

    async def _sse():
        # First event carries the session id so the client can persist it.
        yield f"data: {json.dumps({'type': 'meta', 'session_id': sid})}\n\n"
        active_model = ""
        usage_dict: dict | None = None
        try:
            async for ev in gen:
                if ev.kind == "token":
                    active_model = ev.model or active_model
                    yield f"data: {json.dumps({'type': 'token', 'text': ev.text})}\n\n"
                elif ev.kind == "fallback":
                    yield f"data: {json.dumps({'type': 'fallback', 'model': ev.model})}\n\n"
                    active_model = ev.model
                elif ev.kind == "done":
                    active_model = ev.model or active_model
                    if ev.usage is not None:
                        usage_dict = {
                            "prompt_tokens": ev.usage.prompt_tokens,
                            "completion_tokens": ev.usage.completion_tokens,
                            "total_tokens": ev.usage.total_tokens,
                        }
            yield f"data: {json.dumps({'type': 'done', 'model': active_model, 'usage': usage_dict, 'session_id': sid})}\n\n"
        except LLMError as exc:
            # Stream the failure as a terminal event (status stays 200
            # because headers were already sent — standard SSE practice).
            logger.warning("stream failed: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'error': exc.user_message, 'code': exc.code})}\n\n"

    return StreamingResponse(_sse(), media_type="text/event-stream")


@app.post("/api/chat/reset", response_model=ResetResponse)
async def reset(body: ResetRequest):
    service.sessions.reset(body.session_id)
    logger.info("session reset session=%s", body.session_id[:8])
    return ResetResponse(session_id=body.session_id, cleared=True)


# --- optional static hosting -------------------------------------------------
# If `frontend/dist` has been built, serve it from "/" so a single process
# can demo the whole app (API routes above still take precedence).
try:
    from pathlib import Path

    from fastapi.staticfiles import StaticFiles

    _dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
    if _dist.is_dir():
        app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
        logger.info("serving frontend from %s", _dist)
except Exception as exc:  # pragma: no cover - demo convenience only
    logger.warning("frontend static mount skipped: %s", exc)
