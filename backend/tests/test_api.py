"""App-level rate limiter + API route tests (mocked LLM, no API key)."""

from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app import main
from app.llm.base import LLMRateLimitError, LLMResult


def make_api_client() -> TestClient:
    return TestClient(main.app, raise_server_exceptions=False)


def test_health_endpoint():
    client = make_api_client()
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_config_endpoint_exposes_no_secret():
    client = make_api_client()
    res = client.get("/api/config")
    assert res.status_code == 200
    body = res.text.lower()
    assert "api_key" not in body
    assert "bearer" not in body


def test_valid_chat_request(monkeypatch):
    client = make_api_client()
    monkeypatch.setattr(
        main.service,
        "chat",
        AsyncMock(return_value=(LLMResult(text="Hi there!", model="m"), "sess123")),
    )
    res = client.post("/api/chat", json={"message": "Hello"})
    assert res.status_code == 200
    data = res.json()
    assert data["reply"] == "Hi there!"
    assert data["session_id"] == "sess123"


def test_empty_input_rejected_by_validation():
    client = make_api_client()
    res = client.post("/api/chat", json={"message": "   "})
    # Pydantic min_length passes (non-empty string) but service validation
    # maps to a 400 invalid_request — either way it must NOT reach the LLM.
    assert res.status_code in (400, 422)


def test_missing_message_field_rejected():
    client = make_api_client()
    res = client.post("/api/chat", json={})
    assert res.status_code == 422


def test_provider_rate_limit_maps_to_safe_429(monkeypatch):
    client = make_api_client()

    async def limited(message, session_id):
        raise LLMRateLimitError("provider busy", retry_after=2.0)

    monkeypatch.setattr(main.service, "chat", AsyncMock(side_effect=limited))
    res = client.post("/api/chat", json={"message": "hi"})
    assert res.status_code == 429
    body = res.json()
    assert "error" in body
    # No internals leaked.
    assert "traceback" not in res.text.lower()
    assert "OPENROUTER" not in res.text


def test_reset_endpoint():
    client = make_api_client()
    sid = main.service.sessions.create()
    main.service.sessions.append(sid, "user", "hi")
    res = client.post("/api/chat/reset", json={"session_id": sid})
    assert res.status_code == 200
    assert main.service.get(sid) == [] if hasattr(main.service, "get") else True
    assert main.service.sessions.get(sid) == []


def test_app_rate_limit_blocks_abuse():
    from app.ratelimit import SlidingWindowLimiter

    limiter = SlidingWindowLimiter(max_requests=2, window_seconds=60.0)
    assert limiter.allow("1.2.3.4")[0] is True
    assert limiter.allow("1.2.3.4")[0] is True
    allowed, retry_after = limiter.allow("1.2.3.4")
    assert allowed is False
    assert retry_after > 0


def test_app_rate_limit_middleware_returns_429(monkeypatch):
    from app.ratelimit import SlidingWindowLimiter

    monkeypatch.setattr(
        main, "limiter", SlidingWindowLimiter(max_requests=1, window_seconds=60.0)
    )
    monkeypatch.setattr(
        main.service,
        "chat",
        AsyncMock(return_value=(LLMResult(text="ok", model="m"), "s")),
    )
    client = make_api_client()
    assert client.post("/api/chat", json={"message": "one"}).status_code == 200
    second = client.post("/api/chat", json={"message": "two"})
    assert second.status_code == 429
    assert "Retry-After" in second.headers
