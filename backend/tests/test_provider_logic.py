"""Provider-layer tests: retries, rate limits, fallback, error taxonomy.

No real API calls — `_post_chat` / `_post_stream` are stubbed.
No API key needed: tests inject a fake key via Settings.
"""

import httpx
import pytest

from app.config import Settings
from app.llm import base as llm_base
from app.llm.base import (
    LLMAuthError,
    LLMContextLengthError,
    LLMEmptyResponseError,
    LLMInvalidRequestError,
    LLMNetworkError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResult,
    LLMTimeoutError,
)
from app.llm.openrouter import OpenRouterClient, StreamEvent, _classify_http_error


def make_settings(**overrides) -> Settings:
    defaults: dict = {
        "openrouter_api_key": "test-key",
        "openrouter_model": "primary-model",
        "openrouter_fallback_model": "fallback-model",
        "max_retries": 3,
        "retry_base_delay_seconds": 0.01,  # keep tests fast
    }
    defaults.update(overrides)
    return Settings(**defaults)


def make_client(**overrides) -> OpenRouterClient:
    return OpenRouterClient(make_settings(**overrides))


# --- error classification -------------------------------------------------


def test_auth_errors_classified():
    assert isinstance(_classify_http_error(401, "unauthorized"), LLMAuthError)
    assert isinstance(_classify_http_error(403, "forbidden"), LLMAuthError)


def test_rate_limit_classified():
    assert isinstance(_classify_http_error(429, "slow down"), LLMRateLimitError)


def test_context_length_detected():
    err = _classify_http_error(400, "This model's maximum context length is exceeded")
    assert isinstance(err, LLMContextLengthError)


def test_bad_request_without_context_hint_is_invalid():
    err = _classify_http_error(400, "messages field is required")
    assert isinstance(err, LLMInvalidRequestError)


def test_server_errors_are_provider_errors():
    for status in (500, 502, 503):
        assert isinstance(_classify_http_error(status, "boom"), LLMProviderError)


def test_retry_after_header_parsed():
    headers = httpx.Headers({"retry-after": "7"})
    assert llm_base.LLMRateLimitError("x", retry_after=7).retry_after == 7
    from app.llm.openrouter import _parse_retry_after

    assert _parse_retry_after(headers) == 7.0
    assert _parse_retry_after(httpx.Headers()) is None


# --- retry behaviour --------------------------------------------------------


async def test_retry_succeeds_after_transient_failures(monkeypatch):
    client = make_client()
    calls = {"n": 0}

    async def flaky(key, model, messages):
        calls["n"] += 1
        if calls["n"] < 3:
            raise LLMProviderError("temporary 503")
        return LLMResult(text="recovered", model=model)

    monkeypatch.setattr(client, "_post_chat", flaky)
    result = await client.complete([{"role": "user", "content": "hi"}])
    assert result.text == "recovered"
    assert calls["n"] == 3  # 2 failures + 1 success


async def test_no_retry_on_auth_error(monkeypatch):
    client = make_client()
    calls = {"n": 0}

    async def always_auth(key, model, messages):
        calls["n"] += 1
        raise LLMAuthError("bad key")

    monkeypatch.setattr(client, "_post_chat", always_auth)
    with pytest.raises(LLMAuthError):
        await client.complete([{"role": "user", "content": "hi"}])
    assert calls["n"] == 1  # must NOT retry auth failures


async def test_no_retry_on_invalid_request(monkeypatch):
    client = make_client()
    calls = {"n": 0}

    async def bad_request(key, model, messages):
        calls["n"] += 1
        raise LLMInvalidRequestError("malformed")

    monkeypatch.setattr(client, "_post_chat", bad_request)
    with pytest.raises(LLMInvalidRequestError):
        await client.complete([{"role": "user", "content": "hi"}])
    assert calls["n"] == 1


async def test_timeout_retries_then_raises(monkeypatch):
    client = make_client(max_retries=2)

    async def always_timeout(key, model, messages):
        raise LLMTimeoutError("slow provider")

    monkeypatch.setattr(client, "_post_chat", always_timeout)
    with pytest.raises(LLMTimeoutError):
        await client.complete([{"role": "user", "content": "hi"}])


async def test_rate_limit_error_carries_retry_after():
    err = LLMRateLimitError("limited", retry_after=5.0)
    assert err.retryable is True
    assert err.retry_after == 5.0


# --- fallback behaviour -----------------------------------------------------


async def test_fallback_used_when_primary_fails(monkeypatch):
    client = make_client()
    seen_models: list[str] = []

    async def fail_primary_then_succeed(key, model, messages):
        seen_models.append(model)
        if model == "primary-model":
            raise LLMProviderError("primary down")
        return LLMResult(text="from fallback", model=model)

    monkeypatch.setattr(client, "_post_chat", fail_primary_then_succeed)
    result = await client.complete([{"role": "user", "content": "hi"}])
    assert result.text == "from fallback"
    assert seen_models[0] == "primary-model"
    assert seen_models[-1] == "fallback-model"


async def test_no_fallback_on_auth_error(monkeypatch):
    client = make_client()
    seen_models: list[str] = []

    async def fail_auth(key, model, messages):
        seen_models.append(model)
        raise LLMAuthError("bad key")

    monkeypatch.setattr(client, "_post_chat", fail_auth)
    with pytest.raises(LLMAuthError):
        await client.complete([{"role": "user", "content": "hi"}])
    # Fallback would fail identically — must not even be attempted.
    assert seen_models == ["primary-model"]


async def test_empty_response_triggers_fallback(monkeypatch):
    client = make_client()
    seen_models: list[str] = []

    async def empty_then_ok(key, model, messages):
        seen_models.append(model)
        if model == "primary-model":
            raise LLMEmptyResponseError("empty")
        return LLMResult(text="ok", model=model)

    monkeypatch.setattr(client, "_post_chat", empty_then_ok)
    result = await client.complete([{"role": "user", "content": "hi"}])
    assert result.text == "ok"
    assert "fallback-model" in seen_models


async def test_stream_falls_back_before_first_token(monkeypatch):
    client = make_client()

    async def fake_stream(key, model, messages):
        if model == "primary-model":
            raise LLMProviderError("primary down")
        yield StreamEvent(kind="token", text="hello", model=model)
        yield StreamEvent(kind="done", model=model)

    monkeypatch.setattr(client, "_stream_with_retries", fake_stream)
    events = [ev async for ev in client.stream([{"role": "user", "content": "hi"}])]
    kinds = [e.kind for e in events]
    assert "fallback" in kinds
    assert "token" in kinds


# --- error taxonomy sanity ----------------------------------------------------


def test_non_retryable_errors_flagged():
    assert LLMAuthError.retryable is False
    assert LLMInvalidRequestError.retryable is False
    assert LLMContextLengthError.retryable is False


def test_retryable_errors_flagged():
    assert LLMTimeoutError.retryable is True
    assert LLMRateLimitError.retryable is True
    assert LLMProviderError.retryable is True
    assert LLMNetworkError.retryable is True


def test_network_and_timeout_are_distinct_types():
    assert issubclass(LLMTimeoutError, llm_base.LLMError)
    assert issubclass(LLMNetworkError, llm_base.LLMError)
    assert LLMTimeoutError is not LLMNetworkError
