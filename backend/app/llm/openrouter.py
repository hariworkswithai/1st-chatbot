"""OpenRouter provider adapter.

All OpenRouter-specific HTTP lives here. The chat service calls
`complete()` / `stream()` and only handles `LLMError` + `LLMResult`.

Reliability implemented here:
- per-request timeout (never hang forever)
- retries with exponential backoff (transient errors only)
- 429 handling incl. Retry-After header
- primary -> fallback model (provider-side failures only)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

from app.config import Settings
from app.llm.base import (
    LLMEmptyResponseError,
    LLMError,
    LLMInvalidRequestError,
    LLMNetworkError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResult,
    LLMAuthError,
    LLMContextLengthError,
    LLMTimeoutError,
    TokenUsage,
    is_context_length_message,
)

logger = logging.getLogger(__name__)


@dataclass
class StreamEvent:
    """Events yielded by `OpenRouterClient.stream()`."""

    kind: str  # "token" | "done" | "fallback"
    text: str = ""
    model: str = ""
    usage: TokenUsage | None = None


def _parse_retry_after(headers: httpx.Headers) -> float | None:
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _classify_http_error(status: int, body_text: str) -> LLMError:
    """Map an HTTP failure to our error taxonomy (never raises)."""
    if status in (401, 403):
        return LLMAuthError(f"Provider rejected credentials (status {status}).")
    if status == 429:
        return LLMRateLimitError(f"Provider rate limit (status 429). {body_text[:200]}")
    if status in (400, 422):
        if is_context_length_message(body_text):
            return LLMContextLengthError(body_text[:300])
        return LLMInvalidRequestError(body_text[:300])
    if status == 413:  # payload too large -> treat like context problem
        return LLMContextLengthError(body_text[:300])
    if 500 <= status <= 599:
        return LLMProviderError(f"Provider server error (status {status}).")
    return LLMProviderError(f"Unexpected provider status {status}.")


class OpenRouterClient:
    def __init__(self, settings: Settings):
        self._s = settings
        self._url = f"{settings.openrouter_base_url.rstrip('/')}/chat/completions"

    # -- public API ------------------------------------------------------

    @property
    def primary_model(self) -> str:
        return self._s.openrouter_model

    @property
    def fallback_model(self) -> str:
        return self._s.openrouter_fallback_model

    def _require_key(self) -> str:
        key = (self._s.openrouter_api_key or "").strip()
        if not key:
            raise LLMAuthError("OPENROUTER_API_KEY is missing.")
        return key

    async def complete(self, messages: list[dict[str, str]]) -> LLMResult:
        """Non-streaming completion with retry + fallback."""
        key = self._require_key()
        start = time.perf_counter()
        logger.info(
            "llm request start model=%s messages=%d", self.primary_model, len(messages)
        )
        try:
            try:
                return await self._complete_with_retries(key, self.primary_model, messages)
            except LLMError as exc:
                # Fallback only for provider-side failures, never for
                # auth/invalid-request/context errors.
                if exc.use_fallback and self.fallback_model != self.primary_model:
                    logger.warning(
                        "llm fallback activated primary=%s fallback=%s reason=%s",
                        self.primary_model,
                        self.fallback_model,
                        type(exc).__name__,
                    )
                    return await self._complete_with_retries(
                        key, self.fallback_model, messages
                    )
                raise
        finally:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.info("llm request finished elapsed_ms=%d", elapsed_ms)

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[StreamEvent]:
        """Streaming completion. Falls back to the backup model only if the
        primary fails *before* any token was produced (mid-stream switching
        would corrupt the answer)."""
        key = self._require_key()
        logger.info(
            "llm stream start model=%s messages=%d", self.primary_model, len(messages)
        )
        start = time.perf_counter()
        try:
            got_tokens = False
            try:
                async for ev in self._stream_with_retries(
                    key, self.primary_model, messages
                ):
                    if ev.kind == "token":
                        got_tokens = True
                    yield ev
            except LLMError as exc:
                if (
                    not got_tokens
                    and exc.use_fallback
                    and self.fallback_model != self.primary_model
                ):
                    logger.warning(
                        "llm stream fallback primary=%s fallback=%s reason=%s",
                        self.primary_model,
                        self.fallback_model,
                        type(exc).__name__,
                    )
                    yield StreamEvent(kind="fallback", model=self.fallback_model)
                    async for ev in self._stream_with_retries(
                        key, self.fallback_model, messages
                    ):
                        yield ev
                else:
                    raise
        finally:
            logger.info(
                "llm stream finished elapsed_ms=%d",
                int((time.perf_counter() - start) * 1000),
            )

    # -- non-streaming internals ------------------------------------------

    async def _complete_with_retries(
        self, key: str, model: str, messages: list[dict[str, str]]
    ) -> LLMResult:
        delay = self._s.retry_base_delay_seconds
        last_err: LLMError | None = None
        for attempt in range(1, self._s.max_retries + 1):
            try:
                return await self._post_chat(key, model, messages)
            except LLMError as exc:
                last_err = exc
                if not exc.retryable or attempt == self._s.max_retries:
                    logger.warning(
                        "llm attempt failed model=%s attempt=%d/%d error=%s (no retry)",
                        model,
                        attempt,
                        self._s.max_retries,
                        type(exc).__name__,
                    )
                    raise
                wait = exc.retry_after if exc.retry_after else delay
                logger.warning(
                    "llm attempt failed model=%s attempt=%d/%d error=%s retry_in=%.1fs",
                    model,
                    attempt,
                    self._s.max_retries,
                    type(exc).__name__,
                    wait,
                )
                await asyncio.sleep(wait)
                delay *= 2
        raise last_err or LLMProviderError("LLM request failed.")

    async def _post_chat(
        self, key: str, model: str, messages: list[dict[str, str]]
    ) -> LLMResult:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": self._s.max_output_tokens,
            "stream": False,
        }
        timeout = httpx.Timeout(
            connect=10.0,
            read=self._s.openrouter_timeout_seconds,
            write=10.0,
            pool=10.0,
        )
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    self._url,
                    headers=self._headers(key),
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise LLMNetworkError(str(exc)) from exc

        if resp.status_code == 429:
            raise LLMRateLimitError(
                resp.text[:300], retry_after=_parse_retry_after(resp.headers)
            )
        if resp.status_code != 200:
            raise _classify_http_error(resp.status_code, resp.text)

        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMProviderError("Provider returned non-JSON.") from exc

        text = self._extract_text(data)
        if not text.strip():
            raise LLMEmptyResponseError("Provider returned no content.")
        usage = TokenUsage.from_provider_dict(data.get("usage"))
        if usage:
            logger.info(
                "llm usage model=%s prompt=%d completion=%d total=%d",
                model,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )
        return LLMResult(text=text, model=data.get("model", model), usage=usage)

    @staticmethod
    def _extract_text(data: dict) -> str:
        try:
            choices = data.get("choices") or []
            if not choices:
                return ""
            msg = choices[0].get("message") or {}
            return msg.get("content") or ""
        except (AttributeError, IndexError, TypeError):
            return ""

    # -- streaming internals ----------------------------------------------

    async def _stream_with_retries(
        self, key: str, model: str, messages: list[dict[str, str]]
    ) -> AsyncIterator[StreamEvent]:
        delay = self._s.retry_base_delay_seconds
        last_err: LLMError | None = None
        for attempt in range(1, self._s.max_retries + 1):
            try:
                async for ev in self._post_stream(key, model, messages):
                    yield ev
                return
            except LLMError as exc:
                last_err = exc
                if not exc.retryable or attempt == self._s.max_retries:
                    raise
                wait = exc.retry_after if exc.retry_after else delay
                logger.warning(
                    "llm stream retry model=%s attempt=%d/%d error=%s retry_in=%.1fs",
                    model,
                    attempt,
                    self._s.max_retries,
                    type(exc).__name__,
                    wait,
                )
                await asyncio.sleep(wait)
                delay *= 2
        raise last_err or LLMProviderError("LLM stream failed.")

    async def _post_stream(
        self, key: str, model: str, messages: list[dict[str, str]]
    ) -> AsyncIterator[StreamEvent]:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": self._s.max_output_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        timeout = httpx.Timeout(
            connect=10.0,
            read=self._s.openrouter_timeout_seconds,
            write=10.0,
            pool=10.0,
        )
        full_text: list[str] = []
        usage: TokenUsage | None = None
        returned_model = model
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    self._url,
                    headers=self._headers(key),
                    json=payload,
                ) as resp:
                    if resp.status_code == 429:
                        body = await resp.aread()
                        raise LLMRateLimitError(
                            body.decode("utf-8", "ignore")[:300],
                            retry_after=_parse_retry_after(resp.headers),
                        )
                    if resp.status_code != 200:
                        body = await resp.aread()
                        raise _classify_http_error(
                            resp.status_code, body.decode("utf-8", "ignore")
                        )
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        chunk = line[len("data:"):].strip()
                        if chunk == "[DONE]":
                            break
                        try:
                            data = json.loads(chunk)
                        except ValueError:
                            continue  # skip malformed SSE lines
                        if "model" in data and isinstance(data["model"], str):
                            returned_model = data["model"]
                        chunk_usage = TokenUsage.from_provider_dict(data.get("usage"))
                        if chunk_usage and (
                            chunk_usage.total_tokens or chunk_usage.prompt_tokens
                        ):
                            usage = chunk_usage
                        for choice in data.get("choices") or []:
                            delta = (choice.get("delta") or {}).get("content")
                            if delta:
                                full_text.append(delta)
                                yield StreamEvent(
                                    kind="token", text=delta, model=returned_model
                                )
        except LLMError:
            raise
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise LLMNetworkError(str(exc)) from exc

        if not full_text:
            raise LLMEmptyResponseError("Provider stream produced no content.")
        yield StreamEvent(kind="done", model=returned_model, usage=usage)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _headers(key: str) -> dict[str, str]:
        # Never log these headers — they carry the secret.
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ai-study-assistant",
            "X-Title": "AI Study Assistant",
        }
