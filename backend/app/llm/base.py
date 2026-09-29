"""Shared LLM-provider types and error taxonomy.

The rest of the app only depends on these, never on OpenRouter details.
That is the whole "provider abstraction": small and honest.
"""

from dataclasses import dataclass, field


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_provider_dict(cls, data: dict | None) -> "TokenUsage | None":
        if not isinstance(data, dict):
            return None
        try:
            return cls(
                prompt_tokens=int(data.get("prompt_tokens", 0) or 0),
                completion_tokens=int(data.get("completion_tokens", 0) or 0),
                total_tokens=int(data.get("total_tokens", 0) or 0),
            )
        except (TypeError, ValueError):
            return None


@dataclass
class LLMResult:
    text: str
    model: str
    usage: TokenUsage | None = None


# --- Error taxonomy -------------------------------------------------------
# Each maps to a safe HTTP status + user-friendly message in main.py.
# Only retryable errors are retried; only some trigger model fallback.


class LLMError(Exception):
    """Base class. `retryable` decides retry; `use_fallback` decides fallback."""

    retryable: bool = False
    use_fallback: bool = False
    status_code: int = 502
    code: str = "provider_error"
    user_message: str = "The AI service is temporarily unavailable. Try again shortly."

    def __init__(self, message: str = "", *, retry_after: float | None = None):
        super().__init__(message or self.user_message)
        self.retry_after = retry_after


class LLMAuthError(LLMError):
    """Bad/missing API key. Never retry, never fall back (fallback would fail too)."""

    retryable = False
    use_fallback = False
    status_code = 500  # 500, not 401: don't leak backend-key problems to browsers
    code = "server_misconfigured"
    user_message = "The AI service is not configured correctly. Contact the administrator."


class LLMInvalidRequestError(LLMError):
    retryable = False
    use_fallback = False
    status_code = 400
    code = "invalid_request"
    user_message = "That request was rejected. Try rephrasing or shortening it."


class LLMContextLengthError(LLMError):
    """Conversation too long for the model's context window."""

    retryable = False
    use_fallback = False
    status_code = 400
    code = "context_too_long"
    user_message = (
        "The conversation grew too long for the model. "
        "Clear the chat and try again with a shorter prompt."
    )


class LLMRateLimitError(LLMError):
    retryable = True
    use_fallback = True
    status_code = 429
    code = "rate_limited"
    user_message = "The AI service is busy (rate limit). Please wait a moment and retry."


class LLMTimeoutError(LLMError):
    retryable = True
    use_fallback = True
    status_code = 504
    code = "timeout"
    user_message = "The AI service took too long to respond. Please try again."


class LLMProviderError(LLMError):
    """5xx / unexpected provider failures."""

    retryable = True
    use_fallback = True
    status_code = 502
    code = "provider_error"
    user_message = "The AI service is temporarily unavailable. Try again shortly."


class LLMNetworkError(LLMError):
    retryable = True
    use_fallback = True
    status_code = 502
    code = "network_error"
    user_message = "Could not reach the AI service. Check your connection and retry."


class LLMEmptyResponseError(LLMError):
    retryable = False
    use_fallback = True  # an empty reply from primary may succeed on fallback
    status_code = 502
    code = "empty_response"
    user_message = "The AI service returned an empty response. Please try again."


def is_context_length_message(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "context length",
            "context_length",
            "maximum context",
            "too many tokens",
            "token limit",
            "context window",
        )
    )
