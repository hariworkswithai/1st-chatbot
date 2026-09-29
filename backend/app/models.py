"""Pydantic request/response models for the chat API."""

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """POST /api/chat body."""

    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(
        default=None, max_length=64, description="Omit to start a new session."
    )


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    model: str
    usage: TokenUsage | None = None
    latency_ms: int = 0


class ResetRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)


class ResetResponse(BaseModel):
    session_id: str
    cleared: bool = True


class ErrorResponse(BaseModel):
    error: str
    code: str = "provider_error"
