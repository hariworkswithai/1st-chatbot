"""Conversation history + orchestration between API routes and the LLM.

V1 keeps history in process memory (dict of session_id -> messages).
No database — a deliberate scope choice. History is trimmed so we never
send an unbounded context to the provider.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator

from app.config import Settings
from app.llm.base import LLMInvalidRequestError, LLMResult
from app.llm.openrouter import OpenRouterClient, StreamEvent
from app.prompts import build_messages

logger = logging.getLogger(__name__)

Message = dict[str, str]  # {"role": "user"|"assistant", "content": str}


class SessionStore:
    """Thread-safe in-memory session storage."""

    def __init__(self, max_history: int):
        self._sessions: dict[str, list[Message]] = {}
        self._lock = threading.Lock()
        self._max_history = max_history

    def create(self) -> str:
        session_id = uuid.uuid4().hex[:16]
        with self._lock:
            self._sessions[session_id] = []
        return session_id

    def get(self, session_id: str) -> list[Message]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def ensure(self, session_id: str | None) -> str:
        """Return a valid session id, creating one when needed."""
        if session_id:
            with self._lock:
                if session_id in self._sessions:
                    return session_id
                self._sessions[session_id] = []
                return session_id
        return self.create()

    def append(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            history = self._sessions.setdefault(session_id, [])
            history.append({"role": role, "content": content})
            # Keep only the most recent N messages (user+assistant pairs).
            if len(history) > self._max_history:
                del history[: len(history) - self._max_history]

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions[session_id] = []

    def history_for_prompt(self, session_id: str) -> list[Message]:
        return self.get(session_id)[-self._max_history :]


class ChatService:
    def __init__(self, settings: Settings):
        self._s = settings
        self.sessions = SessionStore(max_history=settings.max_history_messages)
        self.client = OpenRouterClient(settings)

    def _validate(self, message: str) -> str:
        text = (message or "").strip()
        if not text:
            raise LLMInvalidRequestError("Message is empty.")
        if len(message) > self._s.max_input_chars:
            raise LLMInvalidRequestError(
                f"Message too long ({len(message)} chars, "
                f"max {self._s.max_input_chars})."
            )
        return text

    async def chat(self, message: str, session_id: str | None) -> tuple[LLMResult, str]:
        """Non-streaming chat: validate -> build prompt -> call LLM -> store."""
        text = self._validate(message)
        sid = self.sessions.ensure(session_id)
        history = self.sessions.history_for_prompt(sid)
        prompt = build_messages(history, text)

        started = time.perf_counter()
        result = await self.client.complete(prompt)
        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "chat done session=%s model=%s latency_ms=%d reply_chars=%d",
            sid[:8],
            result.model,
            latency_ms,
            len(result.text),
        )

        # Only store after a successful reply so failed calls don't pollute history.
        self.sessions.append(sid, "user", text)
        self.sessions.append(sid, "assistant", result.text)
        return result, sid

    async def stream_chat(
        self, message: str, session_id: str | None
    ) -> tuple[AsyncIterator[StreamEvent], str]:
        """Validate + build prompt now; stream tokens; store full reply at end."""
        text = self._validate(message)
        sid = self.sessions.ensure(session_id)
        history = self.sessions.history_for_prompt(sid)
        prompt = build_messages(history, text)

        async def _gen() -> AsyncIterator[StreamEvent]:
            collected: list[str] = []
            async for ev in self.client.stream(prompt):
                if ev.kind == "token":
                    collected.append(ev.text)
                yield ev
            if collected:
                self.sessions.append(sid, "user", text)
                self.sessions.append(sid, "assistant", "".join(collected))

        return _gen(), sid
