"""History / prompt / validation tests. No API key needed."""

import pytest

from app.chat_service import ChatService
from app.config import Settings
from app.llm.base import LLMInvalidRequestError, LLMResult
from app.prompts import SYSTEM_PROMPT, build_messages


def make_service(**overrides) -> ChatService:
    defaults: dict = {
        "openrouter_api_key": "test-key",
        "max_history_messages": 6,
        "max_input_chars": 100,
    }
    defaults.update(overrides)
    return ChatService(Settings(**defaults))


def test_build_messages_starts_with_system_prompt():
    history = [{"role": "user", "content": "What is 2+2?"}]
    msgs = build_messages(history, "And 3+3?")
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert msgs[1] == {"role": "user", "content": "What is 2+2?"}
    assert msgs[-1] == {"role": "user", "content": "And 3+3?"}


def test_empty_message_rejected():
    svc = make_service()
    with pytest.raises(LLMInvalidRequestError):
        svc._validate("   ")


def test_oversized_message_rejected():
    svc = make_service(max_input_chars=10)
    with pytest.raises(LLMInvalidRequestError):
        svc._validate("x" * 11)


async def test_history_trimmed_to_max(monkeypatch):
    svc = make_service(max_history_messages=4)

    async def fake_complete(messages):
        return LLMResult(text="reply", model="m")

    monkeypatch.setattr(svc.client, "complete", fake_complete)
    sid = svc.sessions.create()
    for i in range(10):
        await svc.chat(f"question {i}", sid)
    # 4 max => last turns only; never unbounded.
    assert len(svc.sessions.get(sid)) <= 4


async def test_conversation_context_sent_to_model(monkeypatch):
    svc = make_service()
    seen: list[list[dict]] = []

    async def capture(messages):
        seen.append(messages)
        return LLMResult(text="ok", model="m")

    monkeypatch.setattr(svc.client, "complete", capture)
    sid = svc.sessions.create()
    await svc.chat("What is photosynthesis?", sid)
    await svc.chat("Give an example", sid)
    second_call = seen[1]
    contents = [m["content"] for m in second_call]
    # Follow-up call includes the earlier exchange.
    assert any("photosynthesis" in c for c in contents)
    assert second_call[0]["role"] == "system"


async def test_sessions_are_isolated(monkeypatch):
    svc = make_service()

    async def fake_complete(messages):
        return LLMResult(text="reply", model="m")

    monkeypatch.setattr(svc.client, "complete", fake_complete)
    sid_a = svc.sessions.create()
    sid_b = svc.sessions.create()
    await svc.chat("topic A", sid_a)
    history_b = svc.sessions.get(sid_b)
    assert all("topic A" not in m["content"] for m in history_b)


async def test_failed_call_does_not_pollute_history(monkeypatch):
    from app.llm.base import LLMProviderError

    svc = make_service()

    async def boom(messages):
        raise LLMProviderError("down")

    monkeypatch.setattr(svc.client, "complete", boom)
    sid = svc.sessions.create()
    with pytest.raises(LLMProviderError):
        await svc.chat("hello", sid)
    assert svc.sessions.get(sid) == []


def test_reset_clears_history():
    svc = make_service()
    sid = svc.sessions.create()
    svc.sessions.append(sid, "user", "hi")
    svc.reset(sid) if hasattr(svc, "reset") else svc.sessions.reset(sid)
    assert svc.sessions.get(sid) == []
