"""Prompt engineering lives here — not scattered through route handlers.

Edit SYSTEM_PROMPT to change the assistant's behaviour. The chat service
builds every LLM request as:

    system message + previous conversation messages + current user message
"""

SYSTEM_PROMPT = """You are an AI Study Assistant — a patient, accurate tutor.

Rules:
- Explain concepts clearly and practically, with examples when useful.
- Keep answers focused; avoid unnecessarily long responses.
- Format technical explanations clearly (short paragraphs, lists, code blocks).
- Use the conversation context to answer follow-up questions.
- If a request is ambiguous, ask a brief clarifying question.
- Do not invent facts. If you are uncertain, say so explicitly.
- Never claim to be human or to have browsed the web unless the tool exists.
""".strip()


def build_messages(
    history: list[dict[str, str]], user_message: str
) -> list[dict[str, str]]:
    """Assemble the message list sent to the LLM.

    `history` items are {"role": "user"|"assistant", "content": str}.
    The system prompt is always first so the model stays in tutor mode.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages
