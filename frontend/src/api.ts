// Thin wrapper around the FastAPI backend. The browser never sees any API key.

export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface StreamResult {
  sessionId: string;
  model: string;
  usage: TokenUsage | null;
  fellBack: boolean;
}

export async function getPublicConfig(): Promise<{
  model: string;
  has_fallback: boolean;
}> {
  const res = await fetch("/api/config");
  if (!res.ok) throw new Error("Could not load app config.");
  return res.json();
}

export async function resetConversation(sessionId: string): Promise<void> {
  await fetch("/api/chat/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
}

interface StreamCallbacks {
  onMeta: (sessionId: string) => void;
  onToken: (text: string) => void;
  onFallback: (model: string) => void;
}

/** POST /api/chat/stream and forward SSE events to callbacks. */
export async function streamChat(
  message: string,
  sessionId: string | null,
  cb: StreamCallbacks,
  signal: AbortSignal,
): Promise<StreamResult> {
  const res = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId }),
    signal,
  });

  // Non-stream errors (validation, app rate limit) come back as plain JSON.
  const contentType = res.headers.get("content-type") ?? "";
  if (!res.ok || !contentType.includes("text/event-stream")) {
    let detail = `Request failed (${res.status}).`;
    try {
      const data = await res.json();
      if (data?.error) detail = data.error;
    } catch {
      /* keep default */
    }
    throw new Error(detail);
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let session = sessionId ?? "";
  let model = "";
  let usage: TokenUsage | null = null;
  let fellBack = false;

  const handleEvent = (raw: string) => {
    const line = raw.trim();
    if (!line.startsWith("data:")) return;
    const payload = JSON.parse(line.slice("data:".length).trim());
    switch (payload.type) {
      case "meta":
        session = payload.session_id ?? session;
        cb.onMeta(session);
        break;
      case "token":
        cb.onToken(payload.text ?? "");
        break;
      case "fallback":
        fellBack = true;
        model = payload.model ?? model;
        cb.onFallback(model);
        break;
      case "done":
        session = payload.session_id ?? session;
        model = payload.model ?? model;
        usage = payload.usage ?? null;
        break;
      case "error":
        throw new Error(payload.error ?? "The AI service failed.");
    }
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // SSE events are separated by a blank line.
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const part of parts) handleEvent(part);
  }
  if (buffer.trim()) handleEvent(buffer);

  return { sessionId: session, model, usage, fellBack };
}
