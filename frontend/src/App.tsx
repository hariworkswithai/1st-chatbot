import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  getPublicConfig,
  resetConversation,
  streamChat,
  type TokenUsage,
} from "./api";
import "./styles.css";

interface Message {
  id: number;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
  model?: string;
  usage?: TokenUsage | null;
  latencyMs?: number;
  fellBack?: boolean;
}

const SESSION_KEY = "asa_session_id";
let nextId = 1;

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false); // true between send and first token
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [model, setModel] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(() =>
    localStorage.getItem(SESSION_KEY),
  );
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getPublicConfig()
      .then((cfg) => setModel(cfg.model))
      .catch(() => setModel(""));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Abort an in-flight stream if the user navigates away.
  useEffect(() => () => abortRef.current?.abort(), []);

  const updateAssistant = (id: number, patch: Partial<Message>) =>
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch } : m)));

  async function send() {
    const text = input.trim();
    if (!text || loading || streaming) return; // ignore empty / double sends
    setError(null);
    setInput("");
    setLoading(true);

    const userMsg: Message = { id: nextId++, role: "user", content: text };
    const assistantId = nextId++;
    setMessages((prev) => [
      ...prev,
      userMsg,
      { id: assistantId, role: "assistant", content: "", streaming: true },
    ]);

    const started = performance.now();
    const controller = new AbortController();
    abortRef.current = controller;
    let collected = "";
    let sawToken = false;

    try {
      const result = await streamChat(
        text,
        sessionId,
        {
          onMeta: (sid) => {
            setSessionId(sid);
            localStorage.setItem(SESSION_KEY, sid);
          },
          onToken: (t) => {
            if (!sawToken) {
              sawToken = true;
              setLoading(false);
              setStreaming(true);
            }
            collected += t;
            const snapshot = collected;
            updateAssistant(assistantId, { content: snapshot });
          },
          onFallback: (fallbackModel) =>
            updateAssistant(assistantId, { model: fallbackModel, fellBack: true }),
        },
        controller.signal,
      );
      const latencyMs = Math.round(performance.now() - started);
      updateAssistant(assistantId, {
        content: collected || "(empty response — please try again)",
        streaming: false,
        model: result.model || model,
        usage: result.usage,
        latencyMs,
        fellBack: result.fellBack,
      });
    } catch (e) {
      const message =
        e instanceof DOMException && e.name === "AbortError"
          ? "Request was interrupted."
          : e instanceof Error
            ? e.message
            : "Something went wrong.";
      setError(message);
      // Remove the empty assistant placeholder on hard failure.
      setMessages((prev) =>
        sawToken
          ? prev.map((m) =>
              m.id === assistantId ? { ...m, streaming: false } : m,
            )
          : prev.filter((m) => m.id !== assistantId),
      );
    } finally {
      setLoading(false);
      setStreaming(false);
      abortRef.current = null;
    }
  }

  async function clearChat() {
    abortRef.current?.abort();
    if (sessionId) {
      try {
        await resetConversation(sessionId);
      } catch {
        /* backend reset is best-effort; still clear the UI */
      }
    }
    const fresh: string | null = null;
    setSessionId(fresh);
    localStorage.removeItem(SESSION_KEY);
    setMessages([]);
    setError(null);
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  }

  const busy = loading || streaming;

  return (
    <div className="page">
      <header className="header">
        <div>
          <h1>AI Study Assistant</h1>
          <p className="subtitle">Ask concepts, get clear explanations with examples.</p>
        </div>
        <div className="header-right">
          {model && <span className="model-badge" title="Primary LLM">{model}</span>}
          <button className="ghost" onClick={() => void clearChat()} disabled={busy && messages.length === 0}>
            Clear chat
          </button>
        </div>
      </header>

      <main className="chat" aria-live="polite">
        {messages.length === 0 && (
          <div className="empty">
            <p>👋 Hi! I&apos;m your study buddy.</p>
            <p>Try: “Explain recursion with a simple example.”</p>
          </div>
        )}
        {messages.map((m) => (
          <div key={m.id} className={`bubble ${m.role}`}>
            <div className="role">{m.role === "user" ? "You" : "Assistant"}</div>
            <div className="content">
              {m.role === "assistant" ? (
                <>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {m.content || (m.streaming ? "" : "(empty response — please try again)")}
                  </ReactMarkdown>
                  {m.streaming && <span className="caret">▍</span>}
                </>
              ) : (
                m.content
              )}
            </div>
            {m.role === "assistant" && !m.streaming && (m.model || m.usage) && (
              <div className="meta">
                {m.fellBack && <span className="fallback">fallback · </span>}
                {m.model && <span>{m.model}</span>}
                {m.usage && m.usage.total_tokens > 0 && (
                  <span> · {m.usage.total_tokens} tokens</span>
                )}
                {m.latencyMs != null && <span> · {(m.latencyMs / 1000).toFixed(1)}s</span>}
              </div>
            )}
          </div>
        ))}
        {loading && <div className="thinking">Thinking…</div>}
        {error && <div className="error">⚠️ {error}</div>}
        <div ref={bottomRef} />
      </main>

      <footer className="composer">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask a study question… (Enter to send, Shift+Enter for new line)"
          rows={2}
          maxLength={4000}
          disabled={busy}
        />
        <button onClick={() => void send()} disabled={busy || !input.trim()}>
          {busy ? "…" : "Send"}
        </button>
      </footer>
    </div>
  );
}
