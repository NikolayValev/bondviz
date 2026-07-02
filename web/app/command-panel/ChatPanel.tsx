"use client";

import * as React from "react";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport, getToolName, isToolUIPart, type UIMessage } from "ai";
import {
  WidgetPreviewCard,
  usePinnedStore,
  type ComponentRegistry,
  type WidgetProposal,
} from "@nikolayvalev/command-panel";

// Replaces the package's ChatPane so the conversation survives reloads:
// settled messages are mirrored to localStorage and seeded back into useChat.
const CHAT_KEY = "command-panel:chat:v1";
const MAX_STORED_MESSAGES = 40;

function loadMessages(): UIMessage[] {
  try {
    const raw: unknown = JSON.parse(localStorage.getItem(CHAT_KEY) ?? "[]");
    return Array.isArray(raw) ? (raw as UIMessage[]) : [];
  } catch {
    return [];
  }
}

function toProposal(id: string, input: unknown): WidgetProposal {
  const p = input as { title: string; description: string; jsx: string; dataSources: string[] };
  return { id, title: p.title, description: p.description, jsx: p.jsx, dataSources: p.dataSources };
}

export function ChatPanel({ registry }: { registry: ComponentRegistry }) {
  const transport = React.useMemo(() => new DefaultChatTransport({ api: "/api/command-panel" }), []);
  const [initialMessages] = React.useState(loadMessages);
  const { messages, sendMessage, setMessages, status, error } = useChat({
    transport,
    messages: initialMessages,
  });
  const { widgets, pin } = usePinnedStore();
  const pinnedIds = React.useMemo(() => new Set(widgets.map((w) => w.id)), [widgets]);
  const [input, setInput] = React.useState("");
  const busy = status === "submitted" || status === "streaming";
  const scrollRef = React.useRef<HTMLDivElement | null>(null);

  // Persist once a turn settles (not on every streamed token).
  React.useEffect(() => {
    if (status !== "ready" && status !== "error") return;
    try {
      localStorage.setItem(CHAT_KEY, JSON.stringify(messages.slice(-MAX_STORED_MESSAGES)));
    } catch {
      /* storage unavailable — chat just won't persist */
    }
  }, [messages, status]);

  React.useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages.length, busy]);

  const clear = () => {
    setMessages([]);
    try {
      localStorage.removeItem(CHAT_KEY);
    } catch {}
  };

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || busy) return;
    sendMessage({ text });
    setInput("");
  };

  return (
    <div className="flex h-[65dvh] min-h-0 flex-col rounded-lg border border-[var(--panel-border)] bg-[var(--panel-2)] p-3 lg:h-full">
      <div className="mb-2 flex items-center justify-between gap-2 border-b border-[var(--panel-border)] pb-2">
        <span className="eyebrow">Chat</span>
        {messages.length > 0 && (
          <button
            type="button"
            onClick={clear}
            className="rounded px-2 py-0.5 text-xs text-[var(--muted)] transition-colors hover:bg-white/5 hover:text-[var(--text)]"
          >
            Clear
          </button>
        )}
      </div>

      <div ref={scrollRef} className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto pr-1">
        {messages.length === 0 && (
          <p className="m-auto max-w-[26ch] text-center text-sm text-[var(--faint)]">
            Ask for a chart or stat — e.g. “chart the treasury yield curve”.
          </p>
        )}
        {messages.map((message) =>
          message.parts.map((part, i) => {
            if (part.type === "text") {
              return message.role === "user" ? (
                <p
                  key={`${message.id}-${i}`}
                  className="ml-8 self-end rounded-lg bg-[var(--accent-dim)] px-3 py-2 text-sm text-[var(--text)]"
                >
                  {part.text}
                </p>
              ) : (
                <p key={`${message.id}-${i}`} className="mr-4 text-sm text-[var(--muted)]">
                  {part.text}
                </p>
              );
            }
            if (
              isToolUIPart(part) &&
              getToolName(part) === "propose_widget" &&
              (part.state === "input-available" || part.state === "output-available")
            ) {
              const proposal = toProposal(part.toolCallId, part.input);
              return (
                <WidgetPreviewCard
                  key={part.toolCallId}
                  proposal={proposal}
                  registry={registry}
                  onPin={pin}
                  isPinned={pinnedIds.has(proposal.id)}
                />
              );
            }
            return null;
          }),
        )}
        {busy && <p className="text-xs text-[var(--faint)]">Thinking…</p>}
        {error && (
          <p role="alert" className="text-sm text-[var(--neg)]">
            {error.message}
          </p>
        )}
      </div>

      <form onSubmit={onSubmit} className="mt-3 flex gap-2">
        <input
          aria-label="Message"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask for a chart or a stat…"
          disabled={busy}
          className="min-w-0 flex-1 rounded-lg border border-[var(--panel-border-strong)] bg-[var(--bg-2)] px-3 py-2 text-sm text-[var(--text)] placeholder:text-[var(--faint)]"
          style={{ fontFamily: "var(--font-sans)" }}
        />
        <button
          type="submit"
          disabled={busy || input.trim() === ""}
          className="rounded-lg bg-[var(--accent)] px-3 py-2 text-sm font-medium text-[var(--on-accent)] transition-opacity disabled:opacity-40"
        >
          Send
        </button>
      </form>
    </div>
  );
}
