"use client";

import * as React from "react";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport, getToolName, isToolUIPart, type UIMessage } from "ai";
import {
  WidgetPreviewCard,
  usePinnedStore,
  type ComponentRegistry,
  type PinnedWidget,
  type WidgetProposal,
} from "@nikolayvalev/command-panel";
import { loadMessages, saveMessages } from "./chatStorage";
import type { ImproveRequest } from "./types";

// One conversation's chat UI: messages, widget proposals, composer. Mounted
// with key=conversation id (see ChatPanel) so switching conversations swaps
// the whole useChat state, seeded from that conversation's stored history.

// Starter prompts for an empty conversation: one per registered data source, so a first-time
// visitor sees what the panel can do and gets a working widget in one click.
const SUGGESTIONS = [
  "Chart today's Treasury yield curve",
  "Show the 2s10s spread over the last two years",
  "Compare carry and roll-down across tenors",
  "Plot the bootstrapped zero curve",
];

function toProposal(id: string, input: unknown): WidgetProposal {
  const p = input as { title: string; description: string; jsx: string; dataSources: string[] };
  return { id, title: p.title, description: p.description, jsx: p.jsx, dataSources: p.dataSources };
}

export function ChatSession({
  id,
  registry,
  improve,
  onSettled,
}: {
  id: string;
  registry: ComponentRegistry;
  improve: ImproveRequest | null;
  onSettled: (id: string, messages: UIMessage[]) => void;
}) {
  const transport = React.useMemo(() => new DefaultChatTransport({ api: "/api/command-panel" }), []);
  const [initialMessages] = React.useState<UIMessage[]>(() => loadMessages(id));
  const { messages, sendMessage, status, error } = useChat({ transport, messages: initialMessages });
  const { widgets, pin } = usePinnedStore();
  const pinnedIds = React.useMemo(() => new Set(widgets.map((w) => w.id)), [widgets]);
  const [input, setInput] = React.useState("");
  const [attached, setAttached] = React.useState<PinnedWidget | null>(null);
  const busy = status === "submitted" || status === "streaming";
  const scrollRef = React.useRef<HTMLDivElement | null>(null);
  const inputRef = React.useRef<HTMLInputElement | null>(null);

  // "Improve in chat" from the dashboard: attach the widget and suggest a
  // prompt. State is adjusted during render (guarded by nonce) per the React
  // "derive state from props" pattern; seeding lastNonce swallows requests
  // that predate this session (e.g. after a conversation switch).
  const [lastNonce, setLastNonce] = React.useState<number | null>(improve?.nonce ?? null);
  if (improve && improve.nonce !== lastNonce) {
    setLastNonce(improve.nonce);
    setAttached(improve.widget);
    setInput((prev) => (prev.trim() ? prev : "What would improve this widget? What else could I add to it?"));
  }

  // Persist once a turn settles (not on every streamed token).
  React.useEffect(() => {
    if (status !== "ready" && status !== "error") return;
    saveMessages(id, messages);
    if (messages.length > 0) onSettled(id, messages);
  }, [messages, status, id, onSettled]);

  React.useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages.length, busy]);

  React.useEffect(() => {
    if (improve) inputRef.current?.focus();
  }, [improve]);

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || busy) return;
    const composed = attached
      ? `${text}\n\nAttached pinned widget “${attached.title}” — improve or extend THIS widget and propose the updated version:\n\`\`\`jsx\n${attached.jsx}\n\`\`\`\nData sources used: ${attached.dataSources.join(", ") || "none"}`
      : text;
    sendMessage({ text: composed });
    setInput("");
    setAttached(null);
  };

  return (
    <>
      <div ref={scrollRef} className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto pr-1">
        {messages.length === 0 && (
          <div className="m-auto flex max-w-sm flex-col items-center gap-3 text-center">
            <p className="text-sm text-[var(--faint)]">
              Ask for a chart or a stat in plain English, or start with one of these. Each
              conversation keeps its own history.
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              {SUGGESTIONS.map((text) => (
                <button
                  key={text}
                  type="button"
                  disabled={busy}
                  onClick={() => sendMessage({ text })}
                  className="rounded-full border border-[var(--panel-border-strong)] px-3 py-1.5 text-xs text-[var(--muted)] transition-colors hover:bg-[var(--accent-dim)] hover:text-[var(--text)] disabled:opacity-40"
                >
                  {text}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((message) =>
          message.parts.map((part, i) => {
            if (part.type === "text") {
              return message.role === "user" ? (
                <p
                  key={`${message.id}-${i}`}
                  className="ml-8 self-end whitespace-pre-wrap break-words rounded-lg bg-[var(--accent-dim)] px-3 py-2 text-sm text-[var(--text)]"
                >
                  {part.text}
                </p>
              ) : (
                <p key={`${message.id}-${i}`} className="mr-4 whitespace-pre-wrap text-sm text-[var(--muted)]">
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

      {attached && (
        <div className="mt-2 flex items-center gap-2 rounded border border-[var(--panel-border-strong)] bg-[var(--accent-dim)] px-2 py-1 text-xs text-[var(--text)]">
          <span aria-hidden="true" className="text-[var(--accent)]">
            ✦
          </span>
          <span className="min-w-0 flex-1 truncate">Improving: {attached.title}</span>
          <button
            type="button"
            aria-label="Detach widget"
            onClick={() => setAttached(null)}
            className="text-[var(--muted)] transition-colors hover:text-[var(--text)]"
          >
            ✕
          </button>
        </div>
      )}

      <form onSubmit={onSubmit} className="mt-3 flex gap-2">
        <input
          ref={inputRef}
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
    </>
  );
}
