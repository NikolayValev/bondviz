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
import {
  DEFAULT_TITLE,
  createChat,
  deleteChat,
  deriveTitle,
  parseChatIndex,
  renameChat,
  selectChat,
  touchChat,
  type ChatIndex,
} from "@/lib/chatStore";

// Multiple persisted conversations ("workflows"): an index of conversations in
// localStorage plus one message-history key per conversation. Each settled
// turn is mirrored to storage and seeded back into useChat on load; switching
// conversations remounts the session (key=id) with that history.
const INDEX_KEY = "command-panel:chats:v1";
const LEGACY_CHAT_KEY = "command-panel:chat:v1"; // pre-conversations history
const MAX_STORED_MESSAGES = 40;
const messagesKey = (id: string) => `command-panel:chat:${id}:v1`;

/** A dashboard widget sent to the chat to be improved (nonce re-triggers repeats). */
export interface ImproveRequest {
  widget: PinnedWidget;
  nonce: number;
}

function loadMessages(id: string): UIMessage[] {
  try {
    const raw: unknown = JSON.parse(localStorage.getItem(messagesKey(id)) ?? "[]");
    return Array.isArray(raw) ? (raw as UIMessage[]) : [];
  } catch {
    return [];
  }
}

function firstUserText(messages: UIMessage[]): string | null {
  for (const m of messages) {
    if (m.role !== "user") continue;
    for (const p of m.parts) {
      if (p.type === "text" && p.text.trim()) return p.text;
    }
  }
  return null;
}

function initIndex(): ChatIndex {
  const existing = parseChatIndex(localStorage.getItem(INDEX_KEY));
  if (existing) return existing;
  const created = createChat(null, crypto.randomUUID(), new Date().toISOString()).index;
  // One-time migration of the single-conversation history from iteration 2.
  try {
    const legacy = localStorage.getItem(LEGACY_CHAT_KEY);
    if (legacy) {
      localStorage.setItem(messagesKey(created.activeId), legacy);
      localStorage.removeItem(LEGACY_CHAT_KEY);
      const first = firstUserText(JSON.parse(legacy) as UIMessage[]);
      if (first) return renameChat(created, created.activeId, deriveTitle(first));
    }
  } catch {
    /* keep the fresh conversation */
  }
  return created;
}

function toProposal(id: string, input: unknown): WidgetProposal {
  const p = input as { title: string; description: string; jsx: string; dataSources: string[] };
  return { id, title: p.title, description: p.description, jsx: p.jsx, dataSources: p.dataSources };
}

const ghostBtn =
  "rounded px-1.5 py-0.5 text-xs text-[var(--muted)] transition-colors hover:bg-white/5 hover:text-[var(--text)] disabled:opacity-30";

export function ChatPanel({
  registry,
  improve = null,
}: {
  registry: ComponentRegistry;
  improve?: ImproveRequest | null;
}) {
  const [index, setIndex] = React.useState<ChatIndex>(initIndex);
  const [renaming, setRenaming] = React.useState(false);
  const [draftTitle, setDraftTitle] = React.useState("");

  React.useEffect(() => {
    try {
      localStorage.setItem(INDEX_KEY, JSON.stringify(index));
    } catch {
      /* storage unavailable — conversations just won't persist */
    }
  }, [index]);

  const active = index.chats.find((c) => c.id === index.activeId) ?? index.chats[0];

  const onNew = () => {
    const { index: next, evictedIds } = createChat(index, crypto.randomUUID(), new Date().toISOString());
    for (const id of evictedIds) {
      try {
        localStorage.removeItem(messagesKey(id));
      } catch {}
    }
    setIndex(next);
    setRenaming(false);
  };

  const onDelete = () => {
    if (!window.confirm(`Delete “${active.title}” and its history?`)) return;
    try {
      localStorage.removeItem(messagesKey(active.id));
    } catch {}
    setIndex(
      deleteChat(index, active.id) ??
        createChat(null, crypto.randomUUID(), new Date().toISOString()).index,
    );
    setRenaming(false);
  };

  const commitRename = () => {
    setIndex((prev) => renameChat(prev, active.id, draftTitle));
    setRenaming(false);
  };

  // A settled turn bumps recency and titles a fresh conversation from its
  // first user message.
  const onSettled = React.useCallback((id: string, messages: UIMessage[]) => {
    setIndex((prev) => {
      let next = touchChat(prev, id, new Date().toISOString());
      if (next.chats.find((c) => c.id === id)?.title === DEFAULT_TITLE) {
        const first = firstUserText(messages);
        if (first) next = renameChat(next, id, deriveTitle(first));
      }
      return next;
    });
  }, []);

  return (
    <div className="flex h-[65dvh] min-h-0 flex-col rounded-lg border border-[var(--panel-border)] bg-[var(--panel-2)] p-3 lg:h-full">
      <div className="mb-2 flex items-center gap-1.5 border-b border-[var(--panel-border)] pb-2">
        {renaming ? (
          <input
            autoFocus
            aria-label="Conversation name"
            value={draftTitle}
            onChange={(e) => setDraftTitle(e.target.value)}
            onBlur={commitRename}
            onKeyDown={(e) => {
              if (e.key === "Enter") commitRename();
              if (e.key === "Escape") setRenaming(false);
            }}
            className="min-w-0 flex-1 rounded border border-[var(--panel-border-strong)] bg-[var(--bg-2)] px-2 py-1 text-sm text-[var(--text)]"
            style={{ fontFamily: "var(--font-sans)" }}
          />
        ) : (
          <select
            aria-label="Conversation"
            value={active.id}
            onChange={(e) => setIndex((prev) => selectChat(prev, e.target.value))}
            className="min-w-0 flex-1 rounded border border-[var(--panel-border)] bg-[var(--bg-2)] px-2 py-1 text-sm text-[var(--text)]"
          >
            {index.chats.map((c) => (
              <option key={c.id} value={c.id}>
                {c.title}
              </option>
            ))}
          </select>
        )}
        <button type="button" aria-label="New conversation" title="New conversation" onClick={onNew} className={ghostBtn}>
          ＋
        </button>
        <button
          type="button"
          aria-label="Rename conversation"
          title="Rename conversation"
          onClick={() => {
            setDraftTitle(active.title);
            setRenaming(true);
          }}
          className={ghostBtn}
        >
          ✎
        </button>
        <button
          type="button"
          aria-label="Delete conversation"
          title="Delete conversation"
          onClick={onDelete}
          className={`${ghostBtn} hover:text-[var(--neg)]`}
        >
          ✕
        </button>
      </div>

      <ChatSession key={active.id} id={active.id} registry={registry} improve={improve} onSettled={onSettled} />
    </div>
  );
}

function ChatSession({
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
    try {
      localStorage.setItem(messagesKey(id), JSON.stringify(messages.slice(-MAX_STORED_MESSAGES)));
    } catch {
      /* storage unavailable — chat just won't persist */
    }
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
          <p className="m-auto max-w-[28ch] text-center text-sm text-[var(--faint)]">
            Ask for a chart or stat — e.g. “chart the treasury yield curve”. Each conversation keeps
            its own history.
          </p>
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
