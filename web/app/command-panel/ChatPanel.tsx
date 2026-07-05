"use client";

import * as React from "react";
import type { UIMessage } from "ai";
import type { ComponentRegistry } from "@nikolayvalev/command-panel";
import {
  DEFAULT_TITLE,
  createChat,
  deleteChat,
  deriveTitle,
  renameChat,
  selectChat,
  touchChat,
  type ChatIndex,
} from "@/lib/chatStore";
import { firstUserText, initIndex, newChatId, removeMessages, saveIndex } from "./chatStorage";
import { ghostBtn } from "./panelChrome";
import { ChatSession } from "./ChatSession";
import type { ImproveRequest } from "./types";

// The conversation switcher: owns the conversation index (select / new /
// rename / delete) and mounts one ChatSession per active conversation
// (key=id remounts useChat with that conversation's history).
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

  React.useEffect(() => saveIndex(index), [index]);

  const active = index.chats.find((c) => c.id === index.activeId) ?? index.chats[0];

  const onNew = () => {
    const { index: next, evictedIds } = createChat(index, newChatId(), new Date().toISOString());
    evictedIds.forEach(removeMessages);
    setIndex(next);
    setRenaming(false);
  };

  const onDelete = () => {
    if (!window.confirm(`Delete “${active.title}” and its history?`)) return;
    removeMessages(active.id);
    setIndex(deleteChat(index, active.id) ?? createChat(null, newChatId(), new Date().toISOString()).index);
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
        <button
          type="button"
          aria-label="New conversation"
          title="New conversation"
          onClick={onNew}
          className={`${ghostBtn} hover:text-[var(--text)]`}
        >
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
          className={`${ghostBtn} hover:text-[var(--text)]`}
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
