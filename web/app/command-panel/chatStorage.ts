// Browser-only localStorage I/O for the chat: one conversation index plus one
// message-history key per conversation. The pure index logic lives in
// lib/chatStore; components go through these helpers instead of localStorage.

import type { UIMessage } from "ai";
import {
  createChat,
  deriveTitle,
  parseChatIndex,
  renameChat,
  type ChatIndex,
} from "@/lib/chatStore";

const INDEX_KEY = "command-panel:chats:v1";
const LEGACY_CHAT_KEY = "command-panel:chat:v1"; // pre-conversations history
const MAX_STORED_MESSAGES = 40;

const messagesKey = (id: string) => `command-panel:chat:${id}:v1`;

export const newChatId = () => crypto.randomUUID();

export function loadMessages(id: string): UIMessage[] {
  try {
    const raw: unknown = JSON.parse(localStorage.getItem(messagesKey(id)) ?? "[]");
    return Array.isArray(raw) ? (raw as UIMessage[]) : [];
  } catch {
    return [];
  }
}

export function saveMessages(id: string, messages: UIMessage[]): void {
  try {
    localStorage.setItem(messagesKey(id), JSON.stringify(messages.slice(-MAX_STORED_MESSAGES)));
  } catch {
    /* storage unavailable — chat just won't persist */
  }
}

export function removeMessages(id: string): void {
  try {
    localStorage.removeItem(messagesKey(id));
  } catch {}
}

export function saveIndex(index: ChatIndex): void {
  try {
    localStorage.setItem(INDEX_KEY, JSON.stringify(index));
  } catch {
    /* storage unavailable — conversations just won't persist */
  }
}

/** First user text part across messages — the auto-title source. */
export function firstUserText(messages: UIMessage[]): string | null {
  for (const m of messages) {
    if (m.role !== "user") continue;
    for (const p of m.parts) {
      if (p.type === "text" && p.text.trim()) return p.text;
    }
  }
  return null;
}

/** Load the conversation index, creating (and migrating) on first run. */
export function initIndex(): ChatIndex {
  const existing = parseChatIndex(localStorage.getItem(INDEX_KEY));
  if (existing) return existing;
  const created = createChat(null, newChatId(), new Date().toISOString()).index;
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
