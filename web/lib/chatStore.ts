// Pure conversation-index logic for the command-panel chat: multiple named
// conversations ("workflows"), each with its own persisted message history.
// Storage I/O stays in the component — these are inputs → outputs.

export interface ChatMeta {
  id: string;
  title: string;
  updatedAt: string; // ISO timestamp of creation or last settled turn
}

export interface ChatIndex {
  activeId: string;
  chats: ChatMeta[]; // newest-created first; order stable thereafter
}

export const MAX_CHATS = 20;
export const DEFAULT_TITLE = "New chat";
export const TITLE_MAX = 40;

/** Parse a persisted index; anything malformed degrades to null (caller creates fresh). */
export function parseChatIndex(json: string | null): ChatIndex | null {
  if (!json) return null;
  try {
    const raw: unknown = JSON.parse(json);
    if (typeof raw !== "object" || raw === null || Array.isArray(raw)) return null;
    const { activeId, chats } = raw as ChatIndex;
    if (typeof activeId !== "string" || !Array.isArray(chats)) return null;
    const clean = chats.filter(
      (c): c is ChatMeta =>
        c !== null &&
        typeof c === "object" &&
        typeof c.id === "string" &&
        c.id.length > 0 &&
        typeof c.title === "string" &&
        typeof c.updatedAt === "string",
    );
    if (clean.length === 0) return null;
    return {
      activeId: clean.some((c) => c.id === activeId) ? activeId : clean[0].id,
      chats: clean,
    };
  } catch {
    return null;
  }
}

/**
 * Prepend a new active conversation. Beyond MAX_CHATS the least recently
 * updated others are evicted; their ids are returned so the caller can drop
 * their message storage too.
 */
export function createChat(
  index: ChatIndex | null,
  id: string,
  now: string,
): { index: ChatIndex; evictedIds: string[] } {
  const chats: ChatMeta[] = [{ id, title: DEFAULT_TITLE, updatedAt: now }, ...(index?.chats ?? [])];
  const evictedIds: string[] = [];
  while (chats.length > MAX_CHATS) {
    let victim = 1; // never the one just created
    for (let i = 2; i < chats.length; i++) {
      if (chats[i].updatedAt < chats[victim].updatedAt) victim = i;
    }
    evictedIds.push(chats[victim].id);
    chats.splice(victim, 1);
  }
  return { index: { activeId: id, chats }, evictedIds };
}

export function selectChat(index: ChatIndex, id: string): ChatIndex {
  return index.chats.some((c) => c.id === id) ? { ...index, activeId: id } : index;
}

/** Rename; blank titles are ignored, long ones truncated. */
export function renameChat(index: ChatIndex, id: string, title: string): ChatIndex {
  const t = title.replace(/\s+/g, " ").trim().slice(0, TITLE_MAX);
  if (!t) return index;
  return { ...index, chats: index.chats.map((c) => (c.id === id ? { ...c, title: t } : c)) };
}

export function touchChat(index: ChatIndex, id: string, now: string): ChatIndex {
  return { ...index, chats: index.chats.map((c) => (c.id === id ? { ...c, updatedAt: now } : c)) };
}

/**
 * Remove a conversation. If it was active, the most recently updated survivor
 * becomes active. Returns null when nothing is left (caller creates fresh).
 */
export function deleteChat(index: ChatIndex, id: string): ChatIndex | null {
  const chats = index.chats.filter((c) => c.id !== id);
  if (chats.length === 0) return null;
  const activeId =
    index.activeId === id
      ? chats.reduce((a, b) => (a.updatedAt >= b.updatedAt ? a : b)).id
      : index.activeId;
  return { activeId, chats };
}

/** Conversation title from its first user message: whitespace collapsed, capped. */
export function deriveTitle(text: string): string {
  const t = text.replace(/\s+/g, " ").trim();
  if (!t) return DEFAULT_TITLE;
  return t.length <= TITLE_MAX ? t : `${t.slice(0, TITLE_MAX - 1).trimEnd()}…`;
}
