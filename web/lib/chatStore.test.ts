import { describe, expect, it } from "vitest";
import {
  DEFAULT_TITLE,
  MAX_CHATS,
  TITLE_MAX,
  createChat,
  deleteChat,
  deriveTitle,
  parseChatIndex,
  renameChat,
  selectChat,
  touchChat,
  type ChatIndex,
} from "./chatStore";

const at = (n: number) => new Date(Date.UTC(2026, 0, 1, 0, n)).toISOString();

function indexOf(...ids: string[]): ChatIndex {
  return {
    activeId: ids[0],
    chats: ids.map((id, i) => ({ id, title: `Chat ${id}`, updatedAt: at(ids.length - i) })),
  };
}

describe("parseChatIndex", () => {
  it("round-trips a valid index", () => {
    const idx = indexOf("a", "b");
    expect(parseChatIndex(JSON.stringify(idx))).toEqual(idx);
  });

  it("returns null for null, garbage, or empty chat lists", () => {
    expect(parseChatIndex(null)).toBeNull();
    expect(parseChatIndex("nope")).toBeNull();
    expect(parseChatIndex(JSON.stringify({ activeId: "a", chats: [] }))).toBeNull();
    expect(parseChatIndex(JSON.stringify([1]))).toBeNull();
  });

  it("drops malformed entries and repairs a dangling activeId", () => {
    const idx = {
      activeId: "gone",
      chats: [{ id: "a", title: "A", updatedAt: at(1) }, { id: "" }, null, { id: "b" }],
    };
    expect(parseChatIndex(JSON.stringify(idx))).toEqual({
      activeId: "a",
      chats: [{ id: "a", title: "A", updatedAt: at(1) }],
    });
  });
});

describe("createChat", () => {
  it("prepends the new chat and makes it active", () => {
    const { index, evictedIds } = createChat(indexOf("a"), "new", at(9));
    expect(index.activeId).toBe("new");
    expect(index.chats.map((c) => c.id)).toEqual(["new", "a"]);
    expect(index.chats[0].title).toBe(DEFAULT_TITLE);
    expect(evictedIds).toEqual([]);
  });

  it("starts a fresh index from null", () => {
    const { index } = createChat(null, "solo", at(1));
    expect(index).toEqual({
      activeId: "solo",
      chats: [{ id: "solo", title: DEFAULT_TITLE, updatedAt: at(1) }],
    });
  });

  it("evicts the least recently updated chat beyond MAX_CHATS", () => {
    let idx: ChatIndex | null = null;
    for (let i = 0; i < MAX_CHATS; i++) idx = createChat(idx, `c${i}`, at(i)).index;
    // c0 is the oldest (updatedAt at(0)) — creating one more evicts it.
    const { index, evictedIds } = createChat(idx, "overflow", at(99));
    expect(evictedIds).toEqual(["c0"]);
    expect(index.chats).toHaveLength(MAX_CHATS);
    expect(index.chats.some((c) => c.id === "c0")).toBe(false);
  });
});

describe("selectChat / renameChat / touchChat", () => {
  it("selects only existing chats", () => {
    const idx = indexOf("a", "b");
    expect(selectChat(idx, "b").activeId).toBe("b");
    expect(selectChat(idx, "ghost")).toEqual(idx);
  });

  it("renames with whitespace collapse and truncation; ignores blank", () => {
    const idx = indexOf("a");
    expect(renameChat(idx, "a", "  Curve   watch  ").chats[0].title).toBe("Curve watch");
    expect(renameChat(idx, "a", "   ").chats[0].title).toBe("Chat a");
    expect(renameChat(idx, "a", "x".repeat(99)).chats[0].title).toHaveLength(TITLE_MAX);
  });

  it("touch bumps updatedAt for the given chat only", () => {
    const idx = indexOf("a", "b");
    const touched = touchChat(idx, "b", at(50));
    expect(touched.chats.find((c) => c.id === "b")?.updatedAt).toBe(at(50));
    expect(touched.chats.find((c) => c.id === "a")?.updatedAt).toBe(idx.chats[0].updatedAt);
  });
});

describe("deleteChat", () => {
  it("activates the most recently updated survivor when deleting the active chat", () => {
    const idx: ChatIndex = {
      activeId: "a",
      chats: [
        { id: "a", title: "A", updatedAt: at(3) },
        { id: "b", title: "B", updatedAt: at(1) },
        { id: "c", title: "C", updatedAt: at(2) },
      ],
    };
    expect(deleteChat(idx, "a")).toEqual({
      activeId: "c",
      chats: [idx.chats[1], idx.chats[2]],
    });
  });

  it("keeps the current active when deleting another chat", () => {
    const idx = indexOf("a", "b");
    expect(deleteChat(idx, "b")?.activeId).toBe("a");
  });

  it("returns null when the last chat is deleted", () => {
    expect(deleteChat(indexOf("a"), "a")).toBeNull();
  });
});

describe("deriveTitle", () => {
  it("collapses whitespace and falls back to the default", () => {
    expect(deriveTitle("  chart   the\ncurve ")).toBe("chart the curve");
    expect(deriveTitle("   ")).toBe(DEFAULT_TITLE);
  });

  it("truncates long text with an ellipsis within the cap", () => {
    const title = deriveTitle("w".repeat(100));
    expect(title.length).toBeLessThanOrEqual(TITLE_MAX);
    expect(title.endsWith("…")).toBe(true);
  });
});
