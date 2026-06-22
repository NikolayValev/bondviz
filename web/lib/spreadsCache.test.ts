import { describe, it, expect, vi, afterEach } from "vitest";
import { loadSpreads, __resetSpreadsCache } from "@/lib/spreadsCache";

afterEach(() => {
  __resetSpreadsCache();
  vi.restoreAllMocks();
});

describe("loadSpreads", () => {
  it("fetches once and shares the result across concurrent callers", async () => {
    const body = JSON.stringify({ points: [{ date: "2020-01-01", s10y3m: -0.1, s2s10s: 0 }] });
    const fetchMock = vi.fn(async () => new Response(body, { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const [a, b] = await Promise.all([loadSpreads(), loadSpreads()]);
    expect(a).toBe(b); // same shared array
    expect(a).toHaveLength(1);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("clears the cache on failure so a later call retries", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("net"); }));
    await expect(loadSpreads()).rejects.toThrow();

    const ok = vi.fn(async () => new Response(JSON.stringify({ points: [] }), { status: 200 }));
    vi.stubGlobal("fetch", ok);
    await expect(loadSpreads()).resolves.toEqual([]);
    expect(ok).toHaveBeenCalledTimes(1);
  });
});
