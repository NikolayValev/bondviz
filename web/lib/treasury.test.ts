import { describe, it, expect, vi, afterEach } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { parseTreasuryXml, fetchTreasuryYear } from "@/lib/treasury";

const xml = readFileSync(
  fileURLToPath(new URL("../test/fixtures/treasury-sample.xml", import.meta.url)),
  "utf-8",
);

describe("parseTreasuryXml", () => {
  it("extracts sorted rows with numeric BC_* fields", () => {
    const rows = parseTreasuryXml(xml);
    expect(rows).toHaveLength(2);
    expect(rows[0].date).toBe("2025-01-02");
    expect(rows[0].BC_10YEAR).toBe(4.5);
    expect(rows[1].date).toBe("2025-01-03");
  });
  it("returns [] for empty/garbage input", () => {
    expect(parseTreasuryXml("<feed></feed>")).toEqual([]);
  });
});

describe("fetchTreasuryYear", () => {
  afterEach(() => vi.restoreAllMocks());
  it("fetches and parses the year feed", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(xml, { status: 200 })));
    const rows = await fetchTreasuryYear(2025);
    expect(rows).toHaveLength(2);
  });
  it("throws on non-OK responses", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("", { status: 500 })));
    await expect(fetchTreasuryYear(2025)).rejects.toThrow();
  });
});
