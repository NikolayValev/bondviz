import { describe, it, expect, vi, afterEach } from "vitest";
import { aggregatesUrl, parseAggregates, fetchAggregates } from "@/lib/polygon";

const sample = {
  ticker: "AAPL",
  results: [
    // intentionally out of order to prove sorting
    { t: 1735948800000, o: 254, h: 258, l: 253, c: 257, v: 1200000, vw: 256 },
    { t: 1735862400000, o: 250, h: 255, l: 249, c: 254, v: 1000000, vw: 252 },
  ],
  status: "OK",
};

describe("aggregatesUrl", () => {
  it("includes ticker, daily range, dates, and query params", () => {
    const url = aggregatesUrl("aapl", "2025-01-01", "2025-01-31", "KEY123");
    expect(url).toContain("/aggs/ticker/AAPL/range/1/day/2025-01-01/2025-01-31");
    expect(url).toContain("adjusted=true");
    expect(url).toContain("sort=asc");
    expect(url).toContain("apiKey=KEY123");
  });
});

describe("parseAggregates", () => {
  it("maps results to sorted StockBar[]", () => {
    const bars = parseAggregates(sample);
    expect(bars).toHaveLength(2);
    expect(bars[0].date).toBe("2025-01-03"); // earlier timestamp first
    expect(bars[0].close).toBe(254);
    expect(bars[1].date).toBe("2025-01-04");
    expect(bars[0].vwap).toBe(252);
  });

  it("returns [] for missing/empty/garbage input", () => {
    expect(parseAggregates({})).toEqual([]);
    expect(parseAggregates({ results: [] })).toEqual([]);
    expect(parseAggregates(null)).toEqual([]);
  });
});

describe("fetchAggregates", () => {
  afterEach(() => vi.restoreAllMocks());

  it("fetches and parses on a 200", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(sample), { status: 200 })));
    const bars = await fetchAggregates("AAPL", "2025-01-01", "2025-01-31", "KEY");
    expect(bars).toHaveLength(2);
  });

  it("throws on a non-OK response", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("", { status: 500 })));
    await expect(fetchAggregates("AAPL", "2025-01-01", "2025-01-31", "KEY")).rejects.toThrow();
  });
});
