import { describe, it, expect, vi, afterEach } from "vitest";
import { GET } from "@/app/api/stocks/aggregates/route";

const sample = JSON.stringify({
  ticker: "AAPL",
  results: [
    { t: 1735862400000, o: 250, h: 255, l: 249, c: 254, v: 1000000, vw: 252 },
    { t: 1735948800000, o: 254, h: 258, l: 253, c: 257, v: 1200000, vw: 256 },
  ],
  status: "OK",
});

const url = "http://x/api/stocks/aggregates?ticker=AAPL&from=2025-01-01&to=2025-01-31";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
});

describe("/api/stocks/aggregates", () => {
  it("reports not configured when the key is missing", async () => {
    vi.stubEnv("POLYGON_API_KEY", "");
    const res = await GET(new Request(url));
    const body = await res.json();
    expect(res.status).toBe(200);
    expect(body.configured).toBe(false);
  });

  it("returns bars when configured", async () => {
    vi.stubEnv("POLYGON_API_KEY", "testkey");
    vi.stubGlobal("fetch", vi.fn(async () => new Response(sample, { status: 200 })));
    const res = await GET(new Request(url));
    const body = await res.json();
    expect(res.status).toBe(200);
    expect(body.configured).toBe(true);
    expect(body.bars).toHaveLength(2);
    expect(body.bars[0].close).toBe(254);
  });

  it("400s without a ticker", async () => {
    vi.stubEnv("POLYGON_API_KEY", "testkey");
    const res = await GET(new Request("http://x/api/stocks/aggregates?from=2025-01-01&to=2025-01-31"));
    expect(res.status).toBe(400);
  });

  it("502s on an upstream error", async () => {
    vi.stubEnv("POLYGON_API_KEY", "testkey");
    vi.stubGlobal("fetch", vi.fn(async () => new Response("", { status: 500 })));
    const res = await GET(new Request(url));
    const body = await res.json();
    expect(res.status).toBe(502);
    expect(body.configured).toBe(true);
  });
});
