import { describe, it, expect, vi, afterEach } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { GET } from "@/app/api/treasury/spreads/route";

const xml = readFileSync(
  fileURLToPath(new URL("../../../../test/fixtures/treasury-sample.xml", import.meta.url)),
  "utf-8",
);

afterEach(() => vi.restoreAllMocks());

describe("/api/treasury/spreads", () => {
  it("returns slim spread points filtered to the window", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(xml, { status: 200 })));
    const req = new Request("http://x/api/treasury/spreads?start=2025-01-03&end=2025-01-03");
    const res = await GET(req);
    const body = await res.json();
    expect(res.status).toBe(200);
    expect(body.points).toHaveLength(1);
    expect(body.points[0].date).toBe("2025-01-03");
    // fixture 2025-01-03: 10Y 4.55, 3M 5.01, 2Y 4.05
    expect(body.points[0].s10y3m).toBeCloseTo(-0.46, 6);
    expect(body.points[0].s2s10s).toBeCloseTo(0.5, 6);
  });

  it("400s without start/end", async () => {
    const res = await GET(new Request("http://x/api/treasury/spreads"));
    expect(res.status).toBe(400);
  });
});
