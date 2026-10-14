import { describe, it, expect, vi, afterEach } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { GET } from "@/app/api/treasury/range/route";

const xml = readFileSync(
  fileURLToPath(new URL("../../../../test/fixtures/treasury-sample.xml", import.meta.url)),
  "utf-8",
);

afterEach(() => vi.restoreAllMocks());

describe("/api/treasury/range", () => {
  it("returns rows filtered to the window", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(xml, { status: 200 })));
    const res = await GET(new Request("http://x/api/treasury/range?start=2025-01-03&end=2025-12-31"));
    const body = await res.json();
    expect(res.status).toBe(200);
    expect(body.rows.map((r: { date: string }) => r.date)).toEqual(["2025-01-03"]);
  });

  it("fetches every year of a multi-year window in parallel and skips failed years", async () => {
    let inFlight = 0;
    let peak = 0;
    const urls: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        urls.push(url);
        inFlight++;
        peak = Math.max(peak, inFlight);
        await new Promise((r) => setTimeout(r, 5));
        inFlight--;
        return url.endsWith("=2024") ? new Response("", { status: 500 }) : new Response(xml, { status: 200 });
      }),
    );
    const res = await GET(new Request("http://x/api/treasury/range?start=2022-01-01&end=2025-12-31"));
    const body = await res.json();
    expect(urls).toHaveLength(4);
    expect(peak).toBe(4);
    expect(res.status).toBe(200);
    // The fixture holds the same two rows for every year: 3 good years x 2 rows; 2024 failed soft.
    expect(body.rows).toHaveLength(6);
  });

  it("400s without start/end", async () => {
    const res = await GET(new Request("http://x/api/treasury/range"));
    expect(res.status).toBe(400);
  });
});
