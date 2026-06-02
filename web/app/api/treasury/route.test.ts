import { describe, it, expect, vi, afterEach } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { GET as latestGET } from "@/app/api/treasury/latest/route";
import { GET as rangeGET } from "@/app/api/treasury/range/route";

const xml = readFileSync(
  fileURLToPath(new URL("../../../test/fixtures/treasury-sample.xml", import.meta.url)),
  "utf-8",
);

afterEach(() => vi.restoreAllMocks());

describe("/api/treasury/latest", () => {
  it("returns the last row", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(xml, { status: 200 })));
    const res = await latestGET();
    const body = await res.json();
    expect(res.status).toBe(200);
    expect(body.row.date).toBe("2025-01-03");
  });
  it("503s when the feed fails", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("", { status: 500 })));
    const res = await latestGET();
    expect(res.status).toBe(503);
  });
});

describe("/api/treasury/range", () => {
  it("returns rows filtered to the requested window", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(xml, { status: 200 })));
    const req = new Request("http://x/api/treasury/range?start=2025-01-03&end=2025-01-03");
    const res = await rangeGET(req);
    const body = await res.json();
    expect(res.status).toBe(200);
    expect(body.rows).toHaveLength(1);
    expect(body.rows[0].date).toBe("2025-01-03");
  });
  it("400s without start/end", async () => {
    const req = new Request("http://x/api/treasury/range");
    const res = await rangeGET(req);
    expect(res.status).toBe(400);
  });
});
