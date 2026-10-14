import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

const seen: string[] = [];
vi.mock("@nikolayvalev/command-panel/server", () => ({
  createCommandPanelHandler: () => async (req: Request) => {
    seen.push(await req.text());
    return new Response("streamed", { status: 200 });
  },
}));
vi.mock("@nikolayvalev/command-panel", () => ({
  defaultComponentRegistry: {},
  createDataRegistry: (entries: unknown) => entries,
}));

const { POST } = await import("@/app/api/command-panel/route");

let ip = 0;
const post = (body = '{"messages":[]}', addr = `10.0.0.${++ip}`) =>
  POST(new Request("http://x/api/command-panel", { method: "POST", body, headers: { "x-forwarded-for": addr } }));

beforeEach(() => {
  seen.length = 0;
  vi.stubEnv("GITHUB_MODELS_TOKEN", "test-token");
});
afterEach(() => vi.unstubAllEnvs());

describe("POST /api/command-panel", () => {
  it("forwards the original body to the chat handler", async () => {
    const res = await post('{"messages":[{"id":"1"}]}');
    expect(res.status).toBe(200);
    expect(seen).toEqual(['{"messages":[{"id":"1"}]}']);
  });

  it("refuses with a readable 503 when the token isn't configured", async () => {
    vi.stubEnv("GITHUB_MODELS_TOKEN", "");
    const res = await post();
    expect(res.status).toBe(503);
    expect(await res.text()).toMatch(/isn't switched on/);
    expect(seen).toHaveLength(0);
  });

  it("rate-limits a client after 5 requests a minute, with Retry-After", async () => {
    const addr = "203.0.113.7";
    for (let i = 0; i < 5; i++) expect((await post(undefined, addr)).status).toBe(200);
    const res = await post(undefined, addr);
    expect(res.status).toBe(429);
    expect(Number(res.headers.get("retry-after"))).toBeGreaterThan(0);
    expect((await post(undefined, "203.0.113.8")).status).toBe(200); // other clients unaffected
  });

  it("rejects oversized conversations", async () => {
    const res = await post(JSON.stringify({ messages: [{ text: "x".repeat(200_001) }] }));
    expect(res.status).toBe(413);
    expect(seen).toHaveLength(0);
  });
});
