import { describe, it, expect } from "vitest";
import { createRateLimiter, clientKey } from "@/lib/rateLimit";

describe("createRateLimiter", () => {
  it("allows up to the limit per window, then blocks with a retry hint", () => {
    const rl = createRateLimiter({ limit: 3, windowMs: 60_000 });
    expect(rl.check("a", 0)).toMatchObject({ allowed: true, remaining: 2 });
    rl.check("a", 1_000);
    expect(rl.check("a", 2_000)).toMatchObject({ allowed: true, remaining: 0 });
    expect(rl.check("a", 10_000)).toEqual({ allowed: false, remaining: 0, retryAfterMs: 50_000 });
  });

  it("resets when the window elapses", () => {
    const rl = createRateLimiter({ limit: 1, windowMs: 1_000 });
    rl.check("a", 0);
    expect(rl.check("a", 999).allowed).toBe(false);
    expect(rl.check("a", 1_000).allowed).toBe(true);
  });

  it("tracks keys independently", () => {
    const rl = createRateLimiter({ limit: 1, windowMs: 1_000 });
    expect(rl.check("a", 0).allowed).toBe(true);
    expect(rl.check("b", 0).allowed).toBe(true);
    expect(rl.check("a", 1).allowed).toBe(false);
  });

  it("evicts the oldest key once maxKeys is reached", () => {
    const rl = createRateLimiter({ limit: 1, windowMs: 60_000, maxKeys: 2 });
    rl.check("a", 0);
    rl.check("b", 0);
    rl.check("c", 0); // evicts "a"
    expect(rl.check("a", 1).allowed).toBe(true);
    expect(rl.check("c", 1).allowed).toBe(false);
  });
});

describe("clientKey", () => {
  it("uses the first x-forwarded-for hop, then x-real-ip, then a fallback", () => {
    expect(clientKey(new Headers({ "x-forwarded-for": "1.2.3.4, 10.0.0.1" }))).toBe("1.2.3.4");
    expect(clientKey(new Headers({ "x-real-ip": "5.6.7.8" }))).toBe("5.6.7.8");
    expect(clientKey(new Headers())).toBe("unknown");
  });
});
