// Fixed-window rate limiter keyed by an arbitrary string (e.g. client IP). Pure apart from the
// Map it owns; time is passed in so tests are deterministic. State lives in the server
// instance's memory, so on serverless it is a per-instance, best-effort guard: enough to stop
// one visitor from draining a shared API quota, not a substitute for an edge rate limiter.

export interface RateLimitResult {
  allowed: boolean;
  remaining: number;
  /** Milliseconds until the current window resets (0 when allowed with quota left). */
  retryAfterMs: number;
}

export interface RateLimiter {
  check(key: string, now?: number): RateLimitResult;
}

export function createRateLimiter({
  limit,
  windowMs,
  maxKeys = 10_000,
}: {
  limit: number;
  windowMs: number;
  maxKeys?: number;
}): RateLimiter {
  const windows = new Map<string, { start: number; count: number }>();

  return {
    check(key, now = Date.now()) {
      let w = windows.get(key);
      if (!w || now - w.start >= windowMs) {
        // Bound memory: drop the oldest entry (Map keeps insertion order) before adding.
        if (!w && windows.size >= maxKeys) windows.delete(windows.keys().next().value as string);
        w = { start: now, count: 0 };
        windows.delete(key);
        windows.set(key, w);
      }
      if (w.count >= limit) {
        return { allowed: false, remaining: 0, retryAfterMs: w.start + windowMs - now };
      }
      w.count++;
      return { allowed: true, remaining: limit - w.count, retryAfterMs: 0 };
    },
  };
}

/** Best-effort client key from proxy headers (Vercel sets x-forwarded-for). */
export function clientKey(headers: Headers): string {
  const fwd = headers.get("x-forwarded-for");
  if (fwd) return fwd.split(",")[0].trim();
  return headers.get("x-real-ip") ?? "unknown";
}
