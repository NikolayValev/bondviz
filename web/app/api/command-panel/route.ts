// Chat/generation route. Server-only: the AI SDK provider and the GitHub Models
// token never reach the client bundle (imports from the /server entry).
//
// Guardrails for a public demo: the token is one shared quota, so the route
// refuses cleanly when it isn't configured, rate-limits per client, and caps
// the request size. Refusals are plain text, which the chat shows as the error.

import { createCommandPanelHandler } from "@nikolayvalev/command-panel/server";
import { defaultComponentRegistry } from "@nikolayvalev/command-panel";
import { createOpenAI } from "@ai-sdk/openai";
import { dataRegistry } from "../../command-panel/data";
import { clientKey, createRateLimiter } from "@/lib/rateLimit";

const github = createOpenAI({
  baseURL: "https://models.github.ai/inference",
  apiKey: process.env.GITHUB_MODELS_TOKEN,
});

// Per-client limits, well under the shared token's own quota. In-memory per
// server instance, so best-effort (see lib/rateLimit.ts).
const perMinute = createRateLimiter({ limit: 5, windowMs: 60_000 });
const perDay = createRateLimiter({ limit: 20, windowMs: 24 * 60 * 60_000 });
const MAX_BODY_CHARS = 200_000;

function refuse(status: number, message: string, retryAfterMs?: number): Response {
  const headers: Record<string, string> = { "content-type": "text/plain; charset=utf-8" };
  if (retryAfterMs !== undefined) headers["retry-after"] = String(Math.ceil(retryAfterMs / 1000));
  return new Response(message, { status, headers });
}

export async function POST(req: Request) {
  if (!process.env.GITHUB_MODELS_TOKEN) {
    return refuse(503, "The AI demo isn't switched on for this deployment. Every other page works without it.");
  }

  const key = clientKey(req.headers);
  const minute = perMinute.check(key);
  if (!minute.allowed) {
    const s = Math.ceil(minute.retryAfterMs / 1000);
    return refuse(429, `That's a lot of requests in a row. Try again in ${s}s.`, minute.retryAfterMs);
  }
  const day = perDay.check(key);
  if (!day.allowed) {
    return refuse(429, "You've reached today's limit for the AI demo. It resets within 24 hours.", day.retryAfterMs);
  }

  const body = await req.text();
  if (body.length > MAX_BODY_CHARS) {
    return refuse(413, "This conversation is too long for the demo. Start a new one to keep going.");
  }

  return createCommandPanelHandler({
    componentRegistry: defaultComponentRegistry,
    dataRegistry,
    model: github.chat(process.env.GITHUB_MODELS_MODEL ?? "openai/gpt-4o"),
  })(new Request(req.url, { method: "POST", headers: req.headers, body }));
}
