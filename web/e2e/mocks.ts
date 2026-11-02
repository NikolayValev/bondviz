import { readFileSync } from "node:fs";
import { join } from "node:path";
import type { Page } from "@playwright/test";

// Real U.S. Treasury data (public domain) captured from the deployed API: ~2 years of daily par
// curves, and the 10y–3m / 2s10s spreads since 1990 thinned to weekly to keep the file small.
const range = readFileSync(join(__dirname, "fixtures", "range.json"), "utf-8");
const spreads = readFileSync(join(__dirname, "fixtures", "spreads.json"), "utf-8");

/** A deterministic daily random walk per ticker, so Stocks renders without Polygon. */
function bars(ticker: string, from: string, to: string) {
  let seed = [...ticker].reduce((a, c) => (a * 31 + c.charCodeAt(0)) % 2147483647, 7);
  const rnd = () => (seed = (seed * 48271) % 2147483647) / 2147483647;
  let px = 50 + rnd() * 400;
  const out = [];
  for (let d = new Date(`${from}T00:00:00Z`); d <= new Date(`${to}T00:00:00Z`); d.setUTCDate(d.getUTCDate() + 1)) {
    if (d.getUTCDay() === 0 || d.getUTCDay() === 6) continue;
    px *= 1 + (rnd() - 0.48) * 0.03;
    const close = Math.round(px * 100) / 100;
    out.push({ date: d.toISOString().slice(0, 10), open: close, high: close, low: close, close, volume: 1e7, vwap: close });
  }
  return out;
}

/** Answer every API route a page calls from fixtures. Call before `page.goto`. */
export async function mockApis(page: Page) {
  await page.route("**/api/treasury/range**", (r) => r.fulfill({ contentType: "application/json", body: range }));
  await page.route("**/api/treasury/spreads**", (r) => r.fulfill({ contentType: "application/json", body: spreads }));
  await page.route("**/api/stocks/aggregates**", (r) => {
    const q = new URL(r.request().url()).searchParams;
    const json = { configured: true, bars: bars(q.get("ticker") ?? "", q.get("from") ?? "", q.get("to") ?? "") };
    return r.fulfill({ json });
  });
  // The AI demo reports itself off, and nothing ever reaches the model.
  await page.route("**/api/command-panel**", (r) =>
    r.request().method() === "GET" && !r.request().url().includes("/data")
      ? r.fulfill({ json: { configured: false } })
      : r.fulfill({ status: 503, json: { error: "disabled in tests" } }),
  );
}

/** Scroll the whole page so lazily mounted charts render, then return to the top. */
export async function walkPage(page: Page) {
  const height = await page.evaluate(() => document.body.scrollHeight);
  for (let y = 0; y < height; y += 500) {
    await page.evaluate((v) => window.scrollTo(0, v), y);
    await page.waitForTimeout(60);
  }
  await page.evaluate(() => window.scrollTo(0, 0));
}
