import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import { StocksClient } from "@/app/stocks/StocksClient";

// Each ticker gets a steady trend so the ranking is known: AAPL +10%, MSFT +20%, GOOGL −5%.
const TRENDS: Record<string, number> = { AAPL: 0.1, MSFT: 0.2, GOOGL: -0.05, TSLA: 0.3 };

function barsFor(ticker: string) {
  const g = TRENDS[ticker] ?? 0;
  return ["2026-10-01", "2026-10-02", "2026-10-05"].map((date, i) => {
    const close = 100 * (1 + (g * i) / 2);
    return { date, open: close, high: close, low: close, close, volume: 1000, vwap: close };
  });
}

function mockApi(configured = true) {
  const fetchMock = vi.fn(async (input: string) => {
    const ticker = new URL(input, "http://x").searchParams.get("ticker") ?? "";
    const body = configured ? { configured: true, bars: barsFor(ticker) } : { configured: false, bars: [] };
    return { json: async () => body };
  });
  vi.stubGlobal("fetch", fetchMock);
  return () => fetchMock.mock.calls.map(([u]) => new URL(u, "http://x").searchParams.get("ticker"));
}

beforeEach(() => window.history.replaceState(null, "", "/stocks"));
afterEach(() => vi.unstubAllGlobals());

describe("StocksClient", () => {
  it("fetches each ticker once and ranks their returns", async () => {
    const tickersFetched = mockApi();
    render(<StocksClient />);
    const table = await screen.findByRole("table", { name: /Return over 3M/ });
    const rows = within(table).getAllByRole("row").slice(1);
    expect(rows.map((r) => r.textContent)).toEqual([
      expect.stringContaining("MSFT+20.00%"),
      expect.stringContaining("AAPL+10.00%"),
      expect.stringContaining("GOOGL−5.00%"),
    ]);
    expect(tickersFetched().sort()).toEqual(["AAPL", "GOOGL", "MSFT"]);
  });

  it("waits for typing to pause before fetching a new ticker", async () => {
    const tickersFetched = mockApi();
    render(<StocksClient />);
    await screen.findByRole("table", { name: /Return over 3M/ });
    const input = screen.getByRole("textbox", { name: /Tickers/ });
    for (const v of ["AAPL, MSFT, GOOGL, T", "AAPL, MSFT, GOOGL, TS", "AAPL, MSFT, GOOGL, TSLA"]) {
      fireEvent.change(input, { target: { value: v } });
    }
    expect(tickersFetched()).not.toContain("TSLA");
    await waitFor(() => expect(tickersFetched()).toContain("TSLA"));
    expect(tickersFetched()).not.toContain("T");
    expect(tickersFetched()).not.toContain("TS");
    expect(tickersFetched().filter((t) => t === "AAPL")).toHaveLength(1);
  });

  it("lets a rate-limited ticker be retried", async () => {
    let msftCalls = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string) => {
        const ticker = new URL(input, "http://x").searchParams.get("ticker") ?? "";
        if (ticker === "MSFT" && msftCalls++ === 0)
          return { json: async () => ({ configured: true, bars: [], error: "rate limited by Polygon" }) };
        return { json: async () => ({ configured: true, bars: barsFor(ticker) }) };
      }),
    );
    render(<StocksClient />);
    expect(await screen.findByText(/Couldn.t load: MSFT \(rate limited by Polygon\)/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    const table = await screen.findByRole("table", { name: /Return over 3M/ });
    await waitFor(() => expect(within(table).getAllByRole("row")).toHaveLength(4)); // header + 3 tickers
    expect(msftCalls).toBe(2);
  });

  it("shows the key notice and no comparison when Polygon isn't configured", async () => {
    mockApi(false);
    render(<StocksClient />);
    expect(await screen.findByText("Polygon API key not set")).toBeInTheDocument();
    expect(screen.queryByText(/Relative performance/)).toBeNull();
  });

  it("skips the comparison for a single ticker", async () => {
    window.history.replaceState(null, "", "/stocks?tickers=AAPL");
    mockApi();
    render(<StocksClient />);
    expect(await screen.findByText("Recent bars")).toBeInTheDocument();
    expect(screen.queryByText(/Relative performance/)).toBeNull();
  });
});
