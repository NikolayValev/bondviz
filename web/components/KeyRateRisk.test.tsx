import { describe, it, expect, vi } from "vitest";

// jsdom has no layout, so give the chart a fixed width.
vi.mock("@/components/charts/useResizeObserver", () => ({
  useResizeObserver: () => ({ ref: { current: null }, width: 600 }),
}));
import { render, screen, fireEvent } from "@testing-library/react";
import { KeyRateRisk } from "@/components/KeyRateRisk";
import { bondCashflows, keyRateDurations, modifiedDuration } from "@/lib/finance";

const { cashflows, times } = bondCashflows(1000, 0.05, 5, 2);
const krds = keyRateDurations(cashflows, times, 0.04);

describe("KeyRateRisk", () => {
  it("charts buckets up to maturity only and states the sum check", () => {
    const { container } = render(<KeyRateRisk krds={krds} subject="this bond" />);
    const labels = Array.from(container.querySelectorAll("svg text")).map((t) => t.textContent);
    expect(labels).toContain("5Y");
    expect(labels).not.toContain("7Y"); // no risk past a 5Y bond's maturity
    expect(screen.getByText(`${modifiedDuration(cashflows, times, 0.04).toFixed(2)} yrs`)).toBeInTheDocument();
  });

  it("switches the chart to dollar DV01s", () => {
    render(<KeyRateRisk krds={krds} subject="this bond" />);
    fireEvent.click(screen.getByRole("radio", { name: "DV01 $" }));
    expect(screen.getByRole("img")).toHaveAttribute("aria-label", expect.stringContaining("DV01 in dollars"));
  });
});
