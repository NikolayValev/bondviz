import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { RelativeValuePanel } from "@/app/signal/RelativeValuePanel";
import type { SpreadPoint } from "@/lib/signal";

// Ten years of a steadily steepening curve: today is the steepest day in any window.
const points: SpreadPoint[] = Array.from({ length: 40 }, (_, i) => ({
  date: `${2016 + Math.floor(i / 4)}-${String((i % 4) * 3 + 1).padStart(2, "0")}-01`,
  s10y3m: -0.5 + i * 0.05,
  s2s10s: 0.3,
}));

describe("RelativeValuePanel", () => {
  it("reports today's percentile, z-score and a plain-English verdict", () => {
    render(<RelativeValuePanel points={points} span="10Y" onSpan={() => {}} />);
    expect(screen.getByText(/10y–3m is in the 100th percentile over the last 10 years/)).toBeInTheDocument();
    expect(screen.getByText(/steeper than usual/)).toBeInTheDocument();
    // A constant 2s10s has no dispersion: z is 0 and it reads as normal.
    expect(screen.getByText(/2s10s is in the 100th percentile .*\(\+0\.0σ\): within its normal range/)).toBeInTheDocument();
  });

  it("changes the window through onSpan", () => {
    const onSpan = vi.fn();
    render(<RelativeValuePanel points={points} span="10Y" onSpan={onSpan} />);
    fireEvent.click(screen.getByRole("radio", { name: "Since 1990" }));
    expect(onSpan).toHaveBeenCalledWith("Since 1990");
  });
});
