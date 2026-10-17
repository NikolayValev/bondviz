import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, within } from "@testing-library/react";
import { ShockScenario } from "@/app/yield-curve/ShockScenario";
import { CurvePoint } from "@/lib/types";

const curve: CurvePoint[] = [
  { label: "3M", years: 0.25, yield: 4.26 },
  { label: "2Y", years: 2, yield: 4.79 },
  { label: "5Y", years: 5, yield: 5.0 },
  { label: "10Y", years: 10, yield: 5.21 },
  { label: "30Y", years: 30, yield: 5.56 },
];

describe("ShockScenario", () => {
  it("lists each benchmark with a signed shift and price change", () => {
    render(<ShockScenario curve={curve} date="2026-10-13" kind="parallel-up" sizeBps={100} onChange={() => {}} />);
    const rows = within(screen.getByRole("table")).getAllByRole("row").slice(1);
    expect(rows.map((r) => r.firstChild?.textContent)).toEqual(["2Y", "5Y", "10Y", "30Y"]);
    for (const r of rows) {
      expect(r).toHaveTextContent("+100 bps");
      expect(r).toHaveTextContent("−"); // every price falls
    }
  });

  it("reports shape and size changes through onChange", () => {
    const onChange = vi.fn();
    render(<ShockScenario curve={curve} date="2026-10-13" kind="parallel-up" sizeBps={50} onChange={onChange} />);
    fireEvent.click(screen.getByRole("radio", { name: "Twist" }));
    expect(onChange).toHaveBeenLastCalledWith({ kind: "twist", sizeBps: 50 });
    fireEvent.click(screen.getByRole("radio", { name: "100 bp" }));
    expect(onChange).toHaveBeenLastCalledWith({ kind: "parallel-up", sizeBps: 100 });
  });
});
