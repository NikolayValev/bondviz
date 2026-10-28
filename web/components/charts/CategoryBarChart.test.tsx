import { describe, it, expect } from "vitest";
import { render, fireEvent } from "@testing-library/react";
import { CategoryBarChart } from "@/components/charts/CategoryBarChart";

describe("CategoryBarChart", () => {
  it("renders an svg with one rect per (series, category) cell", () => {
    const { container, getByRole } = render(
      <CategoryBarChart
        width={600}
        ariaLabel="test bars"
        categories={["2Y", "5Y", "10Y"]}
        series={[
          { id: "carry", label: "Carry", color: "#00d68f", values: [10, 20, 30] },
          { id: "roll", label: "Roll", color: "#5b8def", values: [5, -5, 15] },
        ]}
      />,
    );
    expect(getByRole("img", { name: "test bars" })).toBeTruthy();
    // 2 series × 3 categories = 6 bar rects (class "bar")
    expect(container.querySelectorAll("rect.bar").length).toBe(6);
  });
});

describe("CategoryBarChart readout", () => {
  const chart = (
    <CategoryBarChart
      width={600}
      ariaLabel="carry"
      yUnit=" bps"
      categories={["2Y", "5Y", "10Y"]}
      series={[
        { id: "carry", label: "Carry", color: "#00d68f", values: [119.75, 125, 130.2] },
        { id: "roll", label: "Roll", color: "#5b8def", values: [15, -2, null] },
      ]}
    />
  );

  it("reads each category with a total for stacked bars", () => {
    const { container } = render(chart);
    const svg = container.querySelector("svg") as SVGSVGElement;
    const live = container.querySelector("[aria-live]") as HTMLElement;
    fireEvent.keyDown(svg, { key: "ArrowRight" });
    expect(live.textContent).toBe("2Y: Carry 120 bps, Roll 15 bps, total 135 bps");
    fireEvent.keyDown(svg, { key: "End" });
    expect(live.textContent).toBe("10Y: Carry 130 bps"); // a missing value is skipped, no total
    expect(container.querySelector("rect.hover-band")).toBeTruthy();
  });

  it("uses the chart's own formatter when given", () => {
    const { container } = render(
      <CategoryBarChart
        width={600}
        ariaLabel="krd"
        categories={["10Y"]}
        series={[{ id: "k", label: "KRD", color: "red", values: [6.734] }]}
        formatY={(v) => `${v.toFixed(2)} yrs`}
      />,
    );
    fireEvent.keyDown(container.querySelector("svg") as SVGSVGElement, { key: "Home" });
    expect(container.querySelector("[aria-live]")?.textContent).toBe("10Y: KRD 6.73 yrs");
  });
});

