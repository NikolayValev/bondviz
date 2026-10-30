import { describe, it, expect } from "vitest";
import { render, fireEvent } from "@testing-library/react";
import { LineChart, maturityLabel, timeTickLabels } from "@/components/charts/LineChart";

describe("LineChart", () => {
  it("renders one path per series when given explicit dimensions", () => {
    const { container } = render(
      <LineChart
        ariaLabel="test chart"
        width={400}
        height={200}
        series={[
          { id: "a", label: "A", color: "#00d68f", points: [[0, 1], [1, 2], [2, 1.5]] },
          { id: "b", label: "B", color: "#5b8def", points: [[0, 0.5], [1, 1], [2, 2]] },
        ]}
      />,
    );
    expect(container.querySelectorAll("path.series-line")).toHaveLength(2);
    expect(container.querySelector("svg")?.getAttribute("aria-label")).toBe("test chart");
  });

  it("puts time-axis ticks on UTC month starts within the data range", () => {
    const t0 = Date.parse("2025-01-07");
    const t1 = Date.parse("2025-09-22");
    const { container } = render(
      <LineChart
        ariaLabel="time chart"
        width={600}
        height={200}
        xType="time"
        series={[{ id: "a", label: "A", color: "var(--series-1)", points: [[t0, 1], [Date.parse("2025-04-15"), 2], [t1, 1.5]] }]}
      />,
    );
    const iw = 600 - 48 - 16; // width minus the chart's left/right margins
    const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const ticks = Array.from(container.querySelectorAll("g[transform]"))
      .map((g) => ({ tf: g.getAttribute("transform") ?? "", label: g.textContent ?? "" }))
      .filter((t) => /^[A-Z][a-z]{2}( ’\d{2})?$/.test(t.label));
    expect(ticks.length).toBeGreaterThan(2);
    expect(ticks[0].label).toMatch(/’25$/);
    for (const { tf, label } of ticks) {
      const x = Number(/translate\(([-\d.]+)/.exec(tf)?.[1]);
      const monthStart = Date.UTC(2025, months.indexOf(label.slice(0, 3)), 1);
      expect(x).toBeCloseTo(((monthStart - t0) / (t1 - t0)) * iw, 3);
    }
  });

  it("uses fewer time ticks on a narrow chart", () => {
    const series = [{ id: "a", label: "A", color: "red", points: [[Date.parse("2025-01-07"), 1], [Date.parse("2025-07-20"), 2]] as [number, number][] }];
    const count = (width: number) => {
      const { container, unmount } = render(<LineChart ariaLabel="t" width={width} height={200} xType="time" series={series} />);
      const n = Array.from(container.querySelectorAll("text")).filter((t) => /^[A-Z][a-z]{2}/.test(t.textContent ?? "")).length;
      unmount();
      return n;
    };
    expect(count(240)).toBeLessThan(count(900));
    expect(count(240)).toBeLessThanOrEqual(3);
  });
});

describe("timeTickLabels", () => {
  it("shows the year on the first month tick and at each January", () => {
    const ticks = ["2025-11-01", "2025-12-01", "2026-01-01", "2026-02-01"].map(Date.parse);
    expect(timeTickLabels(ticks)).toEqual(["Nov ’25", "Dec", "Jan ’26", "Feb"]);
  });

  it("labels day ticks with the day", () => {
    expect(timeTickLabels(["2026-10-05", "2026-10-12"].map(Date.parse))).toEqual(["Oct 5", "Oct 12"]);
  });

  it("labels New Year ticks as years", () => {
    expect(timeTickLabels(["2020-01-01", "2022-01-01", "2024-01-01"].map(Date.parse))).toEqual(["2020", "2022", "2024"]);
  });
});

describe("LineChart readout", () => {
  const curve = (
    <LineChart
      ariaLabel="curve"
      width={400}
      height={200}
      xLabel="Maturity (years)"
      yLabel="Yield (%)"
      yUnit="%"
      series={[
        { id: "today", label: "Today", color: "red", points: [[0.25, 4.26], [2, 4.79], [10, 5.21]] },
        { id: "ago", label: "1Y ago", color: "blue", points: [[0.25, 4.5], [10, 4.4]] },
      ]}
    />
  );

  it("steps through values with the arrow keys and announces them", () => {
    const { container } = render(curve);
    const svg = container.querySelector("svg") as SVGSVGElement;
    expect(svg.getAttribute("tabindex")).toBe("0");
    fireEvent.keyDown(svg, { key: "ArrowRight" });
    const live = container.querySelector("[aria-live]") as HTMLElement;
    expect(live.textContent).toBe("3M: Today 4.26%, 1Y ago 4.50%");
    fireEvent.keyDown(svg, { key: "ArrowRight" });
    // 1Y ago has no 2Y point: read off its drawn line between 3M and 10Y.
    expect(live.textContent).toBe("2Y: Today 4.79%, 1Y ago 4.48%");
    fireEvent.keyDown(svg, { key: "End" });
    expect(live.textContent).toBe("10Y: Today 5.21%, 1Y ago 4.40%");
    expect(container.querySelectorAll(".crosshair circle")).toHaveLength(2);
    fireEvent.keyDown(svg, { key: "Escape" });
    expect(container.querySelector(".crosshair")).toBeNull();
  });

  it("clears the crosshair when the data changes instead of pointing at a new value", () => {
    const series = (pts: [number, number][]) => [{ id: "a", label: "A", color: "red", points: pts }];
    const { container, rerender } = render(
      <LineChart ariaLabel="c" width={400} height={200} series={series([[1, 4], [2, 5], [3, 6]])} />,
    );
    const svg = container.querySelector("svg") as SVGSVGElement;
    fireEvent.keyDown(svg, { key: "End" });
    expect(container.querySelector(".crosshair")).toBeTruthy();
    rerender(<LineChart ariaLabel="c" width={400} height={200} series={series([[1, 7], [2, 8], [3, 9], [4, 10]])} />);
    expect(container.querySelector(".crosshair")).toBeNull();
    expect(container.querySelector("[aria-live]")?.textContent).toBe("");
  });

  it("follows the pointer without announcing it", () => {
    const { container, getByText } = render(curve);
    const svg = container.querySelector("svg") as SVGSVGElement;
    // Plot starts 48px in and spans 336px over 0–10 years: 48 + 336 × 0.2 ≈ the 2Y point.
    fireEvent.pointerMove(svg, { clientX: 48 + 336 * 0.2 });
    expect(getByText("2Y")).toBeInTheDocument();
    expect(container.querySelector("[aria-live]")?.textContent).toBe("");
    fireEvent.pointerLeave(svg);
    expect(container.querySelector(".crosshair")).toBeNull();
  });
});

describe("maturityLabel", () => {
  it("reads maturities as tenors", () => {
    expect(maturityLabel(1 / 12)).toBe("1M");
    expect(maturityLabel(0.5)).toBe("6M");
    expect(maturityLabel(7.5)).toBe("7.5Y");
    expect(maturityLabel(30)).toBe("30Y");
  });
});

describe("LineChart √ maturity axis", () => {
  it("spaces maturities by their square root, with round-number ticks", () => {
    const { container } = render(
      <LineChart
        ariaLabel="curve"
        width={400}
        height={200}
        xLabel="Maturity (years)"
        xScale="sqrt"
        series={[{ id: "c", label: "C", color: "red", points: [[1 / 12, 4.1], [2, 4.8], [30, 5.6]] }]}
      />,
    );
    const iw = 400 - 48 - 16;
    // x ticks are the groups holding a 6px tick mark (y ticks hold a full-width grid line).
    const ticks = Array.from(container.querySelectorAll("g[transform]"))
      .filter((g) => g.querySelector(":scope > line")?.getAttribute("y2") === "6")
      .map((g) => ({ tf: g.getAttribute("transform") ?? "", label: g.textContent ?? "" }));
    expect(ticks.map((t) => t.label)).toEqual(["0", "1", "2", "5", "10", "20", "30"]);
    const xOf = (label: string) => Number(/translate\(([\d.]+)/.exec(ticks.find((t) => t.label === label)!.tf)?.[1]);
    expect(xOf("1")).toBeCloseTo(Math.sqrt(1 / 30) * iw, 6);
    expect(xOf("30")).toBeCloseTo(iw, 6);
    expect(container.textContent).toContain("Maturity (years) · √ scale");
  });
});

