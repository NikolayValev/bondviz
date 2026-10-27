import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { LineChart, timeTickLabels } from "@/components/charts/LineChart";

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
