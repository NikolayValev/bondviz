import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { LineChart } from "@/components/charts/LineChart";

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
      .filter((t) => /^[A-Z][a-z]{2} \d{2}$/.test(t.label));
    expect(ticks.length).toBeGreaterThan(2);
    for (const { tf, label } of ticks) {
      const x = Number(/translate\(([-\d.]+)/.exec(tf)?.[1]);
      const monthStart = Date.UTC(2000 + Number(label.slice(4)), months.indexOf(label.slice(0, 3)), 1);
      expect(x).toBeCloseTo(((monthStart - t0) / (t1 - t0)) * iw, 3);
    }
  });
});
