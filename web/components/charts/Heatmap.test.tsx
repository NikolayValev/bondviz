import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { Heatmap, colorRamp } from "@/components/charts/Heatmap";

describe("colorRamp", () => {
  it("maps 0 and 1 to the ramp endpoints", () => {
    expect(colorRamp(0)).toBe("rgb(13, 27, 42)");
    expect(colorRamp(1)).toBe("rgb(0, 214, 143)");
  });
  it("clamps out-of-range input", () => {
    expect(colorRamp(-5)).toBe("rgb(13, 27, 42)");
    expect(colorRamp(5)).toBe("rgb(0, 214, 143)");
  });
});

describe("Heatmap", () => {
  it("renders one cell per (date, tenor) when given explicit dimensions", () => {
    const { container } = render(
      <Heatmap
        ariaLabel="test heatmap"
        width={300}
        height={120}
        dates={["2025-01-02", "2025-01-03"]}
        tenors={["3M", "2Y", "10Y"]}
        values={[
          [5.0, 4.0, 4.5],
          [5.1, 4.1, 4.6],
        ]}
      />,
    );
    expect(container.querySelectorAll("rect.heatmap-cell")).toHaveLength(6);
    expect(container.querySelector("svg")?.getAttribute("aria-label")).toBe("test heatmap");
  });
});
