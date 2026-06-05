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
  it("treats non-finite input as 0", () => {
    expect(colorRamp(NaN)).toBe("rgb(13, 27, 42)");
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

  it("renders all-null rows as grid-colored cells without crashing", () => {
    const { container } = render(
      <Heatmap
        ariaLabel="null heatmap"
        width={200}
        height={100}
        dates={["2025-01-02"]}
        tenors={["3M", "2Y"]}
        values={[[null, null]]}
      />,
    );
    const cells = container.querySelectorAll("rect.heatmap-cell");
    expect(cells).toHaveLength(2);
    cells.forEach((c) => expect(c.getAttribute("fill")).toBe("var(--grid)"));
  });

  it("renders all-equal values without dividing by zero", () => {
    const { container } = render(
      <Heatmap
        ariaLabel="flat heatmap"
        width={200}
        height={100}
        dates={["2025-01-02", "2025-01-03"]}
        tenors={["3M", "2Y"]}
        values={[
          [4, 4],
          [4, 4],
        ]}
      />,
    );
    const cells = container.querySelectorAll("rect.heatmap-cell");
    expect(cells).toHaveLength(4);
    const fills = new Set(Array.from(cells).map((c) => c.getAttribute("fill")));
    expect(fills.size).toBe(1); // all the same color
  });

  it("handles a single date × single tenor", () => {
    const { container } = render(
      <Heatmap
        ariaLabel="single cell heatmap"
        width={120}
        height={80}
        dates={["2025-01-02"]}
        tenors={["10Y"]}
        values={[[4.5]]}
      />,
    );
    expect(container.querySelectorAll("rect.heatmap-cell")).toHaveLength(1);
  });
});
