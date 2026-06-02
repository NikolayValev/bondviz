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
});
